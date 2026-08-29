"""The detector-model registry: a named model class + weights + input contract.

A pose pipeline can drive several detector models (see
:mod:`deeperfly.pose2d.pathways`): each is described in the config by a
``class`` (a registry key here), the ``weights`` to load, the ``input_size`` it
expects, the ``mean`` it subtracts, and the ``n_out_channels`` it emits. A
:class:`ModelSpec` is the parsed, torch-free description; :func:`load_model`
turns it into a :class:`LoadedModel` that owns the model-specific input
preparation (resize to ``input_size`` + normalize) and forward/decode, wrapping
the torch-free seam in :mod:`deeperfly.pose2d.detector`.

The geometry of preparing an input -- a left-right mirror, a crop -- belongs to
the *pathway* (:class:`~deeperfly.preprocessing.FrameTransform`), not here; a
model only ever sees an already-oriented frame and resizes/normalizes it to its
own input contract. Keeping the resize here (anti-aliased, as the original
DeepFly2D did) means a pathway's mirror/crop never perturbs the pixels a model
is trained on -- only the coordinate decode is mapped back through the pathway.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The input ``(height, width)`` every shipped detector is trained at. A class may still
#: be asked for another size, and its loader refuses if the checkpoint disagrees.
DEFAULT_INPUT_SIZE = (256, 512)

#: Every ``class`` spelling -> its canonical name, so aliases share one set of defaults.
CLASS_ALIASES: dict[str, str] = {
    "hrnet": "hrnet",
    "hrnet_timm": "hrnet",
    "mvt": "mvt",
    "multiview_transformer": "mvt",
}

#: What a ``[[pose2d.models]]`` table may leave out, per class.
#:
#: These are not preferences: each is a property of the network that its own loader
#: already **refuses to run against a disagreeing config**. The dense classes carry their
#: normalization in the checkpoint and reject any ``mean`` but 0.0; the multiview
#: transformer rejects any ``precision`` but float32. Writing them in the config was the
#: config restating the artifact under threat of rejection -- information nowhere, friction
#: everywhere -- so the class states them and a table only speaks up to *disagree*.
#:
#: ``n_out_channels = None`` means "as many channels as the skeleton has points", which is
#: what DENSE means -- and it is what every shipped detector is. It is resolved against the
#: skeleton in :func:`class_defaults`, so a config stops restating its own skeleton's size
#: (and cannot get it wrong).
CLASS_DEFAULTS: dict[str, dict] = {
    "hrnet": {"mean": 0.0, "n_out_channels": None},
    "mvt": {"mean": 0.0, "n_out_channels": None, "precision": "float32"},
}


def class_defaults(cls: str, n_points: int | None = None) -> dict:
    """The defaults for model class ``cls``, with ``n_out_channels`` resolved.

    Parameters
    ----------
    cls
        The ``class`` key from a ``[[pose2d.models]]`` table (an alias is fine).
    n_points
        The skeleton's point count. Every shipped class is dense, so this is what its
        channel count resolves to; ``None`` leaves it unresolved for a caller with no
        skeleton in hand (``deeperfly config show``), which never loads a model.

    Returns
    -------
    dict
        ``input_size`` / ``mean`` / ``n_out_channels`` / ``precision`` for the class.

    Raises
    ------
    ValueError
        On a class this build has no loader for. Refused HERE rather than left to fall
        back, because a fallback's defaults are another network's: a typo'd class used to
        inherit DeepFly2D's 19 channels and 0.22 mean and then fail at load with a
        channel-count mismatch, which says nothing about the word that was wrong.
    """
    canonical = CLASS_ALIASES.get(cls)
    if canonical is None:
        raise ValueError(
            f"unknown detector class {cls!r}; this build has "
            f"{sorted(set(CLASS_ALIASES))}"
        )
    out = {
        "input_size": DEFAULT_INPUT_SIZE,
        "mean": 0.0,
        "n_out_channels": None,
        "precision": None,
        **CLASS_DEFAULTS[canonical],
    }
    if out["n_out_channels"] is None and n_points is not None:
        out["n_out_channels"] = n_points
    return out


@dataclass(frozen=True)
class ModelSpec:
    """A torch-free description of one detector model (parsed from ``[[pose2d.models]]``).

    Attributes
    ----------
    name
        The model's name, referenced by a pathway's ``model`` key.
    cls
        The registry key selecting the model class (e.g. ``"mvt"``).
    weights
        Path to a checkpoint. ``None`` is refused at load: nothing auto-provisions.
    input_size
        The network input ``(height, width)``.
    mean
        Scalar subtracted from the ``[0, 1]`` image after the resize.
    n_out_channels
        Number of output heatmap channels (validated against the weights).
    precision
        Per-model forward precision override (``float32``/``float16``/``bfloat16``),
        or ``None`` to inherit the ``[pose2d].precision`` default. Precision is a
        property of running a specific network -- an eager net may tolerate fp16
        autocast while a traced/lite export is pinned to a dtype -- so it can be set
        per model; the fallback keeps the common single-precision config a one-liner.
    kwargs
        Extra class-specific construction kwargs.
    """

    name: str
    cls: str
    weights: str | None
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE
    mean: float = 0.0
    n_out_channels: int | None = None
    precision: str | None = None
    kwargs: dict = field(default_factory=dict)


def _load_hrnet(spec: "ModelSpec"):
    """Load the dense-38 HRNet detector (see :mod:`deeperfly.pose2d.hrnet`).

    There is no auto-provisioned cache: this network is trained per project, so
    ``weights`` is required -- as a bare filename on
    ``$DEEPERFLY_MODELS`` or an outright path (see
    :func:`~deeperfly.pose2d.download.resolve_weights`). ``spec.mean`` is passed through
    to be REFUSED unless it is 0.0 -- the checkpoint carries its own normalization.
    """
    from . import hrnet
    from .download import missing_weights, resolve_weights

    path = resolve_weights(spec.weights, cls=spec.cls, model_name=spec.name)
    if path is None:
        raise missing_weights(spec.cls, spec.name)
    return hrnet.load_hrnet(path, mean=spec.mean, **spec.kwargs)


def _load_mvt(spec: "ModelSpec"):
    """Load the multiview transformer (see :mod:`deeperfly.pose2d.mvt`).

    Like the dense HRNet there is no auto-provisioned cache, so ``weights`` must name an
    exported artifact -- and it must be an *exported* one: a raw Lightning ``.ckpt``
    carries no point names, so nothing could check the channel order it is about to be
    routed through. ``spec.mean`` and ``spec.precision`` are passed through to be REFUSED
    unless they are 0.0 and float32.
    """
    from . import mvt
    from .download import missing_weights, resolve_weights

    path = resolve_weights(spec.weights, cls=spec.cls, model_name=spec.name)
    if path is None:
        raise missing_weights(spec.cls, spec.name)
    return mvt.load_mvt(path, mean=spec.mean, precision=spec.precision, **spec.kwargs)


#: ``class`` name -> loader(spec) -> torch module. New detector architectures register here.
#:
#: ``"hrnet"`` is the DENSE-38 detector: every tracked point in every view, so a camera
#: needs one pathway rather than a pathway and a mirrored twin, and a contralateral
#: point gets a prediction instead of a ``NaN``. Its heatmap is padded and its decode is
#: its own, so it also owns ``predict_points`` -- see :class:`LoadedModel`.
#:
#: ``"mvt"`` is the MULTIVIEW transformer: also 38 channels in every view, but the views
#: of a frame are encoded TOGETHER, so a joint only one camera can see informs the ones
#: that cannot. It is the first class here that is not a per-view function, and it owns
#: its input preparation as well as its decode.
MODEL_CLASSES = {
    "hrnet": _load_hrnet,
    "hrnet_timm": _load_hrnet,
    "mvt": _load_mvt,
    "multiview_transformer": _load_mvt,
}


class LoadedModel:
    """A loaded detector model plus its input contract (resize + normalize).

    Wraps the torch module behind :mod:`deeperfly.pose2d.detector` so the
    orchestration in :mod:`deeperfly.pose2d.inference` stays torch-free. The
    image preparation (resize to :attr:`input_size`, subtract :attr:`mean`,
    CHW) is the model's own; a pathway hands it an already-oriented frame.
    """

    def __init__(self, spec: ModelSpec, module):
        self.spec = spec
        self.module = module

    @property
    def input_size(self) -> tuple[int, int]:
        return self.spec.input_size

    @property
    def n_out_channels(self) -> int:
        return self.spec.n_out_channels

    @property
    def joint_views(self) -> bool:
        """Whether this model's ``V`` axis is COUPLED -- views computed together, not apart.

        ``False`` for a per-view detector (the dense HRNet): its ``V`` axis
        is just more batch, so a caller may put anything there -- other cameras, other
        candidate crops of one camera -- and the results are unchanged. ``True`` for the
        multiview transformer, where attention runs across views: a view's output depends
        on which other views were in the tensor, so the axis carries meaning and cannot be
        borrowed. :mod:`deeperfly.pose2d.autocrop` reads this to decide whether one probe
        is one image or one whole moment.
        """
        return bool(getattr(self.module, "joint_views", False))

    @property
    def padded_field(self) -> bool:
        """Whether a peak may legitimately land OUTSIDE the model input.

        ``True`` when the field covers more than the reported frame, so a joint the crop
        cuts off still has a cell and decodes to a coordinate beyond ``[0, 1]``. Both shipped
        classes are: the dense HRNet pads the head's output by 25% a side, and an ``mvt``
        artifact declaring ``arch.hm_margin_px`` pads the network's INPUT instead.

        ``False`` when the field spans the reported frame exactly -- an ``mvt`` artifact with
        no margin, which is every one through r27. There a cut-off joint has nowhere to go:
        the soft-argmax is an expectation over in-image pixels, so it piles up *against* the
        border instead.

        The distinction matters to anything asking "does this crop cut the animal?" --
        :mod:`deeperfly.pose2d.autocrop` needs a different test in each case, and reading
        the wrong one makes a clipping box look clean.
        """
        return bool(getattr(self.module, "padded_field", False))

    @property
    def accepts_gray(self) -> bool:
        """Whether this model is happy with ONE-channel frames.

        ``True`` only for a model whose own preparation turns the frame grayscale anyway --
        then decoding color-free footage as three identical channels is work done twice, and
        the decoder can hand over the luma plane it already has. ``False`` (the default) for
        a model that reads three channels, and for any model whose relationship to color is
        merely unstated: a detector trained through a color image must keep seeing one.

        Read by :func:`deeperfly.pose2d.stream.detect_2d`, which grants the decoder
        permission only when EVERY model of the plan says yes.
        """
        return bool(getattr(self.module, "accepts_gray", False))

    @property
    def prepares_on_host(self) -> bool:
        """Whether this model's own preparation runs on the CPU, not the device.

        A model that reproduces a host training pipeline (PIL, cv2 -- see
        :func:`deeperfly.pose2d.mvt.prepare_images`) copies any device tensor it is handed
        straight back down. Uploading the raw window for it is then pure round trip: the
        frames go up, come back, and the only thing that needs to be on the device is the
        small normalized input the preparation returns.
        :func:`deeperfly.pose2d.inference.detect_sequence` reads this to leave such a
        source's window on the host.
        """
        return bool(
            getattr(self.module, "owns_prepare", False)
            and getattr(self.module, "prepares_on_host", False)
        )

    @property
    def peak_convention(self) -> str:
        """How this model's normalized peaks map back through a resize.

        ``"half-pixel"`` (the default, and what the dense HRNet wants) is
        ``x' = (x + 0.5) * s - 0.5``, the geometrically correct inverse of a cv2/torch
        resize to a target size, and the convention dfpose's exporter wrote their labels
        with. A model trained on labels written as a pure scale declares ``"pure-scale"``
        instead -- the difference is ``0.5 * (source_w / model_w - 1)``, about half a
        footage pixel on this rig, which is a uniform skeleton shift and not noise.
        """
        return str(getattr(self.module, "peak_convention", "half-pixel"))

    def _impl(self):
        """The module implementing this model's own prepare/forward/decode, or None.

        A class whose input preparation or readout is its own attaches itself here at load
        time. The older ``owns_decode`` flag below predates this and means "the dense
        HRNet"; it is left alone rather than migrated, because the dense path is in
        production and a mechanical refactor of it buys nothing.
        """
        return getattr(self.module, "impl", None)

    def prepare(self, frames):
        """``(..., H, W[, C])`` frame(s) -> ``(..., 1, H_out, W_out)`` normalized input.

        Accepts a NumPy array or an on-device torch tensor and keeps it on its device (a
        GPU-decoded frame is resized and normalized on the GPU). The resize is bilinear +
        anti-aliased; ``mean`` is subtracted last.

        **One plane.** The corpus is monochrome, so every shipped detector is trained on a
        single channel, and a frame arrives either as ``(..., H, W)`` from the decoder's
        gray fast path or as ``(..., H, W, C)`` whose planes are identical -- the decoder
        only takes the gray path when they are. So the first channel is the image; nothing
        is replicated to three to feed a network that would only take one back.
        """
        import torch
        import torch.nn.functional as F

        from .inference import _to_torch_image

        impl = self._impl()
        if impl is not None and getattr(self.module, "owns_prepare", False):
            # The model reproduces its own training pipeline. The shared path below is a
            # resize kernel and a scalar mean; a model trained through a different kernel
            # cannot be served by "an equivalent" one -- see mvt.prepare_images.
            kw = {}
            margin = int(getattr(self.module, "hm_margin_px", 0) or 0)
            if margin:
                import inspect

                # Asked of the signature, as `predict_points_for_views` does: a module whose
                # preparation predates the margin would otherwise take it as a positional
                # surprise, and silently preparing a padded checkpoint's input without the
                # pad is a shift of `margin` px on every point rather than an error.
                try:
                    takes = (
                        "margin" in inspect.signature(impl.prepare_images).parameters
                    )
                except (TypeError, ValueError):
                    takes = False
                if not takes:
                    raise TypeError(
                        f"model {self.spec.name!r} declares hm_margin_px={margin} but "
                        f"{impl.__name__}.prepare_images takes no margin; its input would "
                        "be prepared without the pad its field expects"
                    )
                kw["margin"] = margin
            return impl.prepare_images(
                frames,
                self.input_size,
                self.module.norm_mean,
                self.module.norm_std,
                next(self.module.parameters()).device,
                **kw,
            )

        img = _to_torch_image(frames)
        img = img.float() / 255.0 if not torch.is_floating_point(img) else img.float()
        # One plane in, one plane out. A decoded frame arrives either as (..., H, W) from
        # the gray fast path or as (..., H, W, C) with the planes identical (the decoder
        # only takes the gray path when they are -- see `io.video._frame_array`), so the
        # first channel IS the image and nothing is expanded to feed it.
        if img.ndim >= 3 and img.shape[-1] in (1, 3, 4):
            img = img[..., :1]
        else:
            img = img.unsqueeze(-1)
        chw = img.movedim(-1, -3).contiguous()  # (..., 1, H, W)
        lead = chw.shape[:-3]
        flat = chw.reshape(-1, *chw.shape[-3:])
        resized = F.interpolate(
            flat,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        out = resized.reshape(*lead, *resized.shape[-3:])
        return out - self.spec.mean

    def predict_points(self, inputs, *, method: str = "weighted", radius: int = 2):
        """Fused forward + decode for ``(B, V, 3, H, W)`` input.

        Returns normalized ``(B, V, C_out, 2)`` peaks and ``(B, V, C_out)`` conf (plain 4D
        ``(N, 3, H, W)`` input gives ``(N, C_out, 2)`` / ``(N, C_out)``).

        Every shipped detector owns its own decode, and for the same reason in two
        flavours: the dense HRNet pads its heatmap field 25% a side so an off-frame joint
        still has a cell, and the multiview transformer's soft-argmax cannot leave its
        module. So there is no shared decode left to fall back to -- a class that declares
        neither is a class this build cannot read points from, and says so rather than
        guessing a normalization.

        The returned coordinates are input-normalized, so a pathway inverts them exactly
        as before; they may fall outside ``[0, 1]``, which is an off-frame joint and not
        an error.
        """
        impl = self._impl()
        if impl is not None:
            return impl.predict_points(
                self.module, inputs, method=method, radius=radius
            )
        if getattr(self.module, "owns_decode", False):
            from . import hrnet

            return hrnet.predict_points(
                self.module, inputs, method=method, radius=radius
            )
        raise TypeError(
            f"model {self.spec.name!r} (class {self.spec.cls!r}) neither owns its decode "
            "nor provides one; a heatmap cannot be decoded without knowing whether its "
            "field spans the input"
        )

    def predict_points_for_views(
        self, inputs, views, *, method: str = "weighted", radius: int = 2
    ):
        """:meth:`predict_points`, but decoding only ``views`` of the ``V`` axis.

        Returns the same ``(B, len(views), C_out, 2)`` / ``(B, len(views), C_out)`` a caller
        would get by slicing the full result -- the *forward* is untouched, so a joint-view
        model's cross-view attention still sees every view. Only the readout is narrowed.

        The point is cost, and it is only worth anything for a joint-view model: the
        multiview transformer's decode upsamples every channel to the full input, which for
        eight views costs about as much as the network itself, and a caller after one camera
        pays that eight times over. A model that cannot narrow its decode is asked for
        everything and sliced here, so this is always correct and merely sometimes faster.
        """
        impl = self._impl()
        if impl is not None and getattr(self.module, "joint_views", False):
            import inspect

            # Asked of the signature rather than by catching TypeError: a genuine TypeError
            # from inside the decode would otherwise be swallowed into a silent slow path.
            try:
                offers = "views" in inspect.signature(impl.predict_points).parameters
            except (TypeError, ValueError):  # a builtin/C callable has no signature
                offers = False
            if offers:
                return impl.predict_points(
                    self.module, inputs, method=method, radius=radius, views=views
                )
        xy, conf = self.predict_points(inputs, method=method, radius=radius)
        idx = list(views)
        return xy[:, idx], conf[:, idx]

    def cells_to_normalized(self, cells, field_hw):
        """Sub-pixel FIELD cells ``(..., 2)`` -> input-normalized ``(x, y)``, NumPy.

        The candidate (top-K) path's counterpart to :meth:`predict_points`: it keeps several
        peaks per channel, so it holds cells rather than one decoded point and has to place
        them itself. **A model gets to say how**, because the shared
        ``(c + 0.5) / W_field`` convention is only right when the field spans the reported
        frame. Both shipped classes pad it -- the dense HRNet by 25% a side of its head's
        output, the multiview transformer by padding the network's INPUT -- and for a padded
        field that convention is wrong by the margin AND by the ``(w + 2m) / w`` scale. On
        the r28 multiview transformer that is 46 model px at the frame's edge, which is
        three times what a candidate is allowed to sit from its hypothesis
        (:data:`deeperfly.pipeline.pictorial.DEFAULT_INLIER_PX`), so every edge candidate would be
        silently discarded. This dispatch is what the candidate path used to refuse rather
        than guess.

        ``field_hw`` is the heatmap's own ``(H_out, W_out)``, used only by the fallback --
        a model owning its decode knows its field's geometry without being told.

        Coordinates outside ``[0, 1]`` are meaningful and must not be clipped: on a padded
        field they are a joint outside the reported frame (see :attr:`padded_field`).
        """
        impl = self._impl()
        fn = getattr(impl, "cells_to_input_normalized", None) if impl else None
        if fn is None and getattr(self.module, "owns_decode", False):
            from . import hrnet

            fn = hrnet.cells_to_input_normalized_np
        if fn is not None:
            return fn(self.module, cells)
        # The shared half-pixel cell-center convention: the matched inverse of
        # `inference.heatmap_to_points`, and correct exactly when the field spans the
        # reported frame.
        import numpy as np

        cells = np.asarray(cells, dtype=float)
        hh, ww = int(field_hw[0]), int(field_hw[1])
        return np.stack(
            [(cells[..., 0] + 0.5) / ww, (cells[..., 1] + 0.5) / hh], axis=-1
        )

    def points_from_heatmaps(
        self, heatmaps, *, method: str = "weighted", radius: int = 2
    ):
        """Already-computed heatmaps -> the points :meth:`predict_points` would return.

        So the candidate path reports the SAME arg-max as production instead of a second
        opinion decoded a different way -- which for the multiview transformer it genuinely
        would be, its readout being a global soft-argmax over the upsampled field where the
        shared decode takes a windowed centroid over the raw cells.
        """
        impl = self._impl()
        fn = getattr(impl, "points_from_heatmaps", None) if impl else None
        if fn is None and getattr(self.module, "owns_decode", False):
            from . import hrnet

            fn = hrnet.points_from_heatmaps
        if fn is not None:
            return fn(self.module, heatmaps)
        from .inference import heatmap_to_points

        return heatmap_to_points(heatmaps, method=method, radius=radius)

    def predict_points_and_heatmaps(self, inputs):
        """One forward -> ``(points, conf, heatmaps)`` for the candidate path.

        Fused where the class offers it, because the alternative is forwarding the network
        twice (or decoding a host copy of a field the multiview transformer upsamples 16x,
        which costs more than the forward). Otherwise the same two calls, in order.
        """
        impl = self._impl()
        fn = getattr(impl, "predict_points_and_heatmaps", None) if impl else None
        if fn is not None:
            return fn(self.module, inputs)
        hm = self.predict_heatmaps(inputs)
        xy, conf = self.points_from_heatmaps(hm)
        return xy, conf, hm

    def predict_heatmaps(self, inputs):
        """Final-stack heatmaps ``(B, V, C_out, H_out, W_out)`` (host NumPy) for the candidate path."""
        from . import detector

        impl = self._impl()
        if impl is not None:
            return impl.predict_heatmaps(self.module, inputs)
        if getattr(self.module, "owns_decode", False):
            from . import hrnet

            return hrnet.predict_heatmaps(self.module, inputs)
        return detector.predict_heatmaps(self.module, inputs)

    def set_precision(self, precision: str) -> None:
        """Set the forward precision (``float32``/``float16``/``bfloat16``)."""
        from .runtime import set_precision

        set_precision(self.module, precision)

    def device(self) -> str:
        """The device the model's parameters live on."""
        from . import detector

        return detector.detector_device(self.module)


def load_model(spec: ModelSpec) -> LoadedModel:
    """Build a :class:`LoadedModel` from a :class:`ModelSpec`.

    Looks up ``spec.cls`` in :data:`MODEL_CLASSES`, loads the weights, and
    validates that the loaded module emits ``spec.n_out_channels`` heatmaps.

    Raises
    ------
    SystemExit
        If ``spec.cls`` is not a known model class, or an explicit ``weights``
        path does not exist.
    ValueError
        If the loaded module's output-channel count disagrees with
        ``spec.n_out_channels``.
    """
    loader = MODEL_CLASSES.get(spec.cls)
    if loader is None:
        raise SystemExit(
            f"unknown model class {spec.cls!r} for model {spec.name!r}; "
            f"known classes: {sorted(MODEL_CLASSES)}"
        )
    module = loader(spec)
    num_classes = getattr(module, "num_classes", None)
    if num_classes is not None and int(num_classes) != int(spec.n_out_channels):
        raise ValueError(
            f"model {spec.name!r} declares n_out_channels={spec.n_out_channels} "
            f"but its weights emit {num_classes} channels"
        )
    # An artifact that records the input size it was trained at gets to contradict the
    # config: the resize is what puts the animal at the scale the network learned, so a
    # disagreement here is a silently mis-scaled fly rather than a crash.
    input_hw = getattr(module, "input_hw", None)
    if input_hw is not None and tuple(int(v) for v in input_hw) != tuple(
        spec.input_size
    ):
        raise ValueError(
            f"model {spec.name!r} declares input_size={list(spec.input_size)} but its "
            f"weights were trained at {list(input_hw)}. Drop 'input_size' from its "
            "[[pose2d.models]] table to take the checkpoint's."
        )
    return LoadedModel(spec, module)
