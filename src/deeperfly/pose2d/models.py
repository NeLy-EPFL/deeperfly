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
from pathlib import Path

#: DeepFly2D subtracts this scalar from the ``[0, 1]`` image.
DEFAULT_MEAN = 0.22
#: DeepFly2D network input ``(height, width)``.
DEFAULT_INPUT_SIZE = (256, 512)
#: DeepFly2D body-side detector channels.
DEFAULT_N_OUT_CHANNELS = 19


@dataclass(frozen=True)
class ModelSpec:
    """A torch-free description of one detector model (parsed from ``[[pose2d.models]]``).

    Attributes
    ----------
    name
        The model's name, referenced by a pathway's ``model`` key.
    cls
        The registry key selecting the model class (e.g. ``"hourglass"``).
    weights
        Path to a checkpoint, or ``None`` to use the auto-provisioned cache.
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
    mean: float = DEFAULT_MEAN
    n_out_channels: int = DEFAULT_N_OUT_CHANNELS
    precision: str | None = None
    kwargs: dict = field(default_factory=dict)


def _load_hourglass(spec: "ModelSpec"):
    """Load the DeepFly2D stacked-hourglass detector from a ``.pth`` (or the cache)."""
    from . import detector
    from .download import download_torch_weights

    weights = spec.weights
    if weights is not None and not Path(weights).exists():
        raise SystemExit(
            f"no detector checkpoint at {weights}. Remove the model's 'weights' "
            "to use the auto-provisioned cache, or point it at a valid .pth."
        )
    path = weights or download_torch_weights()
    return detector.load_detector(path, **spec.kwargs)


def _load_hrnet(spec: "ModelSpec"):
    """Load the dense-38 HRNet detector (see :mod:`deeperfly.pose2d.hrnet`).

    Unlike the hourglass there is no auto-provisioned cache: this network is trained
    per-project, so its ``weights`` path is required. ``spec.mean`` is passed through
    to be REFUSED unless it is 0.0 -- the checkpoint carries its own normalization.
    """
    from . import hrnet

    if not spec.weights:
        raise SystemExit(
            "a dense-38 hrnet model needs an explicit 'weights' path in its "
            "[[pose2d.models]] table -- there is no auto-provisioned cache for it"
        )
    return hrnet.load_hrnet(spec.weights, mean=spec.mean, **spec.kwargs)


def _load_mvt(spec: "ModelSpec"):
    """Load the multiview transformer (see :mod:`deeperfly.pose2d.mvt`).

    Like the dense HRNet there is no auto-provisioned cache yet, so ``weights`` must name
    an exported artifact -- and it must be an *exported* one: a raw Lightning ``.ckpt``
    carries no point names, so nothing could check the channel order it is about to be
    routed through. ``spec.mean`` and ``spec.precision`` are passed through to be REFUSED
    unless they are 0.0 and float32.
    """
    from . import mvt

    if not spec.weights:
        raise SystemExit(
            "a multiview-transformer model needs an explicit 'weights' path in its "
            "[[pose2d.models]] table, pointing at an artifact written by dfpose's "
            "scripts/export_mvt_weights.py"
        )
    return mvt.load_mvt(
        spec.weights, mean=spec.mean, precision=spec.precision, **spec.kwargs
    )


#: ``class`` name -> loader(spec) -> torch module. New detector architectures register
#: here; ``"deepfly2d"`` is an alias for ``"hourglass"``.
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
    "hourglass": _load_hourglass,
    "deepfly2d": _load_hourglass,
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
    def peak_convention(self) -> str:
        """How this model's normalized peaks map back through a resize.

        ``"half-pixel"`` (the default, and what the hourglass and the dense HRNet want) is
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
        """``(..., H, W, 3)`` frame(s) -> ``(..., 3, H_out, W_out)`` normalized model input.

        Accepts a NumPy array or an on-device torch tensor and keeps it on its
        device (a GPU-decoded frame is resized and normalized on the GPU). The
        resize is bilinear + anti-aliased to match the original DeepFly2D
        skimage resize closely; ``mean`` is subtracted last.
        """
        import torch
        import torch.nn.functional as F

        from .inference import _to_torch_image

        impl = self._impl()
        if impl is not None and getattr(self.module, "owns_prepare", False):
            # The model reproduces its own training pipeline. The shared path below is a
            # resize kernel and a scalar mean; a model trained through a different kernel
            # cannot be served by "an equivalent" one -- see mvt.prepare_images.
            return impl.prepare_images(
                frames,
                self.input_size,
                self.module.norm_mean,
                self.module.norm_std,
                next(self.module.parameters()).device,
            )

        img = _to_torch_image(frames)
        img = img.float() / 255.0 if not torch.is_floating_point(img) else img.float()
        if img.ndim == 2:  # a single grayscale frame -> 3 channels
            img = img.unsqueeze(-1).expand(-1, -1, 3)
        img = img[..., :3]
        chw = img.movedim(-1, -3).contiguous()  # (..., 3, H, W)
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

        A model whose heatmap does not span its input -- the dense-38 HRNet pads the
        field by 25% a side so an off-frame joint still has a cell -- cannot use the
        shared decode, whose normalization assumes it does. Such a module declares
        ``owns_decode`` and is asked for the points itself. The returned coordinates are
        still input-normalized, so a pathway inverts them exactly as before; they may
        fall outside ``[0, 1]``, which is an off-frame joint and not an error.
        """
        from . import detector

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
        return detector.predict_points(
            self.module, inputs, method=method, radius=radius
        )

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
        from . import detector

        detector.set_precision(self.module, precision)

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
    return LoadedModel(spec, module)
