"""The detection plan: one detector, run once per camera.

Detection is **dense and one-to-one**. One detector emits every skeleton point for the
camera it is given, so a camera IS a source IS a pathway IS a view, and only the camera is
named: the plan is *synthesized* from the camera table rather than declared.

The four counts are still kept as separate structures, because everything downstream reads
them -- ``pts2d`` assembly walks a pathway's mapping and the fingerprint digests it -- but
under v2 they are all the same list:

- **sources** -- one per camera, its ``video`` pattern (see
  :meth:`deeperfly.config.Config.source_patterns`).
- **views** -- the geometric cameras (``[cameras.*]``); the ``V`` axis of the
  ``(V, T, P, 2)`` points array.
- **preprocessors** -- one per camera: its detection window (``[pose2d.crops]``, or a
  searched one), or the identity.
- **models** -- one, from ``[pose2d] class`` / ``weights``.
- **pathways** -- one per camera, mapping channel ``i`` -> point ``i`` of that camera.

The mapping is always the identity and is still materialized, because ``pts2d`` assembly
and the fingerprint read it. A point predicted in a camera's cropped frame is mapped back
into raw footage pixels by inverting the window -- see
:func:`normalized_peaks_to_original_pixels` -- so a camera's intrinsics go on describing
the raw frame and a detection window never moves the principal point.

What is gone with the declaration: ``[[sources]]``, ``[[pose2d.preprocessors]]``,
``[[pose2d.models]]``, ``[[pose2d.pathways]]``, and ``[pose2d.output_points]``'s 38 x V
rows of channel -> point. So is the mirrored twin (one source detected twice, once
flipped) and with it ``check_mirror_consistency``: with no mirrored pathway there is
nothing to be inconsistent. A 19-channel side-agnostic checkpoint is not expressible and
runs under a v1 tag, not a resurrected code path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int

from ..preprocessing import FrameTransform, Resize
from .models import ModelSpec

log = logging.getLogger("deeperfly")


@dataclass(frozen=True)
class Source:
    """One camera's footage: its name and its ``video`` pattern (or patterns)."""

    name: str
    pattern: str | list[str]


@dataclass(frozen=True)
class Pathway:
    """One camera's inference run: its footage through its window into its own view.

    Attributes
    ----------
    name
        The camera's name -- which is also its source's, its preprocessor's and its
        view's, because under v2 they are one thing.
    source, preprocessor, model
        ``source`` is the camera name. ``preprocessor`` is the camera name when it has a
        detection window and ``None`` when it detects on the full frame (``transform`` is
        then the identity). ``model`` is the sole detector.
    transform
        The detection window, as a :class:`~deeperfly.preprocessing.FrameTransform`; the
        identity for a full-frame camera.
    mapping
        An ``(E, 3)`` int array of ``(i, v, p)`` triples: channel ``i`` -> point ``p`` of
        view ``v``. Always the identity into this camera's own view -- materialized
        because ``pts2d`` assembly and the fingerprint read it.
    """

    name: str
    source: str
    preprocessor: str | None
    model: str
    transform: FrameTransform
    mapping: Int[np.ndarray, "E 3"]


def normalized_peaks_to_original_pixels(
    points_norm: Float[np.ndarray, "*lead 2"],
    transform: FrameTransform,
    model_input_hw: tuple[int, int],
    source_size: tuple[int, int],
    peak_convention: str = "half-pixel",
) -> Float[np.ndarray, "*lead 2"]:
    """Map model peaks (normalized ``[0, 1]``) back into the source/view frame.

    Inverts the pathway's geometry: normalized model coords -> model-input
    pixels -> (undo the model's resize) -> preprocessed-frame pixels -> (undo
    the preprocessor, e.g. a mirror) -> raw source pixels, which is the frame
    the view's intrinsics describe.

    Parameters
    ----------
    points_norm
        Peaks of shape ``(..., 2)`` normalized to ``[0, 1]`` of the model input.
    transform
        The pathway's preprocessor.
    model_input_hw
        The model input ``(height, width)``.
    source_size
        The raw source frame ``(height, width)`` the preprocessor is anchored on.
    peak_convention
        How to undo the model's resize. ``"half-pixel"`` uses
        ``x' = (x + 0.5) * s - 0.5``, the correct inverse of a cv2/torch resize and what
        every detector trained on dfpose-written labels needs. ``"pure-scale"`` omits the
        half-pixel term, for a model whose labels were written as ``x * s`` -- the
        multiview transformer, whose labels came from Lightning Pose. The two differ by
        ``0.5 * (source/model - 1)``, roughly half a footage pixel here: small, uniform,
        and indistinguishable by eye from a calibration error.

    Returns
    -------
    np.ndarray
        Peaks of shape ``(..., 2)`` in raw source (view) pixels.
    """
    h_in, w_in = model_input_hw
    model_px = np.asarray(points_norm, dtype=float) * np.array([w_in, h_in])
    prep_size = transform.output_size(source_size)  # (H', W') after the preprocessor
    if peak_convention == "pure-scale":
        # A pure scale, no half-pixel term: the model's labels were written this way, so
        # this is the map that puts its predictions back where its targets were. See the
        # `peak_convention` docstring on LoadedModel for why a model gets to say.
        prep_px = model_px * np.array([prep_size[1] / w_in, prep_size[0] / h_in])
    elif peak_convention == "half-pixel":
        # The model's own resize (preprocessed frame -> input), as a transform so we
        # can invert its pixel map; the image resize itself lives in the model.
        resize = FrameTransform((Resize(width=w_in, height=h_in),))
        prep_px = resize.unmap_points(model_px, prep_size)
    else:
        raise ValueError(
            f"unknown peak convention {peak_convention!r}; "
            "expected 'half-pixel' or 'pure-scale'"
        )
    return transform.unmap_points(prep_px, source_size)


def route_channels_to_points_in_views(
    raw_xy: Float[np.ndarray, "C_out *k 2"],
    conf: Float[np.ndarray, "C_out *k"],
    mapping: Int[np.ndarray, "E 3"],
    out_pts: Float[np.ndarray, "V P *k 2"],
    out_conf: Float[np.ndarray, "V P *k"],
) -> None:
    """Scatter a pathway's channels into ``out_pts`` / ``out_conf`` (in place).

    For each ``(i, v, p)`` mapping triple, writes channel ``i`` to ``[v, p]``.
    Handles both the single-peak arrays (``raw_xy`` ``(C_out, 2)``) and the candidate
    arrays (``(C_out, K, 2)``); any trailing ``K`` axis rides along. Entries no triple
    targets keep their preset values (``NaN`` for points, ``0`` for conf).
    """
    i, v, p = mapping[:, 0], mapping[:, 1], mapping[:, 2]
    out_pts[v, p] = raw_xy[i]
    out_conf[v, p] = conf[i]


@dataclass(frozen=True)
class DetectionPlan:
    """The parsed, validated detection plan (torch-free).

    Attributes
    ----------
    view_names
        The view (camera) order -- the ``V`` axis of the points array.
    n_points
        The skeleton point count -- the ``P`` axis.
    sources
        One per camera, in camera order.
    preprocessors
        ``camera -> FrameTransform`` -- only the cameras that have a detection window.
    models
        The sole detector, keyed by its class name.
    pathways
        One per camera, in camera order.
    point_names
        The skeleton's point names, in order -- the meaning of the ``P`` axis. Carried so
        `deeperfly.pose2d.stream.load_models` can hold a model's own recorded channel
        names against them without re-reading the config.
    """

    view_names: list[str]
    n_points: int
    sources: list[Source]
    preprocessors: dict[str, FrameTransform]
    models: dict[str, ModelSpec]
    pathways: list[Pathway]
    point_names: tuple[str, ...] = ()

    @property
    def n_views(self) -> int:
        return len(self.view_names)

    def source_patterns(self) -> dict[str, str | list[str]]:
        """``camera -> footage pattern(s)`` in camera order.

        A value is one regex, or a list of them concatenated in order (see
        :meth:`deeperfly.config.Config.source_patterns`).
        """
        return {s.name: s.pattern for s in self.sources}

    def model_for(self, pathway: Pathway) -> ModelSpec:
        return self.models[pathway.model]

    def visibility_mask(self) -> np.ndarray:
        """Boolean ``(V, P)`` mask: which ``(view, point)`` pairs any pathway writes."""
        mask = np.zeros((self.n_views, self.n_points), dtype=bool)
        for pw in self.pathways:
            mask[pw.mapping[:, 1], pw.mapping[:, 2]] = True
        return mask

    def view_sources(self) -> dict[str, str]:
        """``view name -> the source feeding it`` (via the pathways targeting it).

        Its own footage, always, since a camera is its own source -- kept as a mapping
        because everything downstream reads one.
        """
        out: dict[str, str] = {}
        for pw in self.pathways:
            for v in np.unique(pw.mapping[:, 1]):
                vname = self.view_names[int(v)]
                if vname not in out:
                    out[vname] = pw.source
                elif out[vname] != pw.source:
                    log.warning(
                        "view %r is fed by multiple sources (%r, %r); using %r",
                        vname,
                        out[vname],
                        pw.source,
                        out[vname],
                    )
        return out

    def view_transforms(self) -> dict[str, tuple[FrameTransform, ...]]:
        """``camera -> its detection window``, as a one-element tuple.

        A tuple because the shape is what callers read (a visualization panel borrowing a
        camera's window, the autocrop resolver); there is only ever one now, since a
        camera has one pathway. Full-frame cameras are absent -- an identity chain
        constrains nothing.
        """
        out: dict[str, list[FrameTransform]] = {}
        for pw in self.pathways:
            if pw.transform.is_identity():
                continue
            for v in np.unique(pw.mapping[:, 1]):
                seen = out.setdefault(self.view_names[int(v)], [])
                if pw.transform not in seen:
                    seen.append(pw.transform)
        return {view: tuple(chains) for view, chains in out.items()}

    @classmethod
    def from_config(cls, config) -> DetectionPlan:
        """Synthesize a plan from the camera table and ``[pose2d]``.

        One source, one preprocessor, one identity-mapped pathway per camera, through the
        single detector ``[pose2d] class`` / ``weights`` names. Nothing to cross-reference,
        so nothing to validate but the two things a config can still get wrong: a
        ``[pose2d.crops]`` or ``auto_crops`` entry naming a camera that does not exist.
        """
        from ..pose2d.models import class_defaults
        from ..preprocessing import AutoCrop, Crop, FrameTransform

        pose2d = config.data.get("pose2d", {})
        cameras = list(config.camera_table()[1])
        if not cameras:
            raise ValueError("the detection plan needs cameras under [cameras.<name>]")
        skeleton = config.skeleton()

        cls_name = pose2d.get("class")
        if not isinstance(cls_name, str) or not cls_name:
            raise ValueError(
                "[pose2d] needs a string 'class' naming the detector, e.g. class = \"mvt\""
            )
        fallback = class_defaults(cls_name, skeleton.n_points)
        size = pose2d.get("input_size") or list(fallback["input_size"])
        if len(size) != 2:
            raise ValueError("[pose2d] input_size must be [height, width]")
        model = ModelSpec(
            name=cls_name,
            cls=cls_name,
            weights=(pose2d.get("weights") or None),  # absent -> refused at load
            input_size=(int(size[0]), int(size[1])),
            mean=float(pose2d.get("mean", fallback["mean"])),
            n_out_channels=int(
                pose2d.get("n_out_channels", fallback["n_out_channels"])
            ),
            precision=(pose2d.get("precision") or fallback["precision"]),
            kwargs={},
        )

        crops = _parse_crops(pose2d.get("crops"), cameras)
        searched = _parse_auto_crops(pose2d.get("auto_crops"), cameras)
        patterns = config.source_patterns()

        preprocessors: dict[str, FrameTransform] = {}
        for name in cameras:
            box = crops.get(name)
            if name in searched:
                # A camera in `auto_crops` WITH a box searches from that box -- which is
                # what a separate `crop_seed` form would have been. There is no third way
                # to say it.
                preprocessors[name] = FrameTransform(
                    (AutoCrop(seed=box, where=f"[pose2d] auto_crops {name!r}"),)
                )
            elif box is not None:
                preprocessors[name] = FrameTransform(
                    (Crop(x=box[0], y=box[1], width=box[2], height=box[3]),)
                )

        identity = np.stack(
            [
                np.arange(skeleton.n_points),
                np.zeros(skeleton.n_points, dtype=int),
                np.arange(skeleton.n_points),
            ],
            axis=1,
        )
        pathways = [
            Pathway(
                name=name,
                source=name,
                preprocessor=(name if name in preprocessors else None),
                model=model.name,
                transform=preprocessors.get(name, FrameTransform(())),
                mapping=(identity + np.array([0, v, 0])).astype(np.int64),
            )
            for v, name in enumerate(cameras)
        ]
        return cls(
            view_names=cameras,
            n_points=skeleton.n_points,
            point_names=tuple(skeleton.point_names),
            sources=[
                Source(name=name, pattern=patterns.get(name, name)) for name in cameras
            ],
            preprocessors=preprocessors,
            models={model.name: model},
            pathways=pathways,
        )


# -- the two things a config can still get wrong ------------------------------


def _parse_crops(raw, cameras: list[str]) -> dict[str, tuple[int, int, int, int]]:
    """``[pose2d.crops]`` -- camera -> ``[x, y, width, height]`` in raw pixels.

    One value type, always a box: which cameras are SEARCHED is ``auto_crops``, a
    separate list, so the map never has to be read for two different things at once.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"[pose2d.crops] must be a table of camera -> [x, y, width, height], "
            f"got {raw!r}"
        )
    out: dict[str, tuple[int, int, int, int]] = {}
    for name, box in raw.items():
        if name not in cameras:
            raise ValueError(
                f"[pose2d.crops] names {name!r}, which is not a camera; "
                f"cameras: {cameras}"
            )
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(
                f"[pose2d.crops] {name!r} must be [x, y, width, height], got {box!r}"
            )
        out[name] = tuple(int(v) for v in box)  # type: ignore[assignment]
    return out


def _parse_auto_crops(raw, cameras: list[str]) -> tuple[str, ...]:
    """``[pose2d] auto_crops`` -- the cameras whose window is searched per recording."""
    if raw is None:
        return ()
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"[pose2d] auto_crops must be a list of camera names, got {raw!r}"
        )
    out = []
    for name in raw:
        if name not in cameras:
            raise ValueError(
                f"[pose2d] auto_crops names {name!r}, which is not a camera; "
                f"cameras: {cameras}"
            )
        out.append(str(name))
    return tuple(out)
