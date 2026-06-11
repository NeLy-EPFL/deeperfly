"""The detection plan: sources, preprocessors, models and pathways from config.

The plan is built from config and keeps four counts independent rather than
fusing them at "one per camera":

- **sources** -- named footage globs, each decoded once.
- **views** -- the geometric cameras (``[cameras.*]``); the ``V`` axis of the
  ``(V, T, P, 2)`` points array.
- **models** -- detector models (see :mod:`deeperfly.pose2d.models`).
- **pathways** -- ``source -> preprocessor -> model``, each a named inference run.

Where a pathway's outputs land is declared separately, in ``[pose2d.output_points.<view>]``
tables keyed by point name: ``point = { pathway, out_channel }`` says point ``point``
of view ``<view>`` is filled by output channel ``out_channel`` of the named pathway.
Keying on ``(view, point)`` makes every point's data come from exactly one place
(a repeat is a TOML error). A ``(view, point)`` no entry names stays ``NaN`` -- that
``NaN`` is how visibility is encoded, so no separate mask is needed. Internally each
pathway carries the resolved ``(i, v, p)`` triples (channel ``i`` -> point ``p`` of
view ``v``).

A source may feed several pathways: the front camera, for instance, is one
source feeding two pathways (one mirrored), each mapping into view ``f``. A point
predicted in a pathway's (possibly mirrored/cropped/resized) model frame is
mapped back into its view's frame by inverting the pathway's preprocessing -- see
:func:`normalized_peaks_to_original_pixels`, which inverts any
:class:`~deeperfly.preprocessing.FrameTransform`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int

from ..preprocessing import FrameTransform, Resize, frame_transform_from_ops
from .models import ModelSpec

log = logging.getLogger("deeperfly")


@dataclass(frozen=True)
class Source:
    """A named footage source: its glob pattern (``[[sources]]`` ``filename``)."""

    name: str
    pattern: str


@dataclass(frozen=True)
class Pathway:
    """One detection pathway: a source through a preprocessor + model into views.

    Attributes
    ----------
    name
        The pathway's name (``[[pose2d.pathways]]`` ``name``), referenced from the
        ``[pose2d.output_points.<view>]`` tables.
    source, preprocessor, model
        The names referenced from ``[[sources]]`` / ``[[pose2d.preprocessors]]`` /
        ``[[pose2d.models]]``. ``preprocessor`` is ``None`` when the pathway omits it
        (no frame ops; ``transform`` is the identity).
    transform
        The resolved preprocessor (the pathway's geometric frame prep); the
        identity when ``preprocessor`` is omitted.
    mapping
        An ``(E, 3)`` int array of ``(i, v, p)`` triples (resolved from
        ``[pose2d.output_points]``): model output channel ``i`` -> point ``p`` of
        view ``v``.
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

    Returns
    -------
    np.ndarray
        Peaks of shape ``(..., 2)`` in raw source (view) pixels.
    """
    h_in, w_in = model_input_hw
    model_px = np.asarray(points_norm, dtype=float) * np.array([w_in, h_in])
    prep_size = transform.output_size(source_size)  # (H', W') after the preprocessor
    # The model's own resize (preprocessed frame -> input), as a transform so we
    # can invert its pixel map; the image resize itself lives in the model.
    resize = FrameTransform((Resize(width=w_in, height=h_in),))
    prep_px = resize.unmap_points(model_px, prep_size)
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
        The footage sources, in config order.
    preprocessors
        ``name -> FrameTransform``.
    models
        ``name -> ModelSpec``.
    pathways
        The pathways, in config order.
    """

    view_names: list[str]
    n_points: int
    sources: list[Source]
    preprocessors: dict[str, FrameTransform]
    models: dict[str, ModelSpec]
    pathways: list[Pathway]

    @property
    def n_views(self) -> int:
        return len(self.view_names)

    def source_patterns(self) -> dict[str, str]:
        """``source name -> footage glob`` in config order."""
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

        A view's intrinsics describe this source's raw frame, and its
        visualization footage comes from it. When several distinct sources feed
        one view, the first is used (and a warning is logged). Views no pathway
        writes are absent.
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

    @classmethod
    def from_config(cls, config) -> DetectionPlan:
        """Build a plan from a :class:`~deeperfly.config.Config`.

        Parses the top-level ``[[sources]]`` plus pose2d's own machinery
        (``[[pose2d.preprocessors]]`` / ``[[pose2d.models]]`` / ``[[pose2d.pathways]]`` /
        ``[pose2d.output_points.<view>]``) and resolves view names from ``[cameras.*]``
        and the points from ``[skeleton]``. Validates every cross-reference and
        index loudly (a config typo fails here, not mid-run).
        """
        data = config.data
        pose2d = data.get("pose2d", {})
        view_names = list(config.camera_table()[1])
        if not view_names:
            raise ValueError(
                "the detection plan needs cameras (views) under [cameras.*]"
            )
        skeleton = config.skeleton()
        point_index = {name: i for i, name in enumerate(skeleton.point_names)}

        sources = _parse_sources(data.get("sources"))
        preprocessors = _parse_preprocessors(pose2d.get("preprocessors"))
        models = _parse_models(pose2d.get("models"))
        pathways = _parse_pathways(
            pose2d.get("pathways"),
            sources={s.name for s in sources},
            preprocessors=preprocessors,
            models=models,
            view_names=view_names,
            point_index=point_index,
            output_points=pose2d.get("output_points"),
        )
        return cls(
            view_names=view_names,
            n_points=skeleton.n_points,
            sources=sources,
            preprocessors=preprocessors,
            models=models,
            pathways=pathways,
        )


# -- parsing helpers ----------------------------------------------------------


def _require_list(value, where: str) -> list:
    if value is None:
        raise ValueError(f"the detection plan is missing {where}")
    if not isinstance(value, list):
        raise ValueError(f"{where} must be a list of tables, got {value!r}")
    if not value:
        raise ValueError(f"{where} is empty")
    return value


def _parse_sources(raw) -> list[Source]:
    out, seen = [], set()
    for i, s in enumerate(_require_list(raw, "[[sources]]")):
        name = s.get("name")
        if not isinstance(name, str):
            raise ValueError(f"[[sources]][{i}] needs a string 'name', got {name!r}")
        if name in seen:
            raise ValueError(f"[[sources]] has a duplicate name {name!r}")
        seen.add(name)
        out.append(Source(name=name, pattern=s.get("filename", name)))
    return out


def _parse_preprocessors(raw) -> dict[str, FrameTransform]:
    out: dict[str, FrameTransform] = {}
    for i, p in enumerate(_require_list(raw, "[[pose2d.preprocessors]]")):
        name = p.get("name")
        if not isinstance(name, str):
            raise ValueError(
                f"[[pose2d.preprocessors]][{i}] needs a string 'name', got {name!r}"
            )
        if name in out:
            raise ValueError(f"[[pose2d.preprocessors]] has a duplicate name {name!r}")
        out[name] = frame_transform_from_ops(
            p.get("ops"), f"[[pose2d.preprocessors]] {name!r} ops"
        )
    return out


def _parse_models(raw) -> dict[str, ModelSpec]:
    fixed = {"name", "class", "weights", "input_size", "mean", "n_out_channels"}
    out: dict[str, ModelSpec] = {}
    for i, m in enumerate(_require_list(raw, "[[pose2d.models]]")):
        name = m.get("name")
        if not isinstance(name, str):
            raise ValueError(
                f"[[pose2d.models]][{i}] needs a string 'name', got {name!r}"
            )
        if name in out:
            raise ValueError(f"[[pose2d.models]] has a duplicate name {name!r}")
        cls = m.get("class")
        if not isinstance(cls, str):
            raise ValueError(
                f"[[pose2d.models]] {name!r} needs a string 'class', got {cls!r}"
            )
        size = m.get("input_size", list(ModelSpec.input_size))
        if len(size) != 2:
            raise ValueError(
                f"[[pose2d.models]] {name!r} input_size must be [height, width]"
            )
        weights = m.get("weights")
        out[name] = ModelSpec(
            name=name,
            cls=cls,
            weights=(weights or None),  # "" / absent -> cached default
            input_size=(int(size[0]), int(size[1])),
            mean=float(m.get("mean", ModelSpec.mean)),
            n_out_channels=int(m.get("n_out_channels", ModelSpec.n_out_channels)),
            kwargs={k: v for k, v in m.items() if k not in fixed},
        )
    return out


def _resolve_view(value, view_names: list[str], where: str) -> int:
    """A view reference (name or index) -> its index into ``view_names``."""
    if isinstance(value, bool):
        raise ValueError(f"{where} view {value!r} is not a name or index")
    if isinstance(value, int):
        if not 0 <= value < len(view_names):
            raise ValueError(f"{where} view index {value} out of range")
        return value
    if value in view_names:
        return view_names.index(value)
    raise ValueError(f"{where} references unknown view {value!r}; views: {view_names}")


def _parse_output_points(
    raw,
    *,
    pathway_models: dict[str, str],
    models: dict[str, ModelSpec],
    view_names: list[str],
    point_index: dict[str, int],
) -> dict[str, np.ndarray]:
    """Resolve ``[pose2d.output_points.<view>]`` into each pathway's ``(E, 3)`` mapping.

    Each ``[pose2d.output_points.<view>]`` table is keyed by point name; an entry
    ``{ pathway, out_channel }`` says output channel ``out_channel`` of that
    pathway fills the named point of ``<view>``. Keying on ``(view, point)``
    means every point has exactly one source (a repeat is a TOML error), so no
    later-write-wins rule is needed. Returns ``pathway name -> (E, 3)`` array of
    ``(out_channel, view, point)`` triples; every pathway must be named at least
    once.
    """
    if not isinstance(raw, dict) or not raw:
        raise ValueError("the detection plan is missing [pose2d.output_points.<view>]")
    triples: dict[str, list[tuple[int, int, int]]] = {n: [] for n in pathway_models}
    for view, table in raw.items():
        v = _resolve_view(view, view_names, f"[pose2d.output_points.{view}]")
        if not isinstance(table, dict):
            raise ValueError(
                f"[pose2d.output_points.{view}] must be a table of "
                "point = {{ pathway, out_channel }}"
            )
        for point_name, entry in table.items():
            where = f"[pose2d.output_points.{view}] {point_name!r}"
            if point_name not in point_index:
                raise ValueError(f"{where} is not a skeleton point")
            if not (
                isinstance(entry, dict)
                and "pathway" in entry
                and "out_channel" in entry
            ):
                raise ValueError(
                    f"{where} must be {{ pathway = ..., out_channel = ... }}"
                )
            pw_name = entry["pathway"]
            if pw_name not in pathway_models:
                raise ValueError(f"{where} references unknown pathway {pw_name!r}")
            i = int(entry["out_channel"])
            n_out = models[pathway_models[pw_name]].n_out_channels
            if not 0 <= i < n_out:
                raise ValueError(
                    f"{where} out_channel {i} outside [0, {n_out}) "
                    "(model n_out_channels)"
                )
            triples[pw_name].append((i, v, point_index[point_name]))
    out: dict[str, np.ndarray] = {}
    for name, t in triples.items():
        if not t:
            raise ValueError(
                f"pathway {name!r} has no [pose2d.output_points] entries; "
                "it maps no points"
            )
        out[name] = np.asarray(t, dtype=np.int64).reshape(-1, 3)
    return out


def _parse_pathways(
    raw,
    *,
    sources: set[str],
    preprocessors: dict[str, FrameTransform],
    models: dict[str, ModelSpec],
    view_names: list[str],
    point_index: dict[str, int],
    output_points,
) -> list[Pathway]:
    specs: list[
        tuple[str, str, str | None, str]
    ] = []  # name, source, preprocessor, model
    seen: set[str] = set()
    for i, pw in enumerate(_require_list(raw, "[[pose2d.pathways]]")):
        where = f"[[pose2d.pathways]][{i}]"
        name = pw.get("name")
        if not isinstance(name, str):
            raise ValueError(f"{where} needs a string 'name', got {name!r}")
        if name in seen:
            raise ValueError(f"[[pose2d.pathways]] has a duplicate name {name!r}")
        seen.add(name)
        src = pw.get("source")
        if src not in sources:
            raise ValueError(f"{where} references unknown source {src!r}")
        prep = pw.get("preprocessor")
        if prep is not None and prep not in preprocessors:
            raise ValueError(f"{where} references unknown preprocessor {prep!r}")
        model = pw.get("model")
        if model not in models:
            raise ValueError(f"{where} references unknown model {model!r}")
        specs.append((name, src, prep, model))

    mappings = _parse_output_points(
        output_points,
        pathway_models={name: model for name, _, _, model in specs},
        models=models,
        view_names=view_names,
        point_index=point_index,
    )
    return [
        Pathway(
            name=name,
            source=src,
            preprocessor=prep,
            model=model,
            transform=preprocessors[prep] if prep is not None else FrameTransform(()),
            mapping=mappings[name],
        )
        for name, src, prep, model in specs
    ]
