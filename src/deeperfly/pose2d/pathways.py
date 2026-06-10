"""The detection plan: sources, preprocessors, models and pathways from config.

This is the config-driven replacement for the old hardcoded fly camera layout
(``fly_camera_layout`` / ``expand_passes`` / ``assemble_skeleton``). It decouples
four counts that used to be fused at "one per camera":

- **sources** -- named footage globs, each decoded once.
- **views** -- the geometric cameras (``[cameras.*]``); the ``V`` axis of the
  ``(V, T, N, 2)`` points array.
- **models** -- detector models (see :mod:`deeperfly.pose2d.models`).
- **pathways** -- ``source -> preprocessor -> model -> mapping``. A pathway runs
  one model over one (preprocessed) source and scatters its output channels into
  the skeleton via a list of ``(i, v, p)`` triples: model output channel ``i``
  becomes point ``p`` of view ``v``. A ``(view, point)`` pair that no pathway
  writes stays ``NaN`` -- that *is* the visibility mask (no separate table).

The front camera is no longer special: it is one source feeding two pathways
(one mirrored), each mapping into view ``f``. A point predicted in a pathway's
(possibly mirrored/cropped/resized) model frame is mapped back into its view's
frame by inverting the pathway's preprocessing -- see :func:`map_to_view`, which
generalizes the old ``x -> 1 - x`` flip undo to any
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
    """A named footage source: its glob pattern (the old ``[cameras.*].input``)."""

    name: str
    pattern: str


@dataclass(frozen=True)
class Pathway:
    """One detection pathway: a source through a preprocessor + model into views.

    Attributes
    ----------
    source, preprocessor, model
        The names referenced from ``[[sources]]`` / ``[[preprocessors]]`` /
        ``[[models]]``.
    transform
        The resolved preprocessor (the pathway's geometric frame prep).
    mapping
        An ``(E, 3)`` int array of ``(i, v, p)`` triples: model output channel
        ``i`` -> point ``p`` of view ``v``.
    """

    source: str
    preprocessor: str
    model: str
    transform: FrameTransform
    mapping: Int[np.ndarray, "E 3"]


def map_to_view(
    points_norm: Float[np.ndarray, "*lead 2"],
    transform: FrameTransform,
    model_input_hw: tuple[int, int],
    source_size: tuple[int, int],
) -> Float[np.ndarray, "*lead 2"]:
    """Map model peaks (normalized ``[0, 1]``) back into the source/view frame.

    Inverts the pathway's geometry: normalized model coords -> model-input
    pixels -> (undo the model's resize) -> preprocessed-frame pixels -> (undo
    the preprocessor, e.g. a mirror) -> raw source pixels, which is the frame
    the view's intrinsics describe. Generalizes the old ``assemble_skeleton``
    ``x -> 1 - x`` flip undo + ``* (w, h)`` scale.

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


def scatter_pathway(
    raw_xy: Float[np.ndarray, "J *k 2"],
    conf: Float[np.ndarray, "J *k"],
    mapping: Int[np.ndarray, "E 3"],
    out_pts: Float[np.ndarray, "V N *k 2"],
    out_conf: Float[np.ndarray, "V N *k"],
) -> None:
    """Scatter a pathway's channels into ``out_pts`` / ``out_conf`` (in place).

    For each ``(i, v, p)`` mapping triple, writes channel ``i`` to ``[v, p]``.
    Handles both the single-peak arrays (``raw_xy`` ``(J, 2)``) and the candidate
    arrays (``(J, K, 2)``); any trailing ``K`` axis rides along. Entries no triple
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
        The skeleton point count -- the ``N`` axis.
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
        """Boolean ``(V, N)`` mask: which ``(view, point)`` pairs any pathway writes."""
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

        Parses ``[[sources]]`` / ``[[preprocessors]]`` / ``[[models]]`` /
        ``[[pathways]]`` and resolves view names from ``[cameras.*]`` and the
        point count from ``[skeleton]``. Validates every cross-reference and
        index loudly (a config typo fails here, not mid-run).
        """
        data = config.data
        view_names = list(config.camera_table()[1])
        if not view_names:
            raise ValueError(
                "the detection plan needs cameras (views) under [cameras.*]"
            )
        n_points = config.skeleton().n_points

        sources = _parse_sources(data.get("sources"))
        preprocessors = _parse_preprocessors(data.get("preprocessors"))
        models = _parse_models(data.get("models"))
        pathways = _parse_pathways(
            data.get("pathways"),
            sources={s.name for s in sources},
            preprocessors=preprocessors,
            models=models,
            view_names=view_names,
            n_points=n_points,
        )
        return cls(
            view_names=view_names,
            n_points=n_points,
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
        out.append(Source(name=name, pattern=s.get("input", name)))
    return out


def _parse_preprocessors(raw) -> dict[str, FrameTransform]:
    out: dict[str, FrameTransform] = {}
    for i, p in enumerate(_require_list(raw, "[[preprocessors]]")):
        name = p.get("name")
        if not isinstance(name, str):
            raise ValueError(
                f"[[preprocessors]][{i}] needs a string 'name', got {name!r}"
            )
        if name in out:
            raise ValueError(f"[[preprocessors]] has a duplicate name {name!r}")
        out[name] = frame_transform_from_ops(
            p.get("ops"), f"[[preprocessors]] {name!r} ops"
        )
    return out


def _parse_models(raw) -> dict[str, ModelSpec]:
    fixed = {"name", "class", "weights", "input_size", "mean", "n_channels"}
    out: dict[str, ModelSpec] = {}
    for i, m in enumerate(_require_list(raw, "[[models]]")):
        name = m.get("name")
        if not isinstance(name, str):
            raise ValueError(f"[[models]][{i}] needs a string 'name', got {name!r}")
        if name in out:
            raise ValueError(f"[[models]] has a duplicate name {name!r}")
        cls = m.get("class")
        if not isinstance(cls, str):
            raise ValueError(f"[[models]] {name!r} needs a string 'class', got {cls!r}")
        size = m.get("input_size", list(ModelSpec.input_size))
        if len(size) != 2:
            raise ValueError(f"[[models]] {name!r} input_size must be [height, width]")
        weights = m.get("weights")
        out[name] = ModelSpec(
            name=name,
            cls=cls,
            weights=(weights or None),  # "" / absent -> cached default
            input_size=(int(size[0]), int(size[1])),
            mean=float(m.get("mean", ModelSpec.mean)),
            n_channels=int(m.get("n_channels", ModelSpec.n_channels)),
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


def _pathway_mapping(
    pw: dict, n_channels: int, view_names: list[str], n_points: int, where: str
) -> np.ndarray:
    """Resolve a pathway's ``map``/``view``+``points`` into an ``(E, 3)`` array."""
    triples: list[tuple[int, int, int]] = []
    if "map" in pw:
        for entry in pw["map"]:
            if len(entry) != 3:
                raise ValueError(f"{where} map entry {entry!r} must be [i, v, p]")
            i, v, p = entry
            triples.append((int(i), _resolve_view(v, view_names, where), int(p)))
    elif "view" in pw and "points" in pw:
        v = _resolve_view(pw["view"], view_names, where)
        for i, p in enumerate(pw["points"]):
            if int(p) >= 0:
                triples.append((i, v, int(p)))
    else:
        raise ValueError(f"{where} needs either 'map' or both 'view' and 'points'")

    arr = np.asarray(triples, dtype=np.int64).reshape(-1, 3)
    if arr.size == 0:
        raise ValueError(f"{where} maps no points")
    if arr[:, 0].min() < 0 or arr[:, 0].max() >= n_channels:
        raise ValueError(
            f"{where} maps a channel outside [0, {n_channels}) (model n_channels)"
        )
    if arr[:, 2].min() < 0 or arr[:, 2].max() >= n_points:
        raise ValueError(f"{where} maps a point outside [0, {n_points}) (skeleton)")
    return arr


def _parse_pathways(
    raw,
    *,
    sources: set[str],
    preprocessors: dict[str, FrameTransform],
    models: dict[str, ModelSpec],
    view_names: list[str],
    n_points: int,
) -> list[Pathway]:
    out: list[Pathway] = []
    seen_targets: set[tuple[int, int]] = set()
    for i, pw in enumerate(_require_list(raw, "[[pathways]]")):
        where = f"[[pathways]][{i}]"
        src = pw.get("source")
        if src not in sources:
            raise ValueError(f"{where} references unknown source {src!r}")
        prep = pw.get("preprocessor")
        if prep not in preprocessors:
            raise ValueError(f"{where} references unknown preprocessor {prep!r}")
        model = pw.get("model")
        if model not in models:
            raise ValueError(f"{where} references unknown model {model!r}")
        mapping = _pathway_mapping(
            pw, models[model].n_channels, view_names, n_points, where
        )
        for v, p in mapping[:, 1:]:
            key = (int(v), int(p))
            if key in seen_targets:
                log.warning(
                    "%s maps to view %r point %d, already written by an earlier "
                    "pathway; the later pathway wins",
                    where,
                    view_names[v],
                    p,
                )
            seen_targets.add(key)
        out.append(
            Pathway(
                source=src,
                preprocessor=prep,
                model=model,
                transform=preprocessors[prep],
                mapping=mapping,
            )
        )
    return out
