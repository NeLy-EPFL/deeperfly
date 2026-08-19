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

Those mirrored pathways are also where the plan's left/right identities are decided,
which is why :func:`check_mirror_consistency` runs at load: with the skeleton's
symmetry pairs declared, a mapping that sends a mirrored channel to the wrong side
is a config error here rather than a silently side-swapped reconstruction later.
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
    pattern: str | list[str]


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
        (no frame ops; ``transform`` is the identity). ``model`` is always resolved --
        a pathway that omits it takes ``[pose2d].model``, or the sole declared model
        (see :func:`_default_model`).
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
        The footage sources, in config order.
    preprocessors
        ``name -> FrameTransform``.
    models
        ``name -> ModelSpec``.
    pathways
        The pathways, in config order.
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
        """``source name -> footage glob(s)`` in config order.

        A value may be a single glob or a list of alternate globs (see
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

    def view_transforms(self) -> dict[str, tuple[FrameTransform, ...]]:
        """``view name -> every DISTINCT preprocessing chain a pathway detects it through``.

        The companion of :meth:`view_sources`, but it hands back all the candidates rather
        than picking one, because unlike a source the right answer depends on what the
        caller wants from the chain. The rig's mirrored twin -- one source feeding two
        pathways into one view, one of them flipped -- is two different chains looking
        through the *same window*, so a caller after the window has no conflict while a
        caller after the chirality has no answer.

        Identity chains are dropped: a pathway that detects on the raw frame constrains
        nothing, and keeping it would make every mirrored twin look like a disagreement.

        Views no pathway writes are absent, in pathway (config) order.

        Returns
        -------
        dict of str to tuple of FrameTransform
            The distinct non-identity chains per view.
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
        models = _parse_models(pose2d.get("models"), n_points=skeleton.n_points)
        pathways = _parse_pathways(
            pose2d.get("pathways"),
            sources={s.name for s in sources},
            preprocessors=preprocessors,
            models=models,
            view_names=view_names,
            point_index=point_index,
            output_points=pose2d.get("output_points"),
            default_model=pose2d.get("model"),
        )
        check_mirror_consistency(pathways, models, skeleton, view_names)
        return cls(
            view_names=view_names,
            n_points=skeleton.n_points,
            point_names=tuple(skeleton.point_names),
            sources=sources,
            preprocessors=preprocessors,
            models=models,
            pathways=pathways,
        )


# -- the mirror check ---------------------------------------------------------


def check_mirror_consistency(
    pathways: list[Pathway],
    models: dict[str, ModelSpec],
    skeleton,
    view_names: list[str],
) -> None:
    """Validate that a **mirrored** pathway lands on the **mirrored** points.

    A detector channel means one anatomical landmark under one chirality convention. The
    rig exploits that: the side cameras all feed the same side-agnostic model, and the
    left-side views reach its convention through a ``fliplr`` preprocessor. Which side a
    channel then represents is decided *only* by ``[pose2d.output_points]`` -- so the
    left/right swap this package relies on is 132 hand-written config rows with nothing
    checking them. A single typo (``rf_femur_tibia`` where ``lf_femur_tibia`` belongs)
    silently swaps a side: the detector still fires, triangulation still converges, and
    the reconstruction looks like a fly with its legs crossed.

    Given the skeleton's symmetry pairs the invariant is decidable, so it is checked. For
    each ``(model, channel)``, every point an **un-mirrored** pathway maps it to must be the
    **symmetry partner** of every point a **mirrored** pathway maps it to.

    Phrased over all combinations rather than over a single point per side on purpose. One
    channel feeding several points is unusual but legal here -- ``[pose2d.output_points]``
    keys on ``(view, point)``, so it constrains where a point's data comes *from*, not how
    many points a channel may feed -- and rejecting that outright would fail configs that
    work today. Comparing every combination costs nothing and is in fact *stricter* where
    it matters: the realistic typo (a row moved from the un-mirrored pathway to the
    mirrored one) leaves a channel mapping to the same point at both parities, and a point
    is never its own partner, so it is caught.

    Skipped entirely when the skeleton declares no ``symmetries`` -- the pairs are the
    premise, and inferring them here would let a rename turn a passing config into a
    failing one. Skipped per channel when only one parity maps it (nothing to compare):
    a one-sided rig's pathways are all un-mirrored, and that is legal.

    Raises
    ------
    ValueError
        Naming the model, channel, pathways and points involved, and the partner that was
        expected -- because "which of these 132 rows is wrong" is the only question the
        operator has at that moment.
    """
    if not getattr(skeleton, "n_symmetries", 0):
        return
    names = tuple(skeleton.point_names)
    # (model, channel, mirrored) -> point -> the pathways/views that said so.
    cells: dict[tuple[str, int, bool], dict[int, list[str]]] = {}
    for pw in pathways:
        mirrored = pw.transform.reverses_handedness
        for i, v, p in np.asarray(pw.mapping).reshape(-1, 3):
            key = (pw.model, int(i), mirrored)
            where = f"{pw.name!r} -> view {view_names[int(v)]!r}"
            cells.setdefault(key, {}).setdefault(int(p), []).append(where)

    problems: list[str] = []
    for model, channel in sorted({(m, c) for m, c, _ in cells}):
        plain = cells.get((model, channel, False)) or {}
        mirror = cells.get((model, channel, True)) or {}
        for p_plain in sorted(plain):
            expected = skeleton.partner(p_plain)
            for p_mirror in sorted(mirror):
                if expected == p_mirror:
                    continue
                want = (
                    f"{names[p_plain]!r} has no symmetry partner, so no mirrored pathway "
                    "may map this channel at all"
                    if expected is None
                    else f"the mirrored pathway must land on {names[expected]!r}"
                )
                problems.append(
                    f"model {model!r} channel {channel}: un-mirrored -> "
                    f"{names[p_plain]!r} ({', '.join(plain[p_plain])}) but mirrored -> "
                    f"{names[p_mirror]!r} ({', '.join(mirror[p_mirror])}). Mirroring the "
                    f"frame mirrors the animal, so {want} -- as declared in "
                    "[skeleton].symmetries."
                )

    if problems:
        raise ValueError(
            "[pose2d.output_points] disagrees with [skeleton].symmetries about "
            "left/right:\n  - " + "\n  - ".join(problems)
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
        from ..config import _source_filename

        out.append(
            Source(name=name, pattern=_source_filename(s.get("filename", name), name))
        )
    return out


def _parse_preprocessors(raw) -> dict[str, FrameTransform]:
    """Parse ``[[pose2d.preprocessors]]``; the section itself is optional.

    A pathway's ``preprocessor`` key is already optional (omitting it means the identity),
    so a plan whose every pathway detects on the raw frame has nothing to declare -- and
    requiring an empty list from it was asking for a line that says "no line".
    """
    if raw is None:
        return {}
    if not isinstance(raw, list):
        raise ValueError(
            f"[[pose2d.preprocessors]] must be a list of tables, got {raw!r}"
        )
    out: dict[str, FrameTransform] = {}
    for i, p in enumerate(raw):
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


def _parse_models(raw, *, n_points: int | None = None) -> dict[str, ModelSpec]:
    """Parse ``[[pose2d.models]]``, filling omitted keys from the model class.

    Only ``name``, ``class`` and ``weights`` are irreducible. ``input_size`` / ``mean`` / ``n_out_channels`` /
    ``precision`` come from :func:`~deeperfly.pose2d.models.class_defaults` when the table
    does not state them -- with the dense classes' channel count resolved against
    ``n_points``, the skeleton this plan routes into.
    """
    from .models import class_defaults

    fixed = {
        "name",
        "class",
        "weights",
        "input_size",
        "mean",
        "n_out_channels",
        "precision",
    }
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
        fallback = class_defaults(cls, n_points)
        size = m.get("input_size") or list(fallback["input_size"])
        if len(size) != 2:
            raise ValueError(
                f"[[pose2d.models]] {name!r} input_size must be [height, width]"
            )
        weights = m.get("weights")
        out[name] = ModelSpec(
            name=name,
            cls=cls,
            weights=(weights or None),  # "" / absent -> resolved at load
            input_size=(int(size[0]), int(size[1])),
            mean=float(m.get("mean", fallback["mean"])),
            n_out_channels=int(m.get("n_out_channels", fallback["n_out_channels"])),
            # "" / absent -> the class's own requirement, else [pose2d].precision
            precision=(m.get("precision") or fallback["precision"]),
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
    triples: dict[str, list[tuple[int, int, int]]] = {n: [] for n in pathway_models}
    for view, table in (raw or {}).items():
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
            t = _identity_triples(
                name,
                pathway_models=pathway_models,
                models=models,
                view_names=view_names,
                n_points=len(point_index),
            )
        out[name] = np.asarray(t, dtype=np.int64).reshape(-1, 3)
    return out


def _identity_triples(
    pw_name: str,
    *,
    pathway_models: dict[str, str],
    models: dict[str, ModelSpec],
    view_names: list[str],
    n_points: int,
) -> list[tuple[int, int, int]]:
    """The default mapping for a pathway no ``[pose2d.output_points]`` table names.

    **Channel ``i`` -> point ``i`` of the view the pathway is named after.** This is what a
    DENSE detector always means -- one that emits every tracked point for the view it was
    given -- and writing it out is 38 x V lines carrying no information, in which a single
    transposition is a wrong limb rather than a crash. So a config for such a detector
    declares no mapping at all.

    The mapping stays REQUIRED for anything that is not dense, and the gate is the channel
    count: the shipped 19-channel detector emits one side of the animal, so which points
    its channels mean genuinely differs per view (and its front camera runs twice,
    mirrored). There is no identity to fall back on and asking for the table is right.

    Two things this cannot check, and one of them is checked elsewhere:

    * That the model's channels are IN the skeleton's order rather than merely as numerous.
      Nothing here can: the plan is parsed torch-free, before any weights are read. The
      loaded module carries its own ``point_names``, and
      :func:`deeperfly.pose2d.stream.load_models` compares them against the skeleton on
      every run -- which also catches a hand-edited config and a swapped weights file,
      neither of which a config generator ever sees.
    * A typo that leaves a pathway unmapped by accident. Before this, an unnamed pathway
      was an error; now a dense one silently gets the identity. The channel-count gate and
      the load-time name check are what make that trade acceptable.
    """
    where = f"pathway {pw_name!r} has no [pose2d.output_points] entry"
    view = _resolve_view(pw_name, view_names, where)
    n_out = models[pathway_models[pw_name]].n_out_channels
    if n_out != n_points:
        raise ValueError(
            f"{where}, so it would default to channel i -> point i of view {pw_name!r} -- "
            f"but its model emits {n_out} channels for a {n_points}-point skeleton. Only a "
            "detector that predicts every point can take the default; give this pathway an "
            "explicit [pose2d.output_points.<view>] table."
        )
    return [(i, view, i) for i in range(n_points)]


def _default_model(declared, models: dict[str, ModelSpec]) -> str | None:
    """The model a pathway that names none gets: ``[pose2d].model``, else the sole one.

    A dense plan is one pathway per camera through *one* detector, so the model name
    was written once per camera and said nothing -- the repetition is only there to be
    a reference. ``[pose2d].model`` hoists it to where it belongs, and a plan with a
    single ``[[pose2d.models]]`` entry needs even that: there is exactly one answer.

    The fallback is deliberately not "the first model". Adding a second model to a plan
    whose pathways are bare must be an error naming both, not a silent pick -- that is
    the moment the default stops being unambiguous.
    """
    if declared is not None:
        if declared not in models:
            raise ValueError(
                f"[pose2d].model references unknown model {declared!r}; "
                f"models: {sorted(models)}"
            )
        return declared
    return next(iter(models)) if len(models) == 1 else None


def _parse_pathways(
    raw,
    *,
    sources: set[str],
    preprocessors: dict[str, FrameTransform],
    models: dict[str, ModelSpec],
    view_names: list[str],
    point_index: dict[str, int],
    output_points,
    default_model=None,
) -> list[Pathway]:
    specs: list[
        tuple[str, str, str | None, str]
    ] = []  # name, source, preprocessor, model
    fallback = _default_model(default_model, models)
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
        model = pw.get("model", fallback)
        if model is None:
            raise ValueError(
                f"{where} ({name!r}) names no model and there is no default: this plan "
                f"declares {len(models)} models ({sorted(models)}), so write 'model' on "
                "the pathway, or [pose2d].model to set one for all of them"
            )
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
