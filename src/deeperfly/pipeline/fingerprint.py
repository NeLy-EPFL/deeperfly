"""Per-stage config fingerprints and the run record that drives cache reuse.

Each stage's *result-affecting* config subset is captured as a plain JSON-able
dict (a *fingerprint*) and recorded in ``<outdir>/run.json`` when the stage
completes. On a later run a stage is reused only when its recorded fingerprint
still matches the current config **and** its output is present -- so editing,
say, ``[triangulation]`` automatically recomputes triangulation (and
everything downstream) while the slow ``pose2d`` cache is reused untouched.
Performance-only knobs (``batch_size``, ``decode_buffer``, ``[io.image]``
workers) are deliberately excluded: they never invalidate a cache.

Fingerprints are stored verbatim rather than hashed so a mismatch can be
reported as a readable diff (:func:`fingerprint_diff`).

Comparison is *subset* semantics (:func:`fingerprint_diff` checks every
expected key against the stored value and ignores extra stored keys), so a key
that drops out of the expected fingerprint -- e.g. ``candidates`` when
``pictorial_structures`` is disabled again -- does not invalidate the cache,
while a key that appears does.

A derived stage's inputs depend on which upstream stages are enabled; the
``*_source`` selectors here name that choice (and embed the config rig geometry
when it is the source), so toggling ``do_bundle_adjustment`` or
``do_pictorial_structures`` invalidates exactly the consumers. The rule is:
``pose2d`` is the data root and its cache always feeds downstream, while a
derived stage's output is used downstream only while that stage is *enabled*.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ..config import STAGES, Config
from ..results import StageStore

log = logging.getLogger("deeperfly")

#: ``run.json`` schema version; an unknown version is reset (recompute all).
RECORD_VERSION = 1

#: The inverse-kinematics solver, recorded in that stage's fingerprint. Because
#: comparison is subset semantics, a *dropped* key cannot invalidate a cache -- so
#: swapping solvers has to announce itself with a key of its own.
IK_SOLVER = "quickik"

#: Bumped by hand when the way deeperfly drives the solver changes the fitted angles
#: without any config key changing (a body-plan structure change, a different neutral
#: pose, a different observation weighting). Deliberately *not* the solver's package
#: version: that would force a full recompute on any dependency bump, and it cannot be
#: read at all on an install without the optional extra.
#:
#: 2 -- the leg chains are parameterised in flygym's own frame. Two DOF axes were swapped
#: and each leg subtree was rotated by the wrong body frame, so EVERY stored leg angle
#: changes convention: a right-leg angle is no longer the negation of flygym's, and the
#: fit that used to saturate the thorax-coxa yaw wall on four of six legs just to
#: represent rest now sits at the rigid-segment floor. An output directory holding
#: revision-1 angles is not a worse fit of the same quantity, it is a different quantity,
#: and it has to recompute rather than validate. (Bumping this here rather than at the
#: commit that made the change is a correction: nothing else in the fingerprint moved, so
#: those trees have been validating ever since.)
IK_SOLVER_REVISION = 2

#: Bumped by hand when the ensemble Kalman smoother's numerics change the fitted
#: trajectory without any ``[eks]`` key changing (a different objective, a different
#: initialization, a different likelihood). Same role as
#: :data:`IK_SOLVER_REVISION`: because a *dropped* fingerprint key cannot
#: invalidate a cache, a behavior change has to announce itself with a key.
EKS_REVISION = 1


# -- the run record (<outdir>/run.json) ---------------------------------------


class RunRecord:
    """The fingerprints of the stage outputs currently cached in an output dir.

    A small JSON sidecar next to ``results.h5``: validity bookkeeping is
    outdir-local run state, kept out of the portable result file (and a viz-only
    run never touches ``results.h5`` at all). Deleting it merely recomputes
    everything.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._stages = self._load()

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        if data.get("format_version") != RECORD_VERSION:
            return {}
        stages = data.get("stages")
        return stages if isinstance(stages, dict) else {}

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(
                {"format_version": RECORD_VERSION, "stages": self._stages}, indent=2
            )
            + "\n"
        )

    def get(self, stage: str) -> dict | None:
        """The recorded fingerprint for ``stage``, or ``None``."""
        entry = self._stages.get(stage)
        return entry.get("fingerprint") if isinstance(entry, dict) else None

    def set(self, stage: str, fingerprint: dict) -> None:
        """Record ``stage`` as freshly computed with ``fingerprint``.

        Every *later* stage's entry is dropped: its inputs just changed, so its
        record is stale even if this run crashes before recomputing it.
        """
        for later in STAGES[STAGES.index(stage) + 1 :]:
            self._stages.pop(later, None)
        self._stages[stage] = {
            "fingerprint": fingerprint,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._save()


# -- fingerprint construction --------------------------------------------------


def _norm(value):
    """JSON-normalize a value (tuples -> lists, ``Path``/datetime -> str)."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _skeleton_digest(config: Config, *, cosmetic: bool = False) -> dict:
    """The skeleton parts that affect geometry (+ drawing, when ``cosmetic``).

    The skeleton's ``name`` is deliberately **not** in here. A name is a label a human
    chose; what decides every stage's answer is the ordered ``point_names`` and the
    ``bones`` between them, and two skeletons agreeing on both compute the same result
    whatever they are called. Including the name meant renaming a preset invalidated
    every cached stage of every existing output tree -- a full re-detection bought by a
    string -- while telling the cache nothing that the point names had not already said.
    """
    skel = config.skeleton()
    digest = {
        "point_names": list(skel.point_names),
        "bones": skel.bones.tolist(),
    }
    if cosmetic:  # the visualization stage also draws the colors
        digest["point_colors"] = list(skel.point_colors)
    return digest


def _camera_geometry(config: Config) -> dict:
    """The rig the config describes -- the orbit tables *and* the calibration it points at.

    ``camera_table()`` deliberately drops the scalar ``calibration`` key (it is a path, not a
    view), and that key is what :meth:`~deeperfly.cameras.CameraGroup.from_config` actually
    builds the rig from when it is set. Fingerprinting only the tables therefore made the
    solved rig **invisible to the cache**: pointing a project at a different calibration, or
    re-solving one in place, changed no stage's fingerprint, so cached
    ``bundle_adjustment`` / ``pictorial_structures`` / ``triangulation`` / ``visualization``
    outputs were all reused against a rig that no longer existed.

    The file's *content* is digested, not just its path, because re-solving a calibration
    rewrites it under the same name -- which is the common case, and the one a path alone
    cannot see.
    """
    defaults, cams = config.camera_table()
    digest: dict = {
        "defaults": dict(defaults),
        "cameras": {n: dict(s) for n, s in cams.items()},
    }
    path = config.calibration_path()
    if path is not None:
        digest["calibration"] = {
            # The name travels too: two rigs that happen to hash alike are still two
            # different artifacts to a human reading `run.json`.
            "name": Path(path).name,
            "content": _file_digest(path),
        }
    return digest


def _file_digest(path) -> str | None:
    """``sha256:<hex>`` of a calibration file or directory, or ``None`` if unreadable.

    Unreadable yields ``None`` rather than raising: a missing calibration is diagnosed by
    the rig builder with a message about the rig, and a fingerprint helper is the wrong
    place to pre-empt it. ``None`` is itself a distinct fingerprint value, so a calibration
    that disappears still invalidates.
    """
    p = Path(path)
    h = hashlib.sha256()
    try:
        files = sorted(p.rglob("*")) if p.is_dir() else [p]
        for f in files:
            if f.is_file():
                h.update(f.name.encode())
                h.update(f.read_bytes())
    except OSError:
        return None
    return f"sha256:{h.hexdigest()[:16]}"


def cameras_source(enabled: dict[str, bool], store: StageStore) -> str:
    """Which rig a downstream stage consumes: ``bundle_adjustment`` or ``config``."""
    if enabled["bundle_adjustment"] and store.has("bundle_adjustment"):
        return "bundle_adjustment"
    return "config"


def pts2d_source(enabled: dict[str, bool], store: StageStore) -> str:
    """Which 2D points triangulation consumes: ``pictorial_structures`` or ``pose2d``."""
    if enabled["pictorial_structures"] and store.has("pictorial_structures"):
        return "pictorial_structures"
    return "pose2d"


def pts3d_source(enabled: dict[str, bool], store: StageStore) -> str | None:
    """Which stage's 3D points a downstream stage consumes, most-derived first."""
    for stage in ("postprocess", "eks", "triangulation", "pictorial_structures"):
        if enabled[stage] and store.has(stage):
            return stage
    return None


def postprocess_source(enabled: dict[str, bool], store: StageStore) -> str | None:
    """Which stage's pose the correction chain consumes -- never its own output.

    Deliberately *not* :func:`pts3d_source`, for the reason spelled out in
    :func:`eks_init_source`: a stage whose fingerprint names a selector that can
    resolve to the stage itself never validates, so its cache would be dead weight.
    """
    for stage in ("eks", "triangulation", "pictorial_structures"):
        if enabled[stage] and store.has(stage):
            return stage
    return None


def eks_init_source(enabled: dict[str, bool], store: StageStore) -> str | None:
    """Which stage's 3D seeds the smoother -- never the smoother's own output.

    Deliberately *not* :func:`pts3d_source`. A stage whose fingerprint names an
    input selector that can resolve to the stage itself never validates: the
    selector says ``triangulation`` on the first run (nothing cached yet) and
    ``eks`` on the second, so the recorded fingerprint disagrees with the expected
    one forever and the stage recomputes on every run.
    """
    for stage in ("triangulation", "pictorial_structures"):
        if enabled[stage] and store.has(stage):
            return stage
    return None


def pose_sources(enabled: dict[str, bool], store: StageStore) -> dict[str, str | None]:
    """Which stage outputs the visualization draws (2D and 3D separately)."""
    for stage in ("postprocess", "eks", "triangulation", "pictorial_structures"):
        if enabled[stage] and store.has(stage):
            return {"pts2d": stage, "pts3d": stage}
    return {"pts2d": "pose2d", "pts3d": None}


def nmf_source(enabled: dict[str, bool], store: StageStore) -> str | None:
    """Whether the fitted IK model is available to draw (the ``skeleton_nmf`` overlay)."""
    if enabled["inverse_kinematics"] and store.has("inverse_kinematics"):
        return "inverse_kinematics"
    return None


def _cameras_entry(config: Config, enabled: dict[str, bool], store: StageStore):
    """The ``cameras_from`` fingerprint entry.

    When the source is the config rig, the rig geometry is embedded so editing
    ``[cameras]`` with bundle adjustment disabled still invalidates the
    consumers; when it is the BA output, geometry changes flow through the BA
    stage's own fingerprint and cascade.
    """
    source = cameras_source(enabled, store)
    if source == "config":
        return {"config": _camera_geometry(config)}
    return source


def stage_fingerprint(
    stage: str, config: Config, enabled: dict[str, bool], store: StageStore
) -> dict:
    """The result-affecting config subset for ``stage``, as a JSON-able dict.

    Evaluated lazily per stage inside the run loop (after the upstream stages
    settled), so the input-source selectors here match what the stage actually
    consumes.

    Parameters
    ----------
    stage
        A :data:`~deeperfly.config.STAGES` name.
    config
        The run config.
    enabled
        The ``do_<stage>`` flags (:meth:`Config.stage_flags`).
    store
        The recording's :class:`~deeperfly.results.StageStore` (for the
        input-source selectors).

    Returns
    -------
    dict
        The fingerprint (JSON-normalized).
    """
    if stage == "pose2d":
        p = config.pose2d
        plan = config.detection_plan()
        fp = {
            "sources": plan.source_patterns(),
            "preprocessors": {
                name: t.to_json() for name, t in plan.preprocessors.items()
            },
            "models": {
                # Precision is result-affecting and now per-model: store the
                # RESOLVED value (override or the [pose2d] fallback), so editing
                # the global default invalidates every inheriting model's cache.
                name: {
                    "class": s.cls,
                    "weights": s.weights,
                    "input_size": list(s.input_size),
                    "mean": s.mean,
                    "n_out_channels": s.n_out_channels,
                    "precision": s.precision or p.precision,
                    "kwargs": s.kwargs,
                }
                for name, s in plan.models.items()
            },
            "pathways": [
                {
                    "source": pw.source,
                    "preprocessor": pw.preprocessor,
                    "model": pw.model,
                    "mapping": pw.mapping.tolist(),
                }
                for pw in plan.pathways
            ],
            "skeleton": _skeleton_digest(config),
        }
        if enabled["pictorial_structures"]:
            # Candidate extraction happens during detection, so needing
            # candidates (and their K) is part of pose2d's contract.
            fp["candidates"] = {
                "k": config.pictorial.k,
                # The peak gate runs during EXTRACTION, so changing it changes the cached
                # candidate set and nothing downstream can recover from a set that was
                # pruned too hard. It belongs to pose2d's contract for the same reason `k`
                # does.
                "peak_threshold": config.pictorial.peak_threshold,
                "peak_threshold_rel": config.pictorial.peak_threshold_rel,
            }
        return _norm(fp)
    if stage == "bundle_adjustment":
        return _norm(
            {
                **dataclasses.asdict(config.bundle_adjustment),
                "cameras": _camera_geometry(config),
                "skeleton": _skeleton_digest(config),
            }
        )
    if stage == "pictorial_structures":
        p = config.pictorial
        return _norm(
            {
                "k": p.k,
                "temporal": p.temporal,
                "lam": p.lam,
                "skeleton": _skeleton_digest(config),
                "cameras_from": _cameras_entry(config, enabled, store),
            }
        )
    if stage == "triangulation":
        return _norm(
            {
                **dataclasses.asdict(config.triangulation),
                "cameras_from": _cameras_entry(config, enabled, store),
                "pts2d_from": pts2d_source(enabled, store),
            }
        )
    if stage == "eks":
        eks = config.eks
        fp = {
            # The smoother has no external solver to name, but it does have
            # numerics of its own; see EKS_REVISION.
            "revision": EKS_REVISION,
            **dataclasses.asdict(eks),
            "cameras_from": _cameras_entry(config, enabled, store),
            "pts2d_from": pts2d_source(enabled, store),
            "init3d_from": eks_init_source(enabled, store),
        }
        if eks.ensemble:
            # The members' *content* is digested, not just their paths: re-running
            # another model's pipeline rewrites its results.h5 under the same name,
            # which a path alone cannot see (as for the calibration in
            # :func:`_camera_geometry`).
            fp["ensemble"] = [
                {"path": str(p), "content": _file_digest(p)} for p in eks.ensemble
            ]
        return _norm(fp)
    if stage == "postprocess":
        return _norm(
            {
                **dataclasses.asdict(config.postprocess),
                # The names are resolved to columns against the skeleton, so a
                # skeleton whose point order changed freezes different points.
                "skeleton": _skeleton_digest(config),
                "cameras_from": _cameras_entry(config, enabled, store),
                "pose_from": postprocess_source(enabled, store),
            }
        )
    if stage == "inverse_kinematics":
        p = config.inverse_kinematics
        return _norm(
            {
                # A *new* key invalidates a cached stage, a dropped one does not (see
                # this module's docstring), so the solver identity has to be recorded
                # explicitly: without it, an output directory written by the previous
                # scipy solver would silently pass as valid QuickIK output.
                "solver": IK_SOLVER,
                "solver_revision": IK_SOLVER_REVISION,
                "template": _ik_template_digest(config),
                "articulation": _ik_articulation_digest(config),
                "n_iterations": p.n_iterations,
                "neutral_weight": p.neutral_weight,
                "damping": p.damping,
                "position_tolerance": p.position_tolerance,
                "angle_tolerance": p.angle_tolerance,
                "fixed_body": p.fixed_body,
                # Changes the plan's segment lengths, hence the whole fit.
                "symmetric_segments": p.symmetric_segments,
                "weigh_by_confidence": p.weigh_by_confidence,
                # Segmentation changes the answer (each segment restarts from neutral),
                # so it belongs here even though it reads like a performance knob.
                "parallel": p.parallel,
                "segment_len": p.segment_len,
                "overlap_len": p.overlap_len,
                "skeleton": _skeleton_digest(config),
                "pts3d_from": pts3d_source(enabled, store),
            }
        )
    if stage == "visualization":
        return _norm(
            {
                "videos": [dataclasses.asdict(spec) for spec in config.videos],
                "mesh_hide": list(config.visualization.get("mesh_hide", ["wings"])),
                "skeleton": _skeleton_digest(config, cosmetic=True),
                "pose_from": pose_sources(enabled, store),
                "nmf_from": nmf_source(enabled, store),
                "cameras_from": _cameras_entry(config, enabled, store),
            }
        )
    raise ValueError(f"unknown stage {stage!r}")


def _ik_template_digest(config: Config) -> dict:
    """The template choice + resolved per-DOF bounds (captures bounds overrides)."""
    t = config.ik_template()
    bounds = {}
    for leg in t.legs:
        lo, hi = leg.bounds
        for name, blo, bhi in zip(leg.dof_names, lo, hi):
            bounds[name] = [float(blo), float(bhi)]
    return {"name": t.name, "dof_names": t.dof_names, "bounds": bounds}


def _ik_articulation_digest(config: Config) -> dict | None:
    """The fitted head/abdomen chains: their per-DOF bounds AND their marker placement.

    The markers belong here as much as the bounds do. A ``[inverse_kinematics.head]`` or
    ``[inverse_kinematics.abdomen]`` table says *where* each tracked keypoint sits on the
    model -- which body it rides and at what offset -- and that decides what the fit is
    fitting. Retargeting a chain and re-running with only the bounds recorded reused the
    previous fit, silently, while the config on disk described a different one; the default
    config's own retarget instructions were a way to hit exactly that.

    Recorded as the resolved neutral positions rather than as the config table, so a
    retarget expressed any of the three ways it can be -- a config table, a rebuilt
    articulation asset, a different base-point nomination -- is one comparison. Rounded
    because these are floats that came from an asset and a config, not from arithmetic: an
    exact compare would make the digest sensitive to a re-export that moved nothing.
    """
    art = config.ik_articulation()
    if art is None:
        return None
    chains = {}
    for chain in art.chains:
        lo, hi = chain.bounds
        chains[chain.name] = {
            "bounds": {
                name: [float(blo), float(bhi)]
                for name, blo, bhi in zip(chain.dof_names, lo, hi)
            },
            "markers": {
                name: [round(float(v), 6) for v in neutral]
                for name, neutral in zip(chain.marker_names, chain.marker_neutral)
            },
            "marker_depth": list(chain.marker_depth),
            "base_point": chain.base_point,
        }
    return chains


# -- comparison ----------------------------------------------------------------


def _short(value, limit: int = 120) -> str:
    s = json.dumps(value, sort_keys=True, default=str)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def fingerprint_diff(stored: dict | None, expected: dict) -> list[str]:
    """Readable ``key: old -> new`` lines where ``expected`` disagrees with ``stored``.

    Subset semantics: every ``expected`` key must match the stored value; keys
    present only in ``stored`` are ignored (see the module docstring). An empty
    list means the cached output is still parameter-valid.
    """
    diffs: list[str] = []

    def walk(path: tuple[str, ...], old, new) -> None:
        if isinstance(old, dict) and isinstance(new, dict):
            for key in new:
                if key not in old:
                    diffs.append(
                        f"{'.'.join((*path, key))}: (absent) -> {_short(new[key])}"
                    )
                elif old[key] != new[key]:
                    walk((*path, key), old[key], new[key])
            return
        diffs.append(f"{'.'.join(path)}: {_short(old)} -> {_short(new)}")

    walk((), stored or {}, _norm(expected))
    return diffs


def stage_valid(
    stage: str,
    config: Config,
    expected: dict,
    store: StageStore,
    record: RunRecord,
    outdir: Path,
) -> tuple[bool, str | None]:
    """Whether ``stage``'s cached output can be reused, with the reason if not.

    Reuse requires a recorded fingerprint that matches ``expected`` *and* the
    output itself to be present (the stage's ``results.h5`` group, or every
    currently-specced MP4 for ``visualization``).

    Returns
    -------
    ok : bool
        ``True`` to reuse the cache.
    reason : str or None
        Why the stage must recompute (``None`` when ``ok``).
    """
    stored = record.get(stage)
    if stored is None:
        return False, "no cached result recorded"
    diff = fingerprint_diff(stored, expected)
    if diff:
        return False, "config changed: " + "; ".join(diff)
    if stage == "visualization":
        missing = [
            spec.video_name
            for spec in config.videos
            if not (Path(outdir) / f"{spec.video_name}.mp4").exists()
        ]
        if missing:
            return False, f"rendered video(s) missing: {', '.join(missing)}"
        return True, None
    if not store.has(stage):
        return False, "output missing from results.h5"
    return True, None
