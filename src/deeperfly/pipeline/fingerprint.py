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
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ..config import STAGES, Config
from ..results import StageStore

log = logging.getLogger("deeperfly")

#: ``run.json`` schema version; an unknown version is reset (recompute all).
RECORD_VERSION = 1


# -- the run record (<outdir>/run.json) ---------------------------------------


class RunRecord:
    """The fingerprints of the stage outputs currently cached in an output dir.

    A small JSON sidecar next to ``poses.h5``: validity bookkeeping is
    outdir-local run state, kept out of the portable result file (and a viz-only
    run never touches ``poses.h5`` at all). Deleting it merely recomputes
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
    """The skeleton parts that affect geometry (+ drawing, when ``cosmetic``)."""
    skel = config.skeleton()
    digest = {
        "name": skel.name,
        "point_names": list(skel.point_names),
        "bones": skel.bones.tolist(),
    }
    if cosmetic:  # the visualization stage also draws limbs/colors
        digest["limb_names"] = list(skel.limb_names)
        digest["limb_id"] = skel.limb_id.tolist()
        digest["palette"] = dict(skel.palette)
    return digest


def _camera_geometry(config: Config) -> dict:
    """The ``[cameras]`` table -- the views' pure geometry (intrinsics/extrinsics)."""
    defaults, cams = config.camera_table()
    return {
        "defaults": dict(defaults),
        "cameras": {n: dict(s) for n, s in cams.items()},
    }


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


def pose_sources(enabled: dict[str, bool], store: StageStore) -> dict[str, str | None]:
    """Which stage outputs the visualization draws (2D and 3D separately)."""
    for stage in ("triangulation", "pictorial_structures"):
        if enabled[stage] and store.has(stage):
            return {"pts2d": stage, "pts3d": stage}
    return {"pts2d": "pose2d", "pts3d": None}


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
            "precision": p.precision,
            "sources": plan.source_patterns(),
            "preprocessors": {
                name: t.to_json() for name, t in plan.preprocessors.items()
            },
            "models": {
                name: {
                    "class": s.cls,
                    "weights": s.weights,
                    "input_size": list(s.input_size),
                    "mean": s.mean,
                    "n_out_channels": s.n_out_channels,
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
            fp["candidates"] = {"k": config.pictorial.k}
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
    if stage == "visualization":
        return _norm(
            {
                "videos": [dataclasses.asdict(spec) for spec in config.videos],
                "skeleton": _skeleton_digest(config, cosmetic=True),
                "pose_from": pose_sources(enabled, store),
                "cameras_from": _cameras_entry(config, enabled, store),
            }
        )
    raise ValueError(f"unknown stage {stage!r}")


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
    output itself to be present (the stage's ``poses.h5`` group, or every
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
        return False, "output missing from poses.h5"
    return True, None
