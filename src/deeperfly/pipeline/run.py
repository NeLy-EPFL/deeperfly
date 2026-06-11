"""Per-recording orchestration: fingerprint-driven cache reuse over the stage loop.

An enabled stage is reused when its recorded fingerprint still matches the
current config and its output is present (see
:mod:`deeperfly.pipeline.fingerprint`); it recomputes when the config changed,
the output is missing, ``--overwrite`` selects it, or any upstream enabled
stage recomputed this run (its inputs changed -- the cascade). Each stage
persists only its own ``poses.h5`` group (:class:`~deeperfly.results.StageStore`),
so downstream re-runs always read pristine upstream outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..config import STAGES, Config
from ..recordings import require_input_footage
from ..results import StageStore
from . import stages
from .fingerprint import RunRecord, stage_fingerprint, stage_valid

log = logging.getLogger("deeperfly")


@dataclass
class _RunContext:
    """Everything a stage runner needs for one recording."""

    config: Config
    enabled: dict[str, bool]
    store: StageStore
    record: RunRecord
    outdir: Path
    sources: dict[str, list[Path]] | None
    input: object
    progress: object


def run_recording(
    config_path: str | None,
    outdir: Path,
    *,
    sources: dict[str, list[Path]] | None = None,
    input=None,
    overwrite: list[str] | None = None,
    progress=None,
) -> None:
    """Run the config's enabled stages for a single recording, reusing cache.

    The config is resolved against ``outdir`` (see :meth:`Config.read_for_run`) and
    its ``[pipeline].do_<stage>`` toggles decide which stages run
    (:meth:`Config.stage_flags`). An enabled stage reuses its cached result when
    its parameters are unchanged and its output is present; editing the config
    recomputes exactly the affected stages (and everything downstream). The
    ``pose2d`` cache always feeds downstream; a *derived* stage's output feeds
    downstream only while that stage is enabled.

    A stage runs only if its input is available -- footage for ``pose2d``, a 2D
    pose for ``bundle_adjustment`` / ``triangulation``, cached candidates for
    ``pictorial_structures``, a result for ``visualization``; a stage whose
    input is missing is skipped with the reason logged.

    Parameters
    ----------
    config_path
        The ``-c`` config path (or ``None`` for the snapshot/default; see
        :meth:`Config.read_for_run`).
    outdir
        The recording's output directory (config snapshot + cached results).
    sources, input
        The recording's footage (see :func:`deeperfly.recordings.camera_sources`);
        ``sources`` is the pre-resolved map (``deeperfly run``), ``input`` a raw
        recording directory a library caller can pass instead.
    overwrite
        Stage names to force-recompute (see
        :func:`deeperfly.pipeline.overwrite_stages`); config changes are
        detected automatically.
    progress
        Optional progress factory threaded into the detector and the compositor.
    """
    outdir = Path(outdir)
    config = Config.read_for_run(config_path, outdir)
    enabled = config.stage_flags()  # config validated at construction
    overwrite_set: set[str] = stages.overwrite_stages(overwrite)

    store = StageStore(outdir / "poses.h5")
    record = RunRecord(outdir / "run.json")

    # Validate the footage *before* creating the output dir, so a fresh run that
    # can't read its input fails cleanly instead of leaving an empty dir behind.
    # Only pose2d decodes the recording, and only when it recomputes; a resume
    # reusing a cached 2D pose needs no footage.
    if enabled["pose2d"] and (
        "pose2d" in overwrite_set
        or not stage_valid(
            "pose2d",
            config,
            stage_fingerprint("pose2d", config, enabled, store),
            store,
            record,
            outdir,
        )[0]
    ):
        require_input_footage(config, sources=sources, input=input)

    outdir.mkdir(parents=True, exist_ok=True)
    log.info("output directory: %s", outdir)
    config.save_snapshot(outdir)
    log.info(
        "stages: %s",
        ", ".join(f"{n}={'on' if enabled[n] else 'off'}" for n in STAGES),
    )

    ctx = _RunContext(
        config=config,
        enabled=enabled,
        store=store,
        record=record,
        outdir=outdir,
        sources=sources,
        input=input,
        progress=progress,
    )

    # Which stages carry a result from a previous run, snapshotted before the
    # loop (record.set drops downstream entries as stages complete). A stage
    # with no prior record runs for the first time -- that is not a "recompute"
    # and warrants no reason.
    had_record = {name: record.get(name) is not None for name in STAGES}

    recomputed = False  # has any enabled stage recomputed this run? -> cascade
    for name in STAGES:
        if not enabled[name]:
            continue
        expected = stage_fingerprint(name, config, enabled, store)
        if name in overwrite_set:
            reason = "--overwrite"
        elif recomputed:
            reason = "an upstream stage recomputed (its inputs changed)"
        else:
            ok, why = stage_valid(name, config, expected, store, record, outdir)
            if ok:
                log.info(
                    "reusing cached %s (pass --overwrite %s to force a recompute)",
                    name,
                    name,
                )
                continue
            reason = why or "unknown"
        if had_record[name]:
            _log_recompute(name, reason)
        else:
            log.info("running %s", name)
        if _RUNNERS[name](ctx):
            record.set(name, expected)
            recomputed = True


def _log_recompute(name: str, reason: str) -> None:
    """Announce that a stage's cached result is stale and being recomputed;
    loudly for the slow detection stage."""
    if name == "pose2d":
        if "candidates" in reason:
            reason += (
                " -- pictorial_structures needs the detector's top-K candidates,"
                " which are extracted during detection"
            )
        log.warning("recomputing pose2d, the slow detection stage (%s)", reason)
    else:
        log.info("recomputing %s (%s)", name, reason)


# -- per-stage runners ---------------------------------------------------------
#
# Each runner gathers its inputs from the store, computes, and persists only its
# own group; it returns False (and logs why) when its inputs are missing, so the
# run record is untouched and the cascade is not triggered.


def _no_2d(ctx: _RunContext, stage: str) -> bool:
    if ctx.store.has("pose2d"):
        return False
    log.warning(
        "skipping %s: no 2D pose available -- enable [pipeline].do_pose2d or leave "
        "a cached poses.h5 with 2D in %s",
        stage,
        ctx.outdir,
    )
    return True


def _run_pose2d(ctx: _RunContext) -> bool:
    cameras, skeleton, pts2d, conf, candidates, image_sizes = stages.stage_pose2d(
        ctx.config,
        sources=ctx.sources,
        input=ctx.input,
        want_candidates=ctx.enabled["pictorial_structures"],
        progress=ctx.progress,
    )
    # Truncates the whole file: a fresh detection invalidates everything downstream.
    ctx.store.write_pose2d(
        cameras=cameras,
        skeleton=skeleton,
        pts2d=pts2d,
        conf=conf,
        image_sizes=image_sizes,
        candidates=candidates,
    )
    log.info(
        "wrote %s  (%d frames, %d views)",
        ctx.store.path,
        pts2d.shape[1],
        pts2d.shape[0],
    )
    return True


def _run_bundle_adjustment(ctx: _RunContext) -> bool:
    if _no_2d(ctx, "bundle_adjustment"):
        return False
    _pose2d = ctx.store.read_pose2d()
    assert _pose2d is not None
    pts2d, conf = _pose2d
    refined = stages.stage_bundle_adjustment(
        ctx.config,
        # Always the un-refined config rig, never a prior BA output, so an edited
        # [cameras] / [bundle_adjustment] re-runs bundle adjustment from the config.
        stages.config_rig_from_store(ctx.config, ctx.store),
        pts2d,
        conf,
        ctx.store.read_skeleton(),
    )
    ctx.store.truncate_from("bundle_adjustment")
    ctx.store.write_cameras("bundle_adjustment", refined)
    return True


def _run_pictorial_structures(ctx: _RunContext) -> bool:
    if _no_2d(ctx, "pictorial_structures"):
        return False
    candidates = ctx.store.read_candidates()
    if candidates is None:
        # Only reachable when pose2d is disabled: an enabled pose2d would have
        # re-detected (the candidates clause of its fingerprint).
        log.warning(
            "skipping pictorial_structures: the cached 2D result stores no top-K "
            "candidates -- enable [pipeline].do_pose2d to re-detect them"
        )
        return False
    _pose2d = ctx.store.read_pose2d()
    assert _pose2d is not None
    pts2d, _ = _pose2d
    new2d, pts3d, reproj = stages.stage_pictorial_structures(
        ctx.config,
        stages.select_cameras(ctx.config, ctx.enabled, ctx.store),
        ctx.store.read_skeleton(),
        candidates,
        pts2d,
    )
    ctx.store.truncate_from("pictorial_structures")
    ctx.store.write_points(
        "pictorial_structures", pts2d=new2d, pts3d=pts3d, reproj_error=reproj
    )
    return True


def _run_triangulation(ctx: _RunContext) -> bool:
    if _no_2d(ctx, "triangulation"):
        return False
    _pose2d = ctx.store.read_pose2d()  # detector confidences for optional weighting
    assert _pose2d is not None
    _, conf = _pose2d
    pts2d, pts3d, reproj = stages.stage_triangulation(
        ctx.config,
        stages.select_cameras(ctx.config, ctx.enabled, ctx.store),
        stages.select_pts2d(ctx.enabled, ctx.store),
        conf,
    )
    ctx.store.truncate_from("triangulation")
    ctx.store.write_points(
        "triangulation", pts2d=pts2d, pts3d=pts3d, reproj_error=reproj
    )
    return True


def _run_visualization(ctx: _RunContext) -> bool:
    result = stages.assemble_result(ctx.config, ctx.enabled, ctx.store)
    if result is None:
        log.warning(
            "skipping visualization: no pose result available -- enable a stage "
            "above or leave a cached poses.h5 in %s",
            ctx.outdir,
        )
        return False
    # MP4s an earlier run rendered that the current config does not spec are
    # left on disk (the output dir may hold user files); just point them out.
    stored = ctx.record.get("visualization") or {}
    stale = sorted(
        {v.get("video_name") for v in stored.get("videos", [])}
        - {spec.video_name for spec in ctx.config.videos}
        - {None}
    )
    if stale:
        log.info(
            "video(s) not in the current config (their MP4s are left in place): %s",
            ", ".join(stale),
        )
    stages.render_videos(
        ctx.config, result, ctx.outdir, sources=ctx.sources, progress=ctx.progress
    )
    return True


_RUNNERS = {
    "pose2d": _run_pose2d,
    "bundle_adjustment": _run_bundle_adjustment,
    "pictorial_structures": _run_pictorial_structures,
    "triangulation": _run_triangulation,
    "visualization": _run_visualization,
}
