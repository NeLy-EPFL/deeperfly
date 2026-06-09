"""Per-recording orchestration: resolve stages, reuse cache, run the pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import STAGES, Config
from ..io import PoseResult
from ..recordings import require_input_footage
from . import stages

log = logging.getLogger("deeperfly")


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
    (:meth:`Config.stage_flags`). An enabled stage reuses its result if it's already
    in the output dir, recomputing only when missing or ``overwrite`` selects it;
    recomputing a stage cascades to every enabled stage downstream (their inputs
    changed).

    Each stage runs only if its input is available -- footage for ``pose2d``, a 2D
    pose for ``bundle_adjustment`` / ``triangulation``, candidates for
    ``pictorial_structures``, a result for ``visualization`` -- from an upstream
    stage or the cached ``poses.h5``; a stage whose input is missing is skipped
    with the reason logged. A disabled stage never runs but its cached output still
    feeds downstream.

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
        Stage names to recompute (see :func:`deeperfly.pipeline.overwrite_stages`).
    progress
        Optional progress factory threaded into the detector and the compositor.
    """
    config = Config.read_for_run(config_path, outdir)
    enabled = config.stage_flags()  # config validated at construction
    overwrite = stages.overwrite_stages(overwrite)

    # `cached` is the result already in the output dir; `result` starts there, a
    # reused stage keeps it, a recomputed stage replaces it. The first recompute
    # flips `recomputed` so every later enabled stage recomputes too (cascade).
    h5_path = outdir / "poses.h5"
    cached = PoseResult.load(h5_path) if h5_path.exists() else None

    # Validate the footage *before* creating the output dir, so a fresh run that
    # can't read its input fails cleanly instead of leaving an empty dir behind.
    # Only pose2d decodes the recording, and only when it recomputes; a resume
    # reusing a cached 2D pose needs no footage.
    if enabled["pose2d"] and (
        "pose2d" in overwrite
        or not stages.stage_cached("pose2d", cached, config, outdir)
    ):
        require_input_footage(config, sources=sources, input=input)

    outdir.mkdir(parents=True, exist_ok=True)
    log.info("output directory: %s", outdir)
    config.save_snapshot(outdir)
    log.info(
        "stages: %s",
        ", ".join(f"{n}={'on' if enabled[n] else 'off'}" for n in STAGES),
    )

    result = cached
    frames = candidates = None
    produced = False  # whether we computed new 2D/3D worth persisting
    recomputed = False  # has any enabled stage recomputed this run? -> cascade
    fresh_rig = (
        False  # are result.cameras the un-refined config rig (vs. cached BA output)?
    )

    def _recompute(stage: str) -> bool:
        """Whether enabled ``stage`` should recompute rather than reuse its cache."""
        if stage in overwrite or recomputed:
            return True
        if stages.stage_cached(stage, cached, config, outdir):
            log.info(
                "reusing cached %s (pass --overwrite %s to recompute)", stage, stage
            )
            return False
        return True

    if enabled["pose2d"] and _recompute("pose2d"):
        result, candidates, frames, _ = stages.stage_pose2d(
            config,
            sources=sources,
            input=input,
            want_candidates=enabled["pictorial_structures"],
            progress=progress,
        )
        produced = recomputed = True
        fresh_rig = True  # pose2d built the cameras from the config

    if enabled["bundle_adjustment"] and _recompute("bundle_adjustment"):
        if stages._has_2d(result):
            # If the rig to refine is itself a *previous* BA output (cached and
            # marked), rebuild the un-refined config rig first, so this recompute
            # starts from the (edited) config instead of re-refining already-refined
            # cameras. Cached cameras never BA-refined are kept as-is.
            if (
                not fresh_rig
                and cached is not None
                and cached.meta.get("bundle_adjustment")
            ):
                try:
                    result.cameras = stages.config_camera_rig(
                        config, sources=sources, input=input
                    )
                    fresh_rig = True
                except SystemExit as exc:
                    log.warning(
                        "bundle_adjustment: could not rebuild the camera rig from the "
                        "config (%s) -- refining the cached cameras instead; re-run "
                        "from pose2d with the recording to recalibrate from scratch",
                        exc,
                    )
            result = stages.stage_bundle_adjustment(config, result)
            produced = recomputed = True
        else:
            log.warning(
                "skipping bundle_adjustment: no 2D pose available -- enable "
                "[pipeline].do_pose2d or leave a cached poses.h5 with 2D in %s",
                outdir,
            )

    if enabled["pictorial_structures"] and _recompute("pictorial_structures"):
        if candidates is not None and stages._has_2d(result):
            result = stages.stage_pictorial_structures(config, result, candidates)
            produced = recomputed = True
        elif not stages._has_2d(result):
            log.warning(
                "skipping pictorial_structures: no 2D pose available -- enable "
                "[pipeline].do_pose2d or leave a cached poses.h5 with 2D in %s",
                outdir,
            )
        else:
            log.warning(
                "skipping pictorial_structures: it needs the detector's top-K "
                "candidates, which a cached 2D result does not store -- enable "
                "[pipeline].do_pose2d to re-detect from the recording"
            )

    if enabled["triangulation"] and _recompute("triangulation"):
        if stages._has_2d(result):
            result = stages.stage_triangulation(config, result)
            produced = recomputed = True
        else:
            log.warning(
                "skipping triangulation: no 2D pose available -- enable "
                "[pipeline].do_pose2d or leave a cached poses.h5 with 2D in %s",
                outdir,
            )

    if produced and result is not None:
        result.save(h5_path)
        log.info(
            "wrote %s  (%d frames, %d views)", h5_path, result.n_frames, result.n_views
        )

    if enabled["visualization"] and _recompute("visualization"):
        if result is not None:
            stages.render_videos(
                config, result, frames, outdir, sources=sources, progress=progress
            )
        else:
            log.warning(
                "skipping visualization: no pose result available -- enable a stage "
                "above or leave a cached poses.h5 in %s",
                outdir,
            )
