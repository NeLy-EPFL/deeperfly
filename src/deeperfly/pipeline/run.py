"""Per-recording orchestration: fingerprint-driven cache reuse over the stage loop.

An enabled stage is reused when its recorded fingerprint still matches the
current config and its output is present (see
:mod:`deeperfly.pipeline.fingerprint`); it recomputes when the config changed,
the output is missing, ``--overwrite`` selects it, or any upstream enabled
stage recomputed this run (its inputs changed -- the cascade). Each stage
persists only its own ``results.h5`` group (:class:`~deeperfly.results.StageStore`),
so downstream re-runs always read pristine upstream outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import STAGES, Config
from ..recordings import require_input_footage
from ..results import StageStore
from . import stages
from .fingerprint import (
    RunRecord,
    postprocess_source,
    stage_fingerprint,
    stage_valid,
)

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
    ``pictorial_structures``, a 3D pose for ``postprocess`` /
    ``inverse_kinematics``, a result for ``visualization``; a stage whose
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

    # Narrow to the footage this recording actually has, BEFORE anything is fingerprinted.
    # The fingerprints are computed from the config and never see footage, so this ordering
    # is what keeps the cache honest: a run that proceeded on seven views records a
    # seven-view fingerprint, and the eighth camera turning up later narrows differently and
    # recomputes. Narrowing any later -- inside `stage_pose2d`, where `auto_crops` is applied
    # -- would store the full fingerprint against the short result, and the next complete
    # run would reuse it.
    if enabled_footage := _resolved_sources(config, sources=sources, input=input):
        config = config.narrowed_to_sources(enabled_footage)
    # And by the rig, for the other way a view can be unusable: a calibration is a
    # measurement of which cameras exist, so a view it does not cover has no extrinsics to
    # project through. Narrowing the rig alone left `pts2d` with more view rows than the rig
    # had cameras, which surfaced as an einsum shape error naming no camera.
    config = config.narrowed_to_covered_views(_rig_coverage(config))
    enabled = config.stage_flags()  # config validated at construction
    overwrite_set: set[str] = stages.overwrite_stages(overwrite)

    store = StageStore(outdir / "results.h5")
    record = RunRecord(outdir / "run.json")

    # Before anything is read or computed: a cached pose on another skeleton makes every
    # array in the file mean something else, and there is no point costing the user a
    # detection pass to find that out.
    _refuse_a_foreign_skeleton(config, store)

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


def _resolved_sources(
    config: Config, *, sources: dict[str, list[Path]] | None, input=None
) -> dict[str, list[Path]] | None:
    """What footage this run has, per source, or ``None`` when that cannot be known yet.

    ``None`` is not "nothing": a resume that reuses a cached 2D pose is handed neither a
    resolved map nor a recording root, and must NOT be narrowed -- there is no evidence of
    absence, and narrowing on no evidence would silently drop every view.
    """
    if sources is not None:
        return sources
    if input is None:
        return None
    from ..recordings import source_sources

    return dict(source_sources(config, input=input))


def _rig_coverage(config: Config) -> list[str]:
    """The views the config's rig can place, which is every declared one unless a solved
    calibration says otherwise.

    An orbit describes whatever the config declares, so there is nothing to narrow; a
    calibration is a measurement, and it can cover fewer. An unreadable one is left to the
    stage that actually needs it, so a config error is reported once, where it belongs.
    """
    declared = [
        v
        for v, spec in (config.data.get("cameras") or {}).items()
        if isinstance(spec, dict)
    ]
    path = config.calibration_path()
    if path is None:
        return declared
    try:
        from ..calibration import Calibration

        have = set(Calibration.load(path).cameras.cameras)
    except Exception:  # noqa: BLE001 -- not this function's error to report
        return declared
    return [v for v in declared if v in have]


def _refuse_a_foreign_skeleton(config: Config, store: StageStore) -> None:
    """Refuse to continue a run whose stored pose is on a different skeleton.

    Every array in ``results.h5`` is ``(..., P, ...)`` with no names beside it, so the
    meaning of the ``P`` axis is whatever skeleton the file records -- and a resume that
    does not recompute ``pose2d`` never rewrites that record. If the config now resolves
    a *different* skeleton, the downstream stages read one skeleton's names against the
    other's columns, and the failure surfaces (if at all) as an unrelated complaint about
    an unknown point name.

    Two 38-point skeletons load each other's files perfectly happily, so a count check
    cannot see this -- only the ordered names can. That is also why the check is by name
    list and not by the skeleton's *name*: a preset can be renamed without a single
    coordinate changing, and a name can be reused for a different point order.
    """
    stored = store.read_skeleton()
    if stored is None:  # a fresh output directory: pose2d will write the record
        return
    want = list(config.skeleton().point_names)
    have = list(stored.point_names)
    if have == want:
        return
    gone = [n for n in have if n not in want]
    extra = [n for n in want if n not in have]
    raise SystemExit(
        f"this output directory holds a pose on a different skeleton than the config "
        f"resolves, so its arrays cannot be read against it.\n"
        f"  stored in results.h5 : {len(have)} points, named {stored.name!r}\n"
        f"  the config resolves  : {len(want)} points, named {config.skeleton().name!r}\n"
        + (f"  only in the stored one: {', '.join(gone[:6])}\n" if gone else "")
        + (f"  only in the config's  : {', '.join(extra[:6])}\n" if extra else "")
        + (
            "  (same points, different ORDER -- every stored column means another point)\n"
            if not gone and not extra
            else ""
        )
        + "Point [skeleton] at the skeleton this pose was detected on, run into a fresh "
        "output directory, or migrate the labels with `deeperfly project skeleton`."
    )


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
        "a cached results.h5 with 2D in %s",
        stage,
        ctx.outdir,
    )
    return True


def _footage_by_view(
    config, sources: dict[str, list[Path]] | None
) -> dict[str, list[Path]] | None:
    """Re-key resolved footage from source name to view (camera) name.

    ``sources`` is keyed by *source* name (config order), but the store records
    footage keyed by *view* name -- matching ``image_sizes`` and ``cameras`` --
    so the viewer can look each camera's footage up by its own name (a view's
    source comes from the detection plan). Returns ``None`` when there is no
    footage to record.
    """
    if not sources:
        return None
    view_sources = config.detection_plan().view_sources()
    return {view: sources[src] for view, src in view_sources.items() if src in sources}


def _run_pose2d(ctx: _RunContext) -> bool:
    cameras, skeleton, pts2d, conf, candidates, image_sizes = stages.stage_pose2d(
        ctx.config,
        sources=ctx.sources,
        input=ctx.input,
        want_candidates=ctx.enabled["pictorial_structures"],
        progress=ctx.progress,
        outdir=ctx.outdir,
        # This runner only runs when pose2d recomputes, and an automatic crop is part of
        # what detection *is*: re-detecting through a box carried over from a run whose
        # inputs have since changed would silently keep a stale window.
        force_autocrop=True,
    )
    # Truncates the whole file: a fresh detection invalidates everything downstream.
    ctx.store.write_pose2d(
        cameras=cameras,
        skeleton=skeleton,
        pts2d=pts2d,
        conf=conf,
        image_sizes=image_sizes,
        footage=_footage_by_view(ctx.config, ctx.sources),
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
    report: dict = {}
    refined = stages.stage_bundle_adjustment(
        ctx.config,
        # Always the un-refined config rig, never a prior BA output, so an edited
        # [cameras] / [bundle_adjustment] re-runs bundle adjustment from the config.
        stages.config_rig_from_store(ctx.config, ctx.store),
        pts2d,
        conf,
        ctx.store.read_skeleton(),
        absent=ctx.store.read_animal()[0],
        report=report,
    )
    ctx.store.truncate_from("bundle_adjustment")
    ctx.store.write_cameras(
        "bundle_adjustment",
        refined,
        image_sizes=ctx.store.read_image_sizes(),
        meta=_rig_meta(ctx, report),
    )
    _write_calibration(ctx, refined, report)
    return True


def _rig_meta(ctx: _RunContext, report: dict) -> dict:
    """The refined rig's units, scale source and provenance -- inherited, not invented.

    A run may build its rig from a **calibration file** (``[cameras].calibration``), which
    already knows what its numbers mean: a board solve in millimeters, a known-distance
    scale, or an orbit prior in arbitrary units. Refining that rig does not change any of
    it, so the answer is *inherited*. Only when the config describes the rig as a bare orbit
    is the honest answer the config-orbit one -- and that is what was being asserted
    unconditionally, which re-labelled a millimeter rig as arbitrary.
    """
    ba = ctx.config.bundle_adjustment
    units, scale_source, parent = "config", "orbit_prior", None
    path = ctx.config.calibration_path()
    if path is not None:
        try:
            from ..calibration import Calibration

            parent_cal = Calibration.load(path)
            units = parent_cal.units
            scale_source = parent_cal.scale_source
            parent = {"name": parent_cal.name, "path": str(path)}
        except Exception as exc:  # a rig that loaded once for the solve, unreadable now
            log.warning("could not read the parent calibration %s: %s", path, exc)
    provenance: dict = {
        "method": "labels_ba",
        "intrinsics": "config",
        "frames": report.get("n_frames"),
        "source": str(ctx.store.path),
        "solver": {
            "weigh_by_confidence": bool(ba.weigh_by_confidence),
            "max_frames": ba.max_frames,
            "frame_sampling": ba.frame_sampling,
            **{k: v for k, v in ba.least_squares.items() if _is_scalar(v)},
        },
    }
    if parent is not None:
        provenance["refined_from"] = parent
    return {
        "units": units,
        "scale_source": scale_source,
        "provenance": provenance,
        "quality": report.get("quality") or {},
    }


def _write_calibration(ctx: _RunContext, refined, report: dict) -> None:
    """Mirror the refined rig into ``<outdir>/calibration.toml``.

    ``results.h5`` already holds these cameras, but only as arrays inside one
    recording's file: nothing can point a *second* recording at them, diff them against
    a later solve, or read their residuals without opening HDF5. The calibration file is
    the same rig in the form that travels (see :mod:`deeperfly.calibration`).

    Best-effort: a rig that cannot be written must not fail a run whose real output
    (``results.h5``) is already committed.
    """
    from ..calibration import CALIBRATION_FILENAME, Calibration

    meta = _rig_meta(ctx, report)
    try:
        Calibration.from_camera_group(
            refined,
            name=ctx.outdir.parent.name or ctx.outdir.name,
            image_sizes=ctx.store.read_image_sizes(),
            # Inherited from whatever rig this refined, so a millimeter calibration stays a
            # millimeter calibration. Only a bare-orbit config yields the unnamed unit --
            # deeperfly was never told what an orbit's `distance` measures.
            units=meta["units"],
            scale_source=meta["scale_source"],
            provenance=meta["provenance"],
            quality=meta["quality"],
        ).save(ctx.outdir / CALIBRATION_FILENAME)
    except Exception:  # pragma: no cover -- a read-only outdir must not fail the run
        log.exception("could not write %s", ctx.outdir / CALIBRATION_FILENAME)


def _is_scalar(value) -> bool:
    """Whether a ``[bundle_adjustment]`` leftover key is a plain TOML scalar.

    The leftovers go straight to ``scipy.optimize.least_squares``, so they may hold
    things the calibration's provenance block has no way to write (a tuple of bounds, a
    callable). Those are dropped rather than stringified -- a provenance record that
    lies about the solver settings is worse than one that omits them.
    """
    return isinstance(value, (str, int, float, bool)) and not isinstance(value, bytes)


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
        absent=ctx.store.read_animal()[0],
    )
    ctx.store.truncate_from("triangulation")
    ctx.store.write_points(
        "triangulation", pts2d=pts2d, pts3d=pts3d, reproj_error=reproj
    )
    return True


def _load_ensemble_members(ctx: _RunContext) -> list | None:
    """``[eks].ensemble`` result files -> ``(pts2d, conf)`` pairs, or ``None`` on error.

    Each entry names another run's ``results.h5`` -- a *different detector* over the
    same recording -- whose 2D becomes another ensemble member. Paths are resolved
    relative to this recording's output directory, so a project can point at a
    sibling run without absolute paths. A member that cannot be read aborts the
    stage rather than silently shrinking the ensemble: an ensemble that quietly
    lost half its members would report the same numbers with different meaning.
    """
    paths = ctx.config.eks.ensemble
    if not paths:
        return []
    members = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = (ctx.outdir / path).resolve()
        other = StageStore(path)
        got = other.read_pose2d()
        if got is None:
            log.warning(
                "skipping eks: [eks].ensemble member %s has no 2D pose (looked in %s)",
                raw,
                path,
            )
            return None
        members.append(got)
    return members


def _run_eks(ctx: _RunContext) -> bool:
    if _no_2d(ctx, "eks"):
        return False
    _pose2d = ctx.store.read_pose2d()
    assert _pose2d is not None
    _, conf = _pose2d
    members = _load_ensemble_members(ctx)
    if members is None:
        return False
    pts2d, pts3d, reproj, result = stages.stage_eks(
        ctx.config,
        stages.select_cameras(ctx.config, ctx.enabled, ctx.store),
        stages.select_pts2d(ctx.enabled, ctx.store),
        conf,
        init3d=stages.select_eks_init(ctx.enabled, ctx.store),
        absent=ctx.store.read_animal()[0],
        members=members,
    )
    ctx.store.truncate_from("eks")
    ctx.store.write_points(
        "eks",
        pts2d=pts2d,
        pts3d=pts3d,
        reproj_error=reproj,
        # The posterior variance is the point of an *uncertainty-aware* smoother:
        # it grows through stretches the views could not pin down. Keeping it means
        # a later consumer can gate on it instead of re-deriving it.
        extra={
            "posterior_var": result.posterior_var,
            "smooth_param": result.smooth_param,
        },
        meta={
            "method": "eks_multiview_nonlinear",
            "n_members": 1 + len(members),
            "n_inflated": result.n_inflated,
            "n_testable": result.n_testable,
            "reproj_error_measured_against": "pose2d observations",
        },
    )
    return True


def _run_postprocess(ctx: _RunContext) -> bool:
    pose = stages.select_postprocess_input(ctx.enabled, ctx.store)
    if pose is None:
        log.warning(
            "skipping postprocess: no 3D pose available to correct -- enable "
            "[pipeline].do_triangulation (or do_eks / do_pictorial_structures)"
        )
        return False
    pts2d_in, pts3d_in = pose
    source = postprocess_source(ctx.enabled, ctx.store)
    base = ctx.store.read_pose2d()
    pts2d, pts3d, reproj, reports = stages.stage_postprocess(
        ctx.config,
        stages.select_cameras(ctx.config, ctx.enabled, ctx.store),
        ctx.store.read_skeleton(),
        pts2d_in,
        pts3d_in,
        obs2d=None if base is None else base[0],
        absent=ctx.store.read_animal()[0],
    )
    ctx.store.truncate_from("postprocess")
    ctx.store.write_points(
        "postprocess",
        pts2d=pts2d,
        pts3d=pts3d,
        reproj_error=reproj,
        meta={
            "pose_from": source,
            "reproj_error_measured_against": "pose2d observations",
            # One entry per op, in order: the same op may appear twice, and what the
            # second one measured depends on what the first one did.
            "ops": reports,
        },
    )
    return True


def _run_inverse_kinematics(ctx: _RunContext) -> bool:
    pts3d = stages.select_pts3d(ctx.enabled, ctx.store)
    if pts3d is None:
        log.warning(
            "skipping inverse_kinematics: no 3D pose available -- enable "
            "[pipeline].do_triangulation (or do_pictorial_structures)"
        )
        return False
    # The solver is an optional extra, and it is now on by DEFAULT -- so a plain install
    # without it has to lose the joint angles, not the run. Skipping matters more than it
    # looks: `inverse_kinematics` precedes `visualization` in STAGES, so an exception here
    # also costs the videos, after detection, bundle adjustment, triangulation, the smoother
    # and the correction chain have all been computed and committed.
    from ..inverse_kinematics._quickik import MissingQuickIK

    conf = None
    if ctx.config.inverse_kinematics.weigh_by_confidence:
        pose2d = ctx.store.read_pose2d()
        conf = None if pose2d is None else pose2d[1]
    try:
        result = stages.stage_inverse_kinematics(
            ctx.config,
            ctx.store.read_skeleton(),
            pts3d,
            conf,
            absent=ctx.store.read_animal()[0],
        )
    except MissingQuickIK as exc:
        log.warning("skipping inverse_kinematics: %s", exc)
        return False
    ctx.store.truncate_from("inverse_kinematics")
    ctx.store.write_ik(
        angles=result.angles,
        angle_names=result.angle_names,
        model_pts3d=result.model_pts3d,
        # The solved plan goes in its own dataset, not the meta: it is far too big for
        # an HDF5 attribute's 64 KB budget to be comfortable, and `write_ik` renders
        # the meta with a `default=str` fallback that would quietly stringify any
        # numpy left in it.
        body_plan=None if result.body_plan is None else result.body_plan.to_json(),
        meta={
            "template": ctx.config.inverse_kinematics.template,
            "solver": "quickik",
            "alignment": result.alignment.to_json(),
            "chain_scales": result.chain_scales,
            "chain_offsets": {
                k: [float(x) for x in np.asarray(v).reshape(3)]
                for k, v in result.chain_offsets.items()
            },
            "body_scale": result.body_scale,
        },
    )
    return True


def _run_visualization(ctx: _RunContext) -> bool:
    result = stages.assemble_result(ctx.config, ctx.enabled, ctx.store)
    if result is None:
        log.warning(
            "skipping visualization: no pose result available -- enable a stage "
            "above or leave a cached results.h5 in %s",
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
        ctx.config,
        result,
        ctx.outdir,
        sources=ctx.sources,
        store=ctx.store,
        progress=ctx.progress,
    )
    return True


_RUNNERS = {
    "pose2d": _run_pose2d,
    "bundle_adjustment": _run_bundle_adjustment,
    "pictorial_structures": _run_pictorial_structures,
    "triangulation": _run_triangulation,
    "eks": _run_eks,
    "postprocess": _run_postprocess,
    "inverse_kinematics": _run_inverse_kinematics,
    "visualization": _run_visualization,
}
