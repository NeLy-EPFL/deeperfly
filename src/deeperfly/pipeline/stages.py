"""The pipeline stages as pure compute wrappers, plus the stage-input selectors.

Each ``stage_*`` function maps explicit inputs to outputs and never mutates a
shared result -- persistence is the caller's job (see
:class:`~deeperfly.results.StageStore` and :func:`deeperfly.pipeline.run_recording`).
The ``select_*`` helpers pick a downstream stage's inputs out of the store,
mirroring the source selectors used by the fingerprints
(:mod:`deeperfly.pipeline.fingerprint`), so what a stage consumes and what its
cache validity is judged against always agree.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..pictorial import Candidates

import numpy as np

from ..cameras import CameraGroup
from ..config import STAGES, Config
from ..pose2d import autocrop
from ..pose2d.stream import _null_progress, detect_2d, load_models, resolve_fps
from ..recordings import source_image_sizes
from ..results import PoseResult, StageStore
from ..skeleton import Skeleton
from . import fingerprint

log = logging.getLogger("deeperfly")


#: Sentinel injected by the CLI's ``--overwrite`` normalization for a bare
#: ``--overwrite`` (no stage names), meaning "recompute every stage".
_OVERWRITE_ALL = "__all__"


def overwrite_stages(overwrite: list[str] | None) -> set[str]:
    """Stage names selected by ``--overwrite`` (empty set = nothing forced).

    ``--overwrite`` is a *manual* force -- config changes are detected
    automatically (see :mod:`deeperfly.pipeline.fingerprint`); use it to redo a
    stage whose parameters did not change.

    Parameters
    ----------
    overwrite
        ``None`` / empty (force nothing), the ``_OVERWRITE_ALL`` sentinel (a
        bare ``--overwrite`` -> every stage), or a list of stage names.

    Returns
    -------
    set of str
        The selected stage names.

    Raises
    ------
    SystemExit
        If a name is not a known stage (:data:`STAGES`).
    """
    if not overwrite:
        return set()
    if _OVERWRITE_ALL in overwrite:
        return set(STAGES)
    unknown = [s for s in overwrite if s not in STAGES]
    if unknown:
        raise SystemExit(
            f"--overwrite got unknown stage(s) {', '.join(unknown)}; choose from "
            f"{', '.join(STAGES)} (or a bare --overwrite to recompute everything)"
        )
    return set(overwrite)


# -- pipeline stages ---------------------------------------------------------


def stage_pose2d(
    config: Config,
    *,
    sources: dict[str, list[Path]] | None = None,
    input=None,
    want_candidates: bool,
    progress=None,
    outdir: Path | None = None,
    force_autocrop: bool = False,
):
    """Run 2D detection over the recording's footage.

    Frames are not held in memory (detection streams them in windows -- see
    :func:`deeperfly.pose2d.stream.detect_2d`); a visualization stage re-sources the
    overlay cameras it needs.

    A plan carrying ``{ op = "crop", auto = true }`` has its window(s) resolved first (see
    :mod:`deeperfly.pose2d.autocrop`) -- reused from ``outdir`` if a previous run recorded
    one, otherwise searched here and recorded. The resolved boxes are stashed on the config
    so every later stage's plan carries them too.

    Parameters
    ----------
    config
        The run config.
    sources, input
        The footage to detect over (see
        :func:`deeperfly.recordings.camera_sources`).
    want_candidates
        Whether to also extract the top-K candidate peaks (for the
        ``pictorial_structures`` stage).
    progress
        Optional progress factory threaded into the streaming detector.
    outdir
        The recording's output directory, where an automatic crop's window is read and
        written. ``None`` searches without persisting (a library call with no run dir).
    force_autocrop
        Re-search an automatic crop even if a window is already recorded.

    Returns
    -------
    cameras : CameraGroup
        The config rig the detection ran with.
    skeleton : Skeleton
        The configured skeleton.
    pts2d, conf : np.ndarray
        The detections ``(V, T, P, 2)`` and confidences ``(V, T, P)``. A
        ``(view, point)`` pair no pathway maps is ``NaN``.
    candidates : deeperfly.pictorial.Candidates or None
        The top-K peak set when ``want_candidates``, else ``None``.
    image_sizes : dict
        ``camera_name -> (height, width)`` of the raw footage frames.
    """
    plan = config.detection_plan()
    source_sizes = source_image_sizes(config, sources=sources, input=input)
    log.info(
        "raw source image sizes (h, w): %s",
        {n: tuple(s) for n, s in source_sizes.items()},
    )
    # Each view's intrinsics describe its source's raw frame; gather per-view sizes
    # to resolve principal points (when omitted) for the rig.
    view_sources = plan.view_sources()
    image_sizes = {
        v: source_sizes[s] for v, s in view_sources.items() if s in source_sizes
    }
    cameras = config.camera_group(image_sizes=image_sizes)
    skeleton = config.skeleton()

    pose2d = config.pose2d
    log.info("loading %d model(s): %s", len(plan.models), ", ".join(plan.models))
    models = load_models(plan)
    # Per-model precision override falls back to the [pose2d] default; float16
    # -> CUDA autocast (a no-op on CPU/MPS, which stay float32).
    resolved_precision = {
        name: (model.spec.precision or pose2d.precision)
        for name, model in models.items()
    }
    for name, model in models.items():
        model.set_precision(resolved_precision[name])
    log.info(
        "detector ready on device %s (precision: %s)",
        next(iter(models.values())).device(),
        resolved_precision,
    )

    # After the models are on the device (the search forwards through them) and before any
    # frame is detected: an automatic crop decides what the detector even sees.
    plan, _searched = autocrop.ensure_resolved(
        config,
        plan,
        models=models,
        cameras=cameras,
        sources=sources,
        input=input,
        outdir=outdir,
        force=force_autocrop,
    )
    # Stash every resolved window on the config, so a later stage that re-derives the plan
    # -- a render borrowing the detector's box with `crop = "pose2d"` -- looks through the
    # one detection actually used. Taken from the PLAN rather than from what was searched,
    # because a box read back from the sidecar needs carrying just as much as a fresh one.
    config.auto_crops = {**config.auto_crops, **autocrop.resolved_boxes(plan)}

    k = config.pictorial.k
    log.info(
        "detecting 2D poses: %d sources, %d pathways, %d views, forward batch %d frames",
        len(plan.sources),
        len(plan.pathways),
        plan.n_views,
        pose2d.batch_size,
    )
    pts2d, conf, candidates = detect_2d(
        config,
        plan,
        models,
        sources=sources,
        input=input,
        want_candidates=want_candidates,
        k=k,
        threshold=config.pictorial.peak_threshold,
        threshold_rel=config.pictorial.peak_threshold_rel,
        progress=progress,
    )
    # A (view, point) pair no pathway writes stays NaN from the scatter, so the
    # cached 2D and every downstream stage agree on what each view observes.
    return cameras, skeleton, pts2d, conf, candidates, image_sizes


def _resolve_bundle_adjustment_points(
    names: list[str] | None, skeleton
) -> list[int] | None:
    """``[bundle_adjustment].points_to_use`` names -> skeleton indices.

    ``None`` (the key omitted) passes through as ``None`` -- bundle-adjust on every
    point. Otherwise each name is resolved against ``skeleton.point_names``.

    Raises
    ------
    ValueError
        If a name is not one of the skeleton's points.
    """
    if names is None:
        return None
    index = {name: i for i, name in enumerate(skeleton.point_names)}
    try:
        return [index[name] for name in names]
    except KeyError as e:
        raise ValueError(
            f"[bundle_adjustment].points_to_use references unknown "
            f"skeleton point {e.args[0]!r}"
        ) from None


def stage_bundle_adjustment(
    config: Config,
    cameras: CameraGroup,
    pts2d,
    conf,
    skeleton,
    absent=None,
    report=None,
) -> CameraGroup:
    """Refine ``cameras`` with bundle adjustment (the fly itself is the target).

    Bundle-adjusts on the arg-max 2D. The caller always hands in the *un-refined*
    config rig (:func:`config_rig_from_store`), so editing the rig or
    ``[bundle_adjustment]`` and recomputing this stage re-runs bundle
    adjustment from the edited config rather than a prior BA output.

    Parameters
    ----------
    config
        The run config (the bundle-adjustment options).
    cameras
        The un-refined config rig.
    pts2d, conf
        The pristine ``pose2d`` detections and confidences.
    skeleton
        The skeleton (the bone-length prior).
    report
        Optional dict the stage fills in with what it measured but does not return:
        ``quality`` (see :func:`deeperfly.calibration.quality_from_errors`) and
        ``n_frames``. The caller writes the calibration artifact
        (:class:`deeperfly.calibration.Calibration`), which needs the residuals this
        stage already computes for its log line -- passing them out is cheaper than
        triangulating the whole recording a second time to recover them.

    Returns
    -------
    CameraGroup
        The refined rig.
    """
    from ..triangulation import reprojection_error, triangulate
    from .core import apply_absent, bundle_adjust_cameras

    ba = config.bundle_adjustment
    # Erase keypoints that are not on this animal before they can influence the rig. A
    # phantom limb's detections are real pixels on something -- usually the contralateral
    # leg -- so left in they would pull the extrinsics toward explaining an object that
    # is not there.
    pts2d, conf = apply_absent(pts2d, conf, absent)
    ba_keypoints = _resolve_bundle_adjustment_points(ba.points_to_use, skeleton)
    weighted = ba.weigh_by_confidence and conf is not None
    v, t = pts2d.shape[:2]
    log.info(
        "bundle adjustment: refining cameras (%d frames, %d views)%s",
        t,
        v,
        " confidence-weighted" if weighted else "",
    )
    # conf is always handed in (frame_sampling="confidence" may use it); whether it
    # also weighs the residuals is the separate weigh_by_confidence switch.
    refined, _ = bundle_adjust_cameras(
        cameras,
        pts2d,
        conf,
        skeleton,
        ba_keypoints=ba_keypoints,
        fixed=ba.fixed,
        shared=ba.shared,
        weigh_by_confidence=ba.weigh_by_confidence,
        max_frames=ba.max_frames,
        frame_sampling=ba.frame_sampling,
        **ba.least_squares,
    )

    # Report the refined rig's pixel reprojection error (triangulate the committed
    # 2D with the new cameras and reproject); the triangulation stage refines it.
    err = reprojection_error(refined, triangulate(refined, pts2d), pts2d)
    log.info(
        "bundle adjustment: reprojection error median %.3f px  max %.3f px",
        np.nanmedian(err),
        np.nanmax(err),
    )
    if report is not None:
        from ..calibration import quality_from_errors

        report["quality"] = quality_from_errors(err, refined.names)
        report["n_frames"] = int(t)
    return refined


def stage_pictorial_structures(
    config: Config,
    cameras: CameraGroup,
    skeleton: Skeleton | None,
    candidates: "Candidates",
    pts2d,
):
    """DeepFly3D pictorial-structures recovery over the detector's top-K candidates.

    Parameters
    ----------
    config
        The run config (the pictorial-structures options).
    cameras
        The rig to triangulate hypotheses with.
    skeleton
        The skeleton (bone-length coupling).
    candidates
        The detector's top-K candidates (cached by ``pose2d`` -- see
        :meth:`deeperfly.results.StageStore.read_candidates`).
    pts2d
        The arg-max 2D the recovery falls back on.

    Returns
    -------
    pts2d, pts3d, reproj_error : np.ndarray
        The corrected per-view 2D, the initial 3D estimate, and its
        reprojection error.
    """
    from .. import pictorial

    if skeleton is None:
        raise ValueError(
            "pictorial_structures requires a skeleton, but none was stored; "
            "re-run with [pipeline].do_pose2d to write one"
        )
    ps = config.pictorial
    v, t = pts2d.shape[:2]
    log.info("pictorial structures: recovering peaks (%d frames, %d views)", t, v)
    pts3d, pts2d, reproj = pictorial.reconstruct(
        cameras,
        skeleton,
        candidates,
        pts2d,
        temporal=ps.temporal,
        lam=ps.lam,
    )
    return pts2d, pts3d, reproj


def stage_triangulation(
    config: Config, cameras: CameraGroup, pts2d, conf=None, absent=None
):
    """Triangulate ``pts2d`` to 3D by the configured method.

    ``ransac`` builds each point from its largest multi-view consensus,
    ``greedy`` drops the worst-reprojecting view, ``dlt`` is plain least squares
    (see :func:`deeperfly.pipeline._validate_triangulation`).

    Parameters
    ----------
    config
        The run config (the triangulation method and thresholds).
    cameras
        The rig to triangulate with.
    pts2d
        The 2D points (pristine ``pose2d`` or pictorial-corrected -- see
        :func:`select_pts2d`).
    conf
        Per-observation detector confidences ``(V, T, P)``. Used as DLT weights
        only when ``[triangulation].weigh_by_confidence`` is set;
        ignored (and may be ``None``) otherwise.

    Returns
    -------
    pts2d, pts3d, reproj_error : np.ndarray
        The (possibly cleaned) 2D, the 3D points, and the reprojection error.
    """
    from ..triangulation import reprojection_error, triangulate
    from .core import (
        _validate_triangulation,
        apply_absent,
        reconstruct,
        reconstruct_ransac,
    )

    opts = config.triangulation
    pts2d, conf = apply_absent(pts2d, conf, absent)
    method = _validate_triangulation(opts.method)
    weights = conf if (opts.weigh_by_confidence and conf is not None) else None
    v, t = pts2d.shape[:2]
    log.info(
        "triangulation: method=%s (%d frames, %d views)%s",
        method,
        t,
        v,
        " confidence-weighted" if weights is not None else "",
    )
    if method == "ransac":
        pts3d, pts2d, reproj = reconstruct_ransac(
            cameras,
            pts2d,
            threshold=opts.ransac_threshold,
            min_inliers=opts.min_inliers,
            weights=weights,
        )
    elif method == "greedy":
        pts3d, pts2d, reproj = reconstruct(
            cameras,
            pts2d,
            reproj_threshold=opts.reproj_threshold,
            max_drops=opts.max_drops,
            weights=weights,
        )
    else:  # "dlt": plain least-squares triangulation, no outlier handling
        pts3d = triangulate(cameras, pts2d, weights)
        reproj = reprojection_error(cameras, pts3d, pts2d)
    return pts2d, pts3d, reproj


def stage_eks(
    config: Config,
    cameras: CameraGroup,
    pts2d,
    conf=None,
    *,
    init3d=None,
    absent=None,
    members: "list[tuple[np.ndarray, np.ndarray | None]] | None" = None,
):
    """Smooth the pose with the ensemble Kalman smoother (see :mod:`deeperfly.eks`).

    Unlike triangulation, which solves each frame independently, this fits one 3D
    trajectory per keypoint jointly against every view's pixels over the whole
    recording. Its 2D output is that trajectory reprojected, so a view whose
    detection blew up is pulled back onto the animal instead of dragging the 3D
    point off it.

    Parameters
    ----------
    config
        The run config (the ``[eks]`` options).
    cameras
        The rig -- its projection *is* the smoother's observation model.
    pts2d, conf
        The 2D observations ``(V, T, P, 2)`` and confidences ``(V, T, P)`` this
        run's detector produced (see :func:`select_pts2d`).
    init3d
        The 3D the filter starts from and the inflation linearizes about --
        triangulation's output when that stage ran (see :func:`select_eks_init`).
        ``None`` falls back to a plain DLT triangulation inside the smoother.
    absent
        The operator's declaration of which keypoints are not on this animal.
    members
        Extra ensemble members as ``(pts2d, conf)`` pairs from other detectors'
        result files, already aligned to this recording's views and frames.

    Returns
    -------
    pts2d, pts3d, reproj_error : np.ndarray
        The reprojected 2D, the smoothed 3D, and the residual.
    result : deeperfly.eks.EksResult
        The full result, whose posterior variance and fitted smoothing parameters
        the caller persists alongside the arrays.
    """
    from ..eks import smooth
    from ..triangulation import reprojection_error
    from .core import apply_absent

    opts = config.eks
    pts2d, conf = apply_absent(pts2d, conf, absent)
    stack2d = [np.asarray(pts2d, dtype=float)]
    stackconf = [
        np.ones(stack2d[0].shape[:3]) if conf is None else np.asarray(conf, dtype=float)
    ]
    for i, (member2d, member_conf) in enumerate(members or []):
        member2d, member_conf = apply_absent(member2d, member_conf, absent)
        member2d = np.asarray(member2d, dtype=float)
        if member2d.shape != stack2d[0].shape:
            raise ValueError(
                f"[eks].ensemble member {i} has 2D of shape {member2d.shape}, but this "
                f"recording's is {stack2d[0].shape}; an ensemble member must be the same "
                "detector plan over the same views and frames"
            )
        stack2d.append(member2d)
        stackconf.append(
            np.ones(member2d.shape[:3])
            if member_conf is None
            else np.asarray(member_conf, dtype=float)
        )

    v, t = stack2d[0].shape[:2]
    log.info(
        "eks: smoothing %d frames x %d views with %d ensemble member(s)%s",
        t,
        v,
        len(stack2d),
        "" if opts.inflate_vars else " (variance inflation off)",
    )
    if len(stack2d) == 1:
        log.info(
            "eks: single ensemble member -- the observation noise is 1/confidence, a "
            "prior rather than a measured spread; the geometric smoother and the "
            "variance inflation are unaffected (they need views, not models)"
        )
    result = smooth(
        cameras,
        np.stack(stack2d),
        np.stack(stackconf),
        init3d=init3d,
        smooth_param=opts.smooth_param,
        avg_mode=opts.avg_mode,
        var_mode=opts.var_mode,
        inflate_vars=opts.inflate_vars,
        inflate_threshold=opts.inflate_threshold,
        inflate_factor=opts.inflate_factor,
        fit_frames=opts.fit_frames,
        fit_iterations=opts.fit_iterations,
        fill_unobserved=opts.fill_unobserved,
    )
    # Measured against the *observations*, not against the stage's own 2D: the latter
    # is the reprojection of `pts3d` by construction, so it would be identically zero.
    # This column therefore reads "how far the smoother moved from the raw detection",
    # which is the number worth looking at.
    reproj = reprojection_error(cameras, result.pts3d, stack2d[0])
    moved = np.linalg.norm(result.pts2d - stack2d[0], axis=-1)
    log.info(
        "eks: 2D correction median %.2f px  p90 %.2f  max %.2f",
        np.nanmedian(moved),
        np.nanpercentile(moved, 90) if np.isfinite(moved).any() else np.nan,
        np.nanmax(moved) if np.isfinite(moved).any() else np.nan,
    )
    return result.pts2d, result.pts3d, reproj, result


def stage_postprocess(
    config: Config,
    cameras: CameraGroup,
    skeleton: Skeleton | None,
    pts2d,
    pts3d,
    *,
    obs2d=None,
    absent=None,
):
    """Apply the ``[postprocess].ops`` chain to a finished 3D pose.

    A thin wrapper: the corrections themselves live in :mod:`deeperfly.postprocess`, one
    pure function per op, so a new correction is an entry in that registry plus a line in
    a config's ``ops`` -- not a new pipeline stage. This function resolves the config,
    runs the chain, re-measures the residual and logs what each op did.

    Parameters
    ----------
    config
        The run config (the ``[postprocess]`` options).
    cameras
        The rig, used only to measure the residual.
    skeleton
        The skeleton, which resolves the configured names to columns.
    pts2d, pts3d
        The upstream stage's pose -- the smoother's when it ran, else the
        triangulation's (see :func:`select_postprocess_input`).
    obs2d
        The pristine ``pose2d`` detections, which ``reproj_error`` is measured against
        (as for :func:`stage_eks`, so the column reads "how far the corrected pose sits
        from the raw detection"). Falls back to ``pts2d``.
    absent
        The operator's declaration of which keypoints are not on this animal.

    Returns
    -------
    pts2d, pts3d, reproj_error : np.ndarray
        The corrected 2D, the corrected 3D, and the residual.
    reports : list of dict
        One entry per op, in order, recording what it measurably did. The caller
        persists them as the stage's metadata.
    """
    from ..postprocess import apply_ops
    from ..triangulation import reprojection_error
    from .core import apply_absent

    if skeleton is None:
        raise ValueError(
            "postprocess requires a skeleton, but none was stored; "
            "re-run with [pipeline].do_pose2d to write one"
        )
    ops = list(config.postprocess.ops)
    if not ops:
        # Deliberately still a stage output rather than a skip: a downstream stage's
        # input must not depend on whether the chain happened to be filled in.
        log.info("postprocess: no ops configured; the pose passes through unchanged")
    pts2d, pts3d, reports = apply_ops(
        pts2d, pts3d, ops=ops, skeleton=skeleton, absent=absent
    )
    for report in reports:
        log.info("postprocess: %s", _describe_op(report))
    observed, _ = apply_absent(
        np.asarray(pts2d if obs2d is None else obs2d, dtype=float), None, absent
    )
    reproj = reprojection_error(cameras, pts3d, observed)
    return pts2d, pts3d, reproj, reports


def _columns_for(names, skeleton, where: str) -> list[int]:
    """Skeleton point names -> column indices, erroring with the config key that failed."""
    from ..postprocess import _columns

    return _columns(names, skeleton, where=where)


def _describe_op(report: dict) -> str:
    """One log line per op: what it did, and the number that says whether it should have.

    Every op reports how far it moved the points it touched, because that is the only
    check on its premise -- a static point that had been drifting a fraction of a pixel
    really was static, and one drifting tens of pixels was moving.
    """
    name = report.get("op", "?")
    if name == "static":
        if not report.get("points"):
            return "static: no points listed; nothing frozen"
        return (
            f"static: froze {len(report['points'])} point(s) to their temporal "
            f"{report['method']}; they had been moving "
            f"{report['moved_2d_median_px']:.2f} px median / "
            f"{report['moved_2d_p90_px']:.2f} p90 in 2D "
            f"({report['moved_3d_median']:.4f} / {report['moved_3d_p90']:.4f} in 3D)"
            + (f", worst {report['worst_point']}" if report.get("worst_point") else "")
        )
    if name == "symmetrize":
        if not report.get("fitted"):
            return "symmetrize: no sagittal plane fitted; the pose is unchanged"
        asym = report.get("asymmetry_before") or {}
        worst = max(asym, key=asym.get) if asym else None
        return (
            f"symmetrize: {len(report['pairs'])} pair(s) + "
            f"{len(report['midline'])} midline point(s) about a "
            f"{'per-frame' if report['per_frame'] else 'single'} sagittal plane "
            f"(strength {report['strength']:g}); moved "
            f"{report['moved_3d_median']:.4f} median / {report['moved_3d_p90']:.4f} p90"
            + (f", worst pair {worst} ({asym[worst]:.4f} apart)" if worst else "")
        )
    return f"{name}: done"


def stage_inverse_kinematics(
    config: Config, skeleton: Skeleton | None, pts3d, conf=None, absent=None
):
    """Fit the NeuroMechFly model's joint angles to the triangulated 3D pose.

    Parameters
    ----------
    config
        The run config (the template, joint limits and solver options).
    skeleton
        The skeleton (resolves the template's point names to ``pts3d`` columns).
    pts3d
        The 3D pose ``(T, P, 3)`` (triangulation, else pictorial -- see
        :func:`select_pts3d`).
    conf
        The per-view 2D confidences ``(V, T, P)``, used only when
        ``[inverse_kinematics].weigh_by_confidence`` is on.

    Returns
    -------
    deeperfly.inverse_kinematics.IKResult
        The joint angles, fitted model joints (world), the measurements and the plan.
    """
    from ..inverse_kinematics import solve_inverse_kinematics

    if skeleton is None:
        raise ValueError(
            "inverse_kinematics requires a skeleton, but none was stored; "
            "re-run with [pipeline].do_pose2d to write one"
        )
    template = config.ik_template()
    articulation = config.ik_articulation()
    p = config.inverse_kinematics
    extra = [c.name for c in articulation.chains] if articulation else []
    log.info(
        "inverse kinematics: fitting %d leg(s)%s over %d frames (template %r, %s body)",
        len(template.legs),
        f" + {'/'.join(extra)}" if extra else "",
        pts3d.shape[0],
        template.name,
        "fixed" if p.fixed_body else "free",
    )
    return solve_inverse_kinematics(
        pts3d,
        skeleton,
        template,
        articulation=articulation,
        weights=_confidence_weights(conf) if p.weigh_by_confidence else None,
        n_iterations=p.n_iterations,
        neutral_weight=p.neutral_weight,
        damping=p.damping,
        position_tolerance=p.position_tolerance,
        angle_tolerance=p.angle_tolerance,
        fixed_body=p.fixed_body,
        symmetric_segments=p.symmetric_segments,
        parallel=p.parallel,
        segment_len=p.segment_len,
        overlap_len=p.overlap_len,
        absent_points=absent,
    )


def _confidence_weights(conf) -> np.ndarray | None:
    """``(T, P)`` per-keypoint solve weights from the ``(V, T, P)`` 2D confidences.

    A 3D point is only as trustworthy as the 2D detections behind it, so the weight is
    the mean confidence over the views. ``None`` when no confidences were stored, which
    weighs every observed point equally.
    """
    if conf is None:
        return None
    with (
        warnings.catch_warnings()
    ):  # a point unseen in every view -> all-NaN (expected)
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.asarray(conf, dtype=float), axis=0)


# -- stage-input selectors -----------------------------------------------------


def config_rig_from_store(config: Config, store: StageStore) -> CameraGroup:
    """The un-refined config rig, rebuilt footage-free from the store.

    Intrinsics resolve against the raw frame sizes ``pose2d`` recorded
    (``image_sizes``) and map through each camera's preprocess chain, so the
    rig matches what a fresh detection would build.
    When the config alone cannot build a rig (no explicit principal point and no
    recorded sizes -- e.g. a result file from an older run), the rig stored by
    ``pose2d`` is used instead, with a note.

    Raises
    ------
    SystemExit
        If no rig can be built at all.
    """
    try:
        return config.camera_group(image_sizes=store.read_image_sizes())
    except ValueError as exc:
        cached = store.read_cameras("pose2d")
        if cached is not None:
            log.warning(
                "could not build the camera rig from the config (%s) -- using the "
                "rig stored by pose2d instead",
                exc,
            )
            return cached
        raise SystemExit(f"cannot build the camera rig from the config: {exc}")


def select_cameras(
    config: Config, enabled: dict[str, bool], store: StageStore
) -> CameraGroup:
    """The rig a downstream stage consumes (BA output if enabled+present, else config)."""
    if fingerprint.cameras_source(enabled, store) == "bundle_adjustment":
        cameras = store.read_cameras("bundle_adjustment")
        assert cameras is not None
        return cameras
    return config_rig_from_store(config, store)


def select_pts2d(enabled: dict[str, bool], store: StageStore) -> np.ndarray | None:
    """The 2D points triangulation consumes (pictorial-corrected if enabled+present)."""
    if fingerprint.pts2d_source(enabled, store) == "pictorial_structures":
        _pts = store.read_points("pictorial_structures")
        assert _pts is not None
        return _pts[0]
    base = store.read_pose2d()
    return None if base is None else base[0]


def select_pts3d(enabled: dict[str, bool], store: StageStore) -> np.ndarray | None:
    """The 3D points inverse kinematics consumes (eks, else triangulation, else pictorial)."""
    source = fingerprint.pts3d_source(enabled, store)
    if source is None:
        return None
    _pts = store.read_points(source)
    return None if _pts is None else _pts[1]  # points3d


def select_postprocess_input(
    enabled: dict[str, bool], store: StageStore
) -> tuple[np.ndarray, np.ndarray] | None:
    """The ``(pts2d, pts3d)`` the postprocess chain consumes, or ``None``.

    Both spaces come from the *same* upstream stage -- the ops correct the two layers of
    one pose, so pairing a smoothed 3D with a triangulated 2D would make the output a
    pose that no stage ever produced. Paired with
    :func:`~deeperfly.pipeline.fingerprint.postprocess_source`, which explains why this
    cannot just be :func:`select_pts3d`.
    """
    source = fingerprint.postprocess_source(enabled, store)
    if source is None:
        return None
    _pts = store.read_points(source)
    if _pts is None or _pts[0] is None or _pts[1] is None:
        return None
    return _pts[0], _pts[1]


def select_eks_init(enabled: dict[str, bool], store: StageStore) -> np.ndarray | None:
    """The 3D the smoother starts from -- never its own output.

    Paired with :func:`~deeperfly.pipeline.fingerprint.eks_init_source`, which
    explains why this cannot just be :func:`select_pts3d`.
    """
    source = fingerprint.eks_init_source(enabled, store)
    if source is None:
        return None
    _pts = store.read_points(source)
    return None if _pts is None else _pts[1]  # points3d


def assemble_result(
    config: Config, enabled: dict[str, bool], store: StageStore
) -> PoseResult | None:
    """The result the visualization stage draws, assembled from the store.

    Like :meth:`PoseResult.load` but *enabled-aware*: a derived stage's output is
    drawn only while that stage is enabled (the fingerprint selectors
    :func:`~deeperfly.pipeline.fingerprint.pose_sources` /
    :func:`~deeperfly.pipeline.fingerprint.cameras_source` make the same choice).

    Returns
    -------
    PoseResult or None
        ``None`` when the store holds no 2D pose at all.
    """
    base = store.read_pose2d()
    if base is None:
        return None
    pts2d, conf = base
    pts3d = reproj = None
    source = fingerprint.pose_sources(enabled, store)
    if source["pts3d"] is not None:
        _pts = store.read_points(source["pts3d"])
        if _pts is not None:
            better2d, pts3d, reproj = _pts
            if better2d is not None:
                pts2d = better2d
    nmf_pts3d = nmf_angles = nmf_angle_names = nmf_body_plan = None
    nmf_chain_scales: dict[str, float] = {}
    nmf_chain_offsets: dict[str, np.ndarray] = {}
    nmf_body_scale = 1.0
    if fingerprint.nmf_source(enabled, store) is not None:
        ik = store.read_ik()
        if ik is not None:
            nmf_angles, nmf_angle_names, nmf_pts3d = ik  # angles, names, model joints
            # The estimated sizes live in the stage's metadata, not its arrays. Reading
            # only the arrays (as this did) rendered the mesh overlay in the run's own
            # videos at model size and a per-frame body scale, while the GUI on the very
            # same file used the fitted ones.
            ik_meta = store.read_ik_meta()
            nmf_chain_scales = {
                str(k): float(v) for k, v in (ik_meta.get("chain_scales") or {}).items()
            }
            nmf_chain_offsets = {
                str(k): np.asarray(v, dtype=float).reshape(3)
                for k, v in (ik_meta.get("chain_offsets") or {}).items()
            }
            if ik_meta.get("body_scale") is not None:
                nmf_body_scale = float(ik_meta["body_scale"])
            nmf_body_plan = ik_meta.get("body_plan")
    # The absence declaration is not a stage output -- it is an operator-authored fact about
    # the specimen -- so it is read straight from the file and carried onto the assembled
    # result. Without this the render path would disagree with the editor on the same file.
    absent, subject_id = store.read_animal()
    return PoseResult(
        cameras=select_cameras(config, enabled, store),
        skeleton=store.read_skeleton(),  # type: ignore[arg-type]
        pts2d=pts2d,
        conf=conf,
        pts3d=pts3d,
        reproj_error=reproj,
        nmf_pts3d=nmf_pts3d,
        nmf_angles=nmf_angles,
        nmf_angle_names=nmf_angle_names,
        nmf_chain_scales=nmf_chain_scales,
        nmf_chain_offsets=nmf_chain_offsets,
        nmf_body_scale=nmf_body_scale,
        nmf_body_plan=nmf_body_plan,
        absent=absent,  # type: ignore[arg-type]
        subject_id=subject_id,
    )


# -- visualization ------------------------------------------------------------


def source_view_frames(
    config: Config,
    result: PoseResult,
    views: list[str],
    *,
    sources: dict[str, list[Path]] | None = None,
    in_memory: list | None = None,
    window: int = 48,
) -> dict[str, Any]:
    """Per-view footage for the visualization stage's ``imshow`` panels.

    Uses ``in_memory`` frames (indexed by camera order) when available; otherwise
    the footage ``deeperfly run`` resolved up front (``sources``). A resume that
    re-renders just re-passes the recording, re-resolving the footage the same way.

    Footage read from disk comes back as :class:`~deeperfly.io.CursorFrames`, one per
    view -- an array-like that decodes on demand rather than a resident array. **This is
    load-bearing rather than an optimization.** A decoded clip costs
    ``frames x height x width x 3`` bytes, and the stage needs one per ``imshow`` view at
    once, so an eight-camera 5900-frame 1984-wide recording came to 155 GB and was
    OOM-killed on a 184 GB host -- while the compositor it feeds already streams its
    output and never holds more than a few frames. Only the input side was eager. Reads
    are near-sequential (see :func:`~deeperfly.visualization.compose._composited_in_order`),
    which is the access pattern a cursor is fastest at, so the peak drops to ``window``
    frames a view for the same work: measured 160.8 GiB (killed) -> 3.28 GiB on that
    recording, and a byte-identical output video on one that could already render.

    Parameters
    ----------
    config
        The run config (I/O backends, per-camera preprocessing).
    result
        The result (for the camera order).
    views
        The camera names whose footage is needed.
    sources
        Optional pre-resolved ``camera_name -> footage files`` map.
    in_memory
        Optional in-memory frames per camera (in ``result.cameras`` order). Passed
        straight through -- a caller that brought its own arrays keeps them.
    window
        Decoded frames kept per view. Must exceed the consumer's look-ahead, or every
        read pays a seek; :func:`render_videos` sizes it from its own render workers.

    Returns
    -------
    dict
        ``view -> footage``, each an ``ndarray`` (from ``in_memory``) or a
        :class:`~deeperfly.io.CursorFrames` presenting the same ``(T, H, W[, 3])``
        surface lazily. Empty when ``views`` is empty. **Close them when done** --
        :func:`render_videos` does this in a ``finally``, since each holds a decoder open.

    Raises
    ------
    SystemExit
        If neither in-memory frames nor resolved footage are available.
    """
    from .. import io

    if not views:
        return {}
    names = result.cameras.names
    workers = config.io.image_workers
    # 2D/3D overlays live in the raw view frame (the detector mapped its points
    # back through the pathway), so the overlay footage is the raw source footage
    # of the source feeding each view -- no transform. Fall back to view==source
    # when the config carries no detection plan (a viz-only library call).
    try:
        view_sources = config.detection_plan().view_sources()
    except (ValueError, KeyError):
        view_sources = {}
    src_for = {v: view_sources.get(v, v) for v in views}

    if in_memory is not None:
        return {v: in_memory[names.index(v)] for v in views}

    sources = sources or {}
    if all(sources.get(src_for[v]) for v in views):
        # One lazy provider per view, each with its own held-open decoder. Opening these
        # is cheap (a container header and one frame apiece), so there is nothing to fan
        # out over threads any more -- the concurrency that used to matter here was eight
        # full clip decodes racing each other, and those are what blew up the host.
        opened: dict[str, Any] = {}
        try:
            for v in views:
                reader = io.open_reader(sources[src_for[v]], workers=workers)
                opened[v] = io.CursorFrames(reader, window=window)
        except BaseException:
            for f in opened.values():  # do not leak decoders on a partial failure
                f.close()
            raise
        return opened
    raise SystemExit(
        "image (imshow) panels need the original frames, but none are in memory and "
        "the run resolved no footage. Re-run with the recording as the input "
        "('deeperfly run <recording>'), or drop the imshow panels from "
        "[[visualization.videos]]."
    )


def _staged_points(specs, store) -> tuple[dict, dict]:
    """Read the per-stage points any panel names, as ``(pts2d, pts3d)`` by stage.

    Only the stages actually referenced are read: each is a full ``(V, T, P, ...)``
    array, so loading every stage a file happens to hold would multiply the render's
    memory for videos that never ask.
    """
    wanted = sorted({p.stage for spec in specs for p in spec.panels if p.stage})
    if not wanted:
        return {}, {}
    if store is None:
        log.warning(
            "panels name stage(s) %s but no results store was passed; those videos "
            "cannot be rendered",
            ", ".join(wanted),
        )
        return {}, {}
    pts2d, pts3d = {}, {}
    for stage in wanted:
        got = store.read_points(stage)
        if got is None:
            continue
        p2, p3, _ = got
        if p2 is not None:
            pts2d[stage] = p2
        if p3 is not None:
            pts3d[stage] = p3
    return pts2d, pts3d


def render_videos(
    config: Config,
    result: PoseResult,
    outdir: Path,
    *,
    sources: dict[str, list[Path]] | None = None,
    store: "StageStore | None" = None,
    progress=None,
) -> None:
    """Render every ``[[visualization.videos]]`` to ``<outdir>/<name>.mp4``.

    Each video is composited by :mod:`deeperfly.visualization.compose` from its panels (see
    the config's ``[visualization]`` section), overwriting any existing MP4.
    A video whose panels reproject the 3D skeleton is skipped with a
    reason when the result has no 3D pose (e.g. no triangulation/pictorial stage
    ran); frames for ``imshow`` panels are sourced only across the videos that
    actually render.

    Parameters
    ----------
    config
        The run config (the video specs and output encoder).
    result
        The pose result drawn from.
    outdir
        The directory the MP4s are written to.
    sources
        Optional pre-resolved footage map for the ``imshow`` overlay panels.
    store
        The recording's stage store, needed only by panels that name a ``stage``
        (``{ plot = "skeleton_3d", stage = "triangulation" }``). Without it such a video
        is skipped rather than silently drawn from the resolved result -- the whole
        reason to name a stage is that the resolved one is a different array.
    progress
        Optional progress factory threaded into the per-video compositor.
    """
    from .. import io
    from ..visualization import compose

    specs = config.videos
    if not specs:
        log.info("no [[visualization.videos]] in the config; nothing to render")
        return

    stage_pts2d, stage_pts3d = _staged_points(specs, store)
    pending = []
    for spec in specs:
        want = {p.stage for p in spec.panels if p.stage}
        short = sorted(
            s
            for s in want
            if (
                any(p.stage == s and p.plot == "skeleton_3d" for p in spec.panels)
                and s not in stage_pts3d
            )
            or (
                any(p.stage == s and p.plot == "skeleton_2d" for p in spec.panels)
                and s not in stage_pts2d
            )
        )
        if short:
            log.warning(
                "skipping video %r: its panels draw stage(s) %s, which this "
                "results.h5 does not have (enable them and re-run)",
                spec.video_name,
                ", ".join(short),
            )
        elif result.pts3d is None and any(p.plot == "skeleton_3d" for p in spec.panels):
            log.warning(
                "skipping video %r: it reprojects the 3D skeleton but the result has "
                "no 3D pose (enable [pipeline].do_triangulation or do_pictorial_structures)",
                spec.video_name,
            )
        elif result.nmf_pts3d is None and any(
            p.plot in ("skeleton_nmf", "mesh_nmf") for p in spec.panels
        ):
            log.warning(
                "skipping video %r: it overlays the fitted NMF model but the result "
                "has no IK pose (enable [pipeline].do_inverse_kinematics)",
                spec.video_name,
            )
        else:
            pending.append(spec)
    if not pending:
        return

    input_fps = resolve_fps(config, sources=sources)
    # The visualization stage *writes* MP4s with PyAV (H.264 / libx264).
    views = sorted(
        {p.view for spec in pending for p in spec.panels if p.plot == "imshow"}
    )
    import os

    cfg_workers = config.io.image_workers or 0  # may be None (auto) or 0
    # Compositing is the render bottleneck (OpenCV, GIL-releasing); fan it out over
    # a few threads. Cap at 8 (diminishing returns past that) and respect an
    # explicit [io.image] workers when set.
    render_workers = min(8, cfg_workers if cfg_workers > 0 else (os.cpu_count() or 4))
    # The footage window has to cover the compositor's look-ahead, or a worker reaching
    # past it makes its view's cursor seek on every frame. `_composited_in_order` keeps
    # `max(2, workers * 2)` frames in flight; double that, so a straggling worker still
    # lands inside the window.
    frame_window = max(8, render_workers * 4)
    src = compose.Sources(
        skeleton=result.skeleton,
        camera_group=result.cameras,
        frames=source_view_frames(
            config, result, views, sources=sources, window=frame_window
        ),
        pts2d=result.pts2d,
        pts3d=result.pts3d,
        conf=result.conf,
        nmf_pts3d=result.nmf_pts3d,
        nmf_angles=result.nmf_angles,
        nmf_angle_names=result.nmf_angle_names,
        nmf_head_scale=result.nmf_head_scale,
        nmf_abdomen_scale=result.nmf_abdomen_scale,
        nmf_chain_offsets=result.nmf_chain_offsets,
        nmf_body_scale=result.nmf_body_scale,
        nmf_hide_parts=tuple(config.visualization.get("mesh_hide", ["wings"])),
        stage_pts2d=stage_pts2d,
        stage_pts3d=stage_pts3d,
    )
    make_progress = progress or _null_progress
    try:
        for spec in pending:
            path = outdir / f"{spec.video_name}.mp4"
            fps = spec.resolve_fps(input_fps)
            log.info("rendering %s -> %s @ %g fps", spec.video_name, path, fps)
            # Composite and encode frame by frame, so a long clip is never fully held
            # in memory (peak is a few in-flight frames plus the encoder's buffers).
            with make_progress(src.n_frames(), f"render {spec.video_name}") as wrap:
                with io.VideoWriter(path, fps=fps) as writer:
                    writer.write_frames(
                        compose.stream_video(
                            spec, src, progress=wrap, workers=render_workers
                        )
                    )
            log.info("wrote %s", path)
    finally:
        # Each footage view holds a decoder open (and, for a video, an av container).
        # Arrays a caller passed in as `in_memory` have no close and are left alone.
        for frames in src.frames.values():
            closer = getattr(frames, "close", None)
            if closer is not None:
                closer()
