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


def _resolve_constant_points(names: list[str] | None, skeleton) -> list[int] | None:
    """``[inverse_kinematics].constant_points`` names -> skeleton indices.

    Empty/omitted passes through as ``None`` (the feature is off). Otherwise each
    name is resolved against ``skeleton.point_names``.

    Raises
    ------
    ValueError
        If a name is not one of the skeleton's points.
    """
    if not names:
        return None
    index = {name: i for i, name in enumerate(skeleton.point_names)}
    try:
        return [index[name] for name in names]
    except KeyError as e:
        raise ValueError(
            f"[inverse_kinematics].constant_points references unknown "
            f"skeleton point {e.args[0]!r}"
        ) from None


def _pin_constant_points(pts3d: np.ndarray, cols: list[int]) -> np.ndarray:
    """Replace the ``cols`` of ``pts3d`` ``(T, P, 3)`` with their temporal median.

    Points declared constant over the recording (a tethered fly's fixed joints) are
    collapsed to their ``nanmedian`` over time, broadcast back over all frames -- so
    the fit sees a steady position and occluded (NaN) frames are filled in. A column
    that is never observed stays all-NaN. Returns a copy; the input is not mutated.
    """
    pts3d = np.array(pts3d, dtype=float)
    with warnings.catch_warnings():  # a never-observed column -> all-NaN (expected)
        warnings.simplefilter("ignore", RuntimeWarning)
        pts3d[:, cols, :] = np.nanmedian(pts3d[:, cols, :], axis=0)
    return pts3d


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
    const_cols = _resolve_constant_points(p.constant_points, skeleton)
    if const_cols:
        pts3d = _pin_constant_points(pts3d, const_cols)
        log.info(
            "inverse kinematics: holding %d point(s) constant over time (median)",
            len(const_cols),
        )
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
    """The 3D points inverse kinematics consumes (triangulation, else pictorial)."""
    source = fingerprint.pts3d_source(enabled, store)
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
) -> dict[str, np.ndarray]:
    """Per-view footage for the visualization stage's ``imshow`` panels.

    Uses ``in_memory`` frames (indexed by camera order) when available; otherwise
    the footage ``deeperfly run`` resolved up front (``sources``). A resume that
    re-renders just re-passes the recording, re-resolving the footage the same way.

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
        Optional in-memory frames per camera (in ``result.cameras`` order).

    Returns
    -------
    dict of str to np.ndarray
        ``view -> preprocessed footage`` (empty when ``views`` is empty).

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
        # Decode the views concurrently (each with its own multithreaded reader),
        # instead of one after another -- the overlay footage is tens of GB.
        from concurrent.futures import ThreadPoolExecutor

        def _load(v: str):
            return v, io.open_reader(sources[src_for[v]], workers=workers)[:]

        with ThreadPoolExecutor(max_workers=max(1, min(len(views), 8))) as pool:
            return dict(pool.map(_load, views))
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
    src = compose.Sources(
        skeleton=result.skeleton,
        camera_group=result.cameras,
        frames=source_view_frames(config, result, views, sources=sources),
        pts2d=result.pts2d,
        pts3d=result.pts3d,
        conf=result.conf,
        nmf_pts3d=result.nmf_pts3d,
        nmf_angles=result.nmf_angles,
        nmf_angle_names=result.nmf_angle_names,
        nmf_head_scale=result.nmf_head_scale,
        nmf_abdomen_scale=result.nmf_abdomen_scale,
        nmf_body_scale=result.nmf_body_scale,
        nmf_hide_parts=tuple(config.visualization.get("mesh_hide", ["wings"])),
        stage_pts2d=stage_pts2d,
        stage_pts3d=stage_pts3d,
    )
    make_progress = progress or _null_progress
    import os

    cfg_workers = config.io.image_workers or 0  # may be None (auto) or 0
    # Compositing is the render bottleneck (OpenCV, GIL-releasing); fan it out over
    # a few threads. Cap at 8 (diminishing returns past that) and respect an
    # explicit [io.image] workers when set.
    render_workers = min(8, cfg_workers if cfg_workers > 0 else (os.cpu_count() or 4))
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
