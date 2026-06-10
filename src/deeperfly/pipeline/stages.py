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
from pathlib import Path

import numpy as np

from ..cameras import CameraGroup
from ..config import STAGES, Config
from ..results import PoseResult, StageStore
from ..pose2d.stream import _null_progress, detect_2d, load_detector, resolve_fps
from ..recordings import camera_image_sizes
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
):
    """Run 2D detection over the recording's footage.

    Frames are not held in memory (detection streams them in windows -- see
    :func:`deeperfly.pose2d.stream.detect_2d`); a visualization stage re-sources the
    overlay cameras it needs.

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

    Returns
    -------
    cameras : CameraGroup
        The config rig the detection ran with.
    skeleton : Skeleton
        The configured skeleton.
    pts2d, conf : np.ndarray
        The visibility-masked detections ``(V, T, N, 2)`` and confidences
        ``(V, T, N)``.
    candidates : deeperfly.pictorial.Candidates or None
        The top-K peak set when ``want_candidates``, else ``None``.
    image_sizes : dict
        ``camera_name -> (height, width)`` of the raw footage frames.
    """
    from ..pose2d import detector, inference
    from ..triangulation import apply_visibility

    transforms = config.frame_transforms()
    image_sizes = camera_image_sizes(config, sources=sources, input=input)
    log.info(
        "raw input image sizes (h, w): %s",
        {n: tuple(s) for n, s in image_sizes.items()},
    )
    cameras = config.camera_group(image_sizes=image_sizes)
    unknown = set(transforms) - set(cameras.names)
    if unknown:
        log.warning(
            "preprocess entries for unknown cameras are ignored: %s",
            sorted(unknown),
        )
    active = {
        n: t
        for n, t in transforms.items()
        if n in cameras.names and not t.is_identity()
    }
    if active:
        log.info(
            "frame preprocessing (per camera): %s",
            {n: t.to_json() for n, t in active.items()},
        )
        log.info(
            "preprocessed image sizes (h, w): %s",
            {n: t.output_size(image_sizes[n]) for n, t in active.items()},
        )
    skeleton = config.skeleton()

    pose2d = config.pose2d
    log.info("loading detector (checkpoint: %s)", pose2d.checkpoint or "cached")
    model = load_detector(pose2d.checkpoint)
    precision = pose2d.precision
    detector.set_precision(
        model, precision
    )  # float16 -> CUDA autocast (no-op on CPU/MPS)
    log.info(
        "detector ready on device %s (precision: %s)",
        detector.detector_device(model),
        precision,
    )

    k = config.pictorial.k
    sides, flips = inference.fly_camera_layout(cameras.names)
    n_passes = len(inference.expand_passes(sides, flips)[0])
    batch_size = pose2d.batch_size
    log.info(
        "detecting 2D poses: %d views, %d forward passes/frame, network input %dx%d, "
        "forward batch %d frames",
        len(image_sizes),
        n_passes,
        inference.IMG_SIZE[0],
        inference.IMG_SIZE[1],
        batch_size,
    )
    pts2d, conf, candidates = detect_2d(
        config,
        model,
        sides,
        flips,
        sources=sources,
        input=input,
        want_candidates=want_candidates,
        k=k,
        progress=progress,
    )
    # Mask (camera, point) pairs the rig cannot see once, here, so the cached 2D
    # and every downstream stage see the same visibility-masked points.
    pts2d = apply_visibility(pts2d, skeleton, cameras.names)
    return cameras, skeleton, pts2d, conf, candidates, image_sizes


def stage_bundle_adjustment(
    config: Config, cameras: CameraGroup, pts2d, conf, skeleton
) -> CameraGroup:
    """Refine ``cameras`` with bundle adjustment (fly-as-calibration-target).

    Calibrates on the arg-max 2D. The caller always hands in the *un-refined*
    config rig (:func:`config_rig_from_store`), so editing the rig or
    ``[pipeline.bundle_adjustment]`` and recomputing this stage recalibrates
    from the edited config rather than a prior BA output.

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

    Returns
    -------
    CameraGroup
        The refined rig.
    """
    from ..triangulation import reprojection_error, triangulate
    from .core import calibrate

    v, t = pts2d.shape[:2]
    log.info("bundle adjustment: refining cameras (%d frames, %d views)", t, v)
    ba = config.bundle_adjustment
    refined, _ = calibrate(
        cameras,
        pts2d,
        conf,
        skeleton,
        ba_keypoints=ba.keypoints,
        fixed=ba.fixed,
        shared=ba.shared,
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
    return refined


def stage_pictorial_structures(
    config: Config, cameras: CameraGroup, skeleton, candidates: object, pts2d
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


def stage_triangulation(config: Config, cameras: CameraGroup, pts2d):
    """Triangulate ``pts2d`` to 3D by the configured method.

    ``ransac`` builds each point from its largest multi-view consensus,
    ``greedy`` drops the worst-reprojecting view, ``dlt`` is plain least squares
    (see :func:`deeperfly.pipeline._resolve_triangulation`).

    Parameters
    ----------
    config
        The run config (the triangulation method and thresholds).
    cameras
        The rig to triangulate with.
    pts2d
        The 2D points (pristine ``pose2d`` or pictorial-corrected -- see
        :func:`select_pts2d`).

    Returns
    -------
    pts2d, pts3d, reproj_error : np.ndarray
        The (possibly cleaned) 2D, the 3D points, and the reprojection error.
    """
    from ..triangulation import reprojection_error, triangulate
    from .core import _resolve_triangulation, reconstruct, reconstruct_ransac

    opts = config.triangulation
    method = _resolve_triangulation(opts.method)
    v, t = pts2d.shape[:2]
    log.info("triangulation: method=%s (%d frames, %d views)", method, t, v)
    if method == "ransac":
        pts3d, pts2d, reproj = reconstruct_ransac(
            cameras,
            pts2d,
            threshold=opts.ransac_threshold,
            min_inliers=opts.min_inliers,
        )
    elif method == "greedy":
        pts3d, pts2d, reproj = reconstruct(
            cameras,
            pts2d,
            reproj_threshold=opts.reproj_threshold,
            max_drops=opts.max_drops,
        )
    else:  # "dlt": plain least-squares triangulation, no outlier handling
        pts3d = triangulate(cameras, pts2d)
        reproj = reprojection_error(cameras, pts3d, pts2d)
    return pts2d, pts3d, reproj


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
        return store.read_cameras("bundle_adjustment")
    return config_rig_from_store(config, store)


def select_pts2d(enabled: dict[str, bool], store: StageStore) -> np.ndarray | None:
    """The 2D points triangulation consumes (pictorial-corrected if enabled+present)."""
    if fingerprint.pts2d_source(enabled, store) == "pictorial_structures":
        return store.read_points("pictorial_structures")[0]
    base = store.read_pose2d()
    return None if base is None else base[0]


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
        better2d, pts3d, reproj = store.read_points(source["pts3d"])
        if better2d is not None:
            pts2d = better2d
    return PoseResult(
        cameras=select_cameras(config, enabled, store),
        skeleton=store.read_skeleton(),
        pts2d=pts2d,
        conf=conf,
        pts3d=pts3d,
        reproj_error=reproj,
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
    from .. import io, preprocessing

    if not views:
        return {}
    names = result.cameras.names
    # The detector ran on transformed frames, so the 2D/3D overlays live in
    # transformed-frame coordinates; the overlay footage must match (apply the
    # same per-camera preprocess transform).
    transforms = config.frame_transforms()
    workers = config.io.image_workers

    def transform(v, frames):
        return transforms.get(v, preprocessing.FrameTransform()).apply(frames)

    if in_memory is not None:
        return {v: transform(v, in_memory[names.index(v)]) for v in views}

    sources = sources or {}
    if all(sources.get(v) for v in views):
        return {
            v: transform(
                v,
                io.open_reader(sources[v], workers=workers)[:],
            )
            for v in views
        }
    raise SystemExit(
        "image (imshow) panels need the original frames, but none are in memory and "
        "the run resolved no footage. Re-run with the recording as the input "
        "('deeperfly run <recording>'), or drop the imshow panels from "
        "[[pipeline.visualization.videos]]."
    )


def render_videos(
    config: Config,
    result: PoseResult,
    outdir: Path,
    *,
    sources: dict[str, list[Path]] | None = None,
    progress=None,
) -> None:
    """Render every ``[[pipeline.visualization.videos]]`` to ``<outdir>/<name>.mp4``.

    Each video is composited by :mod:`deeperfly.visualization.compose` from its panels (see
    the config's ``[pipeline.visualization]`` section), overwriting any existing MP4.
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
    progress
        Optional progress factory threaded into the per-video compositor.
    """
    from .. import io
    from ..visualization import compose

    specs = config.videos
    if not specs:
        log.info(
            "no [[pipeline.visualization.videos]] in the config; nothing to render"
        )
        return

    pending = []
    for spec in specs:
        if result.pts3d is None and any(p.plot == "skeleton_3d" for p in spec.panels):
            log.warning(
                "skipping video %r: it reprojects the 3D skeleton but the result has "
                "no 3D pose (enable [pipeline].do_triangulation or do_pictorial_structures)",
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
    )
    make_progress = progress or _null_progress
    for spec in pending:
        path = outdir / f"{spec.video_name}.mp4"
        fps = spec.resolve_fps(input_fps)
        log.info("rendering %s -> %s @ %g fps", spec.video_name, path, fps)
        # Composite and encode frame by frame, so a long clip is never fully held
        # in memory (peak is one frame plus the encoder's buffers).
        with make_progress(src.n_frames(), f"render {spec.video_name}") as wrap:
            with io.VideoWriter(path, fps=fps) as writer:
                writer.write_frames(compose.stream_video(spec, src, progress=wrap))
        log.info("wrote %s", path)
