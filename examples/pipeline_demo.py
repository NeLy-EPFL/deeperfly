"""Synthetic end-to-end demo of the deeperfly 3D pipeline (no weights needed).

Generates a moving 38-point fly, projects it through the example 7-camera rig,
adds detector-like noise and a gross outlier, then runs the geometry pipeline one
stage at a time -- visibility masking -> (optional) calibration -> robust
triangulation -> a saved HDF5 result plus a 3D-skeleton video.

The stages below mirror :func:`deeperfly.pipeline.run_from_points2d` (the function
``deeperfly run`` calls); here each is its own small function over arrays, so every
intermediate is visible and independently testable.

    uv run python examples/pipeline_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from deeperfly import Config, PoseResult, Skeleton
from deeperfly.pipeline import calibrate, reconstruct_ransac
from deeperfly.triangulation import apply_visibility

HERE = Path(__file__).parent


def fly_motion(rng, n_frames=30, n_pts=38):
    base = rng.uniform(-1.5, 1.5, size=(n_pts, 3))
    t = np.linspace(0, 1, n_frames)[:, None, None]
    wiggle = 0.3 * np.sin(2 * np.pi * (t + np.arange(n_pts)[None, :, None] / n_pts))
    return base[None] + wiggle  # (T, N, 3)


def synthesize_detections(cameras, rng, n_frames=30):
    """Ground-truth 3D motion + noisy per-view 2D, with one gross outlier.

    Returns ``(pts3d_true, pts2d, conf)``: the truth we score against, the
    ``(V, T, N, 2)`` detections the pipeline consumes, and detector-like
    confidences. The outlier lands on a right-leg claw the camera actually sees, so
    it survives visibility masking and is left for the triangulator to reject.
    """
    pts3d_true = fly_motion(rng, n_frames)
    pts2d = np.array(cameras.project(pts3d_true))  # (V, T, N, 2)
    pts2d += rng.normal(scale=0.15, size=pts2d.shape)  # sub-pixel detector noise
    pts2d[2, 5, 23] += [250.0, -200.0]  # a gross outlier RANSAC must reject (rf claw)
    conf = rng.uniform(0.5, 1.0, size=pts2d.shape[:3])
    return pts3d_true, pts2d, conf


def mask_unseen(cameras, skeleton, pts2d):
    """Stage 1 -- NaN out the (camera, point) pairs the rig cannot see."""
    return apply_visibility(pts2d, skeleton, cameras.names)


def calibrate_cameras(cameras, skeleton, pts2d, conf):
    """Stage 2 (optional) -- refine the rig with fly-as-target bundle adjustment."""
    cameras, _ = calibrate(cameras, pts2d, conf, skeleton)
    return cameras


def reconstruct_3d(cameras, pts2d):
    """Stage 3 -- robust 2D->3D via per-point RANSAC consensus.

    Returns ``(pts3d, cleaned_pts2d, reproj_error)``; the gross outlier never
    enters the consensus set, so it is dropped from ``cleaned_pts2d``.
    """
    return reconstruct_ransac(cameras, pts2d, threshold=15.0, min_inliers=2)


def assemble_result(cameras, skeleton, pts2d, conf, pts3d, reproj, *, fps, meta=None):
    """Stage 4 -- bundle the stage outputs into a saveable :class:`PoseResult`."""
    return PoseResult(
        cameras=cameras,
        skeleton=skeleton,
        pts2d=pts2d,
        conf=conf,
        pts3d=pts3d,
        reproj_error=reproj,
        meta={"fps": fps, "triangulation": "ransac", **(meta or {})},
    )


def render_pose3d_video(result, path, *, view="f", fps=15):
    """Reproject the 3D skeleton into one camera view and write it to an MP4.

    Uses the OpenCV panel compositor (:mod:`deeperfly.visualization.compose`) -- the
    same renderer ``deeperfly run`` drives from ``[[pipeline.visualization.videos]]``.
    """
    from deeperfly import io
    from deeperfly.visualization import compose

    spec = compose.VideoSpec(
        video_name=Path(path).stem,
        panels=[compose.Panel(plot="skeleton_3d", view=view)],
    )
    src = compose.Sources(
        skeleton=result.skeleton,
        camera_group=result.cameras,
        frames={},
        pts3d=result.pts3d,
    )
    with io.VideoWriter(path, fps=fps) as writer:
        writer.write_frames(compose.stream_video(spec, src))


def main():
    fps = 100.0
    do_calibrate = False  # the synthetic cameras are already the ground-truth rig
    rng = np.random.default_rng(0)
    config = Config.from_toml(HERE / "cameras.toml")
    # The synthetic rig has no real frames, so give each camera a nominal (H, W):
    # camera_group then fixes its principal point at the image center, exactly as
    # `deeperfly run` does from the footage. View order follows [cameras.*] in the
    # config (rh, rm, rf, f, lf, lm, lh), so camera index 2 is the front-right view.
    _, camera_specs = config.camera_table()
    image_sizes = {name: (512, 1024) for name in camera_specs}
    cameras = config.camera_group(image_sizes=image_sizes)
    skeleton = Skeleton.fly()

    pts3d_true, pts2d, conf = synthesize_detections(cameras, rng)

    # The pipeline, one stage at a time (this is what run_from_points2d does):
    pts2d = mask_unseen(cameras, skeleton, pts2d)
    if do_calibrate:
        cameras = calibrate_cameras(cameras, skeleton, pts2d, conf)
    pts3d, pts2d, reproj = reconstruct_3d(cameras, pts2d)
    result = assemble_result(
        cameras,
        skeleton,
        pts2d,
        conf,
        pts3d,
        reproj,
        fps=fps,
        meta={"source": "pipeline_demo"},
    )

    out = HERE / "demo_result.h5"
    result.save(out)
    err = result.reproj_error
    print(f"saved {out}")
    print(f"frames={result.n_frames} views={result.n_views}")
    print(
        f"median reproj error: {np.nanmedian(err):.3f} px   max: {np.nanmax(err):.3f} px"
    )
    print(
        f"3D recovery RMSE vs truth: {np.sqrt(np.nanmean((result.pts3d - pts3d_true) ** 2)):.4f} mm"
    )

    render_pose3d_video(result, HERE / "demo_pose3d.mp4", fps=15)
    print(f"wrote {HERE / 'demo_pose3d.mp4'}")


if __name__ == "__main__":
    main()
