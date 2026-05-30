"""Synthetic end-to-end demo of the deeperfly 3D pipeline (no weights needed).

Generates a moving 38-point fly, projects it through the example 7-camera rig,
adds detector-like noise and a gross outlier, then runs the geometry pipeline
(visibility masking -> triangulation -> outlier rejection -> smoothing) and
writes an HDF5 result plus a 3D-skeleton video.

    uv run --extra viz python examples/pipeline_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from deeperfly import CameraGroup, Skeleton, run_from_points2d

HERE = Path(__file__).parent


def fly_motion(rng, n_frames=30, n_pts=38):
    base = rng.uniform(-1.5, 1.5, size=(n_pts, 3))
    t = np.linspace(0, 1, n_frames)[:, None, None]
    wiggle = 0.3 * np.sin(2 * np.pi * (t + np.arange(n_pts)[None, :, None] / n_pts))
    return base[None] + wiggle  # (T, N, 3)


def main():
    rng = np.random.default_rng(0)
    cameras = CameraGroup.from_config(HERE / "cameras.toml")
    skeleton = Skeleton.fly()

    pts3d_true = fly_motion(rng)
    pts2d = np.array(cameras.project(pts3d_true))  # (V, T, N, 2)
    pts2d += rng.normal(scale=0.15, size=pts2d.shape)  # sub-pixel detector noise
    pts2d[2, 5, 9] += [250.0, -200.0]  # a gross outlier
    conf = rng.uniform(0.5, 1.0, size=pts2d.shape[:3])

    result = run_from_points2d(
        cameras,
        skeleton,
        pts2d,
        conf,
        do_calibrate=False,
        smooth="one_euro",
        fps=100.0,
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

    try:
        from deeperfly import video

        video.render_pose3d_video(result, HERE / "demo_pose3d.mp4", fps=15)
        print(f"wrote {HERE / 'demo_pose3d.mp4'}")
    except Exception as exc:  # noqa: BLE001 -- viz is optional
        print(f"(skipped video: {exc})")


if __name__ == "__main__":
    main()
