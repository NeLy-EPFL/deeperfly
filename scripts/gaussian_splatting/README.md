# Gaussian-splatting exploration

Reconstruct the fly in 3D at a single video frame with 3D Gaussian Splatting,
starting from the camera parameters deeperfly already infers. This is a
research spike (branch `explore/gaussian-splatting`), not part of the package.

Two eventual goals:
1. **Now:** a visual sanity check that a splat built from the 7 calibrated
   deeperfly cameras is geometrically consistent.
2. **Later:** use the splat's analysis-by-synthesis photometric signal to
   *refine* the camera extrinsics.

## Why two environments

The main `.venv` (torch 2.12+cu130) can't run gsplat: this box has no CUDA
toolkit / no `nvcc`, so gsplat's kernels can't be JIT-compiled, and there is no
prebuilt gsplat wheel for cu130. So the fitting step runs in a separate,
pinned venv `.venv-gsplat` (Python 3.10, torch 2.4.1+cu121) that installs
gsplat's **prebuilt** wheel (ships a compiled `csrc.so`, no `nvcc` needed).

Create it once:

```bash
uv venv .venv-gsplat --python 3.10
.venv-gsplat/bin/python -m ensurepip >/dev/null 2>&1 || true
uv pip install --python .venv-gsplat/bin/python \
  torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv-gsplat/bin/python \
  ninja packaging setuptools wheel numpy "opencv-python-headless<5" h5py imageio imageio-ffmpeg tqdm
uv pip install --python .venv-gsplat/bin/python \
  "gsplat==1.5.3+pt24cu121" --extra-index-url https://docs.gsplat.studio/whl/pt24cu121
```

## Pipeline

**1. Prepare a single-frame dataset** (main `.venv`; reads `results.h5` + videos):

```bash
.venv/bin/python scripts/gaussian_splatting/prepare_data.py --frame 32
```

Writes `gs_data/frame_<N>/` with per-camera `images/`, keypoint-hull fly
`masks/`, `overlays/` (mask + reprojected keypoints, for eyeballing the
calibration), `cameras.npz` (`viewmats`, `Ks` — deeperfly's OpenCV
`R@X+t` extrinsics map 1:1 onto gsplat), `init.npz` (point cloud seeded from
the triangulated keypoints + skeleton + random fill), and `keypoints.npz`.

**2. Fit the splat** (`.venv-gsplat`):

```bash
.venv-gsplat/bin/python scripts/gaussian_splatting/fit_gsplat.py \
  --data gs_data/frame_32 --iters 7000 --strategy default
```

Masked photometric loss (L1 + SSIM) plus an alpha-vs-mask term, with a random
background each step so opacity has to explain the silhouette. Outputs under
`<data>/gs_out/`:

- `train_views_gt_render_alpha.png` — per-view GT | render | alpha montage
  with masked PSNR (the sanity check).
- `orbit.mp4` — novel-view turntable across the covered azimuth arc.
- `gaussians.pt` — the optimized Gaussians.

`--strategy mcmc` swaps grad-based densification for MCMC (fixed cap
`--cap`), which can be steadier under few views.

## Setup notes / caveats

- The 7 cameras only cover the **front hemisphere** — expect artifacts in
  novel views from behind / above / below.
- Cameras are **near-orthographic** (fx≈22388, ~107 units out) — gsplat
  handles it, but per-view perspective cues are weak; depth is constrained by
  the wide *angular* baseline across cameras, not per-view perspective.
- The keypoint-hull mask grabs some of the support ball near the feet and
  drops the translucent wings; tighten `--mask-pad` / segmentation later.
