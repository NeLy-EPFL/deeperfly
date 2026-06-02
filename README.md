# deeperfly

A JAX rewrite of the [DeepFly3D](https://github.com/NeLy-EPFL/DeepFly3D) /
[DeepFly2D](https://github.com/NeLy-EPFL/DeepFly2D) /
[PyBundleAdjustment](https://github.com/semihgunel/PyBundleAdjustment) pipeline
for markerless 3D pose estimation of tethered *Drosophila* from a multi-camera
rig: **2D pose → camera calibration (bundle adjustment) → triangulation → error
correction → visualization.**

The computer-vision core follows OpenCV's conventions (Rodrigues rotations,
`projectPoints` distortion, DLT triangulation) and is cross-checked against
OpenCV in the test suite. Everything geometric is JAX (JIT- and autodiff-
friendly); the 2D detector ships two interchangeable backends behind one
interface — a JAX (Equinox) port of DeepFly2D's stacked hourglass (the default,
and faster on GPU) and the original PyTorch network — selectable with the
config's `[detector].backend = "jax" | "torch"`.

## Pipeline

| Stage | Module | Notes |
| --- | --- | --- |
| 2D pose | `pose2d/` (`backends/{jax,torch}/`) | Stacked hourglass in two backends behind one interface; JAX (Equinox) is the default, PyTorch runs the original weights directly. |
| Calibration | `pipeline.calibrate` → `bundle_adjustment/` | Fly-as-target BA: confidence weights, Huber loss, bone-length prior. |
| Triangulation | `triangulate.py` / `pipeline.reconstruct` | NaN-aware DLT + greedy reprojection-outlier rejection (default). |
| PS correction | `pictorial.py` | Optional DeepFly3D-style pictorial structures: multi-view candidate selection + bone-length priors (`[pipeline].correct = "pictorial"`). |
| Correction | `correction.py` | Procrustes alignment (per side) + One-Euro / Gaussian smoothing. |
| Visualization | `viz.py`, `video.py` | matplotlib 2D overlays, 3D skeleton, MP4 export. |
| Result I/O | `io.py` | Self-contained HDF5 `PoseResult`. |
| Skeleton | `skeleton.py` + `data/skeleton_fly.toml` | 38 points, 10 limbs, 28 bones, per-camera visibility. |

## Usage

Geometry / bundle adjustment only:

```python
from deeperfly import CameraGroup, bundle_adjust

group = CameraGroup.from_config("examples/cameras.toml")
pts2d = group.project(pts3d)                       # (V, N, 2) observations
result, optimized, points = bundle_adjust(group, pts2d, fixed=["*.intr"])
```

2D → 3D pipeline from an existing 2D detection array:

```python
from deeperfly import CameraGroup, Skeleton, run_from_points2d

cameras = CameraGroup.from_config("examples/cameras.toml")
result = run_from_points2d(cameras, Skeleton.fly(), pts2d, conf, smooth="one_euro")
result.save("fly.h5")
```

End to end from images/video via the CLI:

```bash
deeperfly init config.toml          # write a config to edit (cameras, inputs, pipeline, skeleton)
deeperfly run config.toml -i recording/                          # -> recording/deeperfly_outputs/ (2D -> 3D -> video)
deeperfly run config.toml -i recording/ -o out/ --until detect   # 2D only, cached in out/
deeperfly run config.toml -i recording/ -o out/                  # re-run: reuse cached 2D, continue to 3D + video
deeperfly run config.toml -i recording/ -o out/ --overwrite      # recompute everything
deeperfly info --in out/poses.h5
```

`deeperfly run` is the whole pipeline as one linear sequence of stages —
`detect` (2D) → `pose3d` (calibrate + triangulate + correct + smooth) →
`visualize`. By default it runs all three. `-i` is the recording; `-o` is an
output **directory** (default `<input>/deeperfly_outputs`) that collects the
result `poses.h5`, the rendered videos and a copy of the config. Each run reuses
whatever is already cached there and computes only what is missing, so prior work
is never recomputed: a fresh directory starts at `detect`, a cached 2D-only
`poses.h5` resumes at `pose3d`, and one that already has 3D just gets visualized.
`--overwrite` ignores the cache and recomputes everything; `--until <stage>` stops
early (e.g. `--until detect` writes a 2D-only result; `--until pose3d` skips the
video). Detector weights are downloaded and converted to the native JAX
checkpoint **automatically on first use** — nothing to pre-fetch (the `torch`
backend skips the conversion entirely). Add `-v`/`-vv` for progress logging
(`-v` also reports the resolved video backend, image sizes and detector batch) or
`-q` to quiet it.

A single `config.toml` (made by `deeperfly init`) carries everything a run needs:
the camera rig, the input filename→camera map, the 2D detector, the pipeline
options, bundle adjustment, and the skeleton — so a full `run` is just
`config -i <recording>` (outputs default to `<recording>/deeperfly_outputs/`).
The `[inputs]` section maps each camera to
the filename prefix of its frames under `-i` (e.g. `rh = "camera_0"` finds
`camera_0.mp4` or the image sequence `camera_0_img_*.jpg`); knobs like the
detector backend, `correct = "reproject" | "pictorial"`, stripe merging, the
bundle-adjustment keypoints, and smoothing all live in the config too. Sections
are independently usable: `CameraGroup.from_config` reads only the cameras,
`Skeleton.from_config` only `[skeleton]`.

See [`examples/bundle_adjustment.ipynb`](examples/bundle_adjustment.ipynb) for the
BA walkthrough and [`examples/pipeline_demo.py`](examples/pipeline_demo.py) for a
synthetic end-to-end run (no weights required).

## 3D correction: reproject vs pictorial structures

Each view is detected independently; the two views only meet *geometrically*,
and there are two ways to do that (`run_from_points2d(..., correct=...)` or
`[pipeline].correct` in the config):

- **`reproject`** (default) — triangulate the arg-max detections and greedily drop
  the worst-reprojecting view per offending point. Fast, and it *vetoes* a bad
  per-view detection.
- **`pictorial`** — DeepFly3D-style pictorial structures over the detector's top-K
  candidate peaks (`pictorial.py`): per joint it builds multi-view-consistent 3D
  hypotheses, then picks one per joint by exact dynamic programming along each
  limb (the fly's legs/stripes are simple chains) under bone-length priors, with
  an optional temporal term (`[pipeline.pictorial].temporal`). It can *recover* a joint when the
  arg-max landed on the wrong heatmap peak (occlusion, crossing legs, L/R
  confusion) instead of merely dropping it. It needs the full-heatmap detect path
  (so it is slower) and is strictly opt-in; on clean recordings it matches
  `reproject`, earning its keep on the hard frames.

## 2D detector backends

The detector has two interchangeable backends behind one interface, under
`pose2d/backends/{jax,torch}/` — each exposing the same `HourglassNet` /
`load_model` / `predict_heatmaps`. Both are installed by default and selectable
with `[detector].backend`. The PyTorch backend runs the published `sh8`
weights directly; the JAX backend (the default) runs the same weights from a
native checkpoint that `deeperfly run` downloads and converts automatically on
first use, and is validated to match the PyTorch reference numerically (see
`tests/test_pose2d_torch.py`). JAX is the
faster backend on GPU — benchmark them on your own hardware:

```bash
uv run python dev/bench_pose2d.py --batch 7 --frames 8
```

On NVIDIA GPUs both backends use CUDA automatically (JAX via the `gpu` extra). On
**Apple Silicon** the PyTorch backend auto-uses the GPU via Metal (MPS) with no
setup — the simplest accelerated path on macOS. To accelerate the *JAX* backend
on macOS instead, install the optional `mps` extra (`uv pip install
'deeperfly[mps]'`), which adds the experimental [`jax-mps`](https://github.com/tillahoffmann/jax-mps)
Metal plugin; the float32 detector then runs on Metal while geometry and bundle
adjustment stay in float64 on the CPU (MLX is float32-only). Both detector
backends match their CPU output to float32 precision.

## Video I/O backends

`deeperfly.video` reads and writes frames through a pluggable backend registry,
so you can pick where decoding happens and what the frames live in:

| Backend | Read | Write | Seek | Frames | Install |
| --- | :-: | :-: | :-: | --- | --- |
| `imageio` | ✓ | ✓ | – | NumPy (CPU) | `viz` extra |
| `opencv` | ✓ | ✓ | ✓ | NumPy (CPU) | `opencv` extra |
| `pyav` | ✓ | ✓ | ✓ | NumPy (CPU) | `pyav` extra |
| `decord` | ✓ | – | ✓ | NumPy / `torch` (CPU/**CUDA**) | `decord` extra |
| `video_reader_rs` | ✓ | – | ✓ | NumPy (CPU) | `video-reader-rs` extra |
| `torchcodec` | ✓ | – | ✓ | `torch.Tensor` (CPU/**CUDA**) | `torchcodec` extra |
| `dali` | ✓ | – | ✓ | `torch.Tensor` / NumPy (**CUDA**) | `dali` extra |

```python
from deeperfly import video

frames = video.read_video("clip.mp4")                              # auto: NumPy (host)
frames = video.read_video("clip.mp4", backend="pyav")              # frame-accurate
frames = video.read_video("clip.mp4", indices=[0, 50, 120])        # random access
frames = video.read_video("clip.mp4", device="cuda")               # GPU tensor (NVDEC)
video.write_mp4(frames, "out.mp4", fps=30, backend="opencv")
video.available_read_backends()       # what's installed here
```

`backend="auto"` and `device="auto"` (both defaults) pick the **fastest working**
path. With a GPU present and a GPU backend installed, frames are decoded on the
GPU; otherwise the fastest installed CPU decoder is used. Preference order
(fastest first):

- **GPU:** `torchcodec` → `dali` → `decord` (all frame-accurate).
- **CPU:** `decord` → `video_reader_rs` → `torchcodec` → `pyav` → `opencv` →
  `imageio` (last; it forks an `ffmpeg` subprocess, which is slow and trips
  Python 3.13's `os.fork()`-in-a-multithreaded-process warning).

`device="auto"` always returns host NumPy (portable). `device="cuda"` keeps frames
an on-device `torch.Tensor` for zero-copy handoff to JAX, and with `backend="auto"`
it tries each GPU backend until one decodes, gracefully falling back to a CPU read
if none can. Pass `indices=[...]` for random access: seek-capable backends fetch
just those frames, others decode up to `max(indices)` and gather. `deeperfly run`
reads the decoder from `[detector].video_backend`. NVIDIA DALI's wheel is
CUDA-version specific — the `dali` extra pins the CUDA 13 build
(`nvidia-dali-cuda130`); install `nvidia-dali-cuda1NN` directly for another
toolkit.

### Image sequences (jpg / png / …)

A folder or glob of frames is read by `read_images`, which decodes in parallel
across threads (so throughput scales with cores), broadcasts grayscale to RGB and
supports the same `indices` / `start:stop:step` selection. `read_frames` is the
unified entry point — it routes a video file to `read_video` and an image
directory to `read_images` — so the pipeline accepts either input. With
`device="cuda"` JPEGs are decoded on the GPU (torchvision nvJPEG) straight into a
device tensor for `to_jax`:

```python
frames = video.read_images("frames/", workers=8)            # parallel CPU decode
frames = video.read_images("frames/", indices=[0, 10, 20])  # subset
frames = video.read_images("frames/", device="cuda")        # nvJPEG -> GPU tensor
frames = video.read_frames(path)                            # video file *or* image dir
```

### GPU video decoding for `deeperfly run` (opt-in, zero-copy)

`deeperfly run` decodes frames **on the CPU by default** and uploads each window to
the GPU in one shot. The 2D detector — not decode — is the bottleneck, so CPU
decode lands within ~6% of GPU/NVDEC end to end (RTX 4090, 7-cam 480×960; see
`dev/bench_video.py`), and the default needs none of the CUDA-video stack below.

Set `[detector] decode_device = "cuda"` (or `"auto"`) to decode **directly on the
GPU** (NVDEC) instead: the decoded `torch.Tensor` is bridged to JAX via DLPack, so
`preprocess` resizes it on the GPU and frames never touch host memory. It falls
back to CPU decode if no GPU backend can decode here, and is worth it only on the
fastest GPUs — install one of the decoders below first.

**Setup.** The `gpu` extra installs a CUDA-enabled JAX (the detector on the GPU) *and*
the fastest GPU decoder, `torchcodec` on CUDA (the CUDA-13 torchcodec build + NVIDIA
NPP, pinned in the lockfile — see below). **NVIDIA DALI** is an alternative NVDEC
decoder; add `--extra dali` for it. Both are frame-accurate.

```bash
# Recommended — CUDA JAX + torchcodec-on-CUDA, fully managed by the lock:
uv sync --extra gpu

# Add DALI too (also NVDEC, frame-accurate; needs only the NVIDIA driver):
uv sync --extra gpu --extra dali

# …or just grab everything (all decoders + viz + dali):
uv sync --all-extras
```

- **CUDA JAX** (`gpu` extra) puts the detector on the GPU. Check with
  `python -c "import jax; print(jax.devices())"` → it should list a `CudaDevice`.
- **`torchcodec`** is tried first and is the fastest decoder here. CUDA decode
  needs a *CUDA-enabled* torchcodec build — the plain PyPI wheel is CPU-only — plus
  NVIDIA **NPP** (for GPU color conversion). The `gpu` extra handles both: on Linux
  x86-64 `[tool.uv.sources]` routes `torchcodec` to the PyTorch CUDA-13 index
  (`download.pytorch.org/whl/cu130`) and pulls `nvidia-npp`, so a plain `uv sync
  --extra gpu` reproduces a working GPU decoder from the lockfile — no manual
  `uv pip install`. (torch itself is left on PyPI; its wheel already bundles CUDA 13,
  so the cu130 torchcodec is ABI-compatible.) deeperfly `dlopen`s NPP's libs for
  you, so no `LD_LIBRARY_PATH` tweak is needed. If the CUDA build/NPP are missing
  you'll see a one-time `GPU decode via 'torchcodec' backend failed …` and it
  moves on to the next backend.
- **`dali`** (NVIDIA DALI) is a frame-accurate NVDEC decoder that just needs the
  driver. Its wheel is CUDA-version specific — the `dali` extra pins the CUDA 13
  build (`nvidia-dali-cuda130`); for another toolkit install `nvidia-dali-cuda1NN`
  yourself. (It decodes the window straight to a GPU tensor, but rebuilds its
  pipeline per window, so it prefers a larger `chunk_frames`.)

With `decode_device = "cuda"`, if no frame-accurate GPU backend can decode (or
there's no GPU), decoding falls back to the fastest installed CPU backend (decord,
…) — the same path as the default, just selected automatically. Either way each
window is uploaded to the GPU in a single transfer, so the host path runs at full
detector speed.

> **Memory & long videos:** detection **streams** — it decodes and detects
> `[detector] chunk_frames` frames at a time per camera (default 64) and frees each
> window before the next, so peak frame memory is bounded by the window, *not* the
> recording length. A multi-hour video runs in constant memory. `chunk_frames` is a
> **memory** knob, not a speed one: detection is compute-bound (~28 frames/s on an
> RTX 4090) and every decoder is far faster (see `dev/bench_video.py`), so a small
> window costs no throughput — 64 holds ~0.6 GB of frames for a 7-camera 480×960
> rig. Lower it for high-res / many cameras / small GPUs; raise it on the DALI
> fallback or to cut per-window decoder setup. `deeperfly run` also caps JAX's GPU
> pool at half the card (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`, unless you set it),
> and the auto path degrades to CPU decode if a window still won't fit.

#### Doing it yourself

`video.to_jax` is the DLPack bridge used under the hood — on a shared GPU, JAX
wraps the **same** device buffer the decoder produced:

```python
frames = video.read_video("clip.mp4", device="cuda")  # torch.Tensor on the GPU (NVDEC)
x = video.to_jax(frames)                               # jax.Array on the GPU, zero-copy
# keep `frames` alive until JAX has consumed `x`
```

Notes: both sides must share the same CUDA device; modern JAX/torch handle the
DLPack stream synchronization. `video.to_jax` also accepts NumPy (host copy onto
JAX's default device) and decord/DALI GPU tensors. `video.to_numpy` is the
host-side counterpart. `pose2d.preprocess` accepts an on-device tensor directly
(bridging via DLPack), resizes/normalizes with `jax.image.resize`, and keeps the
result on the GPU.

## Development

```bash
uv sync --group test                 # install with test dependencies
uv run --group test pytest           # run the suite (incl. PyTorch-equivalence tests)
```

Optional extras: `viz` (matplotlib + imageio for plotting/video), and the video
read/write backends `opencv` / `pyav` / `decord` / `video-reader-rs` /
`torchcodec`. PyTorch is a core dependency (the second
detector backend), so no extra is needed for it.
