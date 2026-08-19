# Using the library API

deeperfly is a Python library as well as a CLI. The public API lives at the top
level (`from deeperfly import ...`): `Config`, `Camera`, `CameraGroup`,
`Skeleton`, `PoseResult`, `Recording`, `bundle_adjust`,
`bundle_adjust_from_config`, `run_from_points2d`, `run_recording`,
`resolve_recordings`, `detect_2d`, `solve_inverse_kinematics`, `KinematicTemplate`,
and the `geometry`, `triangulation`, `eks`, `pictorial`, `pipeline`, `recordings` and
`io` submodules. This guide shows the common tasks; the
[API reference](../reference/api.md) documents every symbol.

Add it to your project — prefix with `UV_TORCH_BACKEND=auto` so uv picks the
right PyTorch wheel (`uv add` has no `--torch-backend` flag):

```bash
UV_TORCH_BACKEND=auto uv add git+https://github.com/NeLy-EPFL/deeperfly
```

Sections of a `config.toml` are independently usable: load it once with
`Config.from_toml`, then `CameraGroup.from_config(config)` reads only the
cameras, `Skeleton.from_config(config)` only `[skeleton]`. Foreign sections are
ignored, so a config you only use for its rig needs no detector or visualization
keys.

## Geometry and bundle adjustment

```python
from deeperfly import CameraGroup, Config, bundle_adjust

config = Config.from_toml("config.toml")
# image_sizes is required whenever the config omits principal_point_px -- which the
# packaged one does, deliberately: the principal point is each view's own image center,
# so writing it down would be one more number per camera to keep in step with the footage.
group = CameraGroup.from_config(config, image_sizes={"f": (1008, 1600), ...})
pts2d = group.project(pts3d)                       # (V, N, 2) observations
result, optimized, points = bundle_adjust(group, pts2d, fixed=["*.intr"])
```

`image_sizes` maps a view name to its source's raw `(height, width)`; a run records them in
`results.h5`, where `deeperfly.results.StageStore(path).read_image_sizes()` reads them back
(a loaded `PoseResult`'s cameras already carry the inferred principal points). Omitting it
raises `ValueError` naming the intrinsic it could not infer rather than guessing a center.
`[cameras].calibration`, when the config carries one, **wins** over the orbit specs — the
view tables are then read only for their order, which is what the `V` axis of every points
array means.

`bundle_adjust` returns the raw `scipy.optimize.OptimizeResult`, the refined
`CameraGroup`, and the refined 3D points. `fixed` / `shared` anchor the world
gauge with the same grammar as the config (`"*.intr"`, `"f.rvec"`, tying
`[["lf.tvec[2]", "rf.tvec[2]"]]`); extra keyword arguments
(`max_nfev`, `loss`, `f_scale`, ...) pass straight to scipy.
`bundle_adjust_from_config(config, pts2d)` drives the same call from a config's
`[bundle_adjustment]` section.

The lower-level `deeperfly.geometry` module holds the JAX projection /
triangulation / Rodrigues primitives (JIT- and grad-friendly, float64 on the
CPU), and `deeperfly.triangulation` the NaN-aware DLT and RANSAC helpers.

## The full 2D→3D pipeline from a 2D detection array

If you already have 2D detections, `run_from_points2d` runs the array pipeline
(no files): optional bundle adjustment → 3D reconstruction, returning a
`PoseResult`.

```python
from deeperfly import CameraGroup, Config, Skeleton, run_from_points2d

config = Config.from_toml("config.toml")
cameras = CameraGroup.from_config(config)
result = run_from_points2d(cameras, Skeleton.from_config(config), pts2d, conf)
result.save("fly.h5")
```

Read the skeleton from the same config as the cameras, so `P` means one thing.
`Skeleton.fly()` is the shortcut for the packaged one — 38 points, `fly38` — which is what
`Config.default()` resolves to as well.

`run_from_points2d(..., triangulation=..., do_pictorial=...)` selects the 3D
reconstruction: `triangulation` is `ransac` (default), `greedy` or `dlt`, and
`do_pictorial=True` runs pictorial-structures peak recovery first (see the
[pipeline explainer](../explanation/pipeline.md)). Unobserved points are expected
to already be NaN. It does **not** run the smoother, the postprocess chain or the inverse
kinematics — those are pipeline stages, which `run_recording` drives.

## Running a recording with caching

`run_recording(config_path, outdir, *, sources=None, input=None, overwrite=None,
progress=None)` is the staged run behind `deeperfly run`: it resolves the config against an
output directory, narrows it to the footage present, runs the enabled `[pipeline]` stages,
and reuses cached results whose config is unchanged. `overwrite` is the stage-name list
`--overwrite` builds.

```python
from pathlib import Path

from deeperfly import Config, resolve_recordings, run_recording

config = Config.default()
for src, sources in resolve_recordings(
    [Path("recordings/fly1")], recursive=False, config=config
):
    run_recording(None, src / "deeperfly_outputs", sources=sources)
```

`resolve_recordings` returns `(recording directory, source -> footage files)` per recording;
`recursive` is required rather than defaulted, because whether an input is *a* recording or a
*parent of* recordings is not something to guess. A recording holding footage for only some
of the config's sources is still returned, with a warning naming what is absent — the run
then narrows itself to it.

Passing `None` as the first argument means "use the config already in the output directory,
else the packaged default", exactly as `deeperfly run` with no `-c` does; pass a path to a
config TOML to override it.

### Running the detector yourself

Detection is two calls. `deeperfly.pose2d.stream.load_models` loads every model the plan
references and checks each checkpoint's recorded channel names against the config's skeleton;
`detect_2d` then streams the recording through the plan.

```python
from pathlib import Path

from deeperfly import Config
from deeperfly.pose2d.stream import detect_2d, load_models

config = Config.default()
plan = config.detection_plan()
models = load_models(plan)                      # name -> LoadedModel
for model in models.values():
    model.set_precision(config.pose2d.precision)

pts2d, conf, candidates = detect_2d(
    config, plan, models,
    input=Path("recordings/fly1"),
    want_candidates=False, k=0,                 # both required: no silent default
)
```

`pts2d` is `(V, T, P, 2)` in **raw footage pixels** — each pathway's preprocessing is
inverted on the way back — and `conf` is `(V, T, P)`. A `(view, point)` no pathway writes
stays NaN, so the arrays agree with every downstream stage about what each view observed.
`want_candidates=True` (with `k` peaks) additionally returns the top-K candidates
[pictorial structures](../reference/configuration.md#pictorial_structures) needs; both
arguments are keyword-only and have no defaults, so the caller cannot get them by accident.

There is no `load_detector` and nothing auto-downloads: `load_models` resolves each model's
`weights` along `$DEEPERFLY_MODELS` and raises `SystemExit` naming every directory it
searched when it finds none. It also refuses a checkpoint whose recorded channel names are
not the config's skeleton, and one that records no channel names at all.

One step of the real stage is missing above, and it matters with the packaged config: a
`{ op = "crop", auto = true }` preprocessor has no window until something searches for one,
and asking an unresolved automatic crop for its geometry raises rather than quietly falling
back to the whole frame. The `pose2d` stage calls
`deeperfly.pose2d.autocrop.ensure_resolved(config, plan, models=..., cameras=..., ...)`
between loading the models and detecting — after the models are on the device, because the
search forwards through them. Do the same, or use a plan whose crops are explicit boxes.

## Fitting joint angles

`solve_inverse_kinematics` is the stage behind `do_inverse_kinematics`: it fits a
NeuroMechFly model's joint angles to a 3D pose sequence and returns an `IKResult` carrying
`angles` `(T, D)`, `angle_names`, the model's own keypoints `model_pts3d`, the measured
`alignment` / `body_plan`, and the per-chain `chain_scales` / `chain_offsets` / `body_scale`.

```python
from deeperfly import KinematicTemplate, PoseResult, solve_inverse_kinematics
from deeperfly.inverse_kinematics.articulation import load_articulation

result = PoseResult.load("recording/deeperfly_outputs/results.h5")
fit = solve_inverse_kinematics(
    result.pts3d,
    result.skeleton,
    KinematicTemplate.load("neuromechfly"),
    articulation=load_articulation(),   # None fits the legs only
    weights=None,                       # e.g. a (T, P) confidence array
)
```

Its keyword defaults are read from `config.InverseKinematicsParams`, so they cannot drift
from what a run does — including `symmetric_segments=True`. It needs the optional
[`ik` extra](configuration.md#inverse-kinematics); without QuickIK installed it raises
`MissingQuickIK`, which is what the pipeline stage catches to skip with a logged reason
rather than failing the run.

## Inspecting a result

`PoseResult` is the assembled, self-contained result — the cameras, skeleton,
`pts2d` `(V, T, P, 2)`, `conf`, `pts3d` `(T, P, 3)` and `reproj_error`. It
round-trips through HDF5 and reconstructs the cameras and skeleton, so a result
is portable without the original config.

```python
from deeperfly import PoseResult

result = PoseResult.load("recording/deeperfly_outputs/results.h5")
print(result.n_views, result.n_frames)
xyz = result.pts3d              # (T, P, 3), NaN where un-triangulated
```

See the [output-format reference](../reference/output-format.md) for the on-disk
schema and the "best available" assembly rule `load` applies.

## Frame I/O

`deeperfly.io` reads and writes frames through **PyAV** (in-process FFmpeg, with
libx264 bundled in the wheel — no system FFmpeg needed). All decoding and
encoding runs on the CPU and yields `(T, H, W, 3)` uint8 RGB NumPy.

`open_reader(source)` resolves a source to a `VideoReader` (a video file) or an
`ImageSequenceReader` (a directory, glob, or explicit file list), both subclasses
of `FrameReader`. You then index it, stream it, or probe `count` / `fps`. Image
sequences are decoded by OpenCV, in parallel across threads.

```python
from deeperfly import io

reader = io.open_reader(path)                 # video file or image dir/glob/list
frames = reader[:]                            # (T, H, W, 3) uint8 NumPy (host)
clip = io.VideoReader("clip.mp4")[[0, 50]]    # random access (seeks per frame)
for block in io.open_reader(path).stream_blocks(block_size=64):  # forward, low memory
    ...

# gray_ok is permission, not a promise: monochrome footage then arrives as (T, H, W, 1)
# and skips the decoder's YUV->RGB conversion, which is most of what reading a frame
# costs. Color footage -- and TV-range footage, where the conversion rescales the luma
# so the plane is not what RGB would give -- comes back (T, H, W, 3) regardless, so read
# `shape[-1]` rather than assume. thread_count caps the decode pool, which matters when
# several sources are read at once: sizing each from the core count oversubscribes.
for block in io.open_reader(path).stream_blocks(
    block_size=64, gray_ok=True, thread_count=4
):
    ...

# VideoWriter encodes a frame, a batch, or any iterable -- so a long clip can be
# written as it is produced, without ever holding every frame in memory.
with io.VideoWriter("out.mp4", fps=30) as writer:
    writer.write_frames(frames)               # or write_frame() per frame
```

## Examples

Two runnable notebooks, rendered here with their outputs (the Plotly figures
stay interactive):

- [Bundle adjustment](../examples/bundle_adjustment.ipynb) — build a multi-camera
  rig, perturb it, and recover it with `bundle_adjust`.
- [Pipeline walkthrough](../examples/pipeline_walkthrough.ipynb) — the full
  2D→3D pipeline one stage at a time.
