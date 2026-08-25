# Getting started

This walkthrough takes you from a clean machine to a rendered 3D pose video using the
recording bundled with the repository. Three steps do the work: install, put a detector
checkpoint where deeperfly looks for it, run.

!!! warning "Nothing downloads at run time"

    Every detector deeperfly ships is trained per project, so there is no checkpoint for
    it to fetch on demand: the weights must be on this machine **before** the first run.
    That is [step 2](#2-get-the-detector-weights), and it is the one step a fresh install
    cannot skip — without it `deeperfly run` stops before any detection begins.

## 1. Install

Install the CLI with [uv](https://docs.astral.sh/uv/). `--torch-backend=auto`
lets uv pick the right PyTorch wheel for your machine (CUDA, Metal, or CPU):

```bash
uv tool install git+https://github.com/NeLy-EPFL/deeperfly --python 3.13 --torch-backend=auto
```

Python 3.11–3.13 are supported. To follow along with the bundled example, clone the repo
too (the example footage lives under `examples/data/`):

```bash
git clone https://github.com/NeLy-EPFL/deeperfly
cd deeperfly
```

### Joint angles: the `ik` extra

The `inverse_kinematics` stage is **on by default**, and its solver
([QuickIK](https://nely-epfl.github.io/quickik/)) is an optional extra: it publishes no
wheels, so installing it builds a Rust extension and needs a Rust toolchain
([rustup.rs](https://rustup.rs)), which nothing else in deeperfly does. Add it to a tool
install with `--with`:

```bash
uv tool install git+https://github.com/NeLy-EPFL/deeperfly --python 3.13 --torch-backend=auto \
  --with "quickik @ git+https://github.com/NeLy-EPFL/quickik#subdirectory=python"
```

In a clone, `uv sync --extra ik` does the same thing. Without it the stage **skips with
the reason logged** rather than failing the run: you lose the joint angles and the two
NeuroMechFly videos, not the 3D pose. A result file that already holds a fit renders its
overlays on a plain install.

### Development installation

If you want to edit deeperfly itself, install the CLI from your clone in editable
mode instead — source changes then take effect without reinstalling:

```bash
uv tool install ./ --editable --python 3.13 --torch-backend=auto
```

See [CONTRIBUTING.md](https://github.com/NeLy-EPFL/deeperfly/blob/main/CONTRIBUTING.md)
for tests, linting, and the docs site.

### Updating

Upgrade a normal install to the latest `main`:

```bash
uv tool upgrade deeperfly
```

For a development install, pull the latest source — editable code is picked up
automatically. Re-run the install command only if dependencies changed:

```bash
git pull
uv tool install ./ --editable --python 3.13 --torch-backend=auto   # only if deps changed
```

## 2. Get the detector weights

Three checkpoints ship with 0.2.0. All three take **one grayscale channel** and predict
**all 38 points of the `fly38` skeleton in every view**, and all were trained on the same 55
recordings. The two `hrnet` arms saw 465 labeled moments / 138,708 label cells; the MVT saw
those recordings after a further round of labeling, 485 moments / 144,775 label cells:

| checkpoint | `class` | bytes | sha256 |
| --- | --- | --- | --- |
| `mvt_r28_pad48_gray_fly38.pth` | `mvt` | 86,082,205 | `ae482d3a…` |
| `hrnet_w32_r27_gray_fly38.pth` | `hrnet` | 127,076,045 | `13ceb937…` |
| `hgnetv2_b4_r27_gray_fly38.pth` | `hrnet` | 62,452,371 | `fa427062…` |

Each lives in its own directory: the two `hrnet` checkpoints under
`/mnt/upramdya/data/TL/deeperfly-models/260819_*`, the MVT under
`260825_mvt_r28_pad48_gray_fly38`. Each directory holds the `.pth` plus a `README.md` (what
it was trained on and the gates it passed),
`SHA256SUMS`, and `fly38.toml` — the point order the checkpoint was trained in. The
checkpoint records that order inside itself too, and every run checks it against the
config's skeleton: a dense detector's channels *are* a skeleton, and two 38-point
skeletons in different orders would otherwise attach points to the wrong joints — a wrong
limb, not a crash.

Copy one directory to this machine, verify it, and point `$DEEPERFLY_MODELS` at it:

```bash
cd /path/to/models
sha256sum -c SHA256SUMS
export DEEPERFLY_MODELS=/path/to/models     # several dirs allowed, separated like PATH
```

The packaged default config names `mvt_r28_pad48_gray_fly38.pth` as a **bare filename**,
looked up along `$DEEPERFLY_MODELS`. Keep it bare rather than an absolute path: the
filename is a fact about which model a run used and travels with the recording, where
`/mnt/...` is a fact about one machine. An outright path
(`weights = "/path/to/mvt_r28_pad48_gray_fly38.pth"`) works too when you want one.

If the checkpoint is not found, the run stops before any detection, naming every
directory it searched:

```
[[pose2d.models]] 'dense38mv' (class 'mvt'): no checkpoint named 'mvt_r28_pad48_gray_fly38.pth'.
  Searched ($DEEPERFLY_MODELS, then the download cache):
    /home/you/.cache/deeperfly/weights
  Set DEEPERFLY_MODELS=/path/to/models, or write an explicit path:
    weights = "/path/to/mvt_r28_pad48_gray_fly38.pth"
```

That per-user cache directory is always last on the search path even though nothing
writes to it any more — it is the one place to drop a checkpoint without setting an
environment variable.

!!! note "Which of the three"

    `mvt` is the default because it computes a frame's views **together**: a joint only
    one camera can see informs the cameras that cannot. The two `hrnet` entries are
    per-view detectors (the same loader runs both backbones, selecting feature maps by
    stride). See [the dense-38 detectors](explanation/detectors.md) for what each one
    costs and buys.

## 3. Check the install

`deeperfly doctor` reports what this machine can run — accelerators, frame-I/O backends,
where it looks for weights, and the default config path:

```bash
deeperfly doctor
```

The `GPU inference` line tells you whether the detector will use the GPU:

```
inference
  torch             2.12.0+cu130  (CUDA: NVIDIA GeForce RTX 4090)
  GPU inference     available (23.5 GiB memory)
```

On a CPU-only box it reads `not available -- CPU only`. deeperfly still runs on
CPU — just slower.

The `weights` section answers step 2's question directly — every directory searched,
what is in it, and whether the default config's checkpoint resolves:

```
weights
  DEEPERFLY_MODELS  /data/deeperfly-models/260825_mvt_r28_pad48_gray_fly38
  searched [0]      /data/deeperfly-models/260825_mvt_r28_pad48_gray_fly38  (1 .pth)
  searched [1]      /home/you/.cache/deeperfly/weights  (empty)
  default wants     mvt_r28_pad48_gray_fly38.pth  --  found at /data/…/mvt_r28_pad48_gray_fly38.pth
```

With the variable unset the first line reads
`unset -- set it to the directory holding the checkpoints` and the last
`NOT FOUND on the search path above`.

(If you installed as a tool, run the commands below as written; inside a cloned checkout
you can instead use `uv run deeperfly ...`.)

## 4. Run the pipeline

The bundled recording is the 7-camera DeepFly3D rig (`camera_0.mp4` … `camera_6.mp4`),
64 frames. The packaged default config declares **eight** views — the seven orbiting
cameras plus `h`, the axial hind camera that is the rig's only left/right bridge — so
the run narrows itself to the footage it finds and says which views it dropped:

```bash
deeperfly run examples/data/
```

```
WARNING  recording examples/data has footage for 7 of 8 configured source(s) -- absent: ['vid_h'].
         The run will use the ['vid_f', 'vid_lf', ...] it has; check the [[sources]] `filename`
         globs if that is not what you expect
WARNING  narrowing this run to the footage present: source(s) ['vid_h'] resolved no files, so
         pathway(s) ['h'] and view(s) ['h'] are dropped -- running on 7 view(s):
         ['rh', 'rm', 'rf', 'f', 'lf', 'lm', 'lh']
INFO     stages: pose2d=on, bundle_adjustment=on, pictorial_structures=off, triangulation=on,
         eks=on, postprocess=on, inverse_kinematics=on, visualization=on
```

Narrowing happens *before* the snapshot and the fingerprints, so this is recorded as a
7-view run — and when the eighth camera turns up, the pathway list has changed and the
pipeline recomputes from detection down. Below two views the run refuses instead of
reconstructing a confident-looking nothing (one view triangulates to all-NaN without
raising).

!!! warning "`deeperfly run <dir>` does not read `<dir>/config.toml`"

    The config is `-c` if given, else the snapshot a previous run left in the **output**
    directory, else the packaged default. The demo works with no `-c` because it matches
    the packaged default; your own recording almost certainly needs `-c`.

Every stage runs except `pictorial_structures`: detect 2D pose in all seven views,
bundle-adjust the cameras, triangulate, smooth the 3D with the ensemble Kalman smoother
(worth −39% 3D jitter), apply the corrections that come from knowing the animal, fit
NeuroMechFly joint angles, and render the videos. `pictorial_structures` stays off
because it recovers a joint from the top-K peaks of a *one-body-side* detector, and
switching it on would un-densify a dense run.

On one machine (RTX 4090, 64 frames × 7 views) the whole thing takes **45 s**, of which
**25 s** is the automatic crop search on the front view. Nothing is downloaded and
nothing is `torch.compile`d — neither detector class compiles, so there is no warm-up
run to pay for.

!!! note "The 25 s: what the searched crop is doing"

    A detector is trained through a box, and a differently framed camera puts the animal
    at the wrong apparent *scale* — the one thing no augmentation undoes. The packaged
    config therefore gives its two axial views a **searched** crop —
    `{ op = "crop", auto = true }` in `[pose2d].preprocessors`. Here that is only `f`:
    `h`'s preprocessor was dropped along with the view it fed.

    The **search** is a coarse-to-fine grid over (center, width) scored by detector
    confidence, and needs no camera rig at all. What needs a solved rig is the **accept
    gate**, which is reprojection agreement with the other cameras' 3D: on this demo it
    measured 49.7 px → 48.3 px and accepted, while warning that 48 px is still far from
    the other cameras — its own reference comes from a rig only self-consistent to
    15.8 px. Where the gate cannot trust its ruler at all it refuses to choose and falls
    back to confidence alone, which lands a box roughly 1.7x too wide — still far better
    than dropping the whole frame in, which collapses detection outright (255 px).

    Every run writes the bundle-adjusted rig to `deeperfly_outputs/calibration.toml`.
    Point `[cameras].calibration` at it and re-run to give the gate a measured ruler.

Outputs land in `examples/data/deeperfly_outputs/` (override with `-o`):

```
examples/data/deeperfly_outputs/
├── results.h5         # cameras, skeleton, 2D + 3D keypoints, joint angles, reprojection error
├── config.toml        # a snapshot of the exact config this run used (narrowed to 7 views)
├── run.json           # per-stage fingerprints (drives cache reuse)
├── calibration.toml   # the bundle-adjusted rig, in the portable form another recording can use
├── autocrop.json      # the crop window the search settled on, so a re-run neither re-searches nor re-detects
├── pose2d.mp4         # camera montage with the 2D detections drawn on
├── pose3d.mp4         # same montage with the triangulated 3D skeleton reprojected, plus a synthetic plan view
├── pose_nmf.mp4       # the fitted NeuroMechFly skeleton
└── pose_mesh.mp4      # the fitted NeuroMechFly mesh
```

The last two need the `ik` extra; without it those two videos are skipped with the
reason logged and the first two are rendered as usual.

## 5. Inspect the result

```bash
deeperfly inspect examples/data/deeperfly_outputs/results.h5
```

```
file:     examples/data/deeperfly_outputs/results.h5
views:    7  ['rh', 'rm', 'rf', 'f', 'lf', 'lm', 'lh']
frames:   64
skeleton: fly38  (38 points)
has 3D:   True
reproj:   median 2.966 px  max 408.960 px
```

The median is the summary; the maximum is one cell of `views × frames × points`, and on
this recording the tail sits exactly where the rig's geometry says it must. The demo has
no `h` camera, so no camera sees both body sides and a contralateral joint is
triangulated from one side's cameras alone: 91.5% of cells reproject within 10 px, and of
the 463 cells (2.7%) beyond 20 px, **85% are a point on the side away from the camera**.
Open `pose3d.mp4` to see the reconstructed skeleton.

## 6. Re-run after a tweak

Re-running a finished recording is a cheap no-op — **0.91 s** here, every stage's cache
reused. **Editing the config recomputes only the affected stages.** Generate an editable
config, change one value, and re-run:

```bash
deeperfly init config.toml                                     # the packaged config, commented
deeperfly config set eks.inflate_threshold 20 -c config.toml   # validated as a run would
deeperfly run examples/data/ -c config.toml
```

```
INFO  reusing cached pose2d (pass --overwrite pose2d to force a recompute)
INFO  reusing cached bundle_adjustment ...
INFO  reusing cached triangulation ...
INFO  recomputing eks (config changed: inflate_threshold: 30.0 -> 20)
INFO  recomputing postprocess (an upstream stage recomputed ...)
INFO  recomputing inverse_kinematics (an upstream stage recomputed ...)
INFO  recomputing visualization (an upstream stage recomputed ...)
```

**9.7 s** rather than 45: the slow detection and the crop search are reused.
`deeperfly config set` rewrites an existing key in place, stage flags included
(`deeperfly config set pipeline.do_eks false -c config.toml`) — the packaged config
states every `do_<stage>` flag, so that is the only way to turn a default-on stage off.
A key whose table exists but does not state it is still refused, because appending a bare
key after a table header would reparent the keys below it. That resume/recompute model — and
`--overwrite` to force a redo — is the subject of the
[CLI guide](guides/cli.md#resuming-and-recomputing).

!!! note "A fresh output directory for a different skeleton"

    A run refuses an output directory whose stored pose is on a **different point set**,
    because `results.h5` arrays are `(…, P, …)` with no names beside them — two 38-point
    skeletons load each other's files perfectly happily. The message names what is only
    in each. Run into a fresh directory, or point `[skeleton]` at the skeleton the pose
    was detected on.

## Where to go next

- **[CLI usage](guides/cli.md)** — batch runs, output layout, every flag.
- **[Writing configs](guides/configuration.md)** — point deeperfly at your own
  cameras and tune the pipeline.
- **[How it works](explanation/pipeline.md)** — what each stage actually does.
- **[The dense-38 detectors](explanation/detectors.md)** — the two detector classes and
  how a config selects one.
- **[Library API](guides/library.md)** — drive the pipeline from Python.
