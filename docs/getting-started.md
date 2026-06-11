# Getting started

This walkthrough takes you from a clean machine to a rendered 3D pose video using
the recording bundled with the repository. It should take a few minutes plus the
one-time detector-weight download.

## 1. Install

Install the CLI with [uv](https://docs.astral.sh/uv/). `--torch-backend=auto`
lets uv pick the right PyTorch wheel for your machine (CUDA, Metal, or CPU):

```bash
uv tool install git+https://github.com/NeLy-EPFL/deeperfly --python 3.13 --torch-backend=auto
```

To follow along with the bundled example, clone the repo too (the example
footage lives under `examples/data/`):

```bash
git clone https://github.com/NeLy-EPFL/deeperfly
cd deeperfly
```

## 2. Check the install

`deeperfly doctor` reports what this machine can run — accelerators, frame-I/O
backends, the detector weights, and the default config path:

```bash
deeperfly doctor
```

The `GPU inference` line tells you whether the detector will use the GPU:

```
  GPU inference     available (24.0 GiB memory)
```

On a CPU-only box it reads `not available -- CPU only`. deeperfly still runs on
CPU — just slower. (If you installed as a tool rather than cloning, prefix the
commands below with nothing; inside a cloned checkout you can instead use
`uv run deeperfly ...`.)

## 3. Run the pipeline

The example recording is the standard 7-camera rig (`camera_0.mp4` …
`camera_6.mp4`), which the packaged default config already targets — so you can
run it with no `-c`:

```bash
deeperfly run examples/data/
```

This detects 2D pose in every view, bundle-adjusts the cameras, triangulates to
3D, and renders the skeleton videos. The first run downloads the detector weights
and (on CUDA) spends a little time on `torch.compile`; later runs skip both.

Outputs land in `examples/data/deeperfly_outputs/` (override with `-o`):

```
examples/data/deeperfly_outputs/
├── poses.h5        # cameras, skeleton, 2D + 3D keypoints, reprojection error
├── config.toml     # a snapshot of the exact config this run used
├── run.json        # per-stage fingerprints (drives cache reuse)
├── pose2d.mp4      # camera montage with the 2D detections drawn on
└── pose3d.mp4      # same montage with the triangulated 3D skeleton reprojected
```

## 4. Inspect the result

```bash
deeperfly inspect examples/data/deeperfly_outputs/poses.h5
```

```
file:     examples/data/deeperfly_outputs/poses.h5
views:    7  ['rh', 'rm', 'rf', 'f', 'lf', 'lm', 'lh']
frames:   100
skeleton: fly38  (38 points)
has 3D:   True
reproj:   median 2.1 px  max 8.7 px
```

A low median reprojection error means the cameras and 3D points agree well across
views. Open `pose3d.mp4` to see the reconstructed skeleton.

## 5. Re-run after a tweak

Re-running a finished recording is a cheap no-op — every stage's cache is reused.
**Editing the config recomputes only the affected stages.** Generate an editable
config, change something cheap (say the videos), and re-run:

```bash
deeperfly init config.toml                       # a fully commented config
# ...edit [visualization] or [triangulation]...
deeperfly run examples/data/ -c config.toml      # only the changed stages recompute
```

The slow 2D detection is reused; only triangulation/visualization rebuild. That
resume/recompute model — and `--overwrite` to force a redo — is the subject of
the [CLI guide](guides/cli.md#resuming-and-recomputing).

## Where to go next

- **[CLI usage](guides/cli.md)** — batch runs, output layout, every flag.
- **[Writing configs](guides/configuration.md)** — point deeperfly at your own
  cameras and tune the pipeline.
- **[How it works](explanation/pipeline.md)** — what each stage actually does.
- **[Library API](guides/library.md)** — drive the pipeline from Python.
