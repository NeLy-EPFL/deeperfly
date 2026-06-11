# deeperfly

Markerless 3D pose estimation of tethered *Drosophila* from a multi-camera rig.
It estimates camera parameters and 2D/3D keypoint locations from behavioral
recordings through one linear pipeline: 2D pose → bundle adjustment →
triangulation → visualization.

deeperfly is both a command-line tool and a Python library, and a modern rewrite
of [DeepFly3D](https://github.com/NeLy-EPFL/DeepFly3D),
[DeepFly2D](https://github.com/NeLy-EPFL/DeepFly2D) and
[PyBundleAdjustment](https://github.com/semihgunel/PyBundleAdjustment).

📖 **[Documentation](https://nely-epfl.github.io/deeperfly/)**

## Installation

Install the CLI with [uv](https://docs.astral.sh/uv/) (`--torch-backend=auto`
picks the right PyTorch wheel for your machine):

```bash
uv tool install git+https://github.com/NeLy-EPFL/deeperfly --python 3.13 --torch-backend=auto
```

As a library in your own project (prefix with `UV_TORCH_BACKEND=auto`, since
`uv add` has no `--torch-backend` flag):

```bash
UV_TORCH_BACKEND=auto uv add git+https://github.com/NeLy-EPFL/deeperfly
```

## Usage

```bash
deeperfly doctor                                         # what this machine can run
deeperfly run recording/                                 # detect 2D -> 3D -> video
deeperfly inspect recording/deeperfly_outputs/poses.h5   # summarize the result
```

`deeperfly run` does everything in one command: detect 2D pose in every view,
bundle-adjust the cameras, triangulate to 3D, then render skeleton videos.
Outputs land in `recording/deeperfly_outputs/` (override with `-o`): `poses.h5`,
the rendered videos, and a snapshot of the config used. `deeperfly doctor`
reports whether the detector will run on the GPU and where the weights are
cached.

## Configuration

A run is driven by a single, self-contained `config.toml` — the camera rig, which
file belongs to which camera, the detector, the pipeline and the visualization.
It's optional (`deeperfly run recording/` uses sensible defaults); generate a
commented one to edit with `deeperfly init config.toml` and pass it with `-c`.
Each stage is toggled by a `do_<stage>` switch and reuses its cached result when
unchanged, so editing the config recomputes only the affected stages.

```python
from deeperfly import CameraGroup, Config, Skeleton, run_from_points2d

cameras = CameraGroup.from_config(Config.from_toml("config.toml"))
result = run_from_points2d(cameras, Skeleton.fly(), pts2d, conf)
result.save("fly.h5")
```

## Documentation

Full docs are at **[nely-epfl.github.io/deeperfly](https://nely-epfl.github.io/deeperfly/)**:

- [Getting started](https://nely-epfl.github.io/deeperfly/getting-started/) — run the bundled example end to end.
- [CLI usage](https://nely-epfl.github.io/deeperfly/guides/cli/) and [Writing configs](https://nely-epfl.github.io/deeperfly/guides/configuration/).
- [How it works](https://nely-epfl.github.io/deeperfly/explanation/pipeline/) — the pipeline, stage by stage.
- [Library API](https://nely-epfl.github.io/deeperfly/guides/library/) and the complete [reference](https://nely-epfl.github.io/deeperfly/reference/api/).
- [CONTRIBUTING.md](CONTRIBUTING.md) — development install, tests, linting.

## License

GPL-3.0-only. See [LICENSE](LICENSE).
