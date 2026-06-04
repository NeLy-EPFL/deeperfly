# Contributing to deeperfly

Thanks for hacking on deeperfly. This page covers the development setup; for how
the pipeline works, see [docs/architecture.md](docs/architecture.md).

## Requirements

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/) for dependency management

## Development install

Clone the repo and sync a dev environment with the test dependencies:

```bash
git clone https://github.com/tkclam/deeperfly
cd deeperfly
uv sync --group test       # creates .venv with the package (editable) + test deps
```

`uv sync` installs deeperfly from the working tree, so your changes are picked up
without reinstalling. Run the CLI from the env with `uv run deeperfly ...`.

### Optional extras

Layer GPU acceleration and faster video backends on top as needed (see
[docs/video.md](docs/video.md) for the backends):

```bash
uv sync --group test --extra cuda          # NVIDIA CUDA (Linux x86-64)
uv sync --group test --extra mps           # Apple Metal for the JAX detector
uv sync --group test --extra torchcodec    # one optional video backend (repeat --extra for more)
```

PyTorch, OpenCV and PyAV are core dependencies, so no extra is needed for the
PyTorch detector backend or the default OpenCV/PyAV video stack. The other
cross-platform video backends are each their own extra (`video-reader-rs`,
`torchcodec`); add the ones you want to exercise.

## Running the tests

```bash
uv run --group test pytest
```

The suite includes PyTorch-equivalence tests for the JAX detector and OpenCV
cross-checks for the geometry. Some tests download and convert the detector
weights on first run.

## Linting and formatting

Formatting and linting are handled by [ruff](https://docs.astral.sh/ruff/) via
pre-commit. Install the hooks once so they run on every commit:

```bash
uvx pre-commit install
```

The hooks lint and format with ruff, keep `uv.lock` in sync, and strip notebook
outputs with `nbstripout`. To run them across the whole tree on demand:

```bash
uvx pre-commit run --all-files
```

## Benchmarks

Ad-hoc benchmark and experiment scripts live in [`dev/`](dev), e.g.:

```bash
uv run python dev/bench_pose2d.py --batch 7 --frames 8   # 2D detector throughput
uv run python dev/bench_video.py                         # video decode backends
uv run python dev/bench_ba.py                            # bundle adjustment
```

## License

By contributing you agree that your contributions are licensed under the
project's GPL-3.0-only license.
