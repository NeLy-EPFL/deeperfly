# Contributing to deeperfly

Development setup. For how the pipeline works, see
[docs/architecture.md](docs/architecture.md).

## Requirements

- Python ≥ 3.11 (development and CI target 3.14; a `.python-version` pins it)
- [uv](https://docs.astral.sh/uv/) for dependency management

## Development install

Clone the repo and sync a dev environment with the test dependencies:

```bash
git clone https://github.com/tkclam/deeperfly
cd deeperfly
uv sync --group test       # .venv with the editable package + test deps
```

`uv sync` installs from the working tree, so changes are picked up without
reinstalling. Run the CLI with `uv run deeperfly ...`.

### Optional extras

PyTorch, OpenCV and PyAV are core, so no extra is needed for the detector or the
default video stack. Add the alternative (CPU) video backends as needed (see
[docs/video.md](docs/video.md)):

```bash
uv sync --group test --extra torchcodec        # one optional video backend
uv sync --group test --extra video-reader-rs   # (repeat --extra for more)
```

## Running the tests

```bash
uv run --group test pytest
```

The suite covers the PyTorch detector and OpenCV cross-checks for the geometry.
Some tests download the detector weights on first run.

## Linting and formatting

[ruff](https://docs.astral.sh/ruff/) handles formatting and linting via
pre-commit. Install the hooks once:

```bash
uvx pre-commit install
```

The hooks run ruff, keep `uv.lock` in sync, and strip notebook outputs with
`nbstripout`. To run them across the whole tree:

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
