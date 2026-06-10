# Contributing to deeperfly

Development setup. For how the pipeline works, see
[docs/architecture.md](docs/architecture.md).

## Requirements

- Python ≥ 3.11, < 3.14 (development targets 3.13; a `.python-version` pins it)
- [uv](https://docs.astral.sh/uv/) for dependency management

## Development install

Clone the repo and sync a dev environment with the test dependencies:

```bash
git clone https://github.com/NeLy-EPFL/deeperfly
cd deeperfly
uv sync --group test       # .venv with the editable package + test deps
```

`uv sync` installs from the working tree, so changes are picked up without
reinstalling. Run the CLI with `uv run deeperfly ...`. PyTorch, OpenCV and PyAV
are all core dependencies — there are no optional extras.

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
uv run python dev/bench_video.py   # video decode vs detector throughput
uv run python dev/bench_ba.py      # bundle adjustment
```

## License

By contributing you agree that your contributions are licensed under the
project's GPL-3.0-only license.
