# Contributing to deeperfly

Development setup. For how the pipeline works, see
[docs/explanation/pipeline.md](docs/explanation/pipeline.md).

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
uv run --group test pytest          # the whole suite, ~20 s on 32 cores
uv run --group test pytest -n0 tests/test_skeleton.py   # ONE file: use -n0
```

The suite covers the PyTorch detector and OpenCV cross-checks for the geometry.
It runs fully offline — the detector-weight download is mocked.

`-n auto` (the default in `addopts`) is right for the whole suite and wrong for
one file: every worker imports torch, jax, scipy and fastapi to collect a suite
it will not run, so a 6-test file costs 2.6 s and 35 CPU-seconds against 0.67 s
with `-n0`. `-n0` is also what you want for `--pdb` and `-s`.

`tests/test_gui_browser.py` drives the editor in a real headless browser and is
the only gate over the GUI's JavaScript. `playwright` is in the `test` group but
the browser is a separate download, so the file skips itself until you run
`uv run playwright install chromium` once.

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

## Documentation

The docs site is built with [MkDocs](https://www.mkdocs.org/) + Material and
lives in [`docs/`](docs) (configured by [`mkdocs.yml`](mkdocs.yml)). The library
API reference is generated from the source docstrings by
[mkdocstrings](https://mkdocstrings.github.io/). Preview it locally with live
reload:

```bash
uv run --group docs mkdocs serve       # http://127.0.0.1:8000
uv run --group docs mkdocs build --strict   # what CI runs
```

Pushing to `main` rebuilds and publishes the site to GitHub Pages
(see [`.github/workflows/docs.yml`](.github/workflows/docs.yml)).

## License

By contributing you agree that your contributions are licensed under the
project's GPL-3.0-only license.
