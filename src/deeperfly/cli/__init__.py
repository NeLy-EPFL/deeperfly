"""Command-line interface: ``deeperfly <subcommand>``.

This subpackage is **only** the command surface -- argument parsing (Typer) and the
presentation workers (``init`` / ``run`` / ``inspect`` / ``doctor``). The reusable
pipeline logic lives in the library and is callable without importing the CLI:

- footage discovery -- :mod:`deeperfly.recordings`
  (:func:`~deeperfly.recordings.resolve_recordings`).
- detector loading and streaming 2D detection -- :mod:`deeperfly.pose2d.stream`
  (:func:`~deeperfly.pose2d.stream.load_detector`,
  :func:`~deeperfly.pose2d.stream.detect_2d`).
- the per-stage wrappers and the cached/resume run -- :mod:`deeperfly.pipeline`
  (:func:`~deeperfly.pipeline.run_recording`).

The commands:

- ``init`` -- write a default config.toml.
- ``run`` -- the pipeline's enabled stages (``pose2d`` -> ``bundle_adjustment`` ->
  ``pictorial_structures`` -> ``triangulation`` -> ``visualization``) over the
  recordings the inputs resolve to.
- ``inspect`` -- print a summary of a result file.
- ``doctor`` -- print installation/runtime details.

``main`` is the ``deeperfly`` entry point (see ``pyproject.toml``).
"""

from __future__ import annotations

from .app import app, main

__all__ = ["main", "app"]
