"""The ``gui`` command worker: launch the interactive viewer/corrector.

Kept thin and free of any Qt import at module load -- the heavy PySide6 import
happens inside :func:`deeperfly.gui.launch`, so ``deeperfly`` (and this module)
import fine without the optional ``gui`` extra installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _find_results(path: Path) -> Path:
    """Resolve ``path`` to a ``results.h5`` file.

    Accepts the file directly, or a directory containing ``results.h5`` (or a
    ``deeperfly_outputs/results.h5`` beneath it, i.e. a recording directory).

    Parameters
    ----------
    path
        A ``results.h5`` file or a directory to search.

    Returns
    -------
    Path
        The resolved ``results.h5`` path.

    Raises
    ------
    SystemExit
        If no ``results.h5`` can be found at ``path``.
    """
    if path.is_file():
        return path
    if path.is_dir():
        for candidate in (
            path / "results.h5",
            path / "deeperfly_outputs" / "results.h5",
        ):
            if candidate.exists():
                return candidate
    raise SystemExit(
        f"no results.h5 found at {path} -- pass a results.h5 file or a directory "
        "containing one (e.g. <recording>/deeperfly_outputs)"
    )


def _cmd_gui(args: argparse.Namespace) -> None:
    """Launch the GUI on the result resolved from ``args.path``.

    Parameters
    ----------
    args
        The ``gui`` namespace (``path``, ``footage_dir``).

    Raises
    ------
    SystemExit
        If no result is found, or the optional ``gui`` extra is not installed.
    """
    results_path = _find_results(Path(args.path))
    try:
        from ..gui import launch
    except ImportError as exc:  # pragma: no cover -- exercised manually
        raise SystemExit(str(exc)) from exc
    try:
        launch(results_path, footage_dir=args.footage_dir)
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc
