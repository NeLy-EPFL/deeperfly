"""The ``gui`` command worker: launch the interactive web viewer/corrector.

Kept thin and free of any web import at module load -- the FastAPI/uvicorn
import happens inside :func:`deeperfly.gui.serve`, so importing ``deeperfly``
(and this module) stays cheap for every command other than ``gui``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger("deeperfly")

#: Bind addresses that keep the editor on the local machine (no warning).
_LOOPBACK = ("127.0.0.1", "localhost", "::1")


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
    """Serve the web GUI on the result resolved from ``args.path``.

    Parameters
    ----------
    args
        The ``gui`` namespace (``path``, ``footage_dir``, ``host``, ``port``,
        ``no_browser``, ``keep_alive``).

    Raises
    ------
    SystemExit
        If no result is found, or the web stack fails to import (an incomplete
        install -- FastAPI + uvicorn are core dependencies).
    """
    results_path = _find_results(Path(args.path))
    if args.host not in _LOOPBACK:
        log.warning(
            "binding %s exposes the editor on the network without authentication; "
            "prefer the default localhost and an `ssh -L` tunnel for remote use",
            args.host,
        )
    try:
        from ..gui import serve
    except ImportError as exc:  # pragma: no cover -- exercised manually
        raise SystemExit(str(exc)) from exc
    try:
        serve(
            results_path,
            footage_dir=args.footage_dir,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
            exit_on_close=not args.keep_alive,
        )
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc
