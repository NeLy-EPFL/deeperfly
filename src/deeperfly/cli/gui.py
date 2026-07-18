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


def _cmd_labels_export(args: argparse.Namespace) -> None:
    """Export the saved ground-truth labels (``labels.h5``) as an ``.npz`` dataset.

    Resolves ``results.h5`` (a file or a directory holding one), loads the
    ``labels.h5`` beside it (refusing a sidecar from a different recording), and writes
    the provenance-filtered GT + occluded masks in footage pixel space. Raises
    ``SystemExit`` when there are no saved labels to export.
    """
    import numpy as np

    from ..gui.labels import export_gt, labels_identity, load_labels
    from ..results import PoseResult, StageStore

    results_path = _find_results(Path(args.path))
    result = PoseResult.load(results_path)
    store = StageStore(results_path)
    identity = labels_identity(
        point_names=list(result.skeleton.point_names),
        camera_names=list(result.cameras.names),
        n_frames=result.n_frames,
        image_sizes=store.read_image_sizes(),
        footage=store.read_footage(),
    )
    labels_path = results_path.parent / "labels.h5"
    labels = load_labels(labels_path, identity=identity)
    if labels is None:
        raise SystemExit(
            f"no labels.h5 next to {results_path} -- annotate with 'deeperfly gui' "
            "and Save first"
        )
    gt_xy, gt_mask, occluded = export_gt(
        labels, include_projection=args.include_projection
    )
    out = Path(args.output) if args.output else results_path.parent / "labels_gt.npz"
    np.savez(
        out,
        gt_xy=gt_xy,
        gt_mask=gt_mask,
        occluded=occluded,
        point_names=np.array(list(result.skeleton.point_names)),
        camera_names=np.array(list(result.cameras.names)),
    )
    log.info(
        "exported %d ground-truth point(s), %d occlusion(s) (footage space) -> %s",
        int(gt_mask.sum()),
        int(occluded.sum()),
        out,
    )
