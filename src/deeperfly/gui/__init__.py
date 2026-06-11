"""Optional interactive viewer/corrector for deeperfly results (``deeperfly gui``).

This package is the optional GUI extra (``pip install deeperfly[gui]``). Only the
Qt-free core is imported here -- :class:`~deeperfly.gui.state.EditorState`, the
corrections sidecar, and footage resolution -- so importing
:mod:`deeperfly.gui` (and the rest of the package and CLI) never requires
PySide6. :func:`launch` imports the Qt widgets lazily and raises a friendly hint
if the extra is not installed.

The GUI shows every camera view with its 2D skeleton overlay and lets keypoints
be dragged. Corrections are written to a ``corrections.h5`` sidecar and never
overwrite ``results.h5``. In *Edit 3D* mode the triangulated points are
reprojected into each view; dragging one re-solves the 3D point (the point on
the dragged pixel's back-projection ray closest to its old location) and every
other view's reprojection updates.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from ..results import PoseResult, StageStore
from .corrections import Corrections, load_corrections, save_corrections
from .readers import FrameSource, resolve_camera_files, resolve_footage
from .state import EditMode, EditorState

__all__ = [
    "EditMode",
    "EditorState",
    "Corrections",
    "load_corrections",
    "save_corrections",
    "FrameSource",
    "resolve_footage",
    "resolve_camera_files",
    "launch",
]

log = logging.getLogger("deeperfly")

_GUI_IMPORT_HINT = (
    "the deeperfly GUI needs the optional 'gui' extra (PySide6); install it with "
    "`pip install deeperfly[gui]` (or `uv sync --extra gui`)"
)


def launch(results_path: str | Path, footage_dir: str | Path | None = None) -> None:
    """Open the viewer/corrector on a ``results.h5`` file.

    Loads the result, resolves each camera's footage (from the paths recorded in
    ``results.h5``, then ``footage_dir`` / a dialog if needed), loads any
    existing ``corrections.h5`` sidecar, and runs the Qt event loop.

    Parameters
    ----------
    results_path
        Path to a ``results.h5`` file.
    footage_dir
        Optional directory to search for the footage when the recorded paths
        no longer resolve.

    Raises
    ------
    ImportError
        If PySide6 (the ``gui`` extra) is not installed.
    """
    results_path = Path(results_path)
    try:
        from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
    except ImportError as exc:
        raise ImportError(_GUI_IMPORT_HINT) from exc
    from .window import MainWindow

    result = PoseResult.load(results_path)
    store = StageStore(results_path)
    footage = store.read_footage()
    image_sizes = store.read_image_sizes()
    n_points = int(result.pts2d.shape[2])
    results_dir = results_path.parent
    corrections_path = results_dir / "corrections.h5"

    app = QApplication.instance() or QApplication(sys.argv)

    resolved, missing = resolve_footage(footage, results_dir, footage_dir)
    if missing:
        chosen = QFileDialog.getExistingDirectory(
            None, f"Select the footage directory for: {', '.join(missing)}"
        )
        if chosen:
            found, missing = resolve_footage(footage, results_dir, chosen)
            resolved.update(found)
    if not footage:
        log.warning(
            "results.h5 records no footage paths; showing overlays on blank frames "
            "-- re-run 'deeperfly run' to embed them"
        )
    if missing:
        log.warning("footage not found for %s (blank frames)", ", ".join(missing))

    source = FrameSource(resolved, image_sizes=image_sizes)

    corrections = None
    try:
        corrections = load_corrections(
            corrections_path, result.n_views, result.n_frames, n_points
        )
    except ValueError as exc:
        QMessageBox.warning(
            None, "Corrections", f"Ignoring existing corrections: {exc}"
        )
    state = EditorState.from_result(result, corrections)

    window = MainWindow(
        state,
        source,
        results_path=str(results_path),
        corrections_path=corrections_path,
    )
    window.resize(1200, 800)
    window.show()
    app.exec()
