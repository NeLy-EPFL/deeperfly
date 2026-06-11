"""The main editor window: a grid of camera views, a timeline, and edit modes.

:class:`MainWindow` wires the Qt-free :class:`~deeperfly.gui.state.EditorState`
to the widgets: it lays out one :class:`~deeperfly.gui.view.PoseView` per camera,
scrubs frames, switches between View / Edit 2D / Edit 3D, and routes a drag to
the right edit. In Edit 2D a drag moves only that view's 2D point; in Edit 3D a
drag re-solves the 3D point and refreshes every view's reprojection. Save writes
the corrections sidecar; ``results.h5`` is never modified.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..visualization._palette import point_colors_rgb
from .corrections import save_corrections
from .readers import FrameSource
from .state import EditMode, EditorState
from .view import PoseView

__all__ = ["MainWindow"]

log = logging.getLogger("deeperfly")

_MODE_LABELS = [("View", EditMode.view), ("Edit 2D", EditMode.edit_2d)]
_MODE_3D = ("Edit 3D", EditMode.edit_3d)


class MainWindow(QMainWindow):
    """The interactive viewer/corrector window."""

    def __init__(
        self,
        state: EditorState,
        source: FrameSource,
        *,
        results_path: str | Path,
        corrections_path: str | Path,
        parent=None,
    ):
        super().__init__(parent)
        self._state = state
        self._source = source
        self._results_path = str(results_path)
        self._corrections_path = Path(corrections_path)

        n_source = source.n_frames()
        self._n_frames = (
            state.n_frames if n_source is None else min(state.n_frames, n_source)
        )

        self._build_ui()
        self._update_title()
        self._refresh_frame()

    # -- construction ---------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        grid = QGridLayout()
        outer.addLayout(grid, stretch=1)

        skeleton = self._state.result.skeleton
        colors = point_colors_rgb(skeleton)
        names = self._state.camera_names
        cols = max(1, math.ceil(math.sqrt(len(names))))
        self._views: list[PoseView] = []
        for v, name in enumerate(names):
            view = PoseView(v)
            view.set_skeleton(skeleton.bones, colors)
            view.pointDragged.connect(self._on_point_dragged)
            view.pointDragging.connect(self._on_point_dragging)
            grid.addWidget(self._labelled(name, view), v // cols, v % cols)
            self._views.append(view)

        outer.addLayout(self._build_controls())

    def _labelled(self, name: str, view: PoseView) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(name)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        layout.addWidget(view, stretch=1)
        return box

    def _build_controls(self) -> QHBoxLayout:
        controls = QHBoxLayout()

        controls.addWidget(QLabel("Frame"))
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, max(0, self._n_frames - 1))
        self._slider.valueChanged.connect(self._on_frame_changed)
        controls.addWidget(self._slider, stretch=1)

        self._spin = QSpinBox()
        self._spin.setRange(0, max(0, self._n_frames - 1))
        self._spin.valueChanged.connect(self._on_frame_changed)
        controls.addWidget(self._spin)
        self._frame_total = QLabel(f"/ {max(0, self._n_frames - 1)}")
        controls.addWidget(self._frame_total)

        controls.addSpacing(16)
        controls.addWidget(QLabel("Mode"))
        self._mode_combo = QComboBox()
        modes = list(_MODE_LABELS)
        if self._state.has_3d:
            modes.append(_MODE_3D)
        for label, mode in modes:
            self._mode_combo.addItem(label, mode)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        controls.addWidget(self._mode_combo)

        controls.addSpacing(16)
        controls.addWidget(QLabel("Joint"))
        self._joint_combo = QComboBox()
        self._joint_combo.addItem("(none)", -1)
        for i, pname in enumerate(self._state.result.skeleton.point_names):
            self._joint_combo.addItem(pname, i)
        self._joint_combo.currentIndexChanged.connect(self._on_joint_changed)
        controls.addWidget(self._joint_combo)

        self._reset_btn = QPushButton("Reset joint")
        self._reset_btn.clicked.connect(self._on_reset)
        controls.addWidget(self._reset_btn)

        controls.addStretch(1)
        self._save_btn = QPushButton("Save")
        self._save_btn.clicked.connect(self._on_save)
        controls.addWidget(self._save_btn)
        return controls

    # -- current mode / frame -------------------------------------------------

    @property
    def _mode(self) -> EditMode:
        return self._mode_combo.currentData()

    def _refresh_frame(self) -> None:
        """Reload images + points for the current frame and mode."""
        t = self._state.frame
        editable = self._mode in (EditMode.edit_2d, EditMode.edit_3d)
        pts = self._points_for_mode()
        for v, view in enumerate(self._views):
            img = self._source.frame(self._state.camera_names[v], t)
            if img is not None:
                view.set_image(img)
            view.set_points(pts[v])
            view.set_editable(editable)

    def _refresh_points(self) -> None:
        """Repaint just the overlays (after an edit) without reloading images."""
        pts = self._points_for_mode()
        for v, view in enumerate(self._views):
            view.set_points(pts[v])

    def _points_for_mode(self):
        if self._mode == EditMode.edit_3d:
            projected = self._state.display_pts3d_projected()
            if projected is not None:
                return projected
        return self._state.display_pts2d()

    # -- signal handlers ------------------------------------------------------

    def _on_frame_changed(self, value: int) -> None:
        if value == self._state.frame:
            return
        self._state.frame = value
        # keep slider and spinbox in lockstep without re-entrancy
        for widget in (self._slider, self._spin):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self._refresh_frame()

    def _on_mode_changed(self, _index: int) -> None:
        self._state.mode = self._mode
        self._refresh_frame()

    def _on_joint_changed(self, _index: int) -> None:
        point = self._joint_combo.currentData()
        highlight = None if point is None or point < 0 else int(point)
        for view in self._views:
            view.set_highlight(highlight)

    def _on_point_dragging(self, view: int, point: int, x: float, y: float) -> None:
        """Live 3D re-solve mid-drag so every view's reprojection follows the cursor.

        Only meaningful in Edit 3D (a 2D edit is local to its own view, already
        updated by the drag itself). Re-solving the 3D point and reprojecting on
        each mouse-move keeps all the other views in sync as the point moves.
        """
        if self._mode != EditMode.edit_3d:
            return
        self._state.apply_3d_edit(view, point, (x, y))
        self._refresh_points()

    def _on_point_dragged(self, view: int, point: int, x: float, y: float) -> None:
        if self._mode == EditMode.edit_2d:
            self._state.apply_2d_edit(view, point, (x, y))
        elif self._mode == EditMode.edit_3d:
            self._state.apply_3d_edit(view, point, (x, y))
        # Repaint from state: a no-op 3D edit (NaN point) snaps the marker back.
        self._refresh_points()
        self._update_title()

    def _on_reset(self) -> None:
        point = self._joint_combo.currentData()
        if point is None or point < 0:
            return
        self._state.reset_point(int(point))
        self._refresh_points()
        self._update_title()

    def _on_save(self) -> None:
        save_corrections(
            self._corrections_path, self._state.corrections, source=self._results_path
        )
        self.statusBar().showMessage(f"saved {self._corrections_path}", 5000)
        self._update_title()

    # -- misc -----------------------------------------------------------------

    def _update_title(self) -> None:
        mark = " *" if self._state.dirty else ""
        self.setWindowTitle(f"deeperfly gui — {self._results_path}{mark}")
        self._save_btn.setEnabled(self._state.dirty)

    def closeEvent(self, event) -> None:  # noqa: N802 -- Qt override
        if self._state.dirty:
            choice = QMessageBox.question(
                self,
                "Unsaved corrections",
                "Save corrections before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if choice == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if choice == QMessageBox.StandardButton.Save:
                self._on_save()
        self._source.close()
        super().closeEvent(event)
