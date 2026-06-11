"""A single camera view: a frame with a draggable 2D skeleton overlay.

:class:`PoseView` is a :class:`~PySide6.QtWidgets.QGraphicsView` showing one
camera's frame as a background pixmap with the skeleton drawn as persistent
ellipse (joint) and line (bone) items in *pixel* scene coordinates. Editing is
handled at the view level (press picks the nearest joint, move drags it, release
emits :attr:`PoseView.pointDragged`) rather than with movable items, so the
window stays in full control of where a dragged point is allowed to go and what
happens to the 3D point behind it.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

__all__ = ["PoseView"]

#: Drawn joint radius, in scene (pixel) units.
_POINT_RADIUS = 4.0
#: How close (in *screen* pixels) a click must be to grab a joint.
_HIT_TOLERANCE_PX = 14.0


def numpy_to_pixmap(image: np.ndarray) -> QPixmap:
    """Convert an ``(H, W, 3)`` uint8 RGB (or ``(H, W)`` gray) array to a QPixmap."""
    img = np.ascontiguousarray(image)
    height, width = img.shape[:2]
    if img.ndim == 2:
        qimg = QImage(img.data, width, height, width, QImage.Format.Format_Grayscale8)
    else:
        qimg = QImage(img.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    # Copy detaches the pixmap from the (transient) NumPy buffer.
    return QPixmap.fromImage(qimg.copy())


class PoseView(QGraphicsView):
    """One camera's frame plus its draggable skeleton overlay."""

    #: Emitted on drop: ``(view_index, point_index, x, y)`` in pixel coordinates.
    pointDragged = Signal(int, int, float, float)
    #: Emitted continuously while dragging (same payload), for live cross-view updates.
    pointDragging = Signal(int, int, float, float)

    def __init__(self, view_index: int, parent=None):
        super().__init__(parent)
        self._view_index = view_index
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._point_items: list[QGraphicsEllipseItem] = []
        self._bone_items: list[QGraphicsLineItem] = []
        self._colors: list[QColor] = []
        self._bones = np.empty((0, 2), dtype=int)
        self._pts: np.ndarray | None = None
        self._editable = False
        self._highlight: int | None = None
        self._dragging: int | None = None
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMinimumSize(160, 120)

    @property
    def view_index(self) -> int:
        return self._view_index

    # -- setup ----------------------------------------------------------------

    def set_skeleton(self, bones: np.ndarray, colors: np.ndarray) -> None:
        """Create the persistent joint/bone items (once per skeleton).

        Parameters
        ----------
        bones
            ``(B, 2)`` point-index pairs.
        colors
            ``(P, 3)`` RGB floats in ``[0, 1]``, one per point.
        """
        for item in self._point_items + self._bone_items:
            self._scene.removeItem(item)
        self._point_items.clear()
        self._bone_items.clear()
        self._bones = np.asarray(bones, dtype=int).reshape(-1, 2)
        self._colors = [
            QColor(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors
        ]
        for a, _b in self._bones:
            line = QGraphicsLineItem()
            line.setZValue(1.0)
            line.setPen(QPen(self._colors[a], 1.5))
            line.setVisible(False)
            self._scene.addItem(line)
            self._bone_items.append(line)
        for i in range(len(self._colors)):
            dot = QGraphicsEllipseItem()
            dot.setZValue(2.0)
            dot.setBrush(QBrush(self._colors[i]))
            dot.setPen(QPen(Qt.GlobalColor.black, 0.5))
            dot.setVisible(False)
            self._scene.addItem(dot)
            self._point_items.append(dot)

    def set_image(self, image: np.ndarray) -> None:
        """Set the background frame and fit it to the widget."""
        pix = numpy_to_pixmap(image)
        self._pixmap_item.setPixmap(pix)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self._fit()

    def set_points(self, pts2d: np.ndarray) -> None:
        """Move the joint/bone items to ``pts2d`` (``(P, 2)``); hide NaN points."""
        # A writable copy: projected 3D points arrive as a (read-only) JAX buffer,
        # and a drag writes the dragged joint straight into this array.
        self._pts = np.array(pts2d, dtype=float)
        for i, item in enumerate(self._point_items):
            self._place_point(i)
        for j, (a, b) in enumerate(self._bones):
            pa, pb = self._pts[a], self._pts[b]
            line = self._bone_items[j]
            if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                line.setLine(pa[0], pa[1], pb[0], pb[1])
                line.setVisible(True)
            else:
                line.setVisible(False)

    def set_editable(self, editable: bool) -> None:
        self._editable = editable

    def set_highlight(self, point: int | None) -> None:
        """Visually emphasize ``point`` (the active joint), or clear with ``None``."""
        self._highlight = point
        for i in range(len(self._point_items)):
            self._place_point(i)

    # -- drawing helpers ------------------------------------------------------

    def _place_point(self, i: int) -> None:
        item = self._point_items[i]
        if self._pts is None or not np.all(np.isfinite(self._pts[i])):
            item.setVisible(False)
            return
        x, y = self._pts[i]
        radius = _POINT_RADIUS * (2.0 if i == self._highlight else 1.0)
        item.setRect(x - radius, y - radius, 2 * radius, 2 * radius)
        item.setPen(
            QPen(Qt.GlobalColor.white, 1.5)
            if i == self._highlight
            else QPen(Qt.GlobalColor.black, 0.5)
        )
        item.setVisible(True)

    def _fit(self) -> None:
        if not self._pixmap_item.pixmap().isNull():
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event) -> None:  # noqa: N802 -- Qt override
        super().resizeEvent(event)
        self._fit()

    # -- interaction ----------------------------------------------------------

    def _nearest_point(self, scene_pos) -> int | None:
        if self._pts is None:
            return None
        scale = max(self.transform().m11(), 1e-6)
        tol = _HIT_TOLERANCE_PX / scale
        cursor = np.array([scene_pos.x(), scene_pos.y()])
        best, best_d = None, tol
        for i, p in enumerate(self._pts):
            if not np.all(np.isfinite(p)):
                continue
            d = float(np.hypot(*(p - cursor)))
            if d <= best_d:
                best, best_d = i, d
        return best

    def mousePressEvent(self, event) -> None:  # noqa: N802 -- Qt override
        if self._editable and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            point = self._nearest_point(scene_pos)
            if point is not None:
                self._dragging = point
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 -- Qt override
        if self._dragging is not None and self._pts is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._pts[self._dragging] = [scene_pos.x(), scene_pos.y()]
            self._place_point(self._dragging)
            for j, (a, b) in enumerate(self._bones):
                if self._dragging in (a, b):
                    pa, pb = self._pts[a], self._pts[b]
                    if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                        self._bone_items[j].setLine(pa[0], pa[1], pb[0], pb[1])
            self.pointDragging.emit(
                self._view_index, self._dragging, scene_pos.x(), scene_pos.y()
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 -- Qt override
        if self._dragging is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            point = self._dragging
            self._dragging = None
            self.pointDragged.emit(
                self._view_index, point, scene_pos.x(), scene_pos.y()
            )
            event.accept()
            return
        super().mouseReleaseEvent(event)
