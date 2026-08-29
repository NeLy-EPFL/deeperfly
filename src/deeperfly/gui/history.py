"""Undo and redo: per-frame snapshots of the labels, one entry per operator gesture.

A mixin on :class:`~deeperfly.gui.state.EditorState` plus the two entry types it stacks.
Split out because undo is self-contained -- the rest of the editor touches it only
through :meth:`_HistoryMixin._record_undo`, and the correctness argument for it is local:
a snapshot must capture everything a gesture could have changed, including the derived 3D
row, because a drag on a point with fewer than two usable views stores a ray-slide that
the labels alone cannot reproduce.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("deeperfly")


#: How many undo steps to keep (a backstop; one entry per operator gesture).
UNDO_LIMIT = 200


@dataclass
class _UndoEntry:
    """A snapshot of one frame's labels, so an edit can be reverted.

    ``point`` names the target point and ``coalesce`` marks a drag-style edit: the edits
    of one drag *gesture* collapse to a single undo step (so one drag is one undo -- see
    :attr:`EditorState._drag_open`, which decides where a gesture ends), while a discrete
    op (toggle / reset / confirm) always starts a new step.

    ``pts3d`` snapshots the frame's derived 3D alongside the labels, and it is not
    redundant: a drag below two usable views stores a *ray-slide of the prior 3D*
    (:func:`~deeperfly.gui.solve.solve_point_3d_drag`), which no re-derivation from
    the labels can reproduce -- one 2D pixel leaves depth unconstrained. Restoring
    the labels and re-deriving would silently snap every such point in the frame back
    to the run's cached ``pts3d``, hundreds of pixels away, even though the undo never
    touched its labels. ``None`` when the frame was not cached at snapshot time.
    """

    t: int
    point: int | None
    coalesce: bool
    gt: np.ndarray
    occluded: np.ndarray
    #: The frame's instance seeds ``(V, P, 2)``. Authored state like the pixels -- creating
    #: an instance IS a seed write, so without this an undo of the creation would leave the
    #: instance standing.
    seeds: np.ndarray | None = None
    #: Whether the frame had an annotation skeleton. Restored with the seeds, so undoing the
    #: gesture that created one really removes it.
    instance: bool = False
    pts3d: np.ndarray | None = None  # (P, 3), or None if the frame was uncached


@dataclass
class _AbsentEntry:
    """A snapshot of an absence declaration, so one gesture is one undo step.

    Carries **no** cell payload: the declaration only vetoes the derived masks, it never
    destroys a pixel, so restoring the per-``(frame, point)`` bits restores every label
    underneath. ``prev`` is the ``(T, len(points))`` slice the gesture overwrote -- a few
    kilobytes even for a whole-recording declaration, against the megabytes a snapshot of
    the ``(V, T, P)`` overlay would cost.

    ``pts3d`` is the derived 3D for exactly ``points``, per cached frame -- the same
    non-reproducible ray-slide payload :class:`_UndoEntry` carries, restricted to the
    rows an absence declaration can move (``frame -> (len(points), 3)``). A whole-recording
    declaration reaches every frame, so every cached one is snapshotted; at 24 bytes per
    (frame, point) that stays kilobytes even for a long recording.
    """

    points: list[int]
    prev: np.ndarray  # (T, len(points)) bool
    pts3d: dict[int, np.ndarray] = field(default_factory=dict)
    #: Never coalesces with a neighboring entry: an absence declaration is always its own
    #: gesture, and it carries no frame at all (it can span the recording), so none of the
    #: frame-scoped bookkeeping applies to it.
    coalesce: bool = False


class _HistoryMixin:
    """The undo/redo half of :class:`~deeperfly.gui.state.EditorState`."""

    def _snapshot(self, t: int, point: int | None, coalesce: bool) -> _UndoEntry:
        cached = self._pts3d_cache.get(t)
        return _UndoEntry(
            t=t,
            point=point,
            coalesce=coalesce,
            gt=self.labels.gt[:, t].copy(),
            occluded=self.labels.occluded[:, t].copy(),
            seeds=self.labels.seeds[:, t].copy(),
            instance=bool(self.labels.instance[t]),
            pts3d=None if cached is None else cached.copy(),
        )

    def _snapshot_absent_pts3d(self, points: list[int]) -> dict[int, np.ndarray]:
        """The derived 3D rows for ``points`` in every cached frame (see :class:`_AbsentEntry`)."""
        idx = np.asarray(points, dtype=int)
        return {t: arr[idx].copy() for t, arr in self._pts3d_cache.items()}

    def _snapshot_of(self, entry):
        """The current-state counterpart of ``entry``, for the opposite history stack."""
        if isinstance(entry, _AbsentEntry):
            return _AbsentEntry(
                points=list(entry.points),
                prev=self.labels.absent[:, entry.points].copy(),
                pts3d=self._snapshot_absent_pts3d(entry.points),
            )
        return self._snapshot(entry.t, entry.point, entry.coalesce)

    def _record_undo(self, t: int, point: int | None, *, coalesce: bool) -> None:
        """Push a pre-edit snapshot, coalescing the edits of one *drag gesture*.

        Coalescing is keyed on :attr:`_drag_open` -- the gesture the operator currently
        has open -- and deliberately not on "the top of the undo stack has this same
        ``(t, point)``", which is a different question with two wrong answers. It merges
        two *separate* gestures on one point (place a joint in ``f``, come back later,
        nudge it again: one ctrl-z would clear both, in every view either touched), and it
        lets a gesture *resume* across an undo: drag A, drag B, ctrl-z (which pops B's
        entry, leaving A's on top), drag A again -> the same-point test matches A's old
        entry and the new drag silently joins a gesture the operator finished long ago.
        """
        if coalesce and point is not None and self._drag_open == (t, point):
            # Still inside one gesture: keep the existing (older) pre-state. The redo
            # branch still dies -- the operator is authoring, and a surviving redo entry
            # is a whole-frame snapshot that would clobber this very drag.
            self._redo.clear()
            return
        self._undo.append(self._snapshot(t, point, coalesce))
        self._redo.clear()
        self._drag_open = (t, point) if (coalesce and point is not None) else None
        if len(self._undo) > UNDO_LIMIT:
            self._undo.pop(0)

    def _end_gesture(self) -> None:
        """Close the open drag gesture, so the next edit starts a new undo step."""
        self._drag_open = None

    def _apply_snapshot(self, entry) -> None:
        """Restore ``entry``'s labels *and* its derived 3D.

        Restoring the 3D rather than dropping the cache is the whole point: the cache is
        not a pure function of the labels (see :class:`_UndoEntry`), so re-deriving it
        would move points whose labels this entry never touched.
        """
        if isinstance(entry, _AbsentEntry):
            # The entry carries no pixel payload: the veto never destroyed any, so
            # restoring the bits restores everything underneath.
            self.labels.absent[:, entry.points] = entry.prev
            self.labels.dirty = True
            idx = np.asarray(entry.points, dtype=int)
            for t, arr in self._pts3d_cache.items():
                prev = entry.pts3d.get(t)
                if prev is None:
                    # Cached only *after* the declaration, so its rows were derived under
                    # the veto and have no pre-state to restore: re-derive them.
                    for p in entry.points:
                        arr[p] = self._solve_point(t, p)
                else:
                    arr[idx] = prev
            self._model_cache.clear()  # a pure function of the 3D, so refitting is safe
            return
        t = entry.t
        self.labels.gt[:, t] = entry.gt
        self.labels.occluded[:, t] = entry.occluded
        if entry.seeds is not None:
            self.labels.seeds[:, t] = entry.seeds
            self.labels.instance[t] = entry.instance
        self.labels.dirty = True
        if entry.pts3d is None:
            self._invalidate_frame3d(t)  # uncached then, so there is nothing to restore
        else:
            self._pts3d_cache[t] = entry.pts3d.copy()
        self._invalidate_model(t)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> int | None:
        """Revert the last edit; returns the affected frame (so the client navigates)."""
        if not self._undo:
            return None
        self._end_gesture()  # a history op ends any gesture: the next drag is its own step
        entry = self._undo.pop()
        self._redo.append(self._snapshot_of(entry))
        self._apply_snapshot(entry)
        return None if isinstance(entry, _AbsentEntry) else entry.t

    def redo(self) -> int | None:
        """Re-apply the last undone edit; returns the affected frame."""
        if not self._redo:
            return None
        self._end_gesture()
        entry = self._redo.pop()
        self._undo.append(self._snapshot_of(entry))
        self._apply_snapshot(entry)
        return None if isinstance(entry, _AbsentEntry) else entry.t
