"""The editor's data model: a :class:`PoseResult` plus a ground-truth labels overlay.

:class:`EditorState` is deliberately free of any web/Qt dependency -- it is the
testable core the server drives. The editor is a *ground-truth annotation* tool: the
operator's 2D labels are the source of truth and the 3D pose is a pure derived
function of them (see :mod:`deeperfly.gui.solve`), so nothing 3D is stored -- it is
recomputed from the labels + the detector's predictions and cached per frame.

Per ``(view, frame, point)`` the operator authors at most a tri-state
(:class:`~deeperfly.gui.labels.Labels`): a GT pixel, an "occluded" flag, or nothing.
The *displayed* 2D for a view resolves by precedence GT -> prediction, and the 3D
point is ``solve_point_3d(gt, predictions)`` (falling back to the run's cached 3D when
fewer than two views are usable). A drag creates/moves the GT at the dragged view and
re-solves the 3D live via :func:`~deeperfly.gui.solve.solve_point_3d_drag`, which lands
the point under the cursor even with a single usable view.

Orthogonal to that per-cell tri-state, a *point* may be declared **absent** -- not on this
animal, as with an amputated leg. It is view-independent (an amputated joint is missing
from every camera at once) but per *frame*, so a leg lost to autotomy part-way through a
recording is expressible; the editor's default gesture is nonetheless "apply to the whole
recording", which is what almost every real declaration wants. Absence vetoes the derived
masks, forces the point's 2D and 3D to ``NaN`` in the frames it covers, and refuses every
authoring verb there (:meth:`EditorState.absent_refusal`). The ``_solve_point``
short-circuit is what actually removes the limb -- see its docstring for why the run-cache
fallback has to be skipped as well.

The method names ``apply_2d_edit`` / ``apply_3d_edit`` / ``toggle_fixed`` /
``toggle_invisible`` / ``reset_*`` are kept as the wire-compatible surface the server
dispatches to; under the new model ``toggle_fixed`` confirms/clears a GT pixel and
``toggle_invisible`` toggles the occluded flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from jaxtyping import Float

from ..config import AnnotationParams, TriangulationParams
from ..results import PoseResult
from .labels import Labels, LandmarkLabels
from .nmf_live import NmfLive
from .solve import solve_point_3d, solve_point_3d_drag

__all__ = ["EditMode", "EditorState"]

log = logging.getLogger("deeperfly")

#: How many undo steps to keep (a backstop; one entry per operator gesture).
UNDO_LIMIT = 200

#: Views a point needs before its 3D can be triangulated at all -- and so also the
#: threshold that decides whether an "exclude from triangulation" is honored or stands
#: down (see :meth:`EditorState._point_obs`).
_MIN_VIEWS_FOR_3D = 2


def _resolve_view_names(result: PoseResult) -> list[str]:
    """The view names for a result: the rig's, else the recorded ones, else ``viewN``.

    An uncalibrated result has no rig to name its views, so
    :meth:`~deeperfly.results.PoseResult.uncalibrated` records them in ``meta``; a
    hand-built result may carry neither, and positional names are still better than a
    crash, because the names are display-only.
    """
    if result.cameras is not None:
        return list(result.cameras.names)
    recorded = (result.meta or {}).get("view_names")
    if recorded:
        return [str(n) for n in recorded]
    return [f"view{i}" for i in range(result.n_views)]


class EditMode(str, Enum):
    """The interaction mode of the editor."""

    view = "view"  # read-only inspection
    edit_2d = "edit_2d"  # drag per-view 2D keypoints (create GT)
    edit_3d = "edit_3d"  # drag reprojected 3D keypoints (create GT + re-solve)


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


@dataclass
class EditorState:
    """A loaded result plus its ground-truth labels overlay and view state."""

    result: PoseResult
    labels: Labels
    ann: AnnotationParams = field(default_factory=AnnotationParams)
    tri: TriangulationParams = field(default_factory=TriangulationParams)
    frame: int = 0
    mode: EditMode = EditMode.view
    #: Per-frame live NMF re-fit (model joints, angles, names), keyed by frame.
    nmf_live: NmfLive | None = None
    _nmf_cache: dict[int, tuple] = field(default_factory=dict)
    #: Per-frame derived 3D pose (P, 3), recomputed from the labels; the "cache" the
    #: whole design treats the 3D as. Updated per *point* when a point's labels change.
    #:
    #: Almost a pure function of the labels, and the exception matters: a drag on a point
    #: with fewer than two usable views stores a ray-slide of the prior 3D (see
    #: :meth:`_settle_point3d`), which the labels cannot reproduce, because a single
    #: pixel leaves depth unconstrained. So a row here can be the operator's only record
    #: of a hand-placed depth, and anything that drops it loses authored work: invalidate
    #: per point (:meth:`_rederive_points`), snapshot it into the history
    #: (:class:`_UndoEntry`), and clear a whole frame only when every label in it really
    #: did go (:meth:`reset_frame`).
    _pts3d_cache: dict[int, np.ndarray] = field(default_factory=dict)
    #: Undo / redo stacks of per-frame label snapshots (see :class:`_UndoEntry`).
    _undo: list = field(default_factory=list)
    _redo: list = field(default_factory=list)
    #: The ``(frame, point)`` of the drag gesture the operator currently has open, or
    #: ``None`` between gestures. This -- not the top of the undo stack -- is what a
    #: streamed drag coalesces on, so one gesture is one undo step and two gestures never
    #: merge (see :meth:`_record_undo`). Closed by the drag's settle, by any discrete op,
    #: and by undo/redo.
    _drag_open: tuple[int, int] | None = None
    #: The detector's own output ``(V, T, P, 2)`` -- the raw peak, straight from the 2D
    #: network, before triangulation cleaning. This is **the detected layer** (see
    #: :attr:`detections`): a cell is NaN here only when no pathway predicts that keypoint
    #: in that view (an ipsilateral-only model, say), never because a later stage rejected
    #: it. ``None`` when the pristine ``pose2d`` group is unavailable, and then
    #: :attr:`detections` falls back to ``result.pts2d``.
    raw_pts2d: np.ndarray | None = None
    #: Per-view image size as ``(V, 2)`` ``[width, height]``, for a placeholder's
    #: last-resort center. ``None`` when unavailable.
    image_sizes_wh: np.ndarray | None = None
    #: Calibration landmarks for this recording, when the project defines any (see
    #: :mod:`deeperfly.landmarks`). A *separate* overlay from :attr:`labels` and
    #: deliberately so: a landmark is not a skeleton point, must never reach the detector,
    #: the IK plan or the training export, and must not perturb the fingerprinted
    #: ``point_names``. ``None`` when the project declares none.
    landmarks: "LandmarkLabels | None" = None
    #: The view names, in ``V``-axis order. Held here rather than read off the rig
    #: because an **uncalibrated** recording has named views and no geometry at all: the
    #: names are what the operator labels against, and they must not depend on a
    #: calibration existing. Filled from the rig (else the result's recorded view names,
    #: else ``view0 ... viewN``) by ``__post_init__``.
    view_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.view_names:
            self.view_names = _resolve_view_names(self.result)

    @classmethod
    def from_result(
        cls,
        result: PoseResult,
        labels: Labels | None = None,
        *,
        ann: AnnotationParams | None = None,
        tri: TriangulationParams | None = None,
        template=None,
        articulation=None,
        raw_pts2d: np.ndarray | None = None,
        image_sizes: dict[str, tuple[int, int]] | None = None,
        landmarks: "LandmarkLabels | None" = None,
    ) -> EditorState:
        """Build a state for ``result``, with an empty overlay if none is given.

        Unlike the old corrections overlay, a fresh labels overlay seeds *nothing*:
        a view the detector missed is *unobserved* (derived), not a stored occlusion,
        so an untouched session carries no authored state. ``ann`` / ``tri`` are the
        annotation solve policy and shared triangulation params (from the run config
        beside ``results.h5``); both default to the packaged defaults.

        When the result carries a fitted NMF model and 3D pose, a :class:`NmfLive`
        is built so the overlaid model re-fits to the operator's edits. ``template`` /
        ``articulation`` make that re-fit use the *same* model the pipeline did.

        ``raw_pts2d`` is the pristine ``pose2d`` detections (``result.pts2d`` is the
        triangulation-*cleaned* array, so a rejected point is NaN there); it seeds the
        placeholder for an otherwise-unobserved joint. ``image_sizes`` (camera name ->
        ``(height, width)``) supplies the placeholder's last-resort image center.
        """
        if labels is None:
            labels = Labels.empty(
                result.n_views, result.n_frames, cls._n_points(result)
            )
        nmf_live = None
        if result.nmf_pts3d is not None and result.pts3d is not None:
            from ..inverse_kinematics._quickik import INSTALL_HINT, MissingQuickIK

            try:
                nmf_live = NmfLive(result, template=template, articulation=articulation)
            except MissingQuickIK:
                # Expected on a plain install, and harmless: the stored fit still draws
                # the overlay, it just no longer follows edits. One line, not a traceback.
                log.warning(
                    "the NMF overlay will not follow your edits: the live re-fit needs "
                    "the optional QuickIK solver (%s)",
                    INSTALL_HINT,
                )
            except Exception:  # a refit-setup failure just disables the live overlay
                log.exception(
                    "could not set up the live NMF re-fit; using the static fit"
                )
        view_names = _resolve_view_names(result)
        image_sizes_wh = None
        if image_sizes:
            image_sizes_wh = np.array(
                [
                    (image_sizes.get(name, (0, 0))[1], image_sizes.get(name, (0, 0))[0])
                    for name in view_names
                ],
                dtype=float,
            )
        return cls(
            result=result,
            labels=labels,
            ann=ann or AnnotationParams(),
            tri=tri or TriangulationParams(),
            nmf_live=nmf_live,
            raw_pts2d=None if raw_pts2d is None else np.asarray(raw_pts2d, dtype=float),
            image_sizes_wh=image_sizes_wh,
            view_names=view_names,
            landmarks=landmarks,
        )

    @staticmethod
    def _n_points(result: PoseResult) -> int:
        return int(result.pts2d.shape[2])

    # -- dimensions -----------------------------------------------------------

    @property
    def n_views(self) -> int:
        return self.result.n_views

    @property
    def n_frames(self) -> int:
        return self.result.n_frames

    @property
    def n_points(self) -> int:
        return self._n_points(self.result)

    @property
    def has_3d(self) -> bool:
        """Whether the result carries 3D points (enables :attr:`EditMode.edit_3d`)."""
        return self.result.pts3d is not None

    @property
    def has_nmf(self) -> bool:
        """Whether the result carries a fitted NMF model (enables its overlay)."""
        return self.result.nmf_pts3d is not None

    @property
    def has_cameras(self) -> bool:
        """Whether a calibrated rig is available (false = every view is independent).

        With no rig there is no 3D to derive, no reprojection to show, and no way for a
        label in one view to inform another -- so the editor is a set of independent 2D
        canvases. That is the honest state of a project before its rig is solved, and it
        is deliberately *not* papered over with an approximate overlay: an operator
        cannot check a guess they were never shown the basis for.
        """
        return self.result.has_cameras

    @property
    def camera_names(self) -> list[str]:
        return list(self.view_names)

    @property
    def dirty(self) -> bool:
        # Landmarks count: they are authored in the same session and saved by the same
        # button, so a session dirty only in landmarks must still prompt before closing.
        return self.labels.dirty or bool(
            self.landmarks is not None and self.landmarks.dirty
        )

    def _resolve_frame(self, frame: int | None) -> int:
        return self.frame if frame is None else frame

    # -- per-view label masks (for the payload) -------------------------------

    def gt_mask(self, frame: int | None = None) -> np.ndarray:
        """``(V, P)`` boolean: which per-view points carry a GT pixel at ``frame``."""
        return self.labels.has_gt[:, self._resolve_frame(frame)]

    def occluded_mask(self, frame: int | None = None) -> np.ndarray:
        """``(V, P)`` boolean: which per-view points are occluded at ``frame``.

        The *effective* mask: a point declared absent is not "occluded in every view",
        it is not there at all, so it is vetoed out (the front-end draws it as a
        tombstone, not as a rejected observation).
        """
        return self.labels.occluded_effective[:, self._resolve_frame(frame)]

    def absent_mask(self, frame: int | None = None) -> np.ndarray:
        """``(P,)`` boolean: which points are not on this animal at ``frame``."""
        return self.labels.absent_at(self._resolve_frame(frame))

    def absent_points(self) -> list[int]:
        """Sorted indices of the points absent in *every* frame of the recording."""
        return [int(i) for i in np.nonzero(self.labels.absent_all_frames())[0]]

    def absent_points_any(self) -> list[int]:
        """Sorted indices of the points absent in at least one frame."""
        return [int(i) for i in np.nonzero(self.labels.absent_any_frame())[0]]

    def _is_absent(self, point: int) -> bool:
        return bool(self.labels.absent_at(self.frame)[point])

    # -- corrected frames -----------------------------------------------------

    def corrected_frames(self) -> list[dict]:
        """Every frame the operator has touched, with whether it is marked reviewed.

        A frame is listed when any view carries a GT pixel or an occlusion flag for
        some point (an authored human decision), *or* the frame has been marked
        reviewed -- so ticking a frame reviewed keeps it in the list even if its point
        labels are later reset. Returned sorted by frame, each
        ``{"frame": t, "reviewed": bool}`` -- what the GUI's frame list shows so the
        operator can jump back to frames they have worked on and tick off the ones
        they have finished checking.

        A recording-wide absence declaration is deliberately **not** a per-frame label:
        folding it in would mark every frame of the recording "labeled" at a stroke, and
        the suggestion queue (which excludes labeled frames) would empty.
        """
        decided = self.labels.has_gt | self.labels.occluded_effective  # (V, T, P)
        labeled = decided.any(axis=(0, 2))  # (T,) any authored label in the frame
        reviewed = self.labels.reviewed  # (T,)
        return [
            {"frame": int(t), "reviewed": bool(reviewed[t])}
            for t in np.nonzero(labeled | reviewed)[0]
        ]

    @property
    def detections(self) -> Float[np.ndarray, "V T P 2"]:
        """The detected layer: what the 2D network said, per ``(view, frame, point)``.

        Deliberately :attr:`raw_pts2d` and not ``result.pts2d``. The latter is the
        triangulation-*cleaned* array -- NaN wherever the pipeline rejected a peak, and in
        a directory prepared for contralateral labeling it has reprojected geometry written
        *over* the detector's pixels (measured on scape_Fly4_006: 43% of finite cells
        differ from ``pose2d/points``, median 14.9 px). Reading it as "the detection" loses
        the network's opinion exactly where it is most needed and makes a drawn detection
        ambiguous about what it even is.

        Falls back to ``result.pts2d`` only when there is no ``pose2d`` group to read --
        such a file has no separate detector stage, so its points *are* the detections.
        """
        return self.raw_pts2d if self.raw_pts2d is not None else self.result.pts2d

    # -- displayed 2D (labels over predictions) -------------------------------

    def display_pts2d(self, frame: int | None = None) -> Float[np.ndarray, "V P 2"]:
        """The per-view 2D to draw for ``frame``: GT over the detector prediction.

        A GT view shows its pixel; a plain view shows the prediction; an occluded
        view (or one with neither) is ``NaN`` -- the front-end draws the reprojection
        ghost there instead. A point declared absent is ``NaN`` in *every* view: the
        detector still fires somewhere on an amputated limb (an argmax decode always
        emits a peak), and showing that peak would invite the operator to confirm it.
        """
        t = self._resolve_frame(frame)
        gt = self.labels.gt[:, t]  # (V, P, 2)
        has = self.labels.has_gt[:, t]  # (V, P)
        occ = self.labels.occluded_effective[:, t]  # (V, P)
        absent = self.labels.absent_at(t)[None, :]  # (1, P) -> broadcasts over views
        pred = self.detections[:, t]  # (V, P, 2)
        shown = np.where(has[..., None], gt, np.where(~occ[..., None], pred, np.nan))
        return np.where(absent[..., None], np.nan, shown)

    def display_pts2d_refine(
        self, frame: int | None = None
    ) -> Float[np.ndarray, "V P 2"] | None:
        """The per-view 2D drawn in Edit 3D: the derived 3D reprojected into every
        view, with each GT view overridden by its authored pixel, or ``None``.

        GT views hold their pixel (they generally do not all agree with one 3D
        point); every other view -- plain or occluded -- follows the reprojection. An
        absent point has no 3D to reproject, so it stays ``NaN`` here too.
        """
        proj = self.display_pts3d_projected(frame)
        if proj is None:
            return None
        t = self._resolve_frame(frame)
        gt = self.labels.gt[:, t]  # (V, P, 2)
        has = self.labels.has_gt[:, t]  # (V, P)
        absent = self.labels.absent_at(t)[None, :]  # (1, P)
        return np.where(absent[..., None], np.nan, np.where(has[..., None], gt, proj))

    # -- placeholder seeds for unobserved joints ------------------------------

    def placeholder_pts2d(
        self, frame: int | None = None, *, window: int = 30
    ) -> Float[np.ndarray, "V P 2"]:
        """Seed positions for cells with no observation of their own, so a GT can
        always be placed -- the layer that guarantees every joint stays draggable.

        A cell with no GT and no usable detection (the operator marked it Projected, or
        the detector never fired, or triangulation dropped the point) has no position of
        its *own*. It may still be drawn at its reprojection -- but only while the
        reprojection exists AND the operator is showing that overlay. Whenever it is not,
        the canvas draws nothing there: the operator has nothing to grab, hence no way to
        author a GT, and the joint is *unreachable* -- the front-end hit-tests and
        marquee-selects only what is drawn (poseView.js ``grabCandidates``), so the cell
        cannot even be selected to be Reset.

        So this returns a seed for EVERY such cell, whether or not a reprojection exists.
        Deciding when to actually draw one needs the per-layer visibility only the
        front-end knows, so that suppression lives there (poseView.js ``placeholderPos``
        hides a seed under a visible GT / detection / reprojection); this side's job is to
        supply the complete last-resort map it draws from. Withholding a seed here because
        a reprojection happens to exist is what let a joint vanish: hide the reprojected
        overlay (``p``) and the only handle went with it.

        A cell that has a position of its own -- GT, or a usable detection -- is ``NaN``
        (no seed): it is grabbable on its own layer.

        A seed falls back, in order, to:

        1. the reprojection of the derived 3D, when there is one -- the joint's actual
           derived position in this view, so the handle sits where the ghost was,
        2. the raw detector pixel (kept even when triangulation dropped it),
        3. the nearest frame (within ``window``) whose raw/cleaned pixel in this view
           is finite -- a keypoint moves little frame to frame,
        4. the mean of the joint's connected skeleton neighbors shown in this view,
        5. the centroid of the view's shown points,
        6. the image center.

        The last two rungs mean a seed is *always* finite, even for a joint nothing in
        the frame can place. The operator drags the seed to author GT, so the position
        only needs to be a reasonable starting point near where the point belongs.
        """
        t = self._resolve_frame(frame)
        n_views, n_points = self.n_views, self.n_points
        disp = self.display_pts2d(t)  # (V, P, 2): GT over cleaned pred, NaN if absent
        proj = self.display_pts3d_projected(t) if self.has_3d else None
        disp_ok = np.isfinite(disp).all(axis=-1)  # (V, P)
        # Every cell without a position of its own gets a seed. Note the whole-frame
        # bulk-Projected case this has to survive: set every view of a joint Projected and
        # the solve has nothing left to fit (below two usable views
        # :func:`~deeperfly.gui.solve.solve_point_3d` returns NaN) and the run-cache
        # fallback may be NaN too, so there is no reprojection to follow either -- the
        # detection is suppressed, the ghost never appears, and without a seed the joint
        # is gone from every canvas with no way back.
        need = ~disp_ok  # (V, P)
        out = np.full((n_views, n_points, 2), np.nan)
        if not need.any():
            return out

        # A per-cell "shown" position (GT/detected, else the reprojection) -- rung 1 for
        # the cells that need a seed, and what the neighbor/centroid rungs average over.
        shown = np.where(disp_ok[..., None], disp, np.nan)
        if proj is not None:
            fill = np.isfinite(shown).all(axis=-1)
            shown = np.where(fill[..., None], shown, proj)
        shown_ok = np.isfinite(shown).all(axis=-1)  # (V, P)

        bones = np.asarray(self.result.skeleton.bones, dtype=int).reshape(-1, 2)
        lo, hi = max(0, t - window), min(self.n_frames - 1, t + window)
        for v in range(n_views):
            for p in np.nonzero(need[v])[0]:
                out[v, p] = self._seed_position(
                    v, int(p), t, bones, shown, shown_ok, lo, hi
                )
        return out

    def _seed_position(self, v, p, t, bones, shown, shown_ok, lo, hi) -> np.ndarray:
        """One placeholder seed via the fallback chain (see :meth:`placeholder_pts2d`)."""
        raw = self.detections
        # 1. the reprojection of the derived 3D, when this cell has one -- the joint's
        #    own derived position here, so a seed uncovered by hiding the reprojected
        #    overlay lands exactly where its ring was, not on a rejected pixel.
        if shown_ok[v, p]:
            return np.asarray(shown[v, p], dtype=float)
        # 2. the raw detector pixel at this frame.
        if np.all(np.isfinite(raw[v, t, p])):
            return np.asarray(raw[v, t, p], dtype=float)
        # 3. the nearest frame (raw, then cleaned) with a finite pixel in this view.
        for dt in range(1, max(t - lo, hi - t) + 1):
            for tt in (t - dt, t + dt):
                if not lo <= tt <= hi:
                    continue
                if np.all(np.isfinite(raw[v, tt, p])):
                    return np.asarray(raw[v, tt, p], dtype=float)
                cleaned = self.detections[v, tt, p]
                if np.all(np.isfinite(cleaned)):
                    return np.asarray(cleaned, dtype=float)
        # 4. the mean of connected skeleton neighbors shown in this view.
        if bones.size:
            nbrs = np.unique(
                np.concatenate([bones[bones[:, 0] == p, 1], bones[bones[:, 1] == p, 0]])
            )
            npos = [
                shown[v, q] for q in nbrs if 0 <= q < shown.shape[1] and shown_ok[v, q]
            ]
            if npos:
                return np.mean(np.stack(npos), axis=0)
        # 5. the centroid of the view's shown points.
        if shown_ok[v].any():
            return shown[v][shown_ok[v]].mean(axis=0)
        # 6. the image center (else the origin, if even that is unknown).
        if self.image_sizes_wh is not None:
            return self.image_sizes_wh[v] / 2.0
        return np.zeros(2)

    # -- derived 3D (the "cache") ---------------------------------------------

    def _point_obs(self, t: int, point: int):
        """``(gt_obs (V,2), pred_obs (V,2), conf (V,)|None)`` for one point at ``t``.

        GT overrides the detection in its own view, so ``pred_obs`` NaNs out every view
        that carries GT. It also NaNs out views the operator excluded from triangulation --
        but **only while enough evidence survives to solve with**.

        That last clause is the "exclude" verb's actual contract, and it is deliberate.
        Excluding one bad view is a real constraint and is honored. Excluding *everything*
        is how the operator clears the canvas to label against the projected skeleton
        instead, and it must not leave the point unsolvable: the projection they are about
        to drag from is exactly what the excluded detections produce. So an exclusion that
        would drop the point below two usable views is ignored, and the detections keep
        working until GT replaces them -- the first GT takes over its own view, and at two
        GT views the solve no longer needs a detection at all, so every exclusion becomes
        moot on its own.

        A point declared absent contributes nothing from any view: ``gt_obs`` is already
        vetoed via ``has_gt``, and the detector's peaks -- which exist on an amputated limb
        because an argmax decode always emits one -- are dropped rather than triangulated
        into a phantom joint. Absence is a claim about the *animal*, so unlike an exclusion
        (a claim about a view) it is never softened.
        """
        absent = bool(self.labels.absent_at(t)[point])
        has = self.labels.has_gt[:, t, point]  # (V,)
        occ = self.labels.occluded_effective[:, t, point]  # (V,)
        gt_obs = np.where(has[:, None], self.labels.gt[:, t, point], np.nan)
        pred = self.detections[:, t, point].astype(float)  # (V, 2)
        finite = np.isfinite(pred).all(axis=-1)
        pred_ok = finite & ~occ & ~has
        if int(has.sum()) + int(pred_ok.sum()) < _MIN_VIEWS_FOR_3D:
            pred_ok = (
                finite & ~has
            )  # too little left to solve: the exclusions stand down
        if absent:
            pred_ok = np.zeros_like(pred_ok)
        pred_obs = np.where(pred_ok[:, None], pred, np.nan)
        conf = None if self.result.conf is None else self.result.conf[:, t, point]
        return gt_obs, pred_obs, conf

    def _solve_point(self, t: int, point: int) -> np.ndarray:
        """Derive one point's 3D from its labels + predictions (run-cache fallback).

        An absent point short-circuits to ``NaN`` **without** the run-cache fallback.
        That skip is what actually removes the limb: ``solve_point_3d`` already returns
        ``NaN`` once every view is dropped, but the fallback below would immediately
        substitute the run's cached ``pts3d`` -- which is a perfectly finite phantom in
        almost every frame -- and the editor would keep drawing the amputated leg. The
        condition is keyed on *absence*, not on NaN, so the fallback keeps working
        everywhere it is legitimately useful.
        """
        if bool(self.labels.absent_at(t)[point]):
            return np.full(3, np.nan)
        if self.result.cameras is None:
            # Uncalibrated: there is no rig to triangulate through. This is the one 3D
            # path not already gated by `pts3d is None` upstream, because `_ensure_pts3d`
            # can be reached from a placeholder seed computation.
            return np.full(3, np.nan)
        gt_obs, pred_obs, conf = self._point_obs(t, point)
        x = solve_point_3d(
            self.result.cameras, gt_obs, pred_obs, conf, self.ann, self.tri
        )
        if not np.all(np.isfinite(x)) and self.result.pts3d is not None:
            x = np.asarray(self.result.pts3d[t, point], dtype=float)
        return np.asarray(x, dtype=float)

    def _ensure_pts3d(self, t: int) -> np.ndarray:
        """The frame's derived ``(P, 3)`` 3D, building + caching it if needed."""
        pts = self._pts3d_cache.get(t)
        if pts is None:
            pts = np.stack([self._solve_point(t, p) for p in range(self.n_points)])
            self._pts3d_cache[t] = pts
        return pts

    def _rederive_point(self, t: int, point: int) -> None:
        """Recompute one point's 3D in the frame cache (no-op if the frame is uncached)."""
        if t in self._pts3d_cache:
            self._pts3d_cache[t][point] = self._solve_point(t, point)

    def _rederive_points(self, t: int, points) -> None:
        """Recompute just ``points`` in the frame cache -- the batched op's invalidation.

        A batched edit (bulk confirm / reset / occlude / an absence declaration) must
        re-derive the points it *touched*, never the whole frame: dropping the frame
        would also discard the ray-slid 3D of every hand-placed point with fewer than
        two usable views, which is unrecoverable from the labels and would snap those
        points back to the run's cached ``pts3d``. Only :meth:`reset_frame`, which really
        does clear every label in the frame, invalidates wholesale.
        """
        if t not in self._pts3d_cache:
            return
        arr = self._pts3d_cache[t]
        for p in points:
            arr[p] = self._solve_point(t, p)

    def _set_point3d(self, t: int, point: int, xyz) -> None:
        """Store an already-solved 3D for one point (the drag result) in the cache."""
        self._ensure_pts3d(t)[point] = np.asarray(xyz, dtype=float)

    def _invalidate_frame3d(self, t: int) -> None:
        self._pts3d_cache.pop(t, None)

    def display_pts3d(
        self, frame: int | None = None
    ) -> Float[np.ndarray, "P 3"] | None:
        """The derived 3D points for ``frame``, or ``None`` when the result has no 3D."""
        if self.result.pts3d is None:
            return None
        return self._ensure_pts3d(self._resolve_frame(frame))

    def display_pts3d_projected(
        self, frame: int | None = None
    ) -> Float[np.ndarray, "V P 2"] | None:
        """The derived 3D for ``frame`` reprojected into every view, or ``None``.

        Uses the same full forward model (:meth:`CameraGroup.project`) the overlays
        are drawn with, so a drag lands exactly under the cursor.
        """
        pts3d = self.display_pts3d(frame)
        if pts3d is None or self.result.cameras is None:
            return None
        return np.asarray(self.result.cameras.project(pts3d))

    # -- chirality QC ---------------------------------------------------------

    def chirality(self, frame: int | None = None):
        """Left/right swap check for ``frame``, on the **derived 3D** pose.

        Returns a :class:`deeperfly.chirality.Chirality`. 3D rather than per-view 2D
        deliberately: measured over 500 independently-posed flies the 3D test emits 0.002
        false flags per pose and recovers 98-99% of planted swaps exactly, while the same
        test on a side view emits 4-11 false flags per pose from real posture asymmetry
        that no threshold separates from a swap (see :mod:`deeperfly.chirality`). Since
        this rig's derived 3D is what every hand label ultimately moves, judging it also
        answers the operator's actual question.

        An uncalibrated session (no 3D) gets an undecided verdict with a reason, never a
        confident "all clear" -- the two must not look alike.

        Pairs come from the skeleton, falling back to name inference for a ``results.h5``
        written before ``[skeleton].symmetries`` existed, so an old file still gets checked.
        """
        from .. import chirality as chi

        pts3d = self.display_pts3d(frame)
        if pts3d is None:
            return chi.Chirality(
                reason="this session has no 3D, so left/right cannot be judged"
            )
        skel = self.result.skeleton
        pairs = (
            skel.symmetries_or_inferred()
            if hasattr(skel, "symmetries_or_inferred")
            else np.empty((0, 2), np.int64)
        )
        return chi.check(pts3d, pairs)

    # -- NMF overlay (unchanged, driven by the derived 3D) --------------------

    def display_nmf_projected(
        self, frame: int | None = None
    ) -> Float[np.ndarray, "V P 2"] | None:
        """The fitted NMF model joints for ``frame`` reprojected into every view."""
        fit = self.nmf_fit(frame)
        if fit is None or self.result.cameras is None:
            return None
        return np.asarray(self.result.cameras.project(fit[0]))

    def nmf_fit(
        self, frame: int | None = None
    ) -> tuple[np.ndarray, np.ndarray | None, list[str] | None] | None:
        """The NMF fit for ``frame``: ``(model_pts3d, angles, angle_names)`` or ``None``.

        Re-solved from the frame's derived 3D pose when a :class:`NmfLive` is
        available (so it tracks edits) and memoized per frame; otherwise the
        pipeline's static fit.
        """
        t = self._resolve_frame(frame)
        cached = self._nmf_cache.get(t)
        if cached is not None:
            return cached
        out: tuple | None = None
        if self.nmf_live is not None:
            pts3d = self.display_pts3d(t)
            if pts3d is not None:
                # The frame index only picks the seed, and it is what keeps the re-fit a
                # pure function of (frame, labels) -- see NmfLive's module docstring.
                model, angles = self.nmf_live.refit(pts3d, t)
                out = (model, angles, self.nmf_live.angle_names)
        elif self.result.nmf_pts3d is not None:
            angles = (
                None if self.result.nmf_angles is None else self.result.nmf_angles[t]
            )
            out = (self.result.nmf_pts3d[t], angles, self.result.nmf_angle_names)
        if out is not None:
            self._nmf_cache[t] = out
        return out

    def nmf_posed_verts(
        self, frame: int | None = None
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """The frame's posed NMF mesh ``(vertices (Nv, 3), valid_faces (Nf,))``, or ``None``."""
        fit = self.nmf_fit(frame)
        if fit is None:
            return None
        from ..inverse_kinematics.mesh import load_nmf_mesh

        model, angles, names = fit
        return load_nmf_mesh().pose(
            model,
            angles,
            names,
            head_scale=self.result.nmf_head_scale,
            abdomen_scale=self.result.nmf_abdomen_scale,
            body_scale=self.result.nmf_body_scale,
        )

    def _invalidate_nmf(self, t: int) -> None:
        """Drop the cached NMF fit for ``frame`` ``t`` after its 3D pose changed."""
        self._nmf_cache.pop(t, None)

    # -- edits (the wire-compatible surface) ----------------------------------

    def absent_refusal(self, point: int, frame: int | None = None) -> str | None:
        """Why ``point`` cannot be labeled at ``frame``, or ``None`` if it can.

        Every authoring verb consults this. An absence declaration is *refused* rather
        than silently no-op'd: a tool that swallows the gesture reads as broken, and the
        operator needs to be told which fact is blocking them and how to lift it.
        """
        t = self._resolve_frame(frame)
        if not bool(self.labels.absent_at(t)[point]):
            return None
        return (
            f"{self.point_name(point)} is marked absent (not on this animal) -- "
            "un-mark it first to label it"
        )

    def point_name(self, point: int) -> str:
        """The skeleton name of ``point`` (its index, if names are unavailable)."""
        names = getattr(self.result.skeleton, "point_names", None)
        try:
            return str(names[point])  # type: ignore[index]
        except Exception:
            return str(point)

    def apply_2d_edit(
        self, view: int, point: int, xy, frame: int | None = None
    ) -> None:
        """Author a GT pixel for ``point`` in ``view`` at ``xy`` (a 2D drag)."""
        t = self._resolve_frame(frame)
        if self.absent_refusal(point, t):
            return
        self._record_undo(t, point, coalesce=True)
        self.labels.set_gt(view, t, point, xy)
        self._rederive_point(t, point)
        self._invalidate_nmf(t)
        # One message per gesture on this path: with no 3D to re-solve the client sends
        # nothing mid-drag and commits on release (app.js ``onDragged``), so the gesture
        # is over. A future streaming 2D client would mark its settle the way edit_3d does.
        self._end_gesture()

    def apply_3d_edit(
        self, view: int, point: int, xy, frame: int | None = None, *, fix: bool = False
    ) -> Float[np.ndarray, "3"] | None:
        """Create/move the GT at ``view`` from a drag to ``xy`` and re-solve the 3D.

        The dragged view becomes a GT constraint and the 3D re-solves live via
        :func:`~deeperfly.gui.solve.solve_point_3d_drag` -- a weighted DLT with two or
        more usable views, else a ray-slide of the prior 3D so the point lands under
        the cursor. Every drag (``fix`` or not) authors the GT; ``fix`` no longer
        distinguishes a "pin" (GT *is* the constraint) but does mark the *settle*, where
        the point converges to the configured solve policy (:meth:`_settle_point3d`).
        Returns the new 3D point, or ``None`` when no 3D could be derived (the GT pixel
        is still authored -- see below).
        """
        if self.result.pts3d is None:
            return None
        t = self._resolve_frame(frame)
        if self.absent_refusal(point, t):
            return None
        prior = self._ensure_pts3d(t)[point].copy()
        gt_obs, pred_obs, conf = self._point_obs(t, point)
        x_new = solve_point_3d_drag(
            self.result.cameras, gt_obs, pred_obs, conf, view, xy, prior, self.ann
        )
        # The 2D drop is ground truth whether or not a 3D can be derived from it yet.
        # A first view placed on an otherwise-unobserved point (a triangulation reject, or
        # one the detector never fired: no prior 3D and a single usable view) has no
        # solvable 3D, but the authored pixel must still stick so a second view can be
        # added and the point triangulated. So author the GT unconditionally; update
        # the 3D cache when the solve produced one, else re-derive (it may stay NaN
        # until a second view lands).
        self._record_undo(t, point, coalesce=True)
        self.labels.set_gt(view, t, point, xy)
        if x_new is not None:
            if fix:
                x_new = self._settle_point3d(t, point, x_new)
            self._set_point3d(t, point, x_new)
        else:
            self._rederive_point(t, point)
        self._invalidate_nmf(t)
        if fix:
            self._end_gesture()  # the settle: the operator released the point
        return x_new

    def _settle_point3d(self, t: int, point: int, dragged: np.ndarray) -> np.ndarray:
        """The drag-*release* 3D: the configured solve, when the labels determine it.

        The mid-drag stream runs the cheap :func:`~deeperfly.gui.solve.solve_point_3d_drag`
        so a ~60 Hz interaction never pays for a consensus fit. On the settle the point
        converges to the same ``solve_policy`` a plain re-derivation would use, which is
        what keeps the cached 3D a *function of the labels* rather than of how the point
        got there. That divergence is what used to make an unrelated undo appear to move a
        point: the cache said one thing, re-deriving said another.

        It cannot always converge, and must not pretend to: with fewer than two usable
        views the pure solve is ``NaN`` -- one pixel leaves depth free -- and the drag's
        ray-slide is the only position that honors where the operator dropped the point,
        so it stands (and the undo entries snapshot it, see :class:`_UndoEntry`).
        """
        if self.result.cameras is None:
            return dragged
        gt_obs, pred_obs, conf = self._point_obs(t, point)
        settled = solve_point_3d(
            self.result.cameras, gt_obs, pred_obs, conf, self.ann, self.tri
        )
        return settled if np.all(np.isfinite(settled)) else dragged

    def toggle_fixed(
        self, view: int, point: int, frame: int | None = None
    ) -> bool | None:
        """Confirm the view's displayed 2D as GT, or clear it if already GT.

        This is the "finalize / un-finalize" gesture: with no GT it snapshots the
        current reprojected pixel as a confirmed GT; with GT it drops it back to the
        prediction. Either way the 3D re-solves. Returns the new GT state, or ``None``
        if there is no 3D to refine or the point is not visible in this view.
        """
        if self.result.pts3d is None:
            return None
        t = self._resolve_frame(frame)
        if self.absent_refusal(point, t):
            return None
        if self.labels.has_gt[view, t, point]:
            self._record_undo(t, point, coalesce=False)
            self.labels.clear_gt(view, t, point)
            self._rederive_point(t, point)
            self._invalidate_nmf(t)
            return False
        cur = self.display_pts2d_refine(t)
        if cur is None:
            return None
        xy = cur[view, point]
        if not np.all(np.isfinite(xy)):
            return None  # cannot confirm a point that is not visible in this view
        self._record_undo(t, point, coalesce=False)
        self.labels.set_gt(view, t, point, xy)
        self._rederive_point(t, point)
        self._invalidate_nmf(t)
        return True

    def toggle_invisible(
        self, view: int, point: int, frame: int | None = None
    ) -> bool | None:
        """Toggle whether ``point`` in ``view`` is occluded (dropped from the 3D solve).

        An occluded view contributes nothing to the 3D and follows the reprojection;
        toggling re-solves the 3D from the remaining views. Setting it drops any GT for
        that view. Returns the new occluded state, or ``None`` if the point is absent.

        Deliberately **not** gated on a 3D solve. "I cannot place this from this view" is
        a label, not a derived quantity, and on a 2D-only pass (exactly where a hand
        labeling round happens) gating it would leave the operator with no way to reject
        a view except the much stronger anatomical claim that the joint does not exist.
        """
        t = self._resolve_frame(frame)
        if self.absent_refusal(point, t):
            return None
        self._record_undo(t, point, coalesce=False)
        now = not bool(self.labels.occluded[view, t, point])
        self.labels.set_occluded(view, t, point, now)
        self._rederive_point(t, point)
        self._invalidate_nmf(t)
        return now

    def reset_point(self, point: int, frame: int | None = None) -> None:
        """Reset every view of ``point`` at ``frame`` to ``unset`` (drop GT + occlusion)."""
        t = self._resolve_frame(frame)
        self._record_undo(t, point, coalesce=False)
        self.labels.clear_point(t, point)
        self._rederive_point(t, point)
        self._invalidate_nmf(t)

    def reset_point_view(self, view: int, point: int, frame: int | None = None) -> None:
        """Reset just ``view``'s label for ``point`` to ``unset`` (GT + occlusion)."""
        t = self._resolve_frame(frame)
        self._record_undo(t, point, coalesce=False)
        self.labels.clear_view(view, t, point)
        self._rederive_point(t, point)
        self._invalidate_nmf(t)

    def reset_frame(self, frame: int | None = None) -> None:
        """Reset every label in ``frame`` to ``unset`` -- back to the pipeline pose."""
        t = self._resolve_frame(frame)
        self._record_undo(t, None, coalesce=False)
        self.labels.clear_frame(t)
        self._invalidate_frame3d(t)
        self._invalidate_nmf(t)

    # -- calibration landmarks ------------------------------------------------
    #
    # A landmark is authored the same way a keypoint is -- click a pixel in a view -- but it
    # lives in its own namespace and drives only the rig solve. Kept out of the undo stack
    # deliberately: the undo entries snapshot one frame's *skeleton* labels, and widening
    # them to carry a second overlay would make every keypoint undo heavier for a gesture
    # that has its own explicit clear.

    @property
    def has_landmarks(self) -> bool:
        """Whether this project defines calibration landmarks."""
        return self.landmarks is not None and bool(self.landmarks.names)

    def landmark_names(self) -> list[str]:
        return [] if self.landmarks is None else list(self.landmarks.names)

    def display_landmarks(self, frame: int | None = None):
        """``(V, L, 2)`` observed landmark pixels for ``frame``, NaN where unobserved."""
        if self.landmarks is None:
            return None
        return self.landmarks.xy[:, self._resolve_frame(frame)]

    def set_landmark(
        self, view: int, landmark: int, xy, frame: int | None = None
    ) -> bool:
        """Place (or move) one landmark observation. Returns whether anything changed."""
        if self.landmarks is None:
            return False
        self.landmarks.set(view, self._resolve_frame(frame), landmark, xy)
        return True

    def clear_landmark(
        self, view: int, landmark: int, frame: int | None = None
    ) -> bool:
        """Drop one landmark observation. Returns whether anything changed."""
        if self.landmarks is None:
            return False
        self.landmarks.clear(view, self._resolve_frame(frame), landmark)
        return True

    def landmark_counts(self) -> dict[str, int]:
        """``name -> observed (view, frame) cells`` -- the readiness signal for the solve."""
        return {} if self.landmarks is None else self.landmarks.counts()

    def set_reviewed(self, value: bool, frame: int | None = None) -> None:
        """Mark ``frame`` reviewed (or clear it): the operator's "I've checked this" flag.

        A per-frame annotation, orthogonal to the point labels -- it changes no derived
        3D and stays out of the label undo history (undo reverts pixel edits, not review
        bookkeeping). It only records review progress and keeps the frame in the
        corrected-frames list even once its labels are reset.
        """
        t = self._resolve_frame(frame)
        self.labels.set_reviewed(t, value)

    def set_absent(
        self,
        points,
        value: bool,
        frame: int | None = None,
        *,
        whole_recording: bool = False,
    ) -> list[int]:
        """Declare (or un-declare) ``points`` as not being on this animal.

        Scoped to ``frame`` by default -- a keypoint *can* stop existing part-way through
        a recording (autotomy). ``whole_recording=True`` applies it to every frame at once,
        which is the common case and the convenience the editor exposes as its own gesture:
        an animal that arrives with a leg missing keeps it missing.

        View-independent either way. Returns the point indices whose state actually
        changed (an empty list is a no-op and records no undo step).

        A whole-recording change invalidates the *entire* derived-3D and NMF caches; a
        single-frame one only that frame's.
        """
        t = self._resolve_frame(frame)
        idx = sorted({int(p) for p in np.atleast_1d(np.asarray(points, dtype=int))})
        absent = self.labels.absent
        rows = slice(None) if whole_recording else slice(t, t + 1)
        changed = [p for p in idx if bool((absent[rows, p] != bool(value)).any())]
        if not changed:
            return []
        self._undo.append(
            _AbsentEntry(
                points=list(changed),
                prev=absent[:, changed].copy(),
                pts3d=self._snapshot_absent_pts3d(changed),
            )
        )
        self._redo.clear()
        self._end_gesture()  # its own gesture, and it ends any open drag
        if len(self._undo) > UNDO_LIMIT:
            self._undo.pop(0)
        self.labels.set_absent(changed, value, frames=None if whole_recording else [t])
        # Absence only vetoes the declared points, so re-derive exactly those -- clearing
        # the cache would also discard other points' hand-placed (ray-slid) 3D.
        if whole_recording:
            for tt in list(self._pts3d_cache):
                self._rederive_points(tt, changed)
            self._nmf_cache.clear()
        else:
            self._rederive_points(t, changed)
            self._invalidate_nmf(t)
        return changed

    # -- undo / redo + bulk confirm + explicit GT set/clear -------------------

    def _snapshot(self, t: int, point: int | None, coalesce: bool) -> _UndoEntry:
        cached = self._pts3d_cache.get(t)
        return _UndoEntry(
            t=t,
            point=point,
            coalesce=coalesce,
            gt=self.labels.gt[:, t].copy(),
            occluded=self.labels.occluded[:, t].copy(),
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
            self._nmf_cache.clear()  # a pure function of the 3D, so refitting is safe
            return
        t = entry.t
        self.labels.gt[:, t] = entry.gt
        self.labels.occluded[:, t] = entry.occluded
        self.labels.dirty = True
        if entry.pts3d is None:
            self._invalidate_frame3d(t)  # uncached then, so there is nothing to restore
        else:
            self._pts3d_cache[t] = entry.pts3d.copy()
        self._invalidate_nmf(t)

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

    def set_gt(self, view: int, point: int, xy, frame: int | None = None) -> None:
        """Create a GT pixel without a drag re-solve (a click-place)."""
        t = self._resolve_frame(frame)
        self._record_undo(t, point, coalesce=False)
        self.labels.set_gt(view, t, point, xy)
        self._rederive_point(t, point)
        self._invalidate_nmf(t)

    def clear_gt(self, view: int, point: int, frame: int | None = None) -> None:
        """Drop just ``view``'s GT pixel for ``point`` (revert to the prediction)."""
        t = self._resolve_frame(frame)
        self._record_undo(t, point, coalesce=False)
        self.labels.clear_gt(view, t, point)
        self._rederive_point(t, point)
        self._invalidate_nmf(t)

    def _on_image(self, view: int, xy) -> bool:
        """Is ``xy`` inside ``view``'s image (so the operator can see and drag it)?

        ``True`` when the image size is unknown -- refusing to author a pixel because the
        editor cannot check it would be worse than authoring one.
        """
        if self.image_sizes_wh is None:
            return True
        w, h = (float(v) for v in self.image_sizes_wh[view])
        if w <= 0 or h <= 0:
            return True
        return bool(0.0 <= float(xy[0]) <= w - 1.0 and 0.0 <= float(xy[1]) <= h - 1.0)

    def confirm(
        self,
        targets,
        sources: str = "all",
        frame: int | None = None,
    ) -> bool:
        """Create GT at the *displayed* position for many ``(view, point)`` targets at once.

        The bulk form of a drag's first half: it plants a GT pixel where the operator can
        already see a dot, so the joint is authored and can then be nudged. ``sources``
        picks which proposal layers are eligible -- ``"predictions"`` (the detection),
        ``"projections"`` (the reprojected 3D), or ``"all"`` (detection where it fired,
        else the reprojection).

        The stored coordinate is the displayed one, because bulk-creating GT means "these
        dots are right", and storing anything other than the dot the operator saw would
        record a position they never approved.

        A cell with **nothing on screen** is skipped, not invented. An off-image
        reprojection, or a joint with no 3D and no detection, has no dot to approve; its
        Unplaced seed (:meth:`placeholder_pts2d`) is already a draggable handle, so the
        editor does not need to store a fabricated pixel to give the operator something to
        grab. Occluded views, already-GT views and absent points are skipped too -- for an
        absent point a select-all would otherwise author GT on a phantom limb in every
        frame visited. One undo step; the touched points re-derive once. Returns whether
        anything changed.
        """
        t = self._resolve_frame(frame)
        saved_redo = list(self._redo)
        self._record_undo(t, None, coalesce=False)
        want_pred = sources in ("all", "predictions")
        want_proj = sources in ("all", "projections")
        proj = self.display_pts3d_projected(t) if want_proj else None
        changed = False
        touched: set[int] = set()  # the points to re-derive (never the whole frame)
        absent = self.labels.absent_at(t)
        for view, point in targets:
            if (
                absent[point]
                or self.labels.occluded[view, t, point]
                or self.labels.has_gt[view, t, point]
            ):
                continue
            xy = None
            pred = self.detections[view, t, point]
            if want_pred and np.all(np.isfinite(pred)):
                xy = pred
            if xy is None and proj is not None:
                cand = proj[view, point]
                if np.all(np.isfinite(cand)):
                    xy = cand
            if xy is None or not self._on_image(view, xy):
                continue
            self.labels.set_gt(view, t, point, xy)
            touched.add(point)
            changed = True
        if changed:
            self._rederive_points(t, touched)
            self._invalidate_nmf(t)
        else:
            self._undo.pop()  # nothing changed: drop the no-op undo entry
            self._redo[:] = saved_redo  # ... and restore the redo _record_undo cleared
        return changed

    def reset_targets(self, targets, frame: int | None = None) -> None:
        """Reset many ``(view, point)`` cells at ``frame`` to ``unset`` in one undo step.

        The batched counterpart of :meth:`reset_point_view`: it drops GT *and*
        occlusion for every target and re-derives the frame's 3D once, so a
        multi-select "Reset" is a single undoable action. A no-op batch (empty, or
        every target already unset -- e.g. select-all then Reset on a fresh frame) does
        nothing at all: no undo entry, no cleared redo, nothing marked dirty (mirrors
        :meth:`confirm`).
        """
        targets = list(targets)
        if not targets:
            return
        t = self._resolve_frame(frame)
        has_gt = self.labels.has_gt
        occluded = self.labels.occluded_effective
        if not any(
            has_gt[view, t, point] or occluded[view, t, point]
            for view, point in targets
        ):
            return
        self._record_undo(t, None, coalesce=False)
        for view, point in targets:
            self.labels.clear_view(view, t, point)
        self._rederive_points(t, {point for _, point in targets})
        self._invalidate_nmf(t)

    def clear_gt_targets(self, targets, frame: int | None = None) -> None:
        """Delete the GT pixel at many ``(view, point)`` cells, leaving all else alone.

        The batched inverse of a drag, and deliberately **not** the same verb as
        :meth:`reset_targets`, which also drops the exclusion. "I retract the pixel I
        placed" and "I retract my claim that this view is unusable" are different
        retractions; one key doing both silently would undo work the operator did not name.
        Cells with no GT are skipped, so a no-op batch records no undo step.
        """
        targets = list(targets)
        if not targets:
            return
        t = self._resolve_frame(frame)
        has = self.labels.has_gt
        if not any(has[view, t, point] for view, point in targets):
            return
        self._record_undo(t, None, coalesce=False)
        for view, point in targets:
            self.labels.clear_gt(view, t, point)
        self._rederive_points(t, {point for _, point in targets})
        self._invalidate_nmf(t)

    def toggle_exclude_targets(self, targets, frame: int | None = None) -> bool | None:
        """Toggle "exclude this detection from triangulation" over many cells.

        A *toggle*, so one key both marks and un-marks: if every eligible target is already
        excluded the batch clears, otherwise it excludes. Returns the new state, or ``None``
        when nothing was eligible.

        Cells that carry **GT are skipped**, and that is a safety property rather than a
        technicality. Storing an exclusion clears the pixel underneath it
        (:meth:`Labels.set_occluded` -- the cell tri-state is exclusive), so a bulk
        exclusion over a selection that happens to include labeled cells would delete the
        operator's own work on a keystroke. It is also meaningless: GT already overrides the
        detection in its own view, so there is nothing left there to exclude. To exclude a
        view you have labeled, delete the GT first (:meth:`clear_gt_targets`) -- two
        retractions, named separately.
        """
        targets = list(targets)
        if not targets:
            return None
        t = self._resolve_frame(frame)
        absent = self.labels.absent_at(t)
        has = self.labels.has_gt
        eligible = [(v, p) for v, p in targets if not absent[p] and not has[v, t, p]]
        if not eligible:
            return None
        occluded = self.labels.occluded
        now = not all(bool(occluded[v, t, p]) for v, p in eligible)
        self._record_undo(t, None, coalesce=False)
        for view, point in eligible:
            self.labels.set_occluded(view, t, point, now)
        self._rederive_points(t, {point for _, point in eligible})
        self._invalidate_nmf(t)
        return now

    def occlude_targets(self, targets, frame: int | None = None) -> None:
        """Occlude many ``(view, point)`` cells at ``frame`` in one undo step.

        The batched counterpart of :meth:`toggle_invisible`, but a *set* not a toggle:
        every target is flagged occluded (dropping any GT there), and the frame's 3D
        re-derives once. Reversal is :meth:`reset_targets` / undo. Targets on a point
        declared absent are dropped -- an amputated joint is not "occluded everywhere".
        A no-op batch (empty, or every target already occluded) does nothing: no undo
        entry, no cleared redo, nothing marked dirty (mirrors :meth:`confirm`).

        Like :meth:`toggle_invisible`, deliberately not gated on a 3D solve: occlusion is
        an authored label and a hand-labeling pass often starts before any triangulation.
        """
        targets = list(targets)
        if not targets:
            return
        t = self._resolve_frame(frame)
        absent = self.labels.absent_at(t)
        targets = [(v, p) for v, p in targets if not absent[p]]
        occluded = self.labels.occluded
        if not any(not occluded[view, t, point] for view, point in targets):
            return
        self._record_undo(t, None, coalesce=False)
        for view, point in targets:
            self.labels.set_occluded(view, t, point, True)
        self._rederive_points(t, {point for _, point in targets})
        self._invalidate_nmf(t)
