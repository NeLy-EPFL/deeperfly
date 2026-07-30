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
from .labels import Labels, Provenance
from .nmf_live import NmfLive
from .solve import solve_point_3d, solve_point_3d_drag

__all__ = ["EditMode", "EditorState"]

log = logging.getLogger("deeperfly")

#: How many undo steps to keep (a backstop; one entry per operator gesture).
UNDO_LIMIT = 200


class EditMode(str, Enum):
    """The interaction mode of the editor."""

    view = "view"  # read-only inspection
    edit_2d = "edit_2d"  # drag per-view 2D keypoints (create GT)
    edit_3d = "edit_3d"  # drag reprojected 3D keypoints (create GT + re-solve)


@dataclass
class _UndoEntry:
    """A snapshot of one frame's labels, so an edit can be reverted.

    ``point`` names the target point and ``coalesce`` marks a drag-style edit: a run
    of coalescing edits to the *same* ``(t, point)`` collapses to a single undo step
    (so one drag is one undo), while a discrete op (toggle / reset / confirm) always
    starts a new step.
    """

    t: int
    point: int | None
    coalesce: bool
    gt: np.ndarray
    provenance: np.ndarray
    occluded: np.ndarray


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
    #: whole design treats the 3D as. Dropped/updated when a frame's labels change.
    _pts3d_cache: dict[int, np.ndarray] = field(default_factory=dict)
    #: Undo / redo stacks of per-frame label snapshots (see :class:`_UndoEntry`).
    _undo: list = field(default_factory=list)
    _redo: list = field(default_factory=list)
    #: Pristine detector detections ``(V, T, P, 2)`` -- the raw peak *before*
    #: triangulation cleaning, kept even where triangulation dropped the point. The
    #: top-priority seed for a placeholder (see :meth:`placeholder_pts2d`). ``None``
    #: when the pristine ``pose2d`` group is unavailable.
    raw_pts2d: np.ndarray | None = None
    #: Per-view image size as ``(V, 2)`` ``[width, height]``, for a placeholder's
    #: last-resort centre. ``None`` when unavailable.
    image_sizes_wh: np.ndarray | None = None

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
    ) -> EditorState:
        """Build a state for ``result``, with an empty overlay if none is given.

        Unlike the old corrections overlay, a fresh labels overlay seeds *nothing*:
        a view the detector missed is ``absent`` (derived), not a stored occlusion,
        so an untouched session carries no authored state. ``ann`` / ``tri`` are the
        annotation solve policy and shared triangulation params (from the run config
        beside ``results.h5``); both default to the packaged defaults.

        When the result carries a fitted NMF model and 3D pose, a :class:`NmfLive`
        is built so the overlaid model re-fits to the operator's edits. ``template`` /
        ``articulation`` make that re-fit use the *same* model the pipeline did.

        ``raw_pts2d`` is the pristine ``pose2d`` detections (``result.pts2d`` is the
        triangulation-*cleaned* array, so a rejected point is NaN there); it seeds the
        placeholder for an otherwise-absent joint. ``image_sizes`` (camera name ->
        ``(height, width)``) supplies the placeholder's last-resort image centre.
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
        image_sizes_wh = None
        if image_sizes:
            image_sizes_wh = np.array(
                [
                    (image_sizes.get(name, (0, 0))[1], image_sizes.get(name, (0, 0))[0])
                    for name in result.cameras.names
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
    def camera_names(self) -> list[str]:
        return self.result.cameras.names

    @property
    def dirty(self) -> bool:
        return self.labels.dirty

    def _resolve_frame(self, frame: int | None) -> int:
        return self.frame if frame is None else frame

    # -- per-view label masks (for the payload) -------------------------------

    def gt_mask(self, frame: int | None = None) -> np.ndarray:
        """``(V, P)`` boolean: which per-view points carry a GT pixel at ``frame``."""
        return self.labels.has_gt[:, self._resolve_frame(frame)]

    def occluded_mask(self, frame: int | None = None) -> np.ndarray:
        """``(V, P)`` boolean: which per-view points are occluded at ``frame``."""
        return self.labels.occluded[:, self._resolve_frame(frame)]

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
        """
        decided = self.labels.has_gt | self.labels.occluded  # (V, T, P)
        labeled = decided.any(axis=(0, 2))  # (T,) any authored label in the frame
        reviewed = self.labels.reviewed  # (T,)
        return [
            {"frame": int(t), "reviewed": bool(reviewed[t])}
            for t in np.nonzero(labeled | reviewed)[0]
        ]

    # -- displayed 2D (labels over predictions) -------------------------------

    def display_pts2d(self, frame: int | None = None) -> Float[np.ndarray, "V P 2"]:
        """The per-view 2D to draw for ``frame``: GT over the detector prediction.

        A GT view shows its pixel; a plain view shows the prediction; an occluded
        view (or one with neither) is ``NaN`` -- the front-end draws the reprojection
        ghost there instead.
        """
        t = self._resolve_frame(frame)
        gt = self.labels.gt[:, t]  # (V, P, 2)
        has = self.labels.has_gt[:, t]  # (V, P)
        occ = self.labels.occluded[:, t]  # (V, P)
        pred = self.result.pts2d[:, t]  # (V, P, 2)
        return np.where(has[..., None], gt, np.where(~occ[..., None], pred, np.nan))

    def display_pts2d_refine(
        self, frame: int | None = None
    ) -> Float[np.ndarray, "V P 2"] | None:
        """The per-view 2D drawn in Edit 3D: the derived 3D reprojected into every
        view, with each GT view overridden by its authored pixel, or ``None``.

        GT views hold their pixel (they generally do not all agree with one 3D
        point); every other view -- plain or occluded -- follows the reprojection.
        """
        proj = self.display_pts3d_projected(frame)
        if proj is None:
            return None
        t = self._resolve_frame(frame)
        gt = self.labels.gt[:, t]  # (V, P, 2)
        has = self.labels.has_gt[:, t]  # (V, P)
        return np.where(has[..., None], gt, proj)

    # -- placeholder seeds for absent joints ----------------------------------

    def placeholder_pts2d(
        self, frame: int | None = None, *, window: int = 30
    ) -> Float[np.ndarray, "V P 2"]:
        """Seed positions for joints ABSENT from a view, so a GT can still be placed.

        A point rejected by triangulation (or one the detector never fired) has no GT,
        no displayed detection, and no reprojection in a view -- so the canvas draws
        nothing there and the operator has nothing to grab, hence no way to author a
        GT. For exactly those cells this returns a *sensible* draggable seed; every
        other cell -- already grabbable, or deliberately occluded -- is ``NaN`` (no
        placeholder). A seed falls back, in order, to:

        1. the raw detector pixel (kept even when triangulation dropped it),
        2. the nearest frame (within ``window``) whose raw/cleaned pixel in this view
           is finite -- a keypoint moves little frame to frame,
        3. the mean of the joint's connected skeleton neighbours shown in this view,
        4. the centroid of the view's shown points,
        5. the image centre.

        The operator drags the seed to author GT, so the position only needs to be a
        reasonable starting point near where the point belongs.
        """
        t = self._resolve_frame(frame)
        n_views, n_points = self.n_views, self.n_points
        disp = self.display_pts2d(t)  # (V, P, 2): GT over cleaned pred, NaN if absent
        occ = self.labels.occluded[:, t]  # (V, P)
        proj = self.display_pts3d_projected(t) if self.has_3d else None
        disp_ok = np.isfinite(disp).all(axis=-1)  # (V, P)
        proj_ok = (
            np.isfinite(proj).all(axis=-1)
            if proj is not None
            else np.zeros((n_views, n_points), dtype=bool)
        )
        # A cell needs a placeholder iff nothing is grabbable there and it is not
        # occluded (occluding a point is the operator asserting it cannot be placed).
        need = ~disp_ok & ~proj_ok & ~occ  # (V, P)
        out = np.full((n_views, n_points, 2), np.nan)
        if not need.any():
            return out

        # A per-cell "shown" position (GT/detected, else the reprojection) for the
        # neighbour/centroid fallbacks -- what the operator actually sees drawn there.
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
        raw = self.raw_pts2d
        # 1. the raw detector pixel at this frame.
        if raw is not None and np.all(np.isfinite(raw[v, t, p])):
            return np.asarray(raw[v, t, p], dtype=float)
        # 2. the nearest frame (raw, then cleaned) with a finite pixel in this view.
        for dt in range(1, max(t - lo, hi - t) + 1):
            for tt in (t - dt, t + dt):
                if not lo <= tt <= hi:
                    continue
                if raw is not None and np.all(np.isfinite(raw[v, tt, p])):
                    return np.asarray(raw[v, tt, p], dtype=float)
                cleaned = self.result.pts2d[v, tt, p]
                if np.all(np.isfinite(cleaned)):
                    return np.asarray(cleaned, dtype=float)
        # 3. the mean of connected skeleton neighbours shown in this view.
        if bones.size:
            nbrs = np.unique(
                np.concatenate([bones[bones[:, 0] == p, 1], bones[bones[:, 1] == p, 0]])
            )
            npos = [
                shown[v, q] for q in nbrs if 0 <= q < shown.shape[1] and shown_ok[v, q]
            ]
            if npos:
                return np.mean(np.stack(npos), axis=0)
        # 4. the centroid of the view's shown points.
        if shown_ok[v].any():
            return shown[v][shown_ok[v]].mean(axis=0)
        # 5. the image centre (else the origin, if even that is unknown).
        if self.image_sizes_wh is not None:
            return self.image_sizes_wh[v] / 2.0
        return np.zeros(2)

    # -- derived 3D (the "cache") ---------------------------------------------

    def _point_obs(self, t: int, point: int):
        """``(gt_obs (V,2), pred_obs (V,2), conf (V,)|None)`` for one point at ``t``.

        ``pred_obs`` NaNs out occluded views and views that already carry GT (GT
        overrides the prediction there), so it is exactly the prediction contribution
        the solve should see.
        """
        has = self.labels.has_gt[:, t, point]  # (V,)
        occ = self.labels.occluded[:, t, point]  # (V,)
        gt_obs = np.where(has[:, None], self.labels.gt[:, t, point], np.nan)
        pred = self.result.pts2d[:, t, point].astype(float)  # (V, 2)
        pred_ok = np.isfinite(pred).all(axis=-1) & ~occ & ~has
        pred_obs = np.where(pred_ok[:, None], pred, np.nan)
        conf = None if self.result.conf is None else self.result.conf[:, t, point]
        return gt_obs, pred_obs, conf

    def _solve_point(self, t: int, point: int) -> np.ndarray:
        """Derive one point's 3D from its labels + predictions (run-cache fallback)."""
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
        if pts3d is None:
            return None
        return np.asarray(self.result.cameras.project(pts3d))

    # -- NMF overlay (unchanged, driven by the derived 3D) --------------------

    def display_nmf_projected(
        self, frame: int | None = None
    ) -> Float[np.ndarray, "V P 2"] | None:
        """The fitted NMF model joints for ``frame`` reprojected into every view."""
        fit = self.nmf_fit(frame)
        if fit is None:
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

    def apply_2d_edit(
        self, view: int, point: int, xy, frame: int | None = None
    ) -> None:
        """Author a GT pixel for ``point`` in ``view`` at ``xy`` (a 2D drag)."""
        t = self._resolve_frame(frame)
        self._record_undo(t, point, coalesce=True)
        self.labels.set_gt(view, t, point, xy, provenance=Provenance.DRAGGED)
        self._rederive_point(t, point)
        self._invalidate_nmf(t)

    def apply_3d_edit(
        self, view: int, point: int, xy, frame: int | None = None, *, fix: bool = False
    ) -> Float[np.ndarray, "3"] | None:
        """Create/move the GT at ``view`` from a drag to ``xy`` and re-solve the 3D.

        The dragged view becomes a GT constraint and the 3D re-solves live via
        :func:`~deeperfly.gui.solve.solve_point_3d_drag` -- a weighted DLT with two or
        more usable views, else a ray-slide of the prior 3D so the point lands under
        the cursor. Every drag (``fix`` or not) authors the GT; ``fix`` is retained for
        wire compatibility but no longer distinguishes a "pin" (GT *is* the
        constraint). Returns the new 3D point, or ``None`` when no 3D could be derived
        (the GT pixel is still authored -- see below).
        """
        if self.result.pts3d is None:
            return None
        t = self._resolve_frame(frame)
        prior = self._ensure_pts3d(t)[point].copy()
        gt_obs, pred_obs, conf = self._point_obs(t, point)
        x_new = solve_point_3d_drag(
            self.result.cameras, gt_obs, pred_obs, conf, view, xy, prior, self.ann
        )
        # The 2D drop is ground truth whether or not a 3D can be derived from it yet.
        # A first view placed on an otherwise-absent point (a triangulation reject, or
        # one the detector never fired: no prior 3D and a single usable view) has no
        # solvable 3D, but the authored pixel must still stick so a second view can be
        # added and the point triangulated. So author the GT unconditionally; update
        # the 3D cache when the solve produced one, else re-derive (it may stay NaN
        # until a second view lands).
        self._record_undo(t, point, coalesce=True)
        self.labels.set_gt(view, t, point, xy, provenance=Provenance.DRAGGED)
        if x_new is not None:
            self._set_point3d(t, point, x_new)
        else:
            self._rederive_point(t, point)
        self._invalidate_nmf(t)
        return x_new

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
        self.labels.set_gt(
            view, t, point, xy, provenance=Provenance.CONFIRMED_PREDICTION
        )
        self._rederive_point(t, point)
        self._invalidate_nmf(t)
        return True

    def toggle_invisible(
        self, view: int, point: int, frame: int | None = None
    ) -> bool | None:
        """Toggle whether ``point`` in ``view`` is occluded (dropped from the 3D solve).

        An occluded view contributes nothing to the 3D and follows the reprojection;
        toggling re-solves the 3D from the remaining views. Setting it drops any GT for
        that view. Returns the new occluded state, or ``None`` if there is no 3D.
        """
        if self.result.pts3d is None:
            return None
        t = self._resolve_frame(frame)
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

    def set_reviewed(self, value: bool, frame: int | None = None) -> None:
        """Mark ``frame`` reviewed (or clear it): the operator's "I've checked this" flag.

        A per-frame annotation, orthogonal to the point labels -- it changes no derived
        3D and stays out of the label undo history (undo reverts pixel edits, not review
        bookkeeping). It only records review progress and keeps the frame in the
        corrected-frames list even once its labels are reset.
        """
        t = self._resolve_frame(frame)
        self.labels.set_reviewed(t, value)

    # -- undo / redo + bulk confirm + explicit GT set/clear -------------------

    def _snapshot(self, t: int, point: int | None, coalesce: bool) -> _UndoEntry:
        return _UndoEntry(
            t=t,
            point=point,
            coalesce=coalesce,
            gt=self.labels.gt[:, t].copy(),
            provenance=self.labels.gt_provenance[:, t].copy(),
            occluded=self.labels.occluded[:, t].copy(),
        )

    def _record_undo(self, t: int, point: int | None, *, coalesce: bool) -> None:
        """Push a pre-edit snapshot, coalescing a run of drag edits on one point."""
        top = self._undo[-1] if self._undo else None
        if (
            coalesce
            and point is not None
            and top is not None
            and top.coalesce
            and top.t == t
            and top.point == point
        ):
            return  # same drag gesture: keep the existing (older) pre-state
        self._undo.append(self._snapshot(t, point, coalesce))
        self._redo.clear()
        if len(self._undo) > UNDO_LIMIT:
            self._undo.pop(0)

    def _apply_snapshot(self, entry: _UndoEntry) -> None:
        t = entry.t
        self.labels.gt[:, t] = entry.gt
        self.labels.gt_provenance[:, t] = entry.provenance
        self.labels.occluded[:, t] = entry.occluded
        self.labels.dirty = True
        self._invalidate_frame3d(t)
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
        entry = self._undo.pop()
        self._redo.append(self._snapshot(entry.t, entry.point, entry.coalesce))
        self._apply_snapshot(entry)
        return entry.t

    def redo(self) -> int | None:
        """Re-apply the last undone edit; returns the affected frame."""
        if not self._redo:
            return None
        entry = self._redo.pop()
        self._undo.append(self._snapshot(entry.t, entry.point, entry.coalesce))
        self._apply_snapshot(entry)
        return entry.t

    def set_gt(
        self,
        view: int,
        point: int,
        xy,
        frame: int | None = None,
        *,
        provenance: int = Provenance.DRAGGED,
    ) -> None:
        """Author a GT pixel without a drag re-solve (a click-place / confirm-in-place)."""
        t = self._resolve_frame(frame)
        self._record_undo(t, point, coalesce=False)
        self.labels.set_gt(view, t, point, xy, provenance=provenance)
        self._rederive_point(t, point)
        self._invalidate_nmf(t)

    def clear_gt(self, view: int, point: int, frame: int | None = None) -> None:
        """Drop just ``view``'s GT pixel for ``point`` (revert to the prediction)."""
        t = self._resolve_frame(frame)
        self._record_undo(t, point, coalesce=False)
        self.labels.clear_gt(view, t, point)
        self._rederive_point(t, point)
        self._invalidate_nmf(t)

    #: Displayed-vs-``pose2d`` distance below which a pixel is the detector's own output. The
    #: two populations are cleanly separated -- on scape_Fly4_006 no finite cell differs by
    #: between 1e-9 and 1e-3 px, and 454,293 of the 457,481 that differ do so by over 1 px --
    #: so this only has to survive a float round-trip, not discriminate a close call.
    _RAW_IDENTITY_PX = 1e-6

    def _is_raw_detection(self, view: int, t: int, point: int, xy) -> bool:
        """Is the displayed pixel at this cell what the detector actually produced?

        ``False`` means a later stage wrote over it -- in practice the reprojected 3D seeded
        into a contralateral cell. When there is no ``pose2d`` array to compare against the
        answer is ``True``: such a file has no separate detector stage that could have been
        overwritten, so its displayed pixels are the detections.
        """
        raw = self.raw_pts2d
        if raw is None:
            return True
        r = raw[view, t, point]
        if not np.all(np.isfinite(r)):
            return False  # the detector did not fire here, so this pixel came from elsewhere
        return bool(
            np.linalg.norm(np.asarray(xy, dtype=float) - r) <= self._RAW_IDENTITY_PX
        )

    def confirm(
        self,
        targets,
        sources: str = "all",
        frame: int | None = None,
    ) -> bool:
        """Promote suggested positions to GT for many ``(view, point)`` targets at once.

        ``sources`` selects which suggestion to snapshot: ``"predictions"`` (the
        displayed per-view pixel), ``"projections"`` (the current 3D reprojected), or
        ``"all"`` (prediction where present, else projection). A snapshotted pixel is
        tagged by where it actually came from, not by which source asked for it -- see
        :meth:`_is_raw_detection`. Occluded or already-GT views are
        left untouched. One undo step; the 3D re-derives once. Returns whether anything
        changed.
        """
        t = self._resolve_frame(frame)
        saved_redo = list(self._redo)
        self._record_undo(t, None, coalesce=False)
        want_pred = sources in ("all", "predictions")
        want_proj = sources in ("all", "projections")
        proj = self.display_pts3d_projected(t) if want_proj else None
        changed = False
        for view, point in targets:
            if (
                self.labels.occluded[view, t, point]
                or self.labels.has_gt[view, t, point]
            ):
                continue
            xy = prov = None
            pred = self.result.pts2d[view, t, point]
            if want_pred and np.all(np.isfinite(pred)):
                # The COORDINATE is the displayed one: bulk confirm means "I looked at these
                # dots and they are right", so storing anything other than the dot the operator
                # saw would record a position they never approved.
                #
                # The PROVENANCE, though, must not claim more than it knows. ``result.pts2d`` is
                # the most-derived stage, and in a directory prepared for contralateral
                # labeling the reprojected 3D has been written *over* the detector's pixels --
                # measured on scape_Fly4_006: 457,481 of 1,068,256 finite cells (43%) differ
                # from ``pose2d/points``, median 14.9 px, p90 161 px. Calling those
                # CONFIRMED_PREDICTION would be false, and it matters because that is the one
                # provenance ``labels-export`` keeps unconditionally, so geometry would enter
                # training labeled as detector evidence.
                #
                # So the tag follows the pixel's actual origin: unchanged from ``raw_pts2d`` ->
                # the detector really did say this; overwritten -> it is reprojected geometry,
                # tagged CONFIRMED_PROJECTION like any other reprojection.
                prov = (
                    Provenance.CONFIRMED_PREDICTION
                    if self._is_raw_detection(view, t, point, pred)
                    else Provenance.CONFIRMED_PROJECTION
                )
                xy = pred
            if (
                xy is None
                and proj is not None
                and np.all(np.isfinite(proj[view, point]))
            ):
                xy, prov = proj[view, point], Provenance.CONFIRMED_PROJECTION
            if xy is not None:
                self.labels.set_gt(view, t, point, xy, provenance=prov)
                changed = True
        if changed:
            self._invalidate_frame3d(t)
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
        occluded = self.labels.occluded
        if not any(
            has_gt[view, t, point] or occluded[view, t, point]
            for view, point in targets
        ):
            return
        self._record_undo(t, None, coalesce=False)
        for view, point in targets:
            self.labels.clear_view(view, t, point)
        self._invalidate_frame3d(t)
        self._invalidate_nmf(t)

    def occlude_targets(self, targets, frame: int | None = None) -> None:
        """Occlude many ``(view, point)`` cells at ``frame`` in one undo step.

        The batched counterpart of :meth:`toggle_invisible`, but a *set* not a toggle:
        every target is flagged occluded (dropping any GT there), and the frame's 3D
        re-derives once. Reversal is :meth:`reset_targets` / undo. Requires 3D (an
        occluded view only means something when there is a solve to drop it from). A
        no-op batch (empty, or every target already occluded) does nothing: no undo
        entry, no cleared redo, nothing marked dirty (mirrors :meth:`confirm`).
        """
        if self.result.pts3d is None:
            return
        targets = list(targets)
        if not targets:
            return
        t = self._resolve_frame(frame)
        occluded = self.labels.occluded
        if not any(not occluded[view, t, point] for view, point in targets):
            return
        self._record_undo(t, None, coalesce=False)
        for view, point in targets:
            self.labels.set_occluded(view, t, point, True)
        self._invalidate_frame3d(t)
        self._invalidate_nmf(t)
