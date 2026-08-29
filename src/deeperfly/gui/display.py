"""What the editor *draws*: the derived per-view and 3D positions, per frame.

A mixin on :class:`~deeperfly.gui.state.EditorState`, split out because it is the one
slice of the editor's model with a clean seam -- nothing here is reached except through
its own entry points, and nothing here mutates a label. Every method answers one
question: given the labels, the detections and the current 3D, *where does this point go
on screen right now?*

That is not the same question as "where is the ground truth", and the distinction is the
whole design: a cell with no GT still has to be drawn somewhere, and the choice between
the reprojection of the point's 3D and its frozen seed
(:attr:`~deeperfly.gui.state.EditorState.nongt_display`) is a display decision that must
never leak back into the labels.
"""

from __future__ import annotations

import logging

import numpy as np
from jaxtyping import Bool, Float

from .solve import solve_point_3d

log = logging.getLogger("deeperfly")


class _DisplayMixin:
    """The derived-display half of :class:`~deeperfly.gui.state.EditorState`."""

    @property
    def detections(self) -> Float[np.ndarray, "V T P 2"]:
        """The detected layer: what the 2D network said, per ``(view, frame, point)``.

        Deliberately :attr:`raw_pts2d` and not ``result.pts2d``. The latter is whatever the
        *most-derived* stage in the file produced, which is a moving target and never the
        detector: the correction chain's pose when there is one, else the smoother's, else
        the triangulation-cleaned observations. Only the last of those is even in the
        detector's pixel space, and in a directory prepared for contralateral labeling it
        has reprojected geometry written *over* the detector's pixels (measured on
        scape_Fly4_006: 43% of finite cells differ from ``pose2d/points``, median 14.9 px).
        Reading it as "the detection" loses the network's opinion exactly where it is most
        needed and makes a drawn detection ambiguous about what it even is.

        Note what this array is *not* a source of any more: rejection. A cleaned 2D is NaN
        where the pipeline threw a peak out, but a smoothed or corrected one is dense -- it
        fills every cell by construction -- so ``result.pts2d`` says nothing about which
        observations survived triangulation. That question is ``triangulation/points``'s to
        answer, and any caller wanting it must read that group rather than infer it from a
        NaN here.

        Falls back to ``result.pts2d`` only when there is no ``pose2d`` group to read --
        such a file has no separate detector stage, so its points *are* the detections.
        """
        return self.raw_pts2d if self.raw_pts2d is not None else self.result.pts2d

    def display_pts2d(self, frame: int | None = None) -> Float[np.ndarray, "V P 2"]:
        """The per-view 2D to draw for ``frame``: GT over the detector prediction.

        A GT view shows its pixel; every other view shows the prediction, or ``NaN`` where
        the detector fired nothing -- the front-end draws the reprojection ghost there
        instead. A point declared absent is ``NaN`` in *every* view: the detector still
        fires somewhere on an amputated limb (an argmax decode always emits a peak), and
        showing that peak would invite the operator to confirm it.

        The **hidden** flag is deliberately not consulted. It used to suppress the cell's
        detection here, which made the flag a display switch as well as a training one, and
        that cost the operator the position: hiding a cell -- or worse, hiding a whole frame
        with ``a`` then ``e`` -- deleted the only pixel they had to drag, and a joint whose
        3D was unsolvable then had no ghost to fall back on either, so it vanished from every
        canvas with no handle left to select it back. A whole layer of placeholder seeds
        exists because of that. Whether a cell is supervised says nothing about where its
        keypoint is, so it no longer moves or removes one: :meth:`occluded_mask` carries the
        flag to the front-end, which draws it as its own mark over the marker that is there.
        """
        t = self._resolve_frame(frame)
        gt = self.labels.gt[:, t]  # (V, P, 2)
        has = self.labels.has_gt[:, t]  # (V, P)
        absent = self.labels.absent_at(t)[None, :]  # (1, P) -> broadcasts over views
        pred = self.detections[:, t]  # (V, P, 2)
        shown = np.where(has[..., None], gt, pred)
        return np.where(absent[..., None], np.nan, shown)

    def display_pts2d_refine(
        self, frame: int | None = None
    ) -> Float[np.ndarray, "V P 2"] | None:
        """The per-view 2D drawn in Edit 3D: the derived 3D reprojected into every
        view, with each GT view overridden by its authored pixel, or ``None``.

        GT views hold their pixel (they generally do not all agree with one 3D
        point); every other view follows the reprojection. An absent point has no 3D to
        reproject, so it stays ``NaN`` here too. The **hidden** flag is not read: it decides
        what the loss trains on, never where a joint is drawn.
        """
        proj = self.display_pts3d_projected(frame)
        if proj is None:
            return None
        t = self._resolve_frame(frame)
        gt = self.labels.gt[:, t]  # (V, P, 2)
        has = self.labels.has_gt[:, t]  # (V, P)
        absent = self.labels.absent_at(t)[None, :]  # (1, P)
        return np.where(absent[..., None], np.nan, np.where(has[..., None], gt, proj))

    def display_instance_pts2d(
        self, frame: int | None = None
    ) -> Float[np.ndarray, "V P 2"] | None:
        """The annotation skeleton to draw for ``frame``, or ``None`` with no instance.

        Two facts per cell and nothing else: a GT pixel shows where the operator put it,
        and everything else shows the position the model derives for it --  the reprojection
        of the point's current 3D by default, or the instance's frozen seed under
        :attr:`nongt_display` ``= "seed"``.

        An absent point is ``NaN`` in every view: it is not on the animal, so the instance
        has no position to offer, and drawing one would invite the operator to confirm it.
        """
        t = self._resolve_frame(frame)
        if not self.has_instance(t):
            return None
        gt = self.labels.gt[:, t]
        has = self.labels.has_gt[:, t]
        base = np.asarray(self.labels.seeds[:, t], dtype=float)
        # Cells the instance has no evidence-backed seed for (see _seed_instance) are drawn
        # from the placeholder chain: invented, but a joint you cannot see is a joint you
        # cannot drag into place.
        gap = ~np.isfinite(base).all(axis=-1)
        if gap.any():
            base = np.where(
                gap[..., None], np.asarray(self.placeholder_pts2d(t), dtype=float), base
            )
        if self.nongt_display == "reprojection":
            proj = self.display_pts3d_projected(t)
            if proj is not None:
                # A point with no 3D at all (never solvable, or absent) keeps its seed, so
                # the instance stays complete and grabbable.
                ok = np.isfinite(proj).all(axis=-1)
                base = np.where(ok[..., None], np.asarray(proj, dtype=float), base)
        absent = self.labels.absent_at(t)[None, :]
        shown = np.where(has[..., None], gt, base)
        return np.where(absent[..., None], np.nan, shown)

    def placeholder_pts2d(
        self, frame: int | None = None, *, window: int = 30
    ) -> Float[np.ndarray, "V P 2"]:
        """Seed positions for cells with no observation of their own, so a GT can
        always be placed -- the layer that guarantees every joint stays draggable.

        A cell with no GT and no detection (the detector never fired, or triangulation
        dropped the point) has no position of
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
        3. the nearest frame (within ``window``) whose raw detector pixel in this view
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
        # Every cell without a position of its own gets a seed. This layer used to carry a
        # second job -- a joint hidden in every view had its detection suppressed AND, being
        # below two usable views, no 3D to reproject either, so the seed was the only thing
        # left to grab. The hidden flag no longer touches a position, so that case cannot
        # arise; what remains is the honest one, a cell the detector never fired for.
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

        edges = np.asarray(self.result.skeleton.edges, dtype=int).reshape(-1, 2)
        lo, hi = max(0, t - window), min(self.n_frames - 1, t + window)
        for v in range(n_views):
            for p in np.nonzero(need[v])[0]:
                out[v, p] = self._seed_position(
                    v, int(p), t, edges, shown, shown_ok, lo, hi
                )
        return out

    def _seed_position(self, v, p, t, edges, shown, shown_ok, lo, hi) -> np.ndarray:
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
        # 3. the nearest frame with a raw detector pixel in this view.
        #
        # This used to try the *cleaned* array as a second chance per frame, from when
        # ``detections`` was ``result.pts2d``. Two things retired that rung. It was already
        # dead -- the refactor to :attr:`raw_pts2d` left both reads pointing at the same
        # array, so the second test could only repeat the first. And ``result.pts2d`` is no
        # longer a cleaned array to consult: with the smoother and the correction chain in
        # the pipeline it is dense, so reinstating the rung literally would make it fire at
        # ``dt == 1`` for every cell and shadow the neighbor/centroid rungs below.
        for dt in range(1, max(t - lo, hi - t) + 1):
            for tt in (t - dt, t + dt):
                if not lo <= tt <= hi:
                    continue
                if np.all(np.isfinite(raw[v, tt, p])):
                    return np.asarray(raw[v, tt, p], dtype=float)
        # 4. the mean of connected skeleton neighbors shown in this view.
        if edges.size:
            nbrs = np.unique(
                np.concatenate([edges[edges[:, 0] == p, 1], edges[edges[:, 1] == p, 0]])
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

    def invented_mask(self, frame: int | None = None) -> Bool[np.ndarray, "V P"]:
        """``(V, P)`` cells whose drawn position the editor invented rather than derived.

        True where the instance has no evidence-backed seed *and* no reprojection, so
        :meth:`display_instance_pts2d` fell back to the placeholder chain -- whose last rungs
        are the mean of a joint's skeleton neighbours, the view centroid, and the image
        centre. Those exist so the joint stays visible and draggable; they are not
        observations, they never reach the solve (:meth:`_point_obs`), and :meth:`confirm`
        skips them rather than authoring them as GT.

        Which leaves one problem this answers: on screen such a joint is indistinguishable
        from a properly triangulated one. With an ipsilateral-only detector every
        contralateral keypoint is in exactly this state, so it is not an edge case -- the
        front end draws these faintly, the way the retired "Unplaced" layer drew its ghosts.
        All ``False`` before an instance exists (nothing is being claimed yet).
        """
        t = self._resolve_frame(frame)
        if not self.has_instance(t):
            return np.zeros((self.n_views, self.n_points), dtype=bool)
        seeded = np.isfinite(self.labels.seeds[:, t]).all(axis=-1)
        proj = self.display_pts3d_projected(t) if self.has_3d else None
        proj_ok = (
            np.zeros_like(seeded)
            if proj is None
            else np.isfinite(np.asarray(proj, dtype=float)).all(axis=-1)
        )
        return ~seeded & ~proj_ok

    def _project_detections(self, t: int) -> Float[np.ndarray, "V P 2"] | None:
        """Triangulate the *detections* at ``t`` and reproject, ignoring every label."""
        if self.result.cameras is None:
            return None
        det = np.asarray(self.detections[:, t], dtype=float)
        nan_gt = np.full((self.n_views, 2), np.nan)
        pts3d = np.stack(
            [
                solve_point_3d(
                    self.result.cameras, nan_gt, det[:, p], None, self.ann, self.tri
                )
                for p in range(self.n_points)
            ]
        )
        return np.asarray(self.result.cameras.project(pts3d[None, :, :]), dtype=float)[
            :, 0
        ]

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

    def display_model_projected(
        self, frame: int | None = None
    ) -> Float[np.ndarray, "V P 2"] | None:
        """The fitted model joints for ``frame`` reprojected into every view."""
        fit = self.model_fit(frame)
        if fit is None or self.result.cameras is None:
            return None
        return np.asarray(self.result.cameras.project(fit[0]))

    def model_fit(
        self, frame: int | None = None
    ) -> tuple[np.ndarray, np.ndarray | None, list[str] | None] | None:
        """The model fit for ``frame``: ``(model_pts3d, angles, angle_names)`` or ``None``.

        Re-solved from the frame's derived 3D pose when a :class:`ModelLive` is
        available (so it tracks edits) and memoized per frame; otherwise the
        pipeline's static fit.
        """
        t = self._resolve_frame(frame)
        cached = self._model_cache.get(t)
        if cached is not None:
            return cached
        out: tuple | None = None
        if self.model_live is not None:
            pts3d = self.display_pts3d(t)
            if pts3d is not None:
                # The frame index only picks the seed, and it is what keeps the re-fit a
                # pure function of (frame, labels) -- see ModelLive's module docstring.
                model, angles = self.model_live.refit(pts3d, t)
                out = (model, angles, self.model_live.angle_names)
        elif self.result.model_pts3d is not None:
            angles = (
                None
                if self.result.model_angles is None
                else self.result.model_angles[t]
            )
            out = (self.result.model_pts3d[t], angles, self.result.model_angle_names)
        if out is not None:
            self._model_cache[t] = out
        return out

    def model_posed_verts(
        self, frame: int | None = None
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """The frame's posed model mesh ``(vertices (Nv, 3), valid_faces (Nf,))``, or ``None``."""
        fit = self.model_fit(frame)
        if fit is None:
            return None
        from ..inverse_kinematics.mesh import load_model_mesh

        model, angles, names = fit
        return load_model_mesh().pose(
            model,
            angles,
            names,
            chain_scales=dict(self.result.model_chain_scales),
            chain_offsets=self.result.model_chain_offsets,
            body_scale=self.result.model_body_scale,
        )
