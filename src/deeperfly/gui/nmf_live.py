"""Live NeuroMechFly re-fit for the correction GUI.

The pipeline fits the NMF model once, from the triangulated pose. In the editor the
operator corrects the 3D "latent skeleton" frame by frame, so the overlaid model
should follow: :class:`NmfLive` re-solves the inverse kinematics for a single frame
from the *current* (corrected) 3D points; the caller poses the bundled mesh from that
fit and hands the vertices to the browser, which renders them on the GPU (no
server-side rasterization).

The recording-level registration is fixed once from the original full-sequence fit
(the body frame + per-leg segment lengths in :class:`~deeperfly.inverse_kinematics.align.Alignment`,
and the coxa-to-model similarity), so a per-frame refit is just the cheap bounded
least-squares solve for that frame's legs + head + abdomen -- fast enough to run on
demand as the operator edits.
"""

from __future__ import annotations

import logging

import numpy as np

from ..inverse_kinematics.align import body_alignment
from ..inverse_kinematics.articulation import (
    Articulation,
    body_similarity,
    load_articulation,
)
from ..inverse_kinematics.core import solve_chain, solve_leg
from ..inverse_kinematics.template import KinematicTemplate
from ..results import PoseResult

log = logging.getLogger("deeperfly")

__all__ = ["NmfLive"]


class NmfLive:
    """Re-fits the NMF model for one edited frame at a time.

    Built once per session from the full result; :meth:`refit` solves a frame's
    angles + model joints from its (corrected) 3D points. The caller poses the mesh
    from that fit (see :meth:`deeperfly.gui.state.EditorState.nmf_posed_verts`).

    ``template`` / ``articulation`` come from the run config (read from the snapshot
    beside ``results.h5``), so the editor re-fits with the *same* model the pipeline
    used -- restricted legs, custom joint bounds, ``fit_head``/``fit_abdomen``, and
    custom marker placement all carry over. They default to the packaged NeuroMechFly
    model for library callers that have no config.
    """

    def __init__(
        self,
        result: PoseResult,
        *,
        template: KinematicTemplate | None = None,
        articulation: Articulation | None = None,
        max_nfev: int = 100,
        regularization: float = 0.01,
    ) -> None:
        self.skeleton = result.skeleton
        self.index = {n: i for i, n in enumerate(self.skeleton.point_names)}
        self.template = template or KinematicTemplate.load("neuromechfly")
        self.articulation = articulation or load_articulation()
        self.max_nfev = max_nfev
        self.regularization = regularization
        # Head/abdomen size relative to the model: the pipeline's data estimate
        # (fixed per recording, like the body registration below). The caller poses
        # the mesh at these scales (EditorState.nmf_posed_verts).
        self.chain_scales = dict(result.nmf_chain_scales)

        pts3d = np.asarray(result.pts3d, dtype=float)
        # Registration is fixed from the original full-sequence fit, so a per-frame
        # refit only re-solves that frame's joint angles (cheap + temporally stable).
        self.alignment = body_alignment(pts3d, self.skeleton, self.template)
        self.body_sim = self._fit_body_sim(pts3d)
        self.angle_names = [n for leg in self.template.legs for n in leg.dof_names] + [
            n for ch in self.articulation.chains for n in ch.dof_names
        ]

    def _fit_body_sim(self, pts3d: np.ndarray):
        cols = [self.index.get(p, -1) for p in self.articulation.coxa_points]
        n = pts3d.shape[0]
        coxae = np.stack(
            [pts3d[:, c] if c >= 0 else np.full((n, 3), np.nan) for c in cols], axis=1
        )
        with np.errstate(all="ignore"):
            measured = np.nanmedian(coxae, axis=0)
        return body_similarity(self.articulation.coxa_neutral, measured)

    def refit(self, pts3d_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit one frame's NMF model from its (corrected) 3D points.

        Parameters
        ----------
        pts3d_t
            The frame's 3D keypoints ``(P, 3)`` in world coordinates (NaN allowed).

        Returns
        -------
        model_pts3d : np.ndarray
            ``(P, 3)`` fitted model joints (world; skeleton order; NaN where unfit).
        angles : np.ndarray
            ``(D,)`` fitted joint angles, ordered as :attr:`angle_names`.
        """
        pts = np.asarray(pts3d_t, dtype=float)[None]  # (1, P, 3)
        n_points = pts.shape[1]
        model = np.full((n_points, 3), np.nan)
        cols: list[np.ndarray] = []

        for leg in self.template.legs:
            if leg.name not in self.alignment.leg_origin:
                cols.append(np.full(sum(leg.dof_counts), np.nan))
                continue
            angles, world = solve_leg(
                pts, self.index, leg, self.alignment, max_nfev=self.max_nfev
            )
            cols.append(angles[0])
            for j, name in enumerate(leg.point_names):
                if name in self.index:
                    model[self.index[name]] = world[0, j]

        for chain in self.articulation.chains:
            if self.body_sim is None:
                cols.append(np.full(len(chain.dof_names), np.nan))
                continue
            angles, world = solve_chain(
                pts,
                self.index,
                chain,
                self.body_sim,
                max_nfev=self.max_nfev,
                regularization=self.regularization,
                scale=self.chain_scales.get(chain.name, 1.0),
            )
            cols.append(angles[0])
            for m, name in enumerate(chain.marker_names):
                if name in self.index:
                    model[self.index[name]] = world[0, m]

        return model, np.concatenate(cols) if cols else np.zeros(0)
