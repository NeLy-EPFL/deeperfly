"""Live NeuroMechFly re-fit for the annotation editor.

The pipeline fits the model once, over the whole recording. In the editor the operator
authors ground-truth 2D labels and the 3D pose is re-derived from them frame by frame, so
the overlaid model has to follow: :class:`NmfLive` re-solves the inverse kinematics for a
*single* frame from that frame's current 3D points, and the caller poses the bundled mesh
from the result (see :meth:`deeperfly.gui.state.EditorState.nmf_posed_verts`).

It re-solves on **the pipeline's own body plan**, read back from ``results.h5``. That
matters for more than tidiness: the plan carries the measured segment lengths and the coxa
registration, and the pipeline measures those from a pose whose static keypoints have
already been collapsed to one position (the ``[postprocess]`` chain, and/or
``[inverse_kinematics].constant_points``) -- so re-deriving them here from the editor's
un-collapsed pose would quietly fit a slightly different animal than the stored result
did.

Only the frame's joint angles are re-solved, which is cheap: about half a millisecond,
against the tens of milliseconds the previous per-limb scipy fit took (the reason the
editor still skips the overlay mid-drag).

**The fit is a pure function of the frame and its labels.** QuickIK's ``Solver.solve``
mutates the state it is handed, and the editor solves arbitrary frames in arbitrary order
as the operator scrubs and undoes, so a single long-lived solver state would make the
answer depend on history: edit frame 500, undo, redo, and the same labels could give
different angles. The seed here depends only on the frame -- the pipeline's stored angles
for it, else the plan's neutral pose -- which is both deterministic and a better start
than either alternative, since the stored fit is already the answer for the unedited pose.
"""

from __future__ import annotations

import copy
import json
import logging

import numpy as np
from jaxtyping import Float

from ..inverse_kinematics import observations, unfittable_branches
from ..inverse_kinematics._quickik import require_quickik
from ..inverse_kinematics.align import body_alignment
from ..inverse_kinematics.articulation import Articulation, load_articulation
from ..inverse_kinematics.bodyplan import BodyPlan
from ..inverse_kinematics.template import KinematicTemplate
from ..results import PoseResult

log = logging.getLogger("deeperfly")

__all__ = ["NmfLive"]


class NmfLive:
    """Re-fits the NeuroMechFly model for one edited frame at a time.

    Built once per session from the full result; :meth:`refit` solves a frame's angles
    and model joints from its (corrected) 3D points.

    ``template`` / ``articulation`` come from the run config (read from the snapshot
    beside ``results.h5``), and are used only when the result predates stored body plans
    and one has to be rebuilt.

    Raises
    ------
    deeperfly.inverse_kinematics.MissingQuickIK
        If the optional ``deeperfly[ik]`` extra is not installed. The caller treats that
        as "no live re-fit" and falls back to the stored fit, so the editor still opens
        and still draws the overlay.
    """

    def __init__(
        self,
        result: PoseResult,
        *,
        template: KinematicTemplate | None = None,
        articulation: Articulation | None = None,
        n_iterations: int | None = None,
        neutral_weight: float | None = None,
        damping: float | None = None,
        fixed_body: bool | None = None,
        symmetric_segments: bool | None = None,
    ) -> None:
        from ..config import InverseKinematicsParams

        self._quickik = require_quickik()
        defaults = InverseKinematicsParams()
        self.skeleton = result.skeleton
        self.result = result
        self.plan = self._load_plan(
            result, template, articulation, fixed_body, symmetric_segments
        )
        self.kinematics = self.plan.kinematics()
        self.angle_names = list(self.plan.angle_names)
        self.chain_scales = dict(self.plan.chain_scales)
        self._spec = copy.deepcopy(self.plan.plan)
        self._config = self._quickik.SolverConfig(
            n_iterations=int(
                defaults.n_iterations if n_iterations is None else n_iterations
            ),
            neutral_weight=float(
                defaults.neutral_weight if neutral_weight is None else neutral_weight
            ),
            position_tolerance=float(defaults.position_tolerance),
            angle_tolerance=float(defaults.angle_tolerance),
            damping=float(defaults.damping if damping is None else damping),
        )
        # The stored angles are the seed for an unedited frame, so the live overlay
        # agrees with the rendered one until a label actually moves.
        self._stored = self._stored_angles(result)

    # -- setup ---------------------------------------------------------------

    def _load_plan(
        self, result, template, articulation, fixed_body, symmetric_segments=None
    ) -> BodyPlan:
        """The pipeline's stored plan, else one rebuilt from the config and the pose."""
        from ..config import InverseKinematicsParams
        from ..inverse_kinematics import _plan_for

        if result.nmf_body_plan:
            try:
                return BodyPlan.from_json(result.nmf_body_plan, result.skeleton)
            except Exception:
                log.warning(
                    "the stored body plan could not be read; rebuilding it from the "
                    "config (the live overlay may differ slightly from the rendered "
                    "one)",
                    exc_info=True,
                )
        pts3d = np.asarray(result.pts3d, dtype=float)
        tpl = template or KinematicTemplate.load("neuromechfly")
        art = articulation if articulation is not None else load_articulation()
        defaults = InverseKinematicsParams()
        fixed = defaults.fixed_body if fixed_body is None else fixed_body
        symmetric = (
            defaults.symmetric_segments
            if symmetric_segments is None
            else symmetric_segments
        )
        return _plan_for(
            pts3d,
            result.skeleton,
            tpl,
            body_alignment(pts3d, result.skeleton, tpl, symmetric_segments=symmetric),
            art,
            fixed,
        )

    def _stored_angles(self, result) -> np.ndarray | None:
        """``(T, D)`` the pipeline's angles reordered onto this plan's DOF columns.

        ``None`` when the result carries none, or when its names do not line up with the
        plan's -- in which case every frame simply seeds from the neutral pose.
        """
        if result.nmf_angles is None or not result.nmf_angle_names:
            return None
        col = {name: i for i, name in enumerate(result.nmf_angle_names)}
        if not all(name in col for name in self.angle_names):
            return None
        return np.asarray(result.nmf_angles, dtype=float)[
            :, [col[name] for name in self.angle_names]
        ]

    # -- solving -------------------------------------------------------------

    def refit(
        self, pts3d_t: Float[np.ndarray, "P 3"], frame: int | None = None
    ) -> tuple[Float[np.ndarray, "P 3"], Float[np.ndarray, "D"]]:
        """Fit one frame's model from its (corrected) 3D points.

        Parameters
        ----------
        pts3d_t
            The frame's 3D keypoints ``(P, 3)`` in world coordinates (NaN allowed).
        frame
            Which frame this is, used only to pick the seed. ``None`` seeds from the
            plan's neutral pose.

        Returns
        -------
        model_pts3d : np.ndarray
            ``(P, 3)`` fitted model joints (world; skeleton order; NaN where unfit).
        angles : np.ndarray
            ``(D,)`` fitted joint angles, ordered as :attr:`angle_names`.
        """
        pts = np.asarray(pts3d_t, dtype=float)[None]  # (1, P, 3)
        positions, weights = observations(pts, self.plan, None)
        tree = self._quickik.KinematicTree.from_json_str(self._seeded_plan(frame))
        state = self._quickik.State.neutral_pose(tree)
        self._quickik.Solver(tree, self._config).solve(
            state, self._observation_list(positions[0], weights[0])
        )
        angles = np.asarray(state.dof_angles, dtype=float)[None]  # (1, D)

        branches = np.asarray(self.plan.dof_branch)
        for branch, bad in unfittable_branches(weights, self.plan).items():
            if bad[0]:
                angles[0, branches == branch] = np.nan

        world = self.plan.to_world(
            self.kinematics.joint_positions(angles, state.root_pos, state.root_rot)
        )
        model = np.full((pts.shape[1], 3), np.nan)
        rows = self.plan.joint_row
        tracked = rows >= 0
        model[rows[tracked]] = world[0, tracked]
        return model, angles[0]

    def _seeded_plan(self, frame: int | None) -> str:
        """The plan JSON with each DOF's ``neutral`` set to this frame's seed.

        QuickIK's Python ``State`` is read-only -- ``State.neutral_pose`` is its only
        constructor -- so the plan's own neutral values are the only way in to seed a
        solve. That also re-points the toward-neutral prior at the seed, which is what we
        want here: hold the pipeline's pose except where the operator's labels pull away
        from it. Rebuilding the plan costs about a tenth of a millisecond.
        """
        seed = None
        if (
            frame is not None
            and self._stored is not None
            and 0 <= frame < self._stored.shape[0]
        ):
            row = self._stored[frame]
            # Per-DOF, not all-or-nothing. A single permanently-NaN limb -- an amputated
            # leg, an unfittable branch -- would otherwise veto the warm start for the
            # WHOLE body, so every live re-fit would restart from the limit-midpoint
            # neutral pose and the overlay would silently disagree with the rendered fit
            # everywhere, not just on the missing limb.
            if np.isfinite(row).any():
                seed = row
        if seed is None:
            return self.plan.to_json()
        d = 0
        for joint in self._spec["joints"]:
            for dof in joint["dofs"]:
                lo, hi = dof["limits"]
                # A NaN DOF keeps the plan's own neutral; only the fitted ones warm-start.
                if d < seed.shape[0] and np.isfinite(seed[d]):
                    dof["neutral"] = float(np.clip(seed[d], lo, hi))
                d += 1
        return json.dumps(self._spec, separators=(",", ":"))

    def _observation_list(self, positions: np.ndarray, weights: np.ndarray) -> list:
        """One ``KeypointObservation`` per plan joint, in the plan's own order."""
        obs = self._quickik.KeypointObservation
        return [
            obs.position_3d([float(v) for v in positions[i]], float(weights[i]))
            if weights[i] > 0
            else obs.missing()
            for i in range(positions.shape[0])
        ]
