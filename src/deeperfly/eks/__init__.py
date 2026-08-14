"""The Ensemble Kalman Smoother (EKS), in deeperfly's own JAX and camera model.

A post-processing stage for a calibrated multi-view recording: it replaces the
per-frame, per-view argmax with one temporally-coherent 3D trajectory per
keypoint, fitted jointly to every view.

This is the *nonlinear, geometric* EKS of Lightning Pose 3D (Aharon, Whiteway et
al., 2026), which contributes two things over the linear smoother of the 2024
Lightning Pose paper: the observation model becomes the real camera projection
(so the latent state is a 3D point rather than a PCA score, and the rig's
geometry does the cross-view work), and a variance-inflation step tests each
view against its cross-view consensus and down-weights the ones that disagree.

Why this is a reimplementation rather than a dependency
-------------------------------------------------------
The reference package (``ensemble-kalman-smoother``) pins ``jax<=0.4.36``;
deeperfly requires ``jax>=0.10.1``, so the two cannot share an environment at
all. Reimplementing also removes a subtler hazard: the reference drives the
projection through ``aniposelib``, which would give the smoother a *second*
camera model beside the one deeperfly triangulates, bundle-adjusts and draws
overlays with. Here the observation function is
:func:`deeperfly.geometry.project_full_one` -- the same projection, distortion
included -- so the smoother's geometry cannot drift from the run's.

Deliberate differences from the reference implementation
--------------------------------------------------------
- **Missing observations are exact.** deeperfly's 2D is NaN wherever a
  ``(view, point)`` pair is unobserved, which is the common case on a rig where
  each camera sees one side of the animal. Those cells are marginalized out of
  the filter rather than imputed; the reference has no NaN handling on its
  nonlinear path.
- **The inflation uses the projection Jacobian, not factor analysis.** The
  reference *learns* a linear cross-view model because it must also serve
  uncalibrated setups. With a rig in hand the exact loading matrix is already
  known, so the paper's formula is evaluated rather than approximated.
- **The smoothing parameter is fitted by golden-section search** on ``log(s)``
  instead of Adam, and against the time-varying observation noise rather than a
  time-median stand-in. Same objective (the EKF marginal likelihood), same
  bounds, one dimension, no step size to tune.

Where the temporal prior stops helping
--------------------------------------
The latent is a *position* random walk, and the measurement update is a single
Gauss-Newton step -- both as published. Together they mean the smoother's
accuracy gain is confined to keypoints whose per-frame motion is at or below the
detector's localization error: a tethered fly's body and proximal joints, not a
claw mid-swing. On a target moving several times the noise floor the fitted
per-keypoint smoothing parameter backs the prior off, but a small lag survives
(~10% on the median in
``tests/test_eks.py::test_a_fast_target_costs_little``), and raising the
smoothing parameter does not remove it. The outlier repair and the de-jittering
are unaffected -- those are where the method pays on real data. Anyone wanting
the fast keypoints back should look at an iterated update or a
constant-velocity latent, both of which are departures from the paper.

At ensemble size one
--------------------
Only two of the three mechanisms need an ensemble-of-one to work: the geometric
observation model needs a calibration, and the variance inflation needs two
*views*. The ensemble spread does need several models -- with one, the
observation noise degrades to ``1 / confidence``, a prior rather than a
measurement. Pass several members to :func:`smooth` to activate it. Note that
the ensembles in this literature are the same recipe trained differently (data
subsets, initialization, ordering), not different architectures.
"""

from __future__ import annotations

from .ensemble import ensemble_statistics, inflate_variances
from .smoother import EksResult, smooth

__all__ = [
    "smooth",
    "EksResult",
    "ensemble_statistics",
    "inflate_variances",
]
