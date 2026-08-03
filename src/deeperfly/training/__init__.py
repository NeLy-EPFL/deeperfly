"""Training the 2D detector in-tree (``deeperfly[train]``).

Per fork **F3c** of ``docs/project-system-plan.md``, deeperfly grows its own trainer rather
than shelling out to a research repo. What lives here is the part that must be *general*:

:mod:`deeperfly.training.heatmaps`
    The numeric contract -- fractional Gaussian targets, the DARK sub-pixel decode, and the
    masked loss. These three have to stay stable across the whole label -> train -> predict ->
    correct loop, because a change in any of them shifts every reported pixel error without
    touching the loss curve.

**What deliberately does not live here.** The ``dfpose`` research trainer carries a hardcoded
``(camera, (H, W)) -> crop`` table for three specific rig geometries, with measurement notes
about named recordings, and it refuses an unlisted pair on purpose. That is lab policy about
particular cameras, not library code -- and deeperfly already has the general form of it:
per-camera ``preprocess`` in the project's ``rig.toml``. So the crops come from the rig, and
the table stays where it belongs.

The heavier dependencies (``timm`` and friends) belong behind the ``deeperfly[train]`` extra,
mirroring how ``quickik`` is handled for inverse kinematics: a plain install must still run
``gui``, ``run`` and ``calibrate``, and a ``results.h5`` a trained model produced must render
without the trainer present.
"""

from __future__ import annotations

from .heatmaps import (
    STRIDE,
    masked_heatmap_loss,
    refined_argmax,
    render_gaussian_targets,
)

__all__ = [
    "STRIDE",
    "masked_heatmap_loss",
    "refined_argmax",
    "render_gaussian_targets",
]
