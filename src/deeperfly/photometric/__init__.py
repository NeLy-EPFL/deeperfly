"""Photometric cross-side extrinsics refinement.

Fixes the weakly-constrained left/right relative camera pose (the far legs reproject
1-2 leg-widths off) by chamfer-matching reprojected far-leg bones onto image leg pixels,
refining a single 6-DOF rigid transform between the two camera clusters. See
:func:`refine_extrinsics_photometric` for the entry point and :func:`build_leg_maps` for
the image preprocessing.
"""

from __future__ import annotations

from .distance import truncated_dt
from .leg_response import leg_response, paint_out_bones, ridge_response
from .maps import build_leg_maps
from .objective import (
    compensate_left,
    far_leg_bones,
    left_point_indices,
    rigid,
)
from .refine import CrossSideRefinement, build_problem, refine_extrinsics_photometric

__all__ = [
    "leg_response",
    "ridge_response",
    "paint_out_bones",
    "truncated_dt",
    "build_leg_maps",
    "rigid",
    "compensate_left",
    "far_leg_bones",
    "left_point_indices",
    "build_problem",
    "refine_extrinsics_photometric",
    "CrossSideRefinement",
]
