"""Headless visualization backends for 2D / 3D pose overlays.

Two interchangeable backends, imported on demand so neither heavy dependency is
pulled in until it is actually used:

- :mod:`deeperfly.viz.matplotlib` -- figure / ``Axes`` drawing for 3D skeletons
  and montages (needs the ``viz`` extra; ``matplotlib``).
- :mod:`deeperfly.viz.opencv` -- fast overlays drawn straight into image arrays
  with ``cv2``, with painter's-algorithm depth ordering for 3D (needs the
  ``opencv`` extra).
- :mod:`deeperfly.viz.compose` -- a config-driven panel compositor that layers
  the OpenCV primitives into video frames (one MP4 per ``[[viz.videos]]`` entry).

For backward compatibility the matplotlib drawing helpers (``plot_skeleton_2d``,
``plot_skeleton_3d``, ``overlay_grid``, ``limb_colors``, ``apply_background``,
``BACKGROUNDS``) are also reachable directly on this package; they live in
:mod:`deeperfly.viz.matplotlib`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type checkers / IDEs only -- no runtime import
    from . import compose as compose
    from . import matplotlib as matplotlib
    from . import opencv as opencv

_SUBMODULES = frozenset({"matplotlib", "opencv", "compose"})

# Names that used to live in the flat ``viz.py`` (now ``viz.matplotlib``), kept
# importable as ``viz.<name>`` so existing callers do not break.
_MPL_FORWARD = frozenset(
    {
        "plot_skeleton_2d",
        "plot_skeleton_3d",
        "overlay_grid",
        "limb_colors",
        "apply_background",
        "BACKGROUNDS",
    }
)


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    if name in _MPL_FORWARD:
        return getattr(importlib.import_module(f"{__name__}.matplotlib"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_SUBMODULES | _MPL_FORWARD)
