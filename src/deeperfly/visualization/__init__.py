"""Headless visualization for 2D / 3D pose overlays.

Drawing is done with OpenCV (a core dependency), so it stays figure-free and
fast; the two submodules are imported on demand:

- :mod:`deeperfly.visualization.opencv` -- fast overlays drawn straight into image
  arrays with ``cv2``, with painter's-algorithm depth ordering for 3D.
- :mod:`deeperfly.visualization.compose` -- a config-driven panel compositor that layers
  the OpenCV primitives into video frames (one MP4 per
  ``[[visualization.videos]]`` entry).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type checkers / IDEs only -- no runtime import
    from . import compose as compose
    from . import opencv as opencv

_SUBMODULES = frozenset({"opencv", "compose"})


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_SUBMODULES)
