"""The run configuration: one TOML file, one :class:`~deeperfly.config.core.Config`.

:mod:`~deeperfly.config.core` is the class itself and the frozen ``*Params`` dataclasses
it hands out -- the single source of truth for every default, written exactly once.
:mod:`~deeperfly.config.schema` describes that surface *from the code*, so the ``deeperfly
config`` commands and the editor's settings forms cannot drift from it.
"""

from __future__ import annotations

from . import schema  # noqa: E402  (schema reads .core, so core comes first)
from .core import (
    DEFAULT_CONFIG_PATH,
    MIN_VIEWS_FOR_3D,
    SKELETON_PRESET_DIR,
    STAGE_DEFAULTS,
    STAGES,
    AnnotationParams,
    AutoCropParams,
    BundleAdjustmentParams,
    Config,
    EksParams,
    GuiParams,
    InverseKinematicsParams,
    IoParams,
    PictorialParams,
    Pose2dParams,
    PostprocessParams,
    StaticPointsParams,
    SymmetrizeParams,
    TriangulationParams,
    default_skeleton_spec,
    skeleton_presets,
)
from .params import IK_KEYS

__all__ = [
    "schema",
    "Config",
    "Pose2dParams",
    "AutoCropParams",
    "TriangulationParams",
    "EksParams",
    "PostprocessParams",
    "StaticPointsParams",
    "SymmetrizeParams",
    "PictorialParams",
    "IoParams",
    "BundleAdjustmentParams",
    "InverseKinematicsParams",
    "AnnotationParams",
    "GuiParams",
    "STAGES",
    "STAGE_DEFAULTS",
    "IK_KEYS",
    "MIN_VIEWS_FOR_3D",
    "DEFAULT_CONFIG_PATH",
    "SKELETON_PRESET_DIR",
    "skeleton_presets",
    "default_skeleton_spec",
]
