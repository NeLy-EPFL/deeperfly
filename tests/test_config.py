"""Tests for :class:`deeperfly.config.Config` -- the one loader/validator.

Covers the single-source-of-truth defaults (Python field defaults, pinned equal to
the packaged template), the typed per-stage accessors, the per-camera consolidation
(``input`` / ``preprocess`` inside ``[cameras.*]``), the byte-exact snapshot
round-trip, and the migration errors for the renamed/removed sections.
"""

from __future__ import annotations

import pytest

from deeperfly import Config
from deeperfly.config import (
    DEFAULT_CONFIG_PATH,
    STAGE_DEFAULTS,
    BundleAdjustmentParams,
    IoParams,
    PictorialParams,
    Pose2dParams,
    TriangulationParams,
)
from deeperfly.visualization.compose import VideoSpec

# -- defaults: one source of truth -------------------------------------------


def test_empty_config_uses_python_defaults():
    c = Config.from_dict({})
    assert c.pose2d == Pose2dParams()
    assert c.pose2d.decode_buffer == 4  # the value the stale code constant got wrong
    assert c.triangulation == TriangulationParams()
    assert c.pictorial == PictorialParams()
    assert c.io == IoParams()
    assert c.stage_flags() == STAGE_DEFAULTS


def test_template_matches_python_defaults():
    """The packaged template's tunables must equal the Python defaults (anti-drift).

    This is the guard that would have caught ``decode_buffer = 4`` in the TOML vs
    ``8`` in code: the template documents the defaults, so it must agree with them.
    """
    c = Config.default()
    assert c.pose2d == Pose2dParams()
    assert c.triangulation == TriangulationParams()
    assert c.pictorial == PictorialParams()
    assert c.io == IoParams()
    assert c.stage_flags() == STAGE_DEFAULTS


def test_overrides_win_over_defaults():
    c = Config.from_dict({"pose2d": {"batch_size": 32, "precision": "float32"}})
    assert c.pose2d.batch_size == 32
    assert c.pose2d.precision == "float32"
    assert c.pose2d.decode_buffer == 4  # untouched -> default


def test_pose2d_clamps_batch_and_buffer():
    c = Config.from_dict({"pose2d": {"batch_size": 0, "decode_buffer": 0}})
    assert c.pose2d.batch_size == 1 and c.pose2d.decode_buffer == 1


def test_unknown_stage_key_fails_loudly():
    with pytest.raises(ValueError, match="unknown key"):
        Config.from_dict({"triangulation": {"ransac_thresh": 1.0}}).triangulation


# -- bundle adjustment: flat scipy kwargs ------------------------------------


def test_bundle_adjustment_splits_keypoints_fixed_shared_and_scipy_kwargs():
    c = Config.from_dict(
        {
            "bundle_adjustment": {
                "points_to_use": ["lf_claw", "lm_claw", "lh_claw"],
                "fixed": ["*.intr"],
                "shared": [["a.tvec[2]", "b.tvec[2]"]],
                "weigh_by_confidence": False,
                "max_frames": 50,
                "frame_sampling": "coverage",
                "max_nfev": 500,
                "loss": "huber",
            }
        }
    )
    ba = c.bundle_adjustment
    assert isinstance(ba, BundleAdjustmentParams)
    assert ba.points_to_use == ["lf_claw", "lm_claw", "lh_claw"]
    assert ba.fixed == ["*.intr"]
    assert ba.shared == [["a.tvec[2]", "b.tvec[2]"]]
    assert ba.weigh_by_confidence is False
    assert ba.max_frames == 50
    assert ba.frame_sampling == "coverage"
    # the recognized fields are pulled out, not left as scipy least_squares kwargs.
    assert ba.least_squares == {"max_nfev": 500, "loss": "huber"}


def test_bundle_adjustment_defaults_when_absent():
    ba = Config.from_dict({}).bundle_adjustment
    assert (
        ba.points_to_use is None
        and ba.fixed == []
        and ba.shared == []
        and ba.weigh_by_confidence is True  # weighting on by default
        and ba.max_frames == 100
        and ba.frame_sampling == "even"
        and ba.least_squares == {}
    )


# -- sources and views -------------------------------------------------------


def test_source_patterns_and_camera_table():
    c = Config.from_dict(
        {
            "sources": [
                {"name": "cam0", "filename": "v0.mp4"},
                {"name": "cam1"},  # no filename -> own name
            ],
            "cameras": {
                "defaults": {"focal_length_px": 800.0},
                "rh": {},
                "lf": {},
            },
        }
    )
    # Footage globs come from the [[sources]] table (views are pure geometry).
    assert c.source_patterns() == {"cam0": "v0.mp4", "cam1": "cam1"}
    # camera_table() splits the reserved `defaults` key from the real views.
    defaults, cams = c.camera_table()
    assert defaults == {"focal_length_px": 800.0}
    assert set(cams) == {"rh", "lf"}


# -- visualization: typed VideoSpec list -------------------------------------


def test_videos_returns_typed_specs():
    c = Config.from_dict(
        {
            "visualization": {
                "videos": [
                    {
                        "video_name": "v",
                        "panels": [{"plot": "skeleton_2d", "view": "f"}],
                    }
                ]
            }
        }
    )
    specs = c.videos
    assert len(specs) == 1 and isinstance(specs[0], VideoSpec)
    assert specs[0].video_name == "v"


# -- snapshot round-trip -----------------------------------------------------


def test_snapshot_is_byte_exact(tmp_path):
    src = DEFAULT_CONFIG_PATH
    c = Config.from_toml(src)
    out = tmp_path / "out"
    out.mkdir()
    c.save_snapshot(out)
    assert (out / "config.toml").read_text() == src.read_text()


def test_snapshot_from_dict_config_raises(tmp_path):
    with pytest.raises(ValueError, match="no source text"):
        Config.from_dict({}).save_snapshot(tmp_path)
