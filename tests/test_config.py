"""Tests for :class:`deeperfly.config.Config` -- the one loader/validator.

Covers the single-source-of-truth defaults (Python field defaults, pinned equal to
the packaged template), the typed per-stage accessors, the per-camera consolidation
(``input`` / ``preprocess`` inside ``[cameras.*]``), the byte-exact snapshot
round-trip, and the migration errors for the renamed/removed sections.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from deeperfly import Config
from deeperfly.config import (
    DEFAULT_CONFIG_PATH,
    STAGE_DEFAULTS,
    BundleAdjustmentParams,
    InverseKinematicsParams,
    IoParams,
    PictorialParams,
    Pose2dParams,
    TriangulationParams,
)
from deeperfly.visualization.compose import VideoSpec


def _ik_section(text: str) -> str:
    """The ``[inverse_kinematics]`` region of a config TOML, comments included.

    Runs from its leading comment block to the next top-level section that is not one
    of its own sub-tables.
    """
    start = text.index("# Inverse kinematics:")
    rest = re.search(r"(?m)^\[(?!inverse_kinematics)", text[start:])
    assert rest is not None, "no section follows [inverse_kinematics]"
    return text[start : start + rest.start()]


# -- defaults: one source of truth -------------------------------------------


def test_empty_config_uses_python_defaults():
    c = Config.from_dict({})
    assert c.pose2d == Pose2dParams()
    assert c.pose2d.decode_buffer == 4  # the value the stale code constant got wrong
    assert c.triangulation == TriangulationParams()
    assert c.pictorial == PictorialParams()
    assert c.io == IoParams()
    assert c.inverse_kinematics == InverseKinematicsParams()
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
    assert c.inverse_kinematics == InverseKinematicsParams()
    assert c.stage_flags() == STAGE_DEFAULTS


def test_examples_config_inverse_kinematics_matches_the_packaged_template():
    """``examples/config.toml``'s IK section must be the packaged one, verbatim.

    The example config is a hand-maintained copy with no guard of its own, so it drifts
    silently -- and a stale ``[inverse_kinematics]`` there is not a cosmetic problem:
    the GUI reads the config snapshot beside ``results.h5`` to rebuild the model, and a
    key the validator no longer accepts makes it fall back to the packaged default model
    with only a logged warning.
    """
    packaged = _ik_section(DEFAULT_CONFIG_PATH.read_text())
    example = _ik_section(
        (Path(__file__).parents[1] / "examples/config.toml").read_text()
    )
    assert example == packaged


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


def test_inverse_kinematics_defaults_when_absent():
    ik = Config.from_dict({}).inverse_kinematics
    assert (
        ik.template == "neuromechfly"
        and ik.legs is None
        and ik.fit_head is True
        and ik.fit_abdomen is True
        and ik.n_iterations == 60
        and ik.neutral_weight == 1e-3
        # Heavily damped on purpose: light damping overshoots into the joint limits,
        # where QuickIK's clamp-based box handling then deadlocks.
        and ik.damping == 0.1
        and ik.position_tolerance == 1e-3
        and ik.angle_tolerance == 1e-3
        and ik.fixed_body is True  # a tethered fly: the body doesn't move
        and ik.weigh_by_confidence is False
        and ik.parallel is False  # sequential: seam-free and reproducible
        and ik.segment_len == 200
        and ik.overlap_len == 10
        and ik.bounds == {}
        and ik.constant_points == []
    )


def test_inverse_kinematics_reads_overrides():
    ik = Config.from_dict(
        {
            "inverse_kinematics": {
                "template": "neuromechfly",
                "legs": ["rf", "lf"],
                "n_iterations": 50,
                "fixed_body": False,
                "bounds": {"rf_trochanterfemur-rf_tibia-pitch": [10, 160]},
            }
        }
    ).inverse_kinematics
    assert ik.legs == ["rf", "lf"] and ik.n_iterations == 50
    assert ik.fixed_body is False
    assert ik.bounds == {"rf_trochanterfemur-rf_tibia-pitch": [10.0, 160.0]}


def test_inverse_kinematics_unknown_key_fails_loudly():
    with pytest.raises(ValueError, match=r"\[inverse_kinematics\] has unknown key"):
        Config.from_dict({"inverse_kinematics": {"bogus": 1}}).inverse_kinematics


@pytest.mark.parametrize("removed", ["max_nfev", "loss", "f_scale", "regularization"])
def test_inverse_kinematics_rejects_the_removed_scipy_knobs(removed):
    """The pre-QuickIK ``least_squares`` knobs are gone, and say so rather than pass.

    They tuned a solver that no longer exists, so silently ignoring them would leave a
    user believing they were still tuning the fit.
    """
    with pytest.raises(ValueError, match=rf"unknown key\(s\) \['{removed}'\]"):
        Config.from_dict({"inverse_kinematics": {removed: 1}}).inverse_kinematics


def test_solve_inverse_kinematics_signature_defaults_match_the_config():
    """The library entry point's defaults are the configured ones, not a second copy.

    ``solve_inverse_kinematics`` takes each solver knob as a keyword, so its signature
    is a place a default can quietly diverge from ``[inverse_kinematics]`` -- and it did:
    a direct library call solved with a different damping than the pipeline, which showed
    up as a wrong abdomen fit only in the tests that call it directly.
    """
    import inspect

    from deeperfly.inverse_kinematics import solve_inverse_kinematics

    defaults = InverseKinematicsParams()
    params = inspect.signature(solve_inverse_kinematics).parameters
    for name in (
        "n_iterations",
        "neutral_weight",
        "damping",
        "position_tolerance",
        "angle_tolerance",
        "fixed_body",
        "parallel",
        "segment_len",
        "overlap_len",
    ):
        assert params[name].default == getattr(defaults, name), name


def test_inverse_kinematics_allowed_keys_are_derived_from_the_dataclass():
    """The strict-validation message lists the keys actually parsed, not a stale literal."""
    from deeperfly.config import IK_KEYS

    parsed = {f.name for f in InverseKinematicsParams.__dataclass_fields__.values()}
    assert IK_KEYS == (parsed - {"markers"}) | {"head", "abdomen"}


def test_inverse_kinematics_reads_constant_points():
    ik = Config.from_dict(
        {
            "inverse_kinematics": {
                "constant_points": ["lf_thorax_coxa", "rf_thorax_coxa"]
            }
        }
    ).inverse_kinematics
    assert ik.constant_points == ["lf_thorax_coxa", "rf_thorax_coxa"]


def test_gui_mesh_hide_defaults_to_wings_and_reads_overrides():
    assert Config.from_dict({}).gui.mesh_hide == ["wings"]
    cfg = Config.from_dict({"gui": {"mesh_hide": ["wings", "legs"]}})
    assert cfg.gui.mesh_hide == ["wings", "legs"]


def test_gui_unknown_key_fails_loudly():
    with pytest.raises(ValueError, match=r"\[gui\]"):
        Config.from_dict({"gui": {"bogus": 1}}).gui


def test_inverse_kinematics_reads_marker_tables():
    ik = Config.from_dict(
        {
            "inverse_kinematics": {
                "abdomen": {
                    "l_abdomen0": {"body": "c_abdomen3", "offset": [0.0, 0.05, 0.5]},
                },
                "head": {
                    "l_antenna": {"body": "l_pedicel", "offset": [0.0, 0.0, 0.0]},
                },
            }
        }
    ).inverse_kinematics
    assert set(ik.markers) == {"head", "abdomen"}
    assert ik.markers["abdomen"]["l_abdomen0"]["body"] == "c_abdomen3"


def test_ik_articulation_applies_marker_offsets():
    import numpy as np

    base = Config.from_dict({}).ik_articulation().chain("abdomen")
    i = base.marker_names.index("l_abdomen0")
    cfg = Config.from_dict(
        {
            "inverse_kinematics": {
                "fit_head": False,
                "abdomen": {
                    "l_abdomen0": {"body": "c_abdomen3", "offset": [0.0, 0.05, 0.5]},
                    "r_abdomen0": {"body": "c_abdomen3", "offset": [0.0, -0.05, 0.5]},
                },
            }
        }
    )
    art = cfg.ik_articulation()
    assert [c.name for c in art.chains] == ["abdomen"]  # fit_head=False drops head
    ab = art.chain("abdomen")
    assert ab.marker_names == ("l_abdomen0", "r_abdomen0")  # table replaces the set
    # a different offset moves the neutral marker (the default z-offset was 0.3)
    assert not np.allclose(ab.marker_neutral[0], base.marker_neutral[i])


def test_ik_template_applies_legs_and_bounds():
    cfg = Config.from_dict(
        {
            "inverse_kinematics": {
                "legs": ["rf"],
                "bounds": {"rf_trochanterfemur-rf_tibia-pitch": [10, 160]},
            }
        }
    )
    t = cfg.ik_template()
    assert [leg.name for leg in t.legs] == ["rf"]
    fti = next(j for j in t.legs[0].joints if j.name == "FTi").dofs[0]
    import numpy as np

    assert round(np.rad2deg(fti.lo)) == 10 and round(np.rad2deg(fti.hi)) == 160


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
