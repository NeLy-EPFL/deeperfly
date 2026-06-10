"""Tests for the Drosophila skeleton model."""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly.config import Config
from deeperfly.skeleton import Skeleton
from helpers import leg_indices


@pytest.fixture
def fly() -> Skeleton:
    return Skeleton.fly()


def test_counts(fly):
    assert fly.n_points == 38
    assert fly.n_limbs == 10
    assert fly.bones.shape == (28, 2)
    assert len(fly.joint_names) == 38
    assert fly.limb_id.shape == (38,)


def test_bone_indices_in_range(fly):
    assert fly.bones.min() >= 0
    assert fly.bones.max() < fly.n_points


def test_palette(fly):
    # One color per limb, with the bright antenna/stripe cues set in the config.
    assert set(fly.palette) == set(fly.limb_names)
    assert fly.palette["l_antenna"] == "#0a4f6b"
    assert fly.palette["r_antenna"] == "#8c1525"
    assert fly.palette["l_stripe"] == "#a9dbe4"
    assert fly.palette["r_stripe"] == "#e6b3a8"


def test_left_right_legs_disjoint(fly):
    left, right = leg_indices(fly, "l"), leg_indices(fly, "r")
    assert left.size == 15
    assert right.size == 15
    assert set(left).isdisjoint(right)
    # Left legs are the low indices, right legs the high ones.
    assert left.max() < right.min()


def test_bone_index_pairs(fly):
    i, j = fly.bone_index_pairs()
    assert i.shape == j.shape == (28,)
    np.testing.assert_array_equal(np.stack([i, j], axis=1), fly.bones)


def test_from_config_dict_roundtrip(fly):
    spec = {
        "skeleton": {
            "name": "toy",
            "joint_names": ["a", "b", "c"],
            "limb_joints": {"L": [0, 1, 2]},
            "palette": {"L": "#123456"},
        }
    }
    s = Skeleton.from_config(Config.from_dict(spec))
    assert s.n_points == 3
    assert s.limb_names == ("L",)
    np.testing.assert_array_equal(s.limb_id, [0, 0, 0])
    # The limb's three points form a two-edge chain.
    np.testing.assert_array_equal(s.bones, [[0, 1], [1, 2]])
    assert s.palette == {"L": "#123456"}


def test_limb_joints_derive_structure(fly):
    # limb_names / limb_id / bones are all derived from the limb_joints mapping.
    assert fly.limb_names[0] == "lf_leg" and fly.limb_names[3] == "l_antenna"
    assert fly.limb_id[:5].tolist() == [0, 0, 0, 0, 0]
    assert fly.limb_id[15] == 3  # the single-point l_antenna limb
    # A leg's five points become a four-edge chain; an antenna contributes none.
    np.testing.assert_array_equal(fly.bones[:4], [[0, 1], [1, 2], [2, 3], [3, 4]])


def test_out_of_range_limb_joint_raises():
    spec = {"skeleton": {"joint_names": ["a", "b"], "limb_joints": {"L": [0, 2]}}}
    with pytest.raises(ValueError, match="outside"):
        Skeleton.from_config(Config.from_dict(spec))
