"""Tests for the Drosophila skeleton model."""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly.skeleton import Skeleton
from helpers import CAMERA_NAMES


@pytest.fixture
def fly() -> Skeleton:
    return Skeleton.fly()


def test_counts(fly):
    assert fly.n_points == 38
    assert fly.n_limbs == 10
    assert fly.bones.shape == (28, 2)
    assert fly.bones3d.tolist() == [[15, 34]]
    assert len(fly.joint_names) == 38
    assert fly.limb_id.shape == (38,)


def test_bone_indices_in_range(fly):
    for edges in (fly.bones, fly.bones3d):
        assert edges.min() >= 0
        assert edges.max() < fly.n_points


def test_left_right_legs_disjoint(fly):
    assert fly.left_idx.size == 15
    assert fly.right_idx.size == 15
    assert set(fly.left_idx).isdisjoint(fly.right_idx)
    # Right legs are the low indices, left legs the high ones.
    assert fly.right_idx.max() < fly.left_idx.min()


def test_visibility_mask_matches_table(fly):
    mask = fly.visibility_mask(CAMERA_NAMES)
    assert mask.shape == (7, 38)
    # Per-camera visible counts derived from DeepFly3D's camera_see_joint.
    assert mask.sum(axis=1).tolist() == [19, 19, 16, 14, 16, 19, 19]
    # Right cameras never see left-side points and vice versa.
    rh = CAMERA_NAMES.index("rh")
    lh = CAMERA_NAMES.index("lh")
    assert not mask[rh, fly.left_idx].any()
    assert not mask[lh, fly.right_idx].any()


def test_unknown_camera_sees_everything(fly):
    mask = fly.visibility_mask(["mystery_cam"])
    assert mask.shape == (1, 38)
    assert mask.all()


def test_bone_index_pairs(fly):
    i, j = fly.bone_index_pairs()
    assert i.shape == j.shape == (28,)
    np.testing.assert_array_equal(np.stack([i, j], axis=1), fly.bones)
    i3, j3 = fly.bone_index_pairs(include_3d=True)
    assert i3.shape == (29,)
    assert (i3[-1], j3[-1]) == (15, 34)


def test_from_config_dict_roundtrip(fly):
    spec = {
        "skeleton": {
            "name": "toy",
            "joint_names": ["a", "b", "c"],
            "limb_names": ["L"],
            "limb_id": [0, 0, 0],
            "bones": [[0, 1], [1, 2]],
            "bones3d": [],
            "left_points": [0],
            "right_points": [2],
            "visibility": {"cam0": [0, 1], "cam1": [1, 2]},
        }
    }
    s = Skeleton.from_config(spec)
    assert s.n_points == 3
    assert s.bones.shape == (2, 2)
    assert s.bones3d.shape == (0, 2)
    mask = s.visibility_mask(["cam0", "cam1"])
    np.testing.assert_array_equal(mask, [[True, True, False], [False, True, True]])


def test_bad_limb_id_length_raises():
    spec = {"skeleton": {"joint_names": ["a", "b"], "limb_id": [0]}}
    with pytest.raises(ValueError, match="limb_id"):
        Skeleton.from_config(spec)
