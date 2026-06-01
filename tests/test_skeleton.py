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


# -- keypoint categories -----------------------------------------------------


def test_points_in_category(fly):
    legs = fly.points_in_category("legs")
    assert legs.size == 30  # six 5-joint legs
    assert set(legs) == set(fly.right_idx) | set(fly.left_idx)
    np.testing.assert_array_equal(fly.points_in_category(["antennae"]), [15, 34])
    np.testing.assert_array_equal(
        fly.points_in_category(("stripes",)), [16, 17, 18, 35, 36, 37]
    )
    # A multi-category request is the union, and the whole set covers all points.
    everything = fly.points_in_category(["legs", "antennae", "stripes"])
    np.testing.assert_array_equal(everything, np.arange(fly.n_points))


def test_points_in_category_unknown_raises(fly):
    with pytest.raises(ValueError, match="unknown keypoint category 'wings'"):
        fly.points_in_category(["legs", "wings"])


# -- left/right stripe merge -------------------------------------------------


def test_merge_lr_stripes_structure(fly):
    merged, remap = fly.merge_lr_stripes()
    assert merged.n_points == 35  # 38 - 3 left stripe duplicates
    assert merged.joint_names[16:19] == ("Stripe0", "Stripe1", "Stripe2")
    # The left stripes (35..37) collapse onto the right ones (16..18); every
    # other point keeps its index.
    np.testing.assert_array_equal(remap[:35], np.arange(35))
    np.testing.assert_array_equal(remap[35:38], [16, 17, 18])
    # The duplicated stripe chain collapses; the antenna 3D bone is untouched.
    assert merged.bones.shape == (26, 2)
    assert merged.bones3d.tolist() == [[15, 34]]
    # Leg indices are unchanged (all below the dropped points).
    np.testing.assert_array_equal(merged.left_idx, fly.left_idx)
    np.testing.assert_array_equal(merged.right_idx, fly.right_idx)
    # The surviving stripe limb drops its side prefix; the empty one is gone.
    assert merged.n_limbs == 9  # the now-empty left-stripe limb is dropped
    assert "stripe" in merged.limb_names
    assert "R_stripe" not in merged.limb_names and "L_stripe" not in merged.limb_names


def test_merge_lr_stripes_visibility(fly):
    merged, _ = fly.merge_lr_stripes()
    mask = merged.visibility_mask(CAMERA_NAMES)
    assert mask.shape == (7, 35)
    seers = {CAMERA_NAMES[v] for v in range(7) if mask[v, 16]}  # Stripe0
    assert seers == {"rh", "rm", "lm", "lh"}  # all four cameras that see a side


def test_merge_lr_stripes_idempotent(fly):
    merged, _ = fly.merge_lr_stripes()
    again, remap = merged.merge_lr_stripes()
    assert again is merged  # nothing left to merge
    np.testing.assert_array_equal(remap, np.arange(merged.n_points))


def test_merged_points_in_category(fly):
    merged, _ = fly.merge_lr_stripes()
    np.testing.assert_array_equal(merged.points_in_category("stripes"), [16, 17, 18])
    assert merged.points_in_category("legs").size == 30
