"""Tests for the Drosophila skeleton model."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import leg_indices

from deeperfly.config import Config
from deeperfly.skeleton import Skeleton


def test_counts(fly):
    assert fly.n_points == 38
    assert fly.n_limbs == 10
    assert fly.bones.shape == (28, 2)
    assert len(fly.point_names) == 38
    assert fly.limb_id.shape == (38,)


def test_bone_indices_in_range(fly):
    assert fly.bones.min() >= 0
    assert fly.bones.max() < fly.n_points


def test_palette(fly):
    # One color per limb, with the bright antenna cues set in the skeleton.
    assert set(fly.palette) == set(fly.limb_names)
    assert fly.palette["l_antenna"] == "#0a4f6b"
    assert fly.palette["r_antenna"] == "#8c1525"
    # The packaged skeleton's neck and abdomen are MIDLINE structures, so they take a
    # green ramp of their own rather than sitting on the left or the right one.
    assert fly.palette["neck"] == "#15a315"
    assert fly.palette["abdomen"] == "#61e47b"


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
            "point_names": ["a", "b", "c"],
            "limb_points": {"L": [0, 1, 2]},
            "limb_palette": {"L": "#123456"},
        }
    }
    s = Skeleton.from_config(Config.from_dict(spec))
    assert s.n_points == 3
    assert s.limb_names == ("L",)
    np.testing.assert_array_equal(s.limb_id, [0, 0, 0])
    # The limb's three points form a two-edge chain.
    np.testing.assert_array_equal(s.bones, [[0, 1], [1, 2]])
    assert s.palette == {"L": "#123456"}


def test_limb_points_derive_structure(fly):
    # limb_names / limb_id / bones are all derived from the limb_points mapping.
    assert fly.limb_names[0] == "lf_leg" and fly.limb_names[3] == "l_antenna"
    assert fly.limb_id[:5].tolist() == [0, 0, 0, 0, 0]
    assert fly.limb_id[15] == 3  # the single-point l_antenna limb
    # A leg's five points become a four-edge chain; an antenna contributes none.
    np.testing.assert_array_equal(fly.bones[:4], [[0, 1], [1, 2], [2, 3], [3, 4]])


def test_out_of_range_limb_point_raises():
    spec = {"skeleton": {"point_names": ["a", "b"], "limb_points": {"L": [0, 2]}}}
    with pytest.raises(ValueError, match="outside"):
        Skeleton.from_config(Config.from_dict(spec))


def test_limb_points_resolve_names():
    # limb_points may list point names; an unknown name is rejected.
    spec = {
        "skeleton": {"point_names": ["a", "b", "c"], "limb_points": {"L": ["a", "c"]}}
    }
    s = Skeleton.from_config(Config.from_dict(spec))
    np.testing.assert_array_equal(s.bones, [[0, 2]])
    bad = {"skeleton": {"point_names": ["a", "b"], "limb_points": {"L": ["a", "z"]}}}
    with pytest.raises(ValueError, match="unknown point name"):
        Skeleton.from_config(Config.from_dict(bad))


# -- left/right symmetry ------------------------------------------------------


def test_fly38_declares_a_pair_for_every_point(fly38):
    """``fly38`` pairs all 38 points, and pairs them across the halves.

    The left-first block layout means the partner of point ``i`` is ``i + 19``; asserting
    that (rather than just "19 pairs exist") is what would catch an edit that paired two
    points on the same side.
    """
    assert fly38.n_symmetries == 19
    pairs = np.asarray(fly38.symmetries)
    assert pairs.shape == (19, 2)
    assert sorted(pairs.reshape(-1).tolist()) == list(range(38))
    np.testing.assert_array_equal(pairs[:, 1] - pairs[:, 0], np.full(19, 19))
    for a, b in fly38.symmetry_names:
        assert a[0] == "l" and b[0] == "r" and a[1:] == b[1:]


def test_the_packaged_skeleton_pairs_every_point_that_has_a_side(fly):
    """``fly38b``'s unpaired points are exactly the midline ones, and no others.

    A point off the midline with no partner is a real defect -- it silently drops out of
    flip augmentation and the chirality check -- so "16 pairs" is asserted as *which* six
    points are left over, not as a count.
    """
    paired = {fly.point_names[i] for pair in fly.symmetries for i in pair}
    unpaired = [n for n in fly.point_names if n not in paired]
    assert unpaired == [
        "neck",
        "abdomen0",
        "abdomen1",
        "abdomen2",
        "abdomen3",
        "abdomen4",
    ]
    for a, b in fly.symmetry_names:
        assert a[0] == "l" and b[0] == "r" and a[1:] == b[1:]


def test_flip_perm_is_an_involution(fly):
    perm = fly.flip_perm()
    assert perm.shape == (38,)
    # Applying the mirror twice is the identity -- the property every consumer relies on.
    np.testing.assert_array_equal(perm[perm], np.arange(38))
    # A midline point maps to ITSELF, which is what makes mirroring a frame legal for it.
    for name in ("neck", "abdomen0", "abdomen4"):
        i = fly.point_names.index(name)
        assert perm[i] == i


def test_flip_perm_matches_the_fly38_block_layout(fly38):
    """For ``fly38`` the mirror is exactly "swap the halves".

    That is also what the dfpose trainer's ``FLIP_PERM`` is; a divergence here would
    silently retrain every left channel on a right joint.
    """
    np.testing.assert_array_equal(fly38.flip_perm(), np.roll(np.arange(38), 19))


def test_partner(fly):
    assert fly.point_names[fly.partner("lf_claw")] == "rf_claw"
    assert fly.point_names[fly.partner("r_antenna")] == "l_antenna"
    assert fly.partner(0) == fly.point_names.index("rf_thorax_coxa")
    # A midline point has no partner at all -- None, not itself.
    assert fly.partner("neck") is None
    with pytest.raises(ValueError, match="not a point of skeleton"):
        fly.partner("no_such_point")


def test_a_skeleton_with_no_symmetries_disables_the_pair_features():
    """No pairs is legal: the permutation is the identity and nothing else fires.

    This is the side-agnostic detector's case -- a 19-channel model whose channels carry
    no side has nothing to swap -- so it must be a supported state, not a degenerate one.
    """
    skel = Skeleton.from_config(
        Config.from_dict({"skeleton": {"point_names": ["a", "b", "c"]}})
    )
    assert skel.n_symmetries == 0
    assert skel.symmetry_names == ()
    assert skel.partner("a") is None
    np.testing.assert_array_equal(skel.flip_perm(), np.arange(3))


def test_symmetries_are_canonicalized_so_declaration_order_carries_no_meaning():
    """A pair is an unordered set, and two spellings of the same pairing compare equal."""
    names = ["l_a", "r_a", "l_b", "r_b"]

    def build(pairs):
        return Skeleton.from_config(
            Config.from_dict({"skeleton": {"point_names": names, "symmetries": pairs}})
        )

    forward = build([["l_a", "r_a"], ["l_b", "r_b"]])
    shuffled = build([["r_b", "l_b"], ["r_a", "l_a"]])  # reversed rows, reversed order
    by_index = build([[2, 3], [1, 0]])
    np.testing.assert_array_equal(forward.symmetries, shuffled.symmetries)
    np.testing.assert_array_equal(forward.symmetries, by_index.symmetries)


@pytest.mark.parametrize(
    "pairs, message",
    [
        ([["lf_claw", "lf_claw"]], "with itself"),
        ([["lf_claw", "rf_claw"], ["lf_claw", "rm_claw"]], "second symmetry pair"),
        ([["lf_claw", "nope"]], "unknown point name"),
        ([["lf_claw"]], "exactly 2 points"),
        ([["lf_claw", "rf_claw", "rm_claw"]], "exactly 2 points"),
        ([[0, 999]], "outside"),
        (["lf_claw"], "2-element pair"),
        ("lf_claw", "list of 2-element pairs"),
    ],
)
def test_malformed_symmetries_are_rejected_with_a_pointed_message(fly, pairs, message):
    spec = {"skeleton": {"point_names": list(fly.point_names), "symmetries": pairs}}
    with pytest.raises(ValueError, match=message):
        Skeleton.from_config(Config.from_dict(spec))


def test_a_point_in_two_pairs_is_rejected_however_the_skeleton_was_built(fly):
    """The validation is in ``__post_init__``, not only in the config parser.

    A skeleton also arrives from a ``results.h5`` and from the editor, and ``flip_perm``'s
    involution guarantee only holds while no point has two partners -- so the check has to
    sit where every construction path passes through it.
    """
    with pytest.raises(ValueError, match="more than one symmetry pair"):
        Skeleton(
            name="t",
            point_names=fly.point_names,
            limb_names=(),
            limb_id=np.full(38, -1),
            bones=np.empty((0, 2), int),
            palette={},
            symmetries=np.array([[0, 19], [0, 20]]),
        )


def test_symmetries_or_inferred_falls_back_only_when_nothing_is_declared(fly):
    """An old ``results.h5`` carries no pairs, and the chirality QC still has to work.

    The fallback must never *override* a declaration, though -- a skeleton that deliberately
    pairs nothing (or pairs unusually) must be taken at its word.
    """
    np.testing.assert_array_equal(fly.symmetries_or_inferred(), fly.symmetries)

    import dataclasses

    stripped = dataclasses.replace(fly, symmetries=np.empty((0, 2), np.int64))
    np.testing.assert_array_equal(stripped.symmetries_or_inferred(), fly.symmetries)


@pytest.mark.parametrize(
    "names, expected",
    [
        # This project's own scheme: the side letter is glued to the limb letter.
        (["lf_claw", "rf_claw", "l_antenna", "r_antenna"], [(0, 1), (2, 3)]),
        # SLEAP-style suffixes, and the word forms.
        (["Ear_L", "Ear_R", "nose"], [(0, 1)]),
        (["left_wing", "right_wing", "thorax"], [(0, 1)]),
        (["shoulderleft", "shoulderright", "hipL", "hipR"], [(0, 1), (2, 3)]),
        # A lone side-letter-looking name has no counterpart, so it stays unpaired --
        # the guard that keeps the loosest rule from inventing pairs.
        (["l_eye", "r_eye", "rostrum", "labellum"], [(0, 1)]),
        # SLEAP's own mouse skeleton: genuinely no pairs.
        (["head", "torso", "tail_base"], []),
    ],
)
def test_infer_symmetries_by_name(names, expected):
    from deeperfly.skeleton import infer_symmetries_by_name

    got = [tuple(int(x) for x in row) for row in infer_symmetries_by_name(names)]
    assert got == expected


def test_inference_reproduces_the_packaged_declaration(fly):
    """The 19 pairs in ``default_config.toml`` are exactly what inference proposes.

    They are still written out (a rename must not silently re-pair the skeleton), but if
    the two ever disagreed, one of them would be wrong.
    """
    from deeperfly.skeleton import infer_symmetries_by_name

    np.testing.assert_array_equal(
        infer_symmetries_by_name(fly.point_names), fly.symmetries
    )
