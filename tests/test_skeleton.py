"""Tests for the Drosophila skeleton model."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import leg_indices

from deeperfly.config import Config
from deeperfly.skeleton import Skeleton


def test_counts(fly):
    assert fly.n_points == 38
    assert fly.n_bones == 28
    assert fly.bones.shape == (28, 2)
    assert len(fly.point_names) == 38
    assert len(fly.point_colors) == 38


def test_bone_indices_in_range(fly):
    assert fly.bones.min() >= 0
    assert fly.bones.max() < fly.n_points


def test_colors_are_per_point(fly):
    """One color per point, resolved through the selector, with no grouping left over.

    The packaged table is ten keys -- eight `*` patterns and two exact names -- and every
    one of the 38 points has to come out of exactly one of them, because a point that
    fell through would take the colormap and look like a bug in the palette rather than a
    hole in the table.
    """
    color = dict(zip(fly.point_names, fly.point_colors))
    assert color["lf_pretarsus"] == color["lf_thorax_coxa"] == "#0f7399"
    assert color["l_antenna"] == "#0a4f6b"  # exact name, not caught by "l*"
    assert color["r_antenna"] == "#8c1525"
    # The packaged skeleton's neck and abdomen are MIDLINE structures, so they take a
    # green ramp of their own rather than sitting on the left or the right one.
    assert color["neck"] == "#15a315"
    assert color["abdomen0"] == color["abdomen4"] == "#61e47b"
    # Ten distinct colors, and none of them the colormap's -- i.e. nothing fell through.
    from deeperfly.skeleton import TAB10_HEX

    assert len(set(fly.point_colors)) == 10
    assert not set(fly.point_colors) & set(TAB10_HEX)


def test_a_bone_takes_the_color_of_the_point_it_is_written_from(fly):
    """Which is why `edges` is written source-first: one table colors joints and bones."""
    assert len(fly.bone_colors) == fly.n_bones
    color = dict(zip(fly.point_names, fly.point_colors))
    for (a, _b), bone in zip(fly.bones, fly.bone_colors):
        assert bone == color[fly.point_names[int(a)]]


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


def test_from_config_dict_roundtrip():
    spec = {
        "skeleton": {
            "name": "toy",
            "points": ["a", "b", "c"],
            "edges": [["a", "b"], ["b", "c"]],
            "colors": {"a": "#123456"},
        }
    }
    s = Skeleton.from_config(Config.from_dict(spec))
    assert s.n_points == 3
    np.testing.assert_array_equal(s.bones, [[0, 1], [1, 2]])
    assert s.point_colors[0] == "#123456"


def test_edges_are_the_whole_topology(fly):
    """No grouping concept: the bones ARE the edge list, in declaration order.

    A leg's five points are written as a four-edge chain; the antennae and the neck
    appear in no edge at all -- tracked points with no bone, which a chain could not
    express without a one-point chain existing only to be a color key.
    """
    np.testing.assert_array_equal(fly.bones[:4], [[0, 1], [1, 2], [2, 3], [3, 4]])
    touched = set(np.asarray(fly.bones).reshape(-1).tolist())
    loose = [n for i, n in enumerate(fly.point_names) if i not in touched]
    assert loose == ["l_antenna", "r_antenna", "neck"]


def test_an_edge_naming_an_unknown_point_is_refused():
    bad = {"skeleton": {"points": ["a", "b"], "edges": [["a", "z"]]}}
    with pytest.raises(ValueError, match="unknown point name"):
        Skeleton.from_config(Config.from_dict(bad))
    outside = {"skeleton": {"points": ["a", "b"], "edges": [[0, 2]]}}
    with pytest.raises(ValueError, match="outside"):
        Skeleton.from_config(Config.from_dict(outside))


def test_a_skeleton_with_no_points_is_refused():
    with pytest.raises(ValueError, match="declares no 'points'"):
        Skeleton.from_spec({"name": "empty"})


def test_a_repeated_point_is_refused():
    """A point name is a channel, so two of them means two channels with one meaning."""
    with pytest.raises(ValueError, match="repeats"):
        Skeleton.from_spec({"points": ["a", "b", "a"]})


# -- left/right symmetry ------------------------------------------------------


def test_the_deepfly3d_set_declares_a_pair_for_every_point(deepfly3d):
    """``deepfly3d`` pairs all 38 points, and pairs them across the halves.

    The left-first block layout means the partner of point ``i`` is ``i + 19``; asserting
    that (rather than just "19 pairs exist") is what would catch an edit that paired two
    points on the same side.
    """
    assert deepfly3d.n_symmetries == 19
    pairs = np.asarray(deepfly3d.symmetries)
    assert pairs.shape == (19, 2)
    assert sorted(pairs.reshape(-1).tolist()) == list(range(38))
    np.testing.assert_array_equal(pairs[:, 1] - pairs[:, 0], np.full(19, 19))
    for a, b in deepfly3d.symmetry_names:
        assert a[0] == "l" and b[0] == "r" and a[1:] == b[1:]


def test_the_packaged_skeleton_pairs_every_point_that_has_a_side(fly):
    """``fly38b``'s unpaired points are exactly the midline ones, and no others.

    A point off the midline with no partner is a real defect -- it silently drops out of
    flip augmentation and the mirror check -- so "16 pairs" is asserted as *which* six
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


def test_flip_perm_matches_the_deepfly3d_block_layout(deepfly3d):
    """For ``deepfly3d`` the mirror is exactly "swap the halves".

    That is also what the dfpose trainer's ``FLIP_PERM`` is; a divergence here would
    silently retrain every left channel on a right joint.
    """
    np.testing.assert_array_equal(deepfly3d.flip_perm(), np.roll(np.arange(38), 19))


def test_partner(fly):
    assert fly.point_names[fly.partner("lf_pretarsus")] == "rf_pretarsus"
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
        Config.from_dict({"skeleton": {"points": ["a", "b", "c"]}})
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
            Config.from_dict({"skeleton": {"points": names, "symmetries": pairs}})
        )

    forward = build([["l_a", "r_a"], ["l_b", "r_b"]])
    shuffled = build([["r_b", "l_b"], ["r_a", "l_a"]])  # reversed rows, reversed order
    by_index = build([[2, 3], [1, 0]])
    np.testing.assert_array_equal(forward.symmetries, shuffled.symmetries)
    np.testing.assert_array_equal(forward.symmetries, by_index.symmetries)


@pytest.mark.parametrize(
    "pairs, message",
    [
        ([["lf_pretarsus", "lf_pretarsus"]], "with itself"),
        (
            [["lf_pretarsus", "rf_pretarsus"], ["lf_pretarsus", "rm_pretarsus"]],
            "more than one symmetry pair",
        ),
        ([["lf_pretarsus", "nope"]], "unknown point name"),
        ([["lf_pretarsus"]], "exactly 2 points"),
        ([["lf_pretarsus", "rf_pretarsus", "rm_pretarsus"]], "exactly 2 points"),
        ([[0, 999]], "outside"),
        (["lf_pretarsus"], "2-element pair"),
        ("lf_pretarsus", "list of 2-element pairs"),
    ],
)
def test_malformed_symmetries_are_rejected_with_a_pointed_message(fly, pairs, message):
    spec = {"skeleton": {"points": list(fly.point_names), "symmetries": pairs}}
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
            bones=np.empty((0, 2), int),
            symmetries=np.array([[0, 19], [0, 20]]),
        )


# -- the point selector -------------------------------------------------------


def test_a_pattern_stands_in_for_a_group(fly):
    """What a chain name used to do, with nothing declared: `lf_*` is one leg."""
    from deeperfly.skeleton import resolve_points

    legs = resolve_points(
        ["lf_*", "lm_*", "lh_*", "rf_*", "rm_*", "rh_*"],
        fly.point_names,
        where="test",
    )
    assert len(legs) == 30
    assert all("antenna" not in fly.point_names[i] for i in legs)
    # Ascending and deduplicated, so a caller can index straight into a points array.
    assert list(legs) == sorted(legs)


def test_an_exact_name_beats_a_pattern(fly):
    """Otherwise the fly's `l_antenna` could not sit outside the `l*` ramp."""
    from deeperfly.skeleton import resolve_points

    got = resolve_points(["l_antenna", "l*_thorax_coxa"], fly.point_names, where="test")
    assert [fly.point_names[i] for i in got] == [
        "lf_thorax_coxa",
        "lm_thorax_coxa",
        "lh_thorax_coxa",
        "l_antenna",
    ]


@pytest.mark.parametrize(
    "entries, message",
    [
        (["lf_nope"], "not a point of this skeleton"),
        (["zz_*"], "matches no point"),
        (["lf_*", "*_pretarsus"], "both match"),
        ("lf_pretarsus", "not a string"),
    ],
)
def test_the_selector_refuses_what_it_cannot_mean(fly, entries, message):
    """Three typos and a shape mistake. Over-matching is the one it cannot catch, which
    is why the resolved set is logged instead."""
    from deeperfly.skeleton import resolve_points

    with pytest.raises(ValueError, match=message):
        resolve_points(entries, fly.point_names, where="test")


def test_the_selector_logs_what_it_resolved(fly, caplog):
    from deeperfly.skeleton import resolve_points

    with caplog.at_level("INFO", logger="deeperfly"):
        resolve_points(["*_pretarsus"], fly.point_names, where="[test] points")
    assert "[test] points -> 6 points" in caplog.text


# -- the automorphism check ---------------------------------------------------


def test_the_packaged_symmetries_are_an_automorphism_of_the_edges(fly):
    """The whole safety of writing 16 pairs by hand, asserted on the packaged file."""
    perm = fly.flip_perm()
    edges = {frozenset((int(a), int(b))) for a, b in fly.bones}
    assert {frozenset((int(perm[a]), int(perm[b]))) for a, b in fly.bones} == edges


@pytest.mark.parametrize(
    "symmetries",
    [
        # A row carrying the wrong side: pairing lf's coxa with rm's maps the lf femur
        # edge onto a pair that spans two legs.
        [["lf_thorax_coxa", "rm_coxa_trochanter"]],
        # Two joints of one leg exchanged -- the typo a chain-length check could not see.
        [
            ["lf_thorax_coxa", "rf_coxa_trochanter"],
            ["lf_coxa_trochanter", "rf_thorax_coxa"],
        ],
    ],
)
def test_symmetries_that_are_not_a_mirror_of_the_edges_are_refused(fly, symmetries):
    spec = {
        "points": list(fly.point_names),
        "edges": [[int(a), int(b)] for a, b in fly.bones],
        "symmetries": symmetries,
    }
    with pytest.raises(ValueError, match="not a mirror of"):
        Skeleton.from_spec(spec)


def test_a_skeleton_with_no_edges_passes_trivially(fly):
    """Points with no bones have no topology to violate, so the check has to allow it."""
    Skeleton.from_spec(
        {
            "points": list(fly.point_names),
            "symmetries": [list(p) for p in fly.symmetry_names],
        }
    )


# -- color validation ---------------------------------------------------------


def test_a_color_key_that_names_no_point_is_refused():
    with pytest.raises(ValueError, match="not a point of this skeleton"):
        Skeleton.from_spec({"points": ["a", "b"], "colors": {"z": "#fff"}})


def test_two_color_patterns_over_one_point_are_refused():
    with pytest.raises(ValueError, match="both match"):
        Skeleton.from_spec(
            {"points": ["l_a", "l_b"], "colors": {"l_*": "#fff", "*_a": "#000"}}
        )


def test_a_malformed_color_is_refused_at_load():
    with pytest.raises(ValueError, match="not a #rgb or #rrggbb color"):
        Skeleton.from_spec({"points": ["a"], "colors": {"a": "reddish"}})


def test_an_uncolored_point_takes_the_colormap():
    """`colors` is optional and may be partial -- DeepLabCut's default is the colormap."""
    from deeperfly.skeleton import TAB10_HEX

    s = Skeleton.from_spec({"points": ["a", "b"], "colors": {"a": "#123456"}})
    assert s.point_colors == ("#123456", TAB10_HEX[1])
