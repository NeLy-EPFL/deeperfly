"""Tests for the Drosophila skeleton model."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import leg_indices

from deeperfly.config import Config
from deeperfly.skeleton import Skeleton


def test_counts(fly):
    assert fly.n_points == 38
    assert fly.n_edges == 28
    assert fly.edges.shape == (28, 2)
    assert len(fly.point_names) == 38
    assert len(fly.point_colors) == 38


def test_edge_indices_in_range(fly):
    assert fly.edges.min() >= 0
    assert fly.edges.max() < fly.n_points


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


def test_an_undeclared_edge_color_is_the_average_of_its_endpoints(fly):
    """And on `fly38` that reproduces the retired source-first rule exactly.

    Every one of its 28 edges sits inside one colored group (`lf_*` ... `abdomen*`), so
    both endpoints carry the same hex and the average is that hex -- which is what makes
    the derivation safe to adopt without recoloring a single existing drawing.
    """
    assert len(fly.edge_colors) == fly.n_edges
    color = dict(zip(fly.point_names, fly.point_colors))
    for (a, b), edge in zip(fly.edges, fly.edge_colors):
        assert color[fly.point_names[int(a)]] == color[fly.point_names[int(b)]]
        assert edge == color[fly.point_names[int(a)]]


def test_left_right_legs_disjoint(fly):
    left, right = leg_indices(fly, "l"), leg_indices(fly, "r")
    assert left.size == 15
    assert right.size == 15
    assert set(left).isdisjoint(right)
    # Left legs are the low indices, right legs the high ones.
    assert left.max() < right.min()


def test_edge_endpoints(fly):
    i, j = fly.edge_endpoints()
    assert i.shape == j.shape == (28,)
    np.testing.assert_array_equal(np.stack([i, j], axis=1), fly.edges)


def test_from_config_dict_roundtrip():
    spec = {
        "skeleton": {
            "name": "toy",
            "points": ["a", "b", "c"],
            "edges": [["a", "b"], ["b", "c"]],
            "point_colors": {"a": "#123456"},
        }
    }
    s = Skeleton.from_config(Config.from_dict(spec))
    assert s.n_points == 3
    np.testing.assert_array_equal(s.edges, [[0, 1], [1, 2]])
    assert s.point_colors[0] == "#123456"


def test_edges_are_the_whole_topology(fly):
    """No grouping concept: the edge list IS the topology, in declaration order.

    A leg's five points are written as a four-edge chain; the antennae and the neck
    appear in no edge at all -- tracked points with no edge, which a chain could not
    express without a one-point chain existing only to be a color key.
    """
    np.testing.assert_array_equal(fly.edges[:4], [[0, 1], [1, 2], [2, 3], [3, 4]])
    touched = set(np.asarray(fly.edges).reshape(-1).tolist())
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
    pairs = np.asarray(deepfly3d.point_symmetries)
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
    paired = {fly.point_names[i] for pair in fly.point_symmetries for i in pair}
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
            Config.from_dict({"skeleton": {"points": names, "point_symmetries": pairs}})
        )

    forward = build([["l_a", "r_a"], ["l_b", "r_b"]])
    shuffled = build([["r_b", "l_b"], ["r_a", "l_a"]])  # reversed rows, reversed order
    by_index = build([[2, 3], [1, 0]])
    np.testing.assert_array_equal(forward.point_symmetries, shuffled.point_symmetries)
    np.testing.assert_array_equal(forward.point_symmetries, by_index.point_symmetries)


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
    spec = {"skeleton": {"points": list(fly.point_names), "point_symmetries": pairs}}
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
            edges=np.empty((0, 2), int),
            point_symmetries=np.array([[0, 19], [0, 20]]),
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
    edges = {frozenset((int(a), int(b))) for a, b in fly.edges}
    assert {frozenset((int(perm[a]), int(perm[b]))) for a, b in fly.edges} == edges


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
        "edges": [[int(a), int(b)] for a, b in fly.edges],
        "point_symmetries": symmetries,
    }
    with pytest.raises(ValueError, match="not a mirror of"):
        Skeleton.from_spec(spec)


def test_a_skeleton_with_no_edges_passes_trivially(fly):
    """Points with no edges have no topology to violate, so the check has to allow it."""
    Skeleton.from_spec(
        {
            "points": list(fly.point_names),
            "point_symmetries": [list(p) for p in fly.symmetry_names],
        }
    )


# -- color validation ---------------------------------------------------------


def test_a_color_key_that_names_no_point_is_refused():
    with pytest.raises(ValueError, match="not a point of this skeleton"):
        Skeleton.from_spec({"points": ["a", "b"], "point_colors": {"z": "#fff"}})


def test_two_color_patterns_over_one_point_are_refused():
    with pytest.raises(ValueError, match="both match"):
        Skeleton.from_spec(
            {"points": ["l_a", "l_b"], "point_colors": {"l_*": "#fff", "*_a": "#000"}}
        )


def test_a_malformed_color_is_refused_at_load():
    with pytest.raises(ValueError, match="not a #rgb or #rrggbb color"):
        Skeleton.from_spec({"points": ["a"], "point_colors": {"a": "reddish"}})


def test_an_uncolored_point_takes_the_colormap():
    """`colors` is optional and may be partial -- DeepLabCut's default is the colormap."""
    from deeperfly.skeleton import TAB10_HEX

    s = Skeleton.from_spec({"points": ["a", "b"], "point_colors": {"a": "#123456"}})
    assert s.point_colors == ("#123456", TAB10_HEX[1])


# -- edge colors --------------------------------------------------------------


#: Four points in two colour groups, joined in a chain -- so the middle edge is the one
#: that crosses groups and is the only one the average visibly blends.
_TWO_GROUPS = {
    "points": ["a", "b", "c", "d"],
    "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
    "point_colors": {"a": "#ff0000", "b": "#ff0000", "c": "#0000ff", "d": "#0000ff"},
}


def test_an_edge_across_two_color_groups_blends():
    """The only case where the average differs from either endpoint."""
    s = Skeleton.from_spec(_TWO_GROUPS)
    assert s.edge_colors == ("#ff0000", "#800080", "#0000ff")


def test_a_declared_edge_color_wins_and_matches_either_orientation():
    """`c--b` names an edge stored as `b--c`; an edge selector is unordered.

    Requiring the stored order would make a correct-looking key a silent miss, which is
    the one thing a colour table cannot report for itself.
    """
    s = Skeleton.from_spec({**_TWO_GROUPS, "edge_colors": {"c--b": "#404040"}})
    assert s.edge_colors == ("#ff0000", "#404040", "#0000ff")


@pytest.mark.parametrize(
    "table, message",
    [
        ({"a": "#fff"}, "not an edge selector"),
        ({"a--b--c": "#fff"}, "not an edge selector"),
        ({"a--z": "#fff"}, "not a point of this skeleton"),
        ({"a--d": "#fff"}, "matches no edge"),  # both real points, not joined
        ({"a--b": "#fff", "*--b": "#000"}, "two keys both match"),
        ({"a--b": "reddish"}, "not a #rgb or #rrggbb color"),
    ],
)
def test_a_malformed_edge_color_key_is_refused_at_load(table, message):
    with pytest.raises(ValueError, match=message):
        Skeleton.from_spec({**_TWO_GROUPS, "edge_colors": table})


def test_edge_colors_of_the_wrong_length_are_refused():
    """The field is index-aligned with `edges`, so a mismatch is not silently padded."""
    with pytest.raises(ValueError, match="edge_colors has 1 entries for 3 edges"):
        Skeleton.from_spec(_TWO_GROUPS).__class__(
            name="x",
            point_names=("a", "b", "c", "d"),
            edges=np.array([[0, 1], [1, 2], [2, 3]]),
            edge_colors=("#fff",),
        )


# -- identity -----------------------------------------------------------------


def test_the_digest_separates_two_skeletons_a_name_cannot(fly, deepfly3d):
    """The exact collision the project already had: 38 points under one name, twice."""
    assert fly.n_points == deepfly3d.n_points == 38
    assert fly.digest != deepfly3d.digest
    assert fly.label == f"{fly.name}@{fly.digest}"
    assert len(fly.digest) == 8


def test_the_digest_ignores_color_and_name_but_not_structure(fly):
    """Colours change no stage's answer and move no label, so they are out of it.

    The name is out for the same reason it is out of the stage fingerprint: two skeletons
    agreeing on the points and the edges compute the same result whatever they are called.
    """
    import dataclasses

    recolored = dataclasses.replace(
        fly, name="something else", point_colors=("#000000",) * 38, edge_colors=()
    )
    assert recolored.digest == fly.digest

    reordered = Skeleton(
        name=fly.name, point_names=fly.point_names[::-1], edges=fly.edges
    )
    assert reordered.digest != fly.digest
    no_pairs = Skeleton(name=fly.name, point_names=fly.point_names, edges=fly.edges)
    assert no_pairs.digest != fly.digest  # the symmetry pairs are in it
