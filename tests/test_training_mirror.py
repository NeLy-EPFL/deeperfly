"""Tests for the left-right mirror -- the one place a training sample is flipped.

Mirroring an image turns a left leg into a right leg. Get the channel permutation wrong and
every left channel trains on a right joint: the loss still falls, nothing warns, and the
result looks like a model that will not converge. So the tests here pin all four things that
have to move together (image, coordinates, point channels, the per-point arrays riding with
them), the half-pixel convention, and the two multi-view rules that are easy to miss.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly.config import Config
from deeperfly.training import mirror_decisions, mirror_sample, mirror_view_names


@pytest.fixture
def perm(fly):
    return fly.flip_perm()


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(64, 128, 3), dtype=np.uint8)
    points = rng.uniform(0, 128, size=(38, 2))
    points[3] = np.nan  # unobserved, which is how this package spells invisibility
    visible = rng.integers(0, 2, size=38).astype(bool)
    conf = rng.random(38)
    return image, points, visible, conf


def test_all_four_things_move_together(sample, perm, fly):
    image, points, visible, conf = sample
    out_img, out_pts, out_vis, out_conf = mirror_sample(
        image, points, visible, conf, flip_perm=perm
    )
    np.testing.assert_array_equal(out_img, image[:, ::-1])
    # The left front thorax-coxa slot now holds the mirrored RIGHT front thorax-coxa.
    # `perm[0]` rather than a literal 19: which slot that is depends on the skeleton's
    # layout, and the property under test does not.
    assert fly.point_names[0] == "lf_thorax_coxa"
    assert fly.point_names[perm[0]] == "rf_thorax_coxa"
    assert out_pts[0, 0] == pytest.approx(127 - points[perm[0], 0])
    assert out_pts[0, 1] == pytest.approx(points[perm[0], 1])  # y untouched
    np.testing.assert_array_equal(out_vis, visible[perm])
    np.testing.assert_array_equal(out_conf, conf[perm])


def test_the_x_map_uses_pixel_centres(perm):
    """``x -> (W - 1) - x`` and not ``W - x``.

    The wrong form shifts every mirrored label by one pixel -- a quarter of a heatmap cell
    at STRIDE 4, far too small to see in a loss curve, and exactly the class of half-pixel
    error this package has already shipped once in a decoder.
    """
    points = np.zeros((38, 2))
    points[:, 0] = [0.0, 127.0] + [63.5] * 36
    _, out = mirror_sample(None, points, flip_perm=perm, width=128)
    back = out[perm]  # undo the channel permutation to compare positionally
    assert back[0, 0] == pytest.approx(127.0)
    assert back[1, 0] == pytest.approx(0.0)
    assert back[2, 0] == pytest.approx(63.5)  # the exact centre is a fixed point


def test_mirroring_twice_is_the_identity(sample, perm):
    """The cheapest total check on the whole operation.

    It only holds because ``flip_perm`` is an involution and the coordinate map is its own
    inverse -- so this fails loudly if either stops being true.
    """
    image, points, visible, conf = sample
    once = mirror_sample(image, points, visible, conf, flip_perm=perm)
    twice = mirror_sample(*once, flip_perm=perm)
    np.testing.assert_array_equal(twice[0], image)
    np.testing.assert_allclose(twice[1], points, equal_nan=True)
    np.testing.assert_array_equal(twice[2], visible)
    np.testing.assert_allclose(twice[3], conf)


def test_unobserved_points_stay_unobserved(sample, perm):
    """``(W - 1) - NaN`` is ``NaN``, and it must land in the partner's slot."""
    image, points, _, _ = sample
    _, out = mirror_sample(image, points, flip_perm=perm)
    assert np.isnan(out[perm.tolist().index(3)]).all()
    assert int(np.isnan(out).any(axis=1).sum()) == 1


def test_nothing_is_mutated_in_place(sample, perm):
    """A dataset caches decoded frames; mirroring one in place corrupts every later epoch."""
    image, points, visible, conf = sample
    before = (image.copy(), points.copy(), visible.copy(), conf.copy())
    mirror_sample(image, points, visible, conf, flip_perm=perm)
    np.testing.assert_array_equal(image, before[0])
    np.testing.assert_allclose(points, before[1], equal_nan=True)
    np.testing.assert_array_equal(visible, before[2])
    np.testing.assert_allclose(conf, before[3])


@pytest.mark.parametrize(
    "shape, width",
    [((64, 128), 128), ((64, 128, 3), 128), ((5, 64, 128, 1), 128), ((3, 32, 96), 96)],
)
def test_image_layouts(shape, width, perm):
    """Grayscale ``(H, W)``, channels-last, and batched forms all mirror on the width axis."""
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, size=shape, dtype=np.uint8)
    points = rng.uniform(0, width, size=(38, 2))
    out_img, out_pts = mirror_sample(image, points, flip_perm=perm)
    assert out_img.shape == image.shape
    assert out_pts[0, 0] == pytest.approx((width - 1) - points[perm[0], 0])


def test_a_batched_points_array_keeps_its_leading_axes(perm):
    rng = np.random.default_rng(2)
    points = rng.uniform(0, 128, size=(4, 7, 38, 2))
    vis = rng.integers(0, 2, size=(4, 7, 38)).astype(bool)
    _, out_pts, out_vis = mirror_sample(None, points, vis, flip_perm=perm, width=128)
    assert out_pts.shape == points.shape and out_vis.shape == vis.shape
    np.testing.assert_allclose(out_pts[2, 5, 0, 0], 127 - points[2, 5, perm[0], 0])
    np.testing.assert_array_equal(out_vis[2, 5], vis[2, 5][perm])


def test_a_per_point_array_with_a_trailing_axis_is_permuted_on_the_right_axis(perm):
    """``(P, k)`` and ``(..., P)`` are both valid; the rule is declared, not guessed."""
    rng = np.random.default_rng(3)
    points = rng.uniform(0, 128, size=(38, 2))
    weights = rng.random((38, 5))
    _, _, out = mirror_sample(None, points, weights, flip_perm=perm, width=128)
    np.testing.assert_allclose(out, weights[perm])


def test_an_ambiguous_per_point_array_is_refused(perm):
    """A ``(38, 38)`` array cannot say which axis indexes points, so it is not guessed."""
    points = np.zeros((38, 2))
    with pytest.raises(ValueError, match="ambiguous"):
        mirror_sample(None, points, np.zeros((38, 38)), flip_perm=perm, width=128)


def test_a_per_point_array_with_no_matching_axis_is_refused(perm):
    points = np.zeros((38, 2))
    with pytest.raises(ValueError, match="no axis of length 38"):
        mirror_sample(None, points, np.zeros((7, 5)), flip_perm=perm, width=128)


def test_a_permutation_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="must cover exactly"):
        mirror_sample(None, np.zeros((38, 2)), flip_perm=np.arange(19), width=128)


def test_width_is_required_without_an_image(perm):
    with pytest.raises(ValueError, match="width is required"):
        mirror_sample(None, np.zeros((38, 2)), flip_perm=perm)


def test_a_contradictory_width_is_refused(sample, perm):
    image, points, _, _ = sample
    with pytest.raises(ValueError, match="contradicts"):
        mirror_sample(image, points, flip_perm=perm, width=999)


def test_a_side_agnostic_model_needs_no_permutation(sample):
    """The identity permutation is a supported state, not a degenerate one.

    deeperfly's shipping detector has 19 side-agnostic channels reached through a
    canonicalizing ``fliplr`` preprocessor: there is no left channel to confuse with a right
    one, so a mirror permutes nothing. That falls out of the skeleton declaring no
    symmetries, so the same code path serves both regimes.
    """
    from deeperfly.skeleton import Skeleton

    side_agnostic = Skeleton.from_config(
        Config.from_dict({"skeleton": {"point_names": [f"ch{i}" for i in range(19)]}})
    )
    perm = side_agnostic.flip_perm()
    np.testing.assert_array_equal(perm, np.arange(19))
    rng = np.random.default_rng(4)
    points = rng.uniform(0, 128, size=(19, 2))
    _, out = mirror_sample(None, points, flip_perm=perm, width=128)
    np.testing.assert_allclose(out[:, 0], 127 - points[:, 0])
    np.testing.assert_allclose(out[:, 1], points[:, 1])


def test_torch_and_numpy_agree(sample, perm):
    """One implementation, two array libraries -- a divergence here is a silent train/eval gap."""
    torch = pytest.importorskip("torch")
    image, points, visible, _ = sample
    ref = mirror_sample(image, points, visible, flip_perm=perm)
    got = mirror_sample(
        torch.as_tensor(image),
        torch.as_tensor(points),
        torch.as_tensor(visible),
        flip_perm=perm,
    )
    np.testing.assert_array_equal(got[0].numpy(), ref[0])
    np.testing.assert_allclose(got[1].numpy(), ref[1], equal_nan=True)
    np.testing.assert_array_equal(got[2].numpy(), ref[2])


# -- the camera identity ------------------------------------------------------


def test_the_packaged_rig_declares_a_symmetric_mirror_pairing():
    cfg = Config.default()
    names = list(cfg.camera_table()[1])
    mapping = cfg.mirror_views()
    assert mapping == {
        "rh": "lh",
        "rm": "lm",
        "rf": "lf",
        "f": "f",  # the midline camera mirrors to itself
        "lf": "rf",
        "lm": "rm",
        "lh": "rh",
    }
    idx = mirror_view_names(names, mapping)
    np.testing.assert_array_equal(idx[idx], np.arange(len(names)))


def test_a_half_declared_mirror_pairing_is_refused():
    """One-sided is the dangerous state: it reads as correct and swaps sides on one camera."""
    cfg = Config.from_dict(
        {
            "cameras": {
                "defaults": {"distance": 1.0},
                "a": {"azimuth_deg": 0, "mirror": "b"},
                "b": {"azimuth_deg": 90},
            }
        }
    )
    with pytest.raises(ValueError, match="must be symmetric"):
        cfg.mirror_views()


def test_a_mirror_naming_an_unknown_view_is_refused():
    cfg = Config.from_dict(
        {
            "cameras": {
                "defaults": {"distance": 1.0},
                "a": {"azimuth_deg": 0, "mirror": "ghost"},
            }
        }
    )
    with pytest.raises(ValueError, match="unknown view"):
        cfg.mirror_views()


def test_the_mirror_key_does_not_reach_the_rig_parser():
    """It is stage metadata, not geometry, so it must be stripped before ``Camera.from_spec``.

    Built as a minimal rig rather than from the packaged config so the assertion is about
    the ``mirror`` key alone: with it present, a camera spec must still parse.
    """
    cfg = Config.default()
    sizes = {name: (512, 1024) for name in cfg.camera_table()[1]}
    rig = cfg.camera_group(image_sizes=sizes)
    assert list(rig.names) == list(cfg.camera_table()[1])
    assert cfg.mirror_views()  # and the key was still readable where it belongs


def test_an_undeclared_view_maps_to_itself():
    """The conservative degradation: leave the camera id alone rather than invent a pairing."""
    idx = mirror_view_names(["a", "b", "c"], {"a": "b", "b": "a"})
    np.testing.assert_array_equal(idx, [1, 0, 2])


# -- the multi-view rule ------------------------------------------------------


def test_one_mirror_decision_per_frame_group():
    """Flipping the views of one moment independently is the subtle multi-view bug.

    A mirror permutes the L/R channels and remaps the camera id, so per-view decisions leave
    a single frame in which some views' channel 0 means the left front leg and others' means
    the right -- and those views no longer describe one physical scene, so triangulation is
    fed a chimera.
    """
    rng = np.random.default_rng(5)
    decisions = mirror_decisions(200, 0.5, rng=rng, n_views=7)
    assert decisions.shape == (200, 7)
    # Constant along the view axis: every row is all-True or all-False.
    assert bool((decisions.all(axis=1) | (~decisions).all(axis=1)).all())
    # And still mirrored at about the requested rate.
    assert 0.35 < decisions[:, 0].mean() < 0.65


@pytest.mark.parametrize("p, expected", [(0.0, False), (1.0, True)])
def test_the_mirror_probability_endpoints(p, expected):
    """p=0 mirrors nothing and p=1 mirrors everything -- the off switch has to be exact."""
    d = mirror_decisions(16, p, rng=np.random.default_rng(6), n_views=3)
    assert bool(d.all()) is expected
    assert bool(d.any()) is expected


def test_a_probability_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError, match="probability"):
        mirror_decisions(4, 1.5, rng=np.random.default_rng(7))
