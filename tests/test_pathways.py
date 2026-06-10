"""Tests for the detection plan: parsing/validation, the (i, v, p) scatter and
the coordinate inverse that maps a model peak back into its view frame."""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly.config import Config
from deeperfly.preprocessing import FrameTransform, Fliplr, Resize
from deeperfly.pose2d.pathways import map_to_view, scatter_pathway


def _plan(pathways, cameras=None, models=None):
    skel = Config.default().data["skeleton"]
    data = {
        "sources": [{"name": "s0", "input": "a"}, {"name": "s1", "input": "b"}],
        "preprocessors": [
            {"name": "plain", "ops": []},
            {"name": "mirror", "ops": [{"op": "fliplr"}]},
        ],
        "models": models
        or [
            {
                "name": "m",
                "class": "hourglass",
                "input_size": [256, 512],
                "n_channels": 19,
            }
        ],
        "pathways": pathways,
        "cameras": cameras
        or {
            "rh": {"azimuth_deg": 0, "distance": 100, "focal_length_px": 1},
            "lf": {"azimuth_deg": 1, "distance": 100, "focal_length_px": 1},
        },
        "skeleton": skel,
    }
    return Config.from_dict(data).detection_plan()


# -- coordinate inverse (map_to_view) ----------------------------------------


def test_map_to_view_plain_scales_to_source_pixels():
    # No preprocessor: a model peak at normalized (x, y) maps to ~ (x*W, y*H) in
    # source pixels (within the half-pixel resize convention).
    pts = np.array([[0.5, 0.5]])
    out = map_to_view(pts, FrameTransform(()), (256, 512), (480, 960))
    np.testing.assert_allclose(out, [[0.5 * 960, 0.5 * 480]], atol=1.0)


def test_map_to_view_mirror_undoes_flip():
    # With a fliplr preprocessor the x coordinate is reflected back into the source
    # frame; y is unchanged.
    transform = FrameTransform((Fliplr(),))
    src = (480, 960)
    plain = map_to_view(np.array([[0.3, 0.4]]), FrameTransform(()), (256, 512), src)
    mirrored = map_to_view(np.array([[0.3, 0.4]]), transform, (256, 512), src)
    # The mirrored x is the reflection of the plain x about the image centre.
    np.testing.assert_allclose(mirrored[0, 0], (960 - 1) - plain[0, 0], atol=1e-6)
    np.testing.assert_allclose(mirrored[0, 1], plain[0, 1], atol=1e-6)


def test_map_to_view_roundtrips_with_map_points():
    transform = FrameTransform((Fliplr(),))
    src = (480, 960)
    norm = np.array([[0.2, 0.7], [0.9, 0.1]])
    view_px = map_to_view(norm, transform, (256, 512), src)
    # Forward map (source -> mirrored -> model input) then normalize recovers norm.
    mirror_px = transform.map_points(view_px, src)
    resize = FrameTransform((Resize(width=512, height=256),))
    back = resize.map_points(mirror_px, transform.output_size(src)) / np.array(
        [512, 256]
    )
    np.testing.assert_allclose(back, norm, atol=1e-6)


# -- scatter ------------------------------------------------------------------


def test_scatter_pathway_routes_channels_and_leaves_nan():
    mapping = np.array([[0, 0, 5], [2, 1, 7]])  # (i, v, p)
    raw_xy = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    conf = np.array([0.1, 0.2, 0.3])
    out_pts = np.full((2, 10, 2), np.nan)
    out_conf = np.zeros((2, 10))
    scatter_pathway(raw_xy, conf, mapping, out_pts, out_conf)
    np.testing.assert_array_equal(out_pts[0, 5], [1.0, 2.0])  # channel 0 -> (0, 5)
    np.testing.assert_array_equal(out_pts[1, 7], [5.0, 6.0])  # channel 2 -> (1, 7)
    assert out_conf[0, 5] == 0.1 and out_conf[1, 7] == 0.3
    assert np.isnan(out_pts[0, 0]).all()  # untouched stays NaN


def test_scatter_pathway_candidate_axis():
    # The same scatter handles a trailing K (candidate) axis.
    mapping = np.array([[1, 0, 3]])
    raw_xy = np.zeros((2, 4, 2))
    raw_xy[1] = 9.0
    score = np.zeros((2, 4))
    score[1] = 0.5
    out_pts = np.full((1, 5, 4, 2), np.nan)
    out_conf = np.zeros((1, 5, 4))
    scatter_pathway(raw_xy, score, mapping, out_pts, out_conf)
    np.testing.assert_array_equal(out_pts[0, 3], 9.0)
    np.testing.assert_array_equal(out_conf[0, 3], 0.5)


# -- visibility derived from the plan ----------------------------------------


def test_visibility_mask_is_union_of_pathways():
    plan = _plan(
        [
            {
                "source": "s0",
                "preprocessor": "plain",
                "model": "m",
                "view": "rh",
                "points": [19, 20, -1] + [-1] * 16,
            },
            {
                "source": "s1",
                "preprocessor": "mirror",
                "model": "m",
                "view": "lf",
                "points": [0, -1, 2] + [-1] * 16,
            },
        ]
    )
    vm = plan.visibility_mask()
    assert vm.shape == (2, 38)
    assert sorted(np.where(vm[0])[0]) == [19, 20]
    assert sorted(np.where(vm[1])[0]) == [0, 2]


def test_view_sources_links_each_view_to_its_source():
    plan = _plan(
        [
            {
                "source": "s0",
                "preprocessor": "plain",
                "model": "m",
                "view": "rh",
                "points": list(range(19, 38)),
            },
            {
                "source": "s1",
                "preprocessor": "mirror",
                "model": "m",
                "view": "lf",
                "points": list(range(0, 19)),
            },
        ]
    )
    assert plan.view_sources() == {"rh": "s0", "lf": "s1"}


# -- validation ---------------------------------------------------------------


def test_pathway_rejects_unknown_references():
    with pytest.raises(ValueError, match="unknown source"):
        _plan(
            [
                {
                    "source": "nope",
                    "preprocessor": "plain",
                    "model": "m",
                    "view": "rh",
                    "points": list(range(19, 38)),
                }
            ]
        )
    with pytest.raises(ValueError, match="unknown preprocessor"):
        _plan(
            [
                {
                    "source": "s0",
                    "preprocessor": "nope",
                    "model": "m",
                    "view": "rh",
                    "points": list(range(19, 38)),
                }
            ]
        )
    with pytest.raises(ValueError, match="unknown model"):
        _plan(
            [
                {
                    "source": "s0",
                    "preprocessor": "plain",
                    "model": "nope",
                    "view": "rh",
                    "points": list(range(19, 38)),
                }
            ]
        )


def test_pathway_rejects_unknown_view_and_out_of_range_point():
    with pytest.raises(ValueError, match="unknown view"):
        _plan(
            [
                {
                    "source": "s0",
                    "preprocessor": "plain",
                    "model": "m",
                    "view": "ghost",
                    "points": list(range(19, 38)),
                }
            ]
        )
    with pytest.raises(ValueError, match="outside"):
        _plan(
            [
                {
                    "source": "s0",
                    "preprocessor": "plain",
                    "model": "m",
                    "view": "rh",
                    "points": [999] + [-1] * 18,
                }
            ]
        )


def test_pathway_channel_count_must_fit_model():
    # A points list longer than the model's channel count maps a channel out of
    # range.
    with pytest.raises(ValueError, match="channel outside"):
        _plan(
            [
                {
                    "source": "s0",
                    "preprocessor": "plain",
                    "model": "m",
                    "view": "rh",
                    "points": list(range(19, 38)) + [0],
                }
            ]
        )


def test_map_form_with_view_index_resolves():
    plan = _plan(
        [
            {
                "source": "s0",
                "preprocessor": "plain",
                "model": "m",
                "map": [[0, 0, 19], [1, 1, 0]],
            }
        ]
    )
    pw = plan.pathways[0]
    np.testing.assert_array_equal(pw.mapping, [[0, 0, 19], [1, 1, 0]])
