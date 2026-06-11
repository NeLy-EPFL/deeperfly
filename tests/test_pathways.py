"""Tests for the detection plan: parsing/validation, the (i, v, p) scatter and
the coordinate inverse that maps a model peak back into its view frame."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import point_sources_table

from deeperfly.config import Config
from deeperfly.pose2d.pathways import (
    normalized_peaks_to_original_pixels,
    route_channels_to_points_in_views,
)
from deeperfly.preprocessing import Fliplr, FrameTransform, Resize


def _config(pathways, point_sources, cameras=None, models=None):
    """Build a plan from explicit ``[[pathways]]`` and ``[point_sources]`` tables."""
    skel = Config.default().data["skeleton"]
    data = {
        "sources": [{"name": "s0", "filename": "a"}, {"name": "s1", "filename": "b"}],
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
                "n_out_channels": 19,
            }
        ],
        "pathways": pathways,
        "point_sources": point_sources,
        "cameras": cameras
        or {
            "rh": {"azimuth_deg": 0, "distance": 100, "focal_length_px": 1},
            "lf": {"azimuth_deg": 1, "distance": 100, "focal_length_px": 1},
        },
        "skeleton": skel,
    }
    return Config.from_dict(data).detection_plan()


def _plan(specs, **kwargs):
    """Build a plan from ``(view, source, preprocessor, points)`` pathway specs.

    Each spec gets a pathway named ``"<view>_p"`` and a ``[point_sources.<view>]``
    table derived from ``points`` (``points[i]`` = the point index channel ``i``
    fills, ``-1`` to drop).
    """
    point_names = Config.default().data["skeleton"]["point_names"]
    pathways, ps_specs = [], []
    for s in specs:
        name = f"{s['view']}_p"
        pathways.append(
            {
                "name": name,
                "source": s["source"],
                "preprocessor": s["preprocessor"],
                "model": s.get("model", "m"),
            }
        )
        ps_specs.append((s["view"], name, s["points"]))
    return _config(pathways, point_sources_table(point_names, ps_specs), **kwargs)


# -- coordinate inverse (normalized_peaks_to_original_pixels) ----------------


def test_normalized_peaks_to_original_pixels_plain_scales_to_source_pixels():
    # No preprocessor: a model peak at normalized (x, y) maps to ~ (x*W, y*H) in
    # source pixels (within the half-pixel resize convention).
    pts = np.array([[0.5, 0.5]])
    out = normalized_peaks_to_original_pixels(
        pts, FrameTransform(()), (256, 512), (480, 960)
    )
    np.testing.assert_allclose(out, [[0.5 * 960, 0.5 * 480]], atol=1.0)


def test_normalized_peaks_to_original_pixels_mirror_undoes_flip():
    # With a fliplr preprocessor the x coordinate is reflected back into the source
    # frame; y is unchanged.
    transform = FrameTransform((Fliplr(),))
    src = (480, 960)
    plain = normalized_peaks_to_original_pixels(
        np.array([[0.3, 0.4]]), FrameTransform(()), (256, 512), src
    )
    mirrored = normalized_peaks_to_original_pixels(
        np.array([[0.3, 0.4]]), transform, (256, 512), src
    )
    # The mirrored x is the reflection of the plain x about the image centre.
    np.testing.assert_allclose(mirrored[0, 0], (960 - 1) - plain[0, 0], atol=1e-6)
    np.testing.assert_allclose(mirrored[0, 1], plain[0, 1], atol=1e-6)


def test_normalized_peaks_to_original_pixels_roundtrips_with_map_points():
    transform = FrameTransform((Fliplr(),))
    src = (480, 960)
    norm = np.array([[0.2, 0.7], [0.9, 0.1]])
    view_px = normalized_peaks_to_original_pixels(norm, transform, (256, 512), src)
    # Forward map (source -> mirrored -> model input) then normalize recovers norm.
    mirror_px = transform.map_points(view_px, src)
    resize = FrameTransform((Resize(width=512, height=256),))
    back = resize.map_points(mirror_px, transform.output_size(src)) / np.array(
        [512, 256]
    )
    np.testing.assert_allclose(back, norm, atol=1e-6)


# -- scatter ------------------------------------------------------------------


def test_route_channels_to_points_in_views_routes_channels_and_leaves_nan():
    mapping = np.array([[0, 0, 5], [2, 1, 7]])  # (i, v, p)
    raw_xy = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    conf = np.array([0.1, 0.2, 0.3])
    out_pts = np.full((2, 10, 2), np.nan)
    out_conf = np.zeros((2, 10))
    route_channels_to_points_in_views(raw_xy, conf, mapping, out_pts, out_conf)
    np.testing.assert_array_equal(out_pts[0, 5], [1.0, 2.0])  # channel 0 -> (0, 5)
    np.testing.assert_array_equal(out_pts[1, 7], [5.0, 6.0])  # channel 2 -> (1, 7)
    assert out_conf[0, 5] == 0.1 and out_conf[1, 7] == 0.3
    assert np.isnan(out_pts[0, 0]).all()  # untouched stays NaN


def test_route_channels_to_points_in_views_candidate_axis():
    # The same scatter handles a trailing K (candidate) axis.
    mapping = np.array([[1, 0, 3]])
    raw_xy = np.zeros((2, 4, 2))
    raw_xy[1] = 9.0
    score = np.zeros((2, 4))
    score[1] = 0.5
    out_pts = np.full((1, 5, 4, 2), np.nan)
    out_conf = np.zeros((1, 5, 4))
    route_channels_to_points_in_views(raw_xy, score, mapping, out_pts, out_conf)
    np.testing.assert_array_equal(out_pts[0, 3], 9.0)
    np.testing.assert_array_equal(out_conf[0, 3], 0.5)


# -- visibility derived from the plan ----------------------------------------


def test_visibility_mask_is_union_of_pathways():
    plan = _plan(
        [
            {
                "source": "s0",
                "preprocessor": "plain",
                "view": "rh",
                "points": [19, 20, -1] + [-1] * 16,
            },
            {
                "source": "s1",
                "preprocessor": "mirror",
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
                "view": "rh",
                "points": list(range(19, 38)),
            },
            {
                "source": "s1",
                "preprocessor": "mirror",
                "view": "lf",
                "points": list(range(0, 19)),
            },
        ]
    )
    assert plan.view_sources() == {"rh": "s0", "lf": "s1"}


# -- validation ---------------------------------------------------------------


def _ps(view, pathway, **points):
    """A single ``[point_sources.<view>]`` table (point_name=out_channel kwargs)."""
    return {
        view: {n: {"pathway": pathway, "out_channel": c} for n, c in points.items()}
    }


def _one_pathway(source="s0", preprocessor="plain", model="m", name="rh_p"):
    return [
        {"name": name, "source": source, "preprocessor": preprocessor, "model": model}
    ]


def test_pathway_rejects_unknown_references():
    ps = _ps("rh", "rh_p", rf_thorax_coxa=0)
    with pytest.raises(ValueError, match="unknown source"):
        _config(_one_pathway(source="nope"), ps)
    with pytest.raises(ValueError, match="unknown preprocessor"):
        _config(_one_pathway(preprocessor="nope"), ps)
    with pytest.raises(ValueError, match="unknown model"):
        _config(_one_pathway(model="nope"), ps)


def test_pathway_omitting_preprocessor_defaults_to_identity():
    # No `preprocessor` key -> no frame ops (identity), like an empty `ops = []`.
    pathway = [{"name": "rh_p", "source": "s0", "model": "m"}]
    plan = _config(pathway, _ps("rh", "rh_p", rf_thorax_coxa=0))
    (pw,) = plan.pathways
    assert pw.preprocessor is None
    assert pw.transform == FrameTransform(())


def test_duplicate_pathway_name_rejected():
    with pytest.raises(ValueError, match="duplicate name"):
        _config(
            _one_pathway(name="dup") + _one_pathway(name="dup", source="s1"),
            _ps("rh", "dup", rf_thorax_coxa=0),
        )


def test_point_sources_rejects_unknown_view():
    with pytest.raises(ValueError, match="unknown view"):
        _config(_one_pathway(), _ps("ghost", "rh_p", rf_thorax_coxa=0))


def test_point_sources_rejects_unknown_point_name():
    with pytest.raises(ValueError, match="not a skeleton point"):
        _config(_one_pathway(), _ps("rh", "rh_p", not_a_point=0))


def test_point_sources_rejects_unknown_pathway():
    with pytest.raises(ValueError, match="unknown pathway"):
        _config(_one_pathway(), _ps("rh", "ghost_pw", rf_thorax_coxa=0))


def test_out_channel_must_fit_model():
    with pytest.raises(ValueError, match="out_channel 19 outside"):
        _config(_one_pathway(), _ps("rh", "rh_p", rf_thorax_coxa=19))


def test_pathway_without_point_sources_rejected():
    # A pathway named by no [point_sources] entry maps nothing -> error.
    with pytest.raises(ValueError, match="no \\[point_sources\\] entries"):
        _config(
            _one_pathway(name="rh_p") + _one_pathway(name="idle", source="s1"),
            _ps("rh", "rh_p", rf_thorax_coxa=0),
        )
