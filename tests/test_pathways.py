"""Tests for the detection plan: parsing/validation, the (i, v, p) scatter and
the coordinate inverse that maps a model peak back into its view frame."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import DEEPFLY3D_SKELETON_PATH, output_points_table

from deeperfly.config import Config
from deeperfly.pose2d.pathways import (
    normalized_peaks_to_original_pixels,
    route_channels_to_points_in_views,
)
from deeperfly.preprocessing import Fliplr, FrameTransform, Resize


def _fly38_table() -> dict:
    """The retired DeepFly3D skeleton table.

    These plans are the SPARSE, mirrored ones -- a pathway detects one body side and its
    twin supplies the other -- which only means anything against a skeleton whose 38
    points are two mirrored 19-point halves. That is the DeepFly3D set, not "whatever
    ships": on the packaged midline ``fly38`` a mirrored channel landing on point
    ``i + 19`` is a genuine left/right error and the mirror check correctly says so.
    """
    return Config.from_dict(
        {"skeleton": {"include": str(DEEPFLY3D_SKELETON_PATH)}}
    ).data["skeleton"]


def _config(pathways, output_points, cameras=None, models=None):
    """Build a plan from explicit ``[[pose2d.pathways]]`` and ``[pose2d.output_points]`` tables."""
    skel = _fly38_table()
    data = {
        "sources": [{"name": "s0", "filename": "a"}, {"name": "s1", "filename": "b"}],
        "pose2d": {
            "preprocessors": [
                {"name": "plain", "ops": []},
                {"name": "mirror", "ops": [{"op": "fliplr"}]},
            ],
            "models": models
            or [
                {
                    "name": "m",
                    "class": "hrnet",
                    "input_size": [256, 512],
                    "n_out_channels": 19,
                }
            ],
            "pathways": pathways,
            "output_points": output_points,
        },
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

    Each spec gets a pathway named ``"<view>_p"`` and a ``[pose2d.output_points.<view>]``
    table derived from ``points`` (``points[i]`` = the point index channel ``i``
    fills, ``-1`` to drop).
    """
    point_names = _fly38_table()["points"]
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
    return _config(pathways, output_points_table(point_names, ps_specs), **kwargs)


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


def test_footage_by_view_rekeys_source_footage_to_view_names():
    # The pipeline resolves footage keyed by *source* name, but records it keyed by
    # *view* name so the viewer can match each camera to its footage (the regression
    # behind the GUI showing blank frames). Two sources s0/s1 feed views rh/lf.
    from pathlib import Path

    from deeperfly.pipeline.run import _footage_by_view

    point_names = _fly38_table()["points"]
    pathways = [
        {"name": "rh_p", "source": "s0", "preprocessor": "plain", "model": "m"},
        {"name": "lf_p", "source": "s1", "preprocessor": "mirror", "model": "m"},
    ]
    output_points = output_points_table(
        point_names,
        [("rh", "rh_p", list(range(19, 38))), ("lf", "lf_p", list(range(0, 19)))],
    )
    config = Config.from_dict(
        {
            "sources": [
                {"name": "s0", "filename": "a"},
                {"name": "s1", "filename": "b"},
            ],
            "pose2d": {
                "preprocessors": [
                    {"name": "plain", "ops": []},
                    {"name": "mirror", "ops": [{"op": "fliplr"}]},
                ],
                "models": [
                    {
                        "name": "m",
                        "class": "hrnet",
                        "input_size": [256, 512],
                        "n_out_channels": 19,
                    }
                ],
                "pathways": pathways,
                "output_points": output_points,
            },
            "cameras": {
                "rh": {"azimuth_deg": 0, "distance": 100, "focal_length_px": 1},
                "lf": {"azimuth_deg": 1, "distance": 100, "focal_length_px": 1},
            },
            "skeleton": _fly38_table(),
        }
    )

    sources = {"s0": [Path("/foot/cam0.mp4")], "s1": [Path("/foot/cam1.mp4")]}
    assert _footage_by_view(config, sources) == {
        "rh": [Path("/foot/cam0.mp4")],
        "lf": [Path("/foot/cam1.mp4")],
    }
    # No footage to record -> no footage attr written.
    assert _footage_by_view(config, None) is None
    assert _footage_by_view(config, {}) is None


# -- validation ---------------------------------------------------------------


def _ps(view, pathway, **points):
    """A single ``[pose2d.output_points.<view>]`` table (point_name=out_channel kwargs)."""
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


def test_unmapped_pathway_not_named_after_a_view_is_rejected():
    """An unmapped pathway falls back to "channel i -> point i of my own view", so a
    pathway whose name is not a view has no view to default INTO."""
    with pytest.raises(ValueError, match="unknown view 'idle'"):
        _config(
            _one_pathway(name="rh_p") + _one_pathway(name="idle", source="s1"),
            _ps("rh", "rh_p", rf_thorax_coxa=0),
        )


def test_unmapped_pathway_of_a_partial_channel_model_is_rejected():
    """The identity default is only meaningful for a detector that predicts EVERY point.

    The shipped 19-channel detector emits one side of the animal, so which points its
    channels mean differs per view and there is no identity to fall back on. The refusal
    names both counts rather than saying the table is missing, because the missing table
    is the symptom and the channel count is the reason.
    """
    with pytest.raises(ValueError, match="19 channels for a 38-point skeleton"):
        _config(_one_pathway(name="rh"), {})


def test_a_dense_pathway_needs_no_output_points_table():
    """channel i -> point i of the pathway's own view, with no table written at all.

    This is what lets a dense config drop 38 x V lines of identity mapping. The check that
    the model's channels really ARE this skeleton's, in this order, cannot happen here --
    the plan is parsed torch-free, before any weights are read -- so it lives in
    `stream.load_models` and runs on every load.
    """
    dense = [
        {
            "name": "m",
            "class": "hrnet",
            "input_size": [256, 512],
            "n_out_channels": 38,
            "weights": "unused.pt",
        }
    ]
    plan = _config(_one_pathway(name="rh"), {}, models=dense)
    (pathway,) = plan.pathways
    mapping = pathway.mapping
    assert mapping.shape == (38, 3)
    rh = plan.view_names.index("rh")
    # (channel, view, point) == (i, rh, i) for every i, and NOTHING lands in another view.
    assert np.array_equal(mapping[:, 0], np.arange(38))
    assert np.array_equal(mapping[:, 2], np.arange(38))
    assert set(mapping[:, 1].tolist()) == {rh}


def test_an_explicit_table_and_the_identity_default_agree_for_a_dense_pathway():
    """The default is not a different mapping, it is the same one unwritten.

    Written out because the whole argument for deleting the generated table is that it
    carried no information; if these two disagreed, it carried some.
    """
    dense = [
        {
            "name": "m",
            "class": "hrnet",
            "input_size": [256, 512],
            "n_out_channels": 38,
            "weights": "unused.pt",
        }
    ]
    names = list(_fly38_table()["points"])
    explicit = {
        "rh": {n: {"pathway": "rh", "out_channel": i} for i, n in enumerate(names)}
    }
    a = _config(_one_pathway(name="rh"), explicit, models=dense).pathways[0].mapping
    b = _config(_one_pathway(name="rh"), {}, models=dense).pathways[0].mapping
    assert np.array_equal(a, b)


def _model(**over):
    m = {
        "name": "m",
        "class": "hrnet",
        "input_size": [256, 512],
        "n_out_channels": 19,
    }
    m.update(over)
    return [m]


def test_model_precision_parsed_as_first_class_field_not_kwargs():
    # A per-model `precision` is a real ModelSpec field, NOT swept into kwargs
    # (kwargs is forwarded to the model loader constructor, so it would be
    # mis-delivered instead of reaching set_precision).
    plan = _config(
        _one_pathway(),
        _ps("rh", "rh_p", rf_thorax_coxa=0),
        models=_model(precision="bfloat16"),
    )
    spec = plan.models["m"]
    assert spec.precision == "bfloat16"
    assert "precision" not in spec.kwargs


@pytest.mark.parametrize("over", [{}, {"precision": ""}])
def test_model_precision_absent_or_empty_inherits(over):
    # Omitted or "" -> None, so the model inherits the [pose2d].precision default.
    plan = _config(
        _one_pathway(), _ps("rh", "rh_p", rf_thorax_coxa=0), models=_model(**over)
    )
    assert plan.models["m"].precision is None


# -- the mirror check ---------------------------------------------------------
#
# A mirrored pathway must land on the MIRRORED points. Nothing enforced that before the
# skeleton declared its symmetry pairs, so a one-word typo in one of the packaged config's
# 132 output_points rows swapped a body side silently: the detector still fires, the
# triangulation still converges, and the reconstruction is a fly with its legs crossed.


def _mirror_config(left_point, *, symmetries=None, mirror_left=True):
    """A two-view rig: one plain pathway onto ``r_a``, one mirrored onto ``left_point``."""
    skel = {"points": ["l_a", "r_a", "l_b", "r_b"]}
    if symmetries is not None:
        skel["symmetries"] = symmetries
    return Config.from_dict(
        {
            "skeleton": skel,
            "default_camera": {"distance": 1.0},
            "cameras": {
                "left": {"azimuth_deg": 90},
                "right": {"azimuth_deg": -90},
            },
            "sources": [{"name": "vid", "filename": "vid*.mp4"}],
            "pose2d": {
                "preprocessors": [{"name": "flip", "ops": [{"op": "fliplr"}]}],
                "models": [{"name": "m", "class": "hrnet", "n_out_channels": 2}],
                "pathways": [
                    {"name": "plain", "source": "vid", "model": "m"},
                    {
                        "name": "mir",
                        "source": "vid",
                        **({"preprocessor": "flip"} if mirror_left else {}),
                        "model": "m",
                    },
                ],
                "output_points": {
                    "right": {"r_a": {"pathway": "plain", "out_channel": 0}},
                    "left": {left_point: {"pathway": "mir", "out_channel": 0}},
                },
            },
        }
    )


PAIRS = [["l_a", "r_a"], ["l_b", "r_b"]]


def test_a_real_mirrored_plan_satisfies_the_mirror_invariant():
    """Every mirrored channel of a real 19-channel config lands on the symmetric partner.

    Run against the shipped-before-dense plan rather than the packaged one: the packaged
    detector is dense, so it has no mirrored pathway and nothing for this check to
    compare. That is not the invariant weakening -- it is the config no longer relying on
    132 hand-written rows for its left/right identities, which is what the check existed
    to police.
    """
    from helpers import sparse_config

    plan = sparse_config().detection_plan()
    mirrored = {pw.name for pw in plan.pathways if pw.transform.reverses_handedness}
    # The check is only meaningful if the config actually has both parities.
    assert mirrored and len(mirrored) < len(plan.pathways)


def test_the_packaged_plan_is_dense_and_needs_no_mirror_check():
    """The shipped plan states its left/right identities structurally, not in a table."""
    plan = Config.default().detection_plan()
    assert not any(pw.transform.reverses_handedness for pw in plan.pathways)
    assert [pw.name for pw in plan.pathways] == plan.view_names
    assert plan.visibility_mask().all()


def test_a_mirrored_pathway_on_the_partner_is_accepted():
    plan = _mirror_config("l_a", symmetries=PAIRS).detection_plan()
    assert len(plan.pathways) == 2


@pytest.mark.parametrize("wrong", ["l_b", "r_b", "r_a"])
def test_a_mirrored_pathway_on_the_wrong_point_is_refused(wrong):
    """``r_a`` is the important case: it is what moving a row to the flipped pathway does.

    A point is never its own symmetry partner, so a channel that maps to the same point at
    both parities is caught even though nothing about that mapping is locally malformed.
    """
    with pytest.raises(ValueError, match="left/right"):
        _mirror_config(wrong, symmetries=PAIRS).detection_plan()


def test_the_check_is_skipped_when_the_skeleton_declares_no_pairs():
    """The pairs are the premise. Without them there is nothing to check against, and
    guessing them here would let renaming a point turn a passing config into a failing one.
    """
    plan = _mirror_config("l_b", symmetries=None).detection_plan()
    assert len(plan.pathways) == 2


def test_two_un_mirrored_pathways_are_never_compared():
    """A one-sided rig maps every channel at one parity only, which is legal.

    Also pins that one channel feeding several points stays legal -- ``output_points`` keys
    on ``(view, point)``, so it constrains a point's source, not a channel's fan-out.
    """
    plan = _mirror_config("l_b", symmetries=PAIRS, mirror_left=False).detection_plan()
    assert len(plan.pathways) == 2


def test_an_even_number_of_reflections_is_not_a_mirror():
    """``fliplr`` + ``flipud`` is a half-turn: it preserves handedness, so no swap is due."""
    cfg = Config.from_dict(
        {
            "skeleton": {"points": ["l_a", "r_a"], "symmetries": [["l_a", "r_a"]]},
            "default_camera": {"distance": 1.0},
            "cameras": {"a": {"azimuth_deg": 0}},
            "sources": [{"name": "vid", "filename": "v"}],
            "pose2d": {
                "preprocessors": [
                    {"name": "half_turn", "ops": [{"op": "fliplr"}, {"op": "flipud"}]}
                ],
                "models": [{"name": "m", "class": "hrnet", "n_out_channels": 1}],
                "pathways": [
                    {"name": "plain", "source": "vid", "model": "m"},
                    {
                        "name": "turned",
                        "source": "vid",
                        "preprocessor": "half_turn",
                        "model": "m",
                    },
                ],
                "output_points": {
                    "a": {"r_a": {"pathway": "plain", "out_channel": 0}},
                },
            },
        }
    )
    # Both pathways are un-mirrored, so `turned` may target the SAME point as `plain`.
    cfg.data["pose2d"]["output_points"]["a"]["l_a"] = {
        "pathway": "turned",
        "out_channel": 0,
    }
    plan = cfg.detection_plan()
    assert not any(pw.transform.reverses_handedness for pw in plan.pathways)
