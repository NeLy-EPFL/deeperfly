"""Tests for the 2D pose detector (PyTorch) and the plan-driven orchestration.

Architecture/shape, the torch weight round-trip, heatmap decoding, the model's
input preparation and the plan-driven source -> pathway -> skeleton detection are
checked here. The pathway scatter / coordinate inverse and plan parsing live in
``test_pathways.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from helpers import point_sources_table

from deeperfly.config import Config
from deeperfly.pose2d import detector, inference
from deeperfly.pose2d.model import HourglassNet
from deeperfly.pose2d.models import LoadedModel, ModelSpec
from deeperfly.pose2d.weights import load_model


@pytest.fixture
def model() -> HourglassNet:
    # A small 2-stack model keeps these mechanics tests fast; the published
    # config is 8 stacks (see test_default_is_sh8).
    return HourglassNet(num_stacks=2).eval()


# -- model -------------------------------------------------------------------


def test_forward_shapes(model):
    x = torch.randn(1, 3, 256, 512)
    with torch.inference_mode():
        outs = model(x)
    assert len(outs) == 2  # one heatmap stack per hourglass
    assert tuple(outs[-1].shape) == (1, 19, 64, 128)


def test_forward_view_axis_folds_into_batch(model):
    # The standard input is (B, V, C, H, W): the V views run in parallel and
    # independent, so the result must equal folding them into one (B*V) batch.
    x = torch.randn(2, 4, 3, 256, 512)
    with torch.inference_mode():
        out5 = model(x)[-1]
        out4 = model(x.reshape(8, 3, 256, 512))[-1]
    assert tuple(out5.shape) == (2, 4, 19, 64, 128)
    np.testing.assert_allclose(
        np.asarray(out5).reshape(8, 19, 64, 128), np.asarray(out4), atol=1e-6
    )


def test_default_is_sh8():
    # The shipped DeepFly2D checkpoint is "sh8": the default must be 8 stacks so
    # load_model builds a matching architecture for the published weights.
    assert HourglassNet().num_stacks == 8


def test_batched_inference(model):
    inputs = torch.randn(3, 3, 256, 512)
    hm = detector.predict_heatmaps(model, inputs)
    assert hm.shape == (3, 19, 64, 128)
    assert isinstance(hm, np.ndarray)  # backend always returns host NumPy


def test_predict_view_axis_matches_folded(model):
    # predict_* carry the (B, V) axes through to their outputs and must match the
    # equivalent flat (B*V, ...) batch to float32 epsilon.
    inputs = torch.randn(2, 3, 3, 256, 512)
    hm = detector.predict_heatmaps(model, inputs)
    pts, conf = detector.predict_points(model, inputs)
    assert hm.shape == (2, 3, 19, 64, 128)
    assert pts.shape == (2, 3, 19, 2)
    assert conf.shape == (2, 3, 19)
    hm_flat = detector.predict_heatmaps(model, inputs.reshape(6, 3, 256, 512))
    np.testing.assert_allclose(hm.reshape(6, 19, 64, 128), hm_flat, atol=1e-6)


# -- weight I/O --------------------------------------------------------------


def test_infer_num_stacks_counts_score_heads(model):
    sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    assert detector.infer_num_stacks(sd) == 2


def test_infer_num_stacks_rejects_foreign_state_dict():
    with pytest.raises(KeyError, match="not a HourglassNet"):
        detector.infer_num_stacks({"conv.weight": np.zeros((1,))})


def test_checkpoint_save_load(model, tmp_path):
    # The torch backend loads the original DeepFly2D state_dict directly; saving
    # and reloading must reproduce the same forward (architecture inferred from
    # the checkpoint's score heads).
    path = tmp_path / "model.pth"
    torch.save(model.state_dict(), path)
    loaded = load_model(path, dev="cpu")  # match the CPU fixture for comparison
    x = torch.randn(1, 3, 256, 512)
    with torch.inference_mode():
        np.testing.assert_allclose(
            np.asarray(model(x)[-1]), np.asarray(loaded(x)[-1]), atol=1e-6
        )


# -- heatmap decoding --------------------------------------------------------


def test_heatmap_to_points_argmax_and_conf():
    hm = np.zeros((1, 2, 64, 128), dtype=np.float32)
    hm[0, 0, 10, 20] = 5.0
    hm[0, 1, 30, 100] = 3.0
    # A lone spike has no neighbourhood mass, so every method returns its cell.
    for method in ("argmax", "weighted", "taylor"):
        points, conf = inference.heatmap_to_points(hm, method=method)
        np.testing.assert_allclose(points[0, 0], [20 / 128, 10 / 64])
        np.testing.assert_allclose(points[0, 1], [100 / 128, 30 / 64])
        np.testing.assert_allclose(conf[0], [5.0, 3.0])


def test_predict_points_matches_heatmaps_decode(model):
    # The fused forward+decode (arg-max on the forward's device) must match the
    # reference "predict_heatmaps then heatmap_to_points" path to float32 epsilon,
    # for every sub-pixel method. Exercised on the CPU here, so no GPU is required.
    inputs = torch.randn(3, 3, 256, 512)
    hm = detector.predict_heatmaps(model, inputs)
    for method in ("argmax", "weighted", "taylor"):
        ref_pts, ref_conf = inference.heatmap_to_points(hm, method=method)
        pts, conf = detector.predict_points(model, inputs, method=method)
        np.testing.assert_allclose(pts, ref_pts, atol=1e-4)
        np.testing.assert_allclose(conf, ref_conf, atol=1e-4)


def test_set_precision_accepts_and_rejects(model):
    for p in ("float32", "float16", "bfloat16"):
        detector.set_precision(model, p)  # all valid; autocast is a CUDA no-op here
    with pytest.raises(ValueError, match="unknown detector precision"):
        detector.set_precision(model, "int8")


def test_heatmap_to_points_subpixel_recovers_offgrid_gaussian():
    hh, ww = 64, 128
    ys, xs = np.mgrid[0:hh, 0:ww]
    true_r, true_c = 10.7, 20.3  # centre between cells
    hm = np.exp(-((ys - true_r) ** 2 + (xs - true_c) ** 2) / 2.0)[None, None]

    def err(method):
        pts, _ = inference.heatmap_to_points(hm, method=method)
        return abs(pts[0, 0, 0] * ww - true_c), abs(pts[0, 0, 1] * hh - true_r)

    ax, ay = err("argmax")
    sx, sy = err("weighted")
    tx, ty = err("taylor")
    assert ax >= 0.3 and ay >= 0.3  # arg-max is quantized to the cell
    assert sx < 0.1 and sy < 0.1  # centroid lands well inside the cell
    assert tx < 1e-2 and ty < 1e-2  # Taylor is near-exact on a clean Gaussian


# -- model input preparation -------------------------------------------------


def _loaded(model, n_out_channels=19, input_size=(256, 512), mean=0.22):
    spec = ModelSpec(
        name="m",
        cls="hourglass",
        weights=None,
        input_size=input_size,
        mean=mean,
        n_out_channels=n_out_channels,
    )
    return LoadedModel(spec, model)


def test_model_prepare_shape_and_mean(model):
    gray = np.full((200, 100, 3), 128, dtype=np.uint8)  # 128/255 ~ 0.502
    out = _loaded(model).prepare(gray)
    assert tuple(out.shape) == (3, 256, 512)
    np.testing.assert_allclose(np.asarray(out), 128 / 255 - 0.22, atol=1e-4)


def test_model_prepare_accepts_on_device_tensor(model):
    # A caller may hand frames in as a torch.Tensor (e.g. already on the GPU);
    # prepare must keep them on the tensor's device and match the NumPy path.
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
    lm = _loaded(model)
    from_numpy = np.asarray(lm.prepare(img))
    from_tensor = np.asarray(lm.prepare(torch.from_numpy(img)))
    np.testing.assert_array_equal(from_numpy, from_tensor)


# -- plan-driven detection ---------------------------------------------------


def _model_list():
    return [
        {
            "name": "m",
            "class": "hourglass",
            "input_size": [256, 512],
            "n_out_channels": 19,
        }
    ]


def _mini_plan():
    """A 2-source / 2-pathway / 2-view plan: rh (plain) and lf (mirrored)."""
    skel = Config.default().data["skeleton"]
    point_names = skel["point_names"]
    data = {
        "sources": [{"name": "s0", "filename": "a"}, {"name": "s1", "filename": "b"}],
        "preprocessors": [
            {"name": "plain", "ops": []},
            {"name": "mirror", "ops": [{"op": "fliplr"}]},
        ],
        "models": _model_list(),
        "pathways": [
            {"name": "rh_p", "source": "s0", "preprocessor": "plain", "model": "m"},
            {"name": "lf_p", "source": "s1", "preprocessor": "mirror", "model": "m"},
        ],
        "point_sources": point_sources_table(
            point_names,
            [("rh", "rh_p", list(range(19, 38))), ("lf", "lf_p", list(range(0, 19)))],
        ),
        "cameras": {
            "rh": {
                "azimuth_deg": -120,
                "distance": 100,
                "focal_length_px": 1000,
                "principal_point_px": [63.5, 31.5],
            },
            "lf": {
                "azimuth_deg": 45,
                "distance": 100,
                "focal_length_px": 1000,
                "principal_point_px": [63.5, 31.5],
            },
        },
        "skeleton": skel,
    }
    return Config.from_dict(data).detection_plan()


def _front_plan():
    """A 1-source / 2-pathway / 1-view plan: the front source bridges both sides."""
    skel = Config.default().data["skeleton"]
    point_names = skel["point_names"]
    data = {
        "sources": [{"name": "fcam", "filename": "f"}],
        "preprocessors": [
            {"name": "plain", "ops": []},
            {"name": "mirror", "ops": [{"op": "fliplr"}]},
        ],
        "models": _model_list(),
        "pathways": [
            {
                "name": "f_plain",
                "source": "fcam",
                "preprocessor": "plain",
                "model": "m",
            },
            {
                "name": "f_mirror",
                "source": "fcam",
                "preprocessor": "mirror",
                "model": "m",
            },
        ],
        "point_sources": point_sources_table(
            point_names,
            [
                ("f", "f_plain", list(range(19, 38))),
                ("f", "f_mirror", list(range(0, 19))),
            ],
        ),
        "cameras": {
            "f": {
                "azimuth_deg": 0,
                "distance": 100,
                "focal_length_px": 1000,
                "principal_point_px": [63.5, 31.5],
            }
        },
        "skeleton": skel,
    }
    return Config.from_dict(data).detection_plan()


def _models(plan, model):
    return {name: LoadedModel(spec, model) for name, spec in plan.models.items()}


def test_detect_sequence_shapes_and_scatter(model):
    plan = _mini_plan()
    models = _models(plan, model)
    rng = np.random.default_rng(0)
    windows = {
        s.name: rng.uniform(size=(2, 64, 128, 3)).astype(np.float32)
        for s in plan.sources
    }
    pts, conf = inference.detect_sequence(plan, models, windows)
    assert pts.shape == (2, 2, 38, 2)
    assert conf.shape == (2, 2, 38)
    # rh (view 0) fills the right half (19..37); lf (view 1) the left half (0..18).
    assert not np.isnan(pts[0, :, 19:]).any()
    assert np.isnan(pts[0, :, :19]).all()
    assert not np.isnan(pts[1, :, :19]).any()
    assert np.isnan(pts[1, :, 19:]).all()


def test_detect_sequence_chunking_is_equivalent(model):
    # Detection is per-frame independent, so processing a clip in windows and
    # concatenating along time must equal one full pass.
    plan = _mini_plan()
    models = _models(plan, model)
    rng = np.random.default_rng(2)
    windows = {
        s.name: rng.uniform(size=(7, 48, 64, 3)).astype(np.float32)
        for s in plan.sources
    }
    full_pts, full_conf = inference.detect_sequence(plan, models, windows)
    a = 4
    w0 = {n: w[:a] for n, w in windows.items()}
    w1 = {n: w[a:] for n, w in windows.items()}
    p0, c0 = inference.detect_sequence(plan, models, w0)
    p1, c1 = inference.detect_sequence(plan, models, w1)
    np.testing.assert_allclose(
        np.concatenate([p0, p1], axis=1), full_pts, atol=1e-5, equal_nan=True
    )
    np.testing.assert_allclose(np.concatenate([c0, c1], axis=1), full_conf, atol=1e-5)


def test_detect_sequence_batched_matches_per_frame(model, monkeypatch):
    # Batching the forward over more frames per call only regroups inputs, so it must
    # yield the same skeletons as the per-frame path. Stub the fused forward+decode
    # with a deterministic peak keyed on each input's content (identical however the
    # (B, V, ...) input is chunked), carrying the (B, V) leading axes through.
    def fake_predict_points(_model, inputs, *, method="weighted", radius=2):
        x = np.asarray(inputs.cpu() if hasattr(inputs, "cpu") else inputs)
        lead, flat = x.shape[:-3], x.reshape(-1, *x.shape[-3:])
        pts = np.zeros((flat.shape[0], 19, 2), np.float32)
        conf = np.ones((flat.shape[0], 19), np.float32)
        for i in range(flat.shape[0]):
            r, c = divmod(int(abs(flat[i]).sum() * 1e3) % (64 * 128), 128)
            pts[i, :, 0], pts[i, :, 1] = c / 128, r / 64
        return pts.reshape(*lead, 19, 2), conf.reshape(*lead, 19)

    monkeypatch.setattr(detector, "predict_points", fake_predict_points)
    plan = _mini_plan()
    models = _models(plan, model)
    rng = np.random.default_rng(3)
    windows = {
        s.name: rng.uniform(size=(5, 64, 64, 3)).astype(np.float32)
        for s in plan.sources
    }
    ref_pts, ref_conf = inference.detect_sequence(plan, models, windows)
    for bs in (1, 3, 64):  # < pathways, straddling, and >> the whole window
        p, c = inference.detect_sequence(plan, models, windows, batch_size=bs)
        np.testing.assert_array_equal(p, ref_pts)
        np.testing.assert_array_equal(c, ref_conf)


def test_front_source_two_pathways_bridge_both_sides(model):
    # One front source feeds two pathways (one mirrored) into a single view, so the
    # front row carries BOTH body halves with no NaN.
    plan = _front_plan()
    models = _models(plan, model)
    rng = np.random.default_rng(1)
    windows = {"fcam": rng.uniform(size=(2, 64, 128, 3)).astype(np.float32)}
    pts, conf = inference.detect_sequence(plan, models, windows)
    assert pts.shape == (1, 2, 38, 2)
    assert not np.isnan(pts[0]).any()  # both halves filled on the bridging view
    # The untrained model can emit negative peak values; just require all filled.
    assert np.isfinite(conf[0]).all()


def test_detect_single_frame_matches_sequence(model):
    plan = _mini_plan()
    models = _models(plan, model)
    rng = np.random.default_rng(4)
    images = {
        s.name: rng.uniform(size=(64, 96, 3)).astype(np.float32) for s in plan.sources
    }
    pts, conf = inference.detect(plan, models, images)
    windows = {n: im[None] for n, im in images.items()}
    seq_pts, seq_conf = inference.detect_sequence(plan, models, windows)
    np.testing.assert_allclose(pts, seq_pts[:, 0], atol=1e-5, equal_nan=True)
    np.testing.assert_allclose(conf, seq_conf[:, 0], atol=1e-5)
