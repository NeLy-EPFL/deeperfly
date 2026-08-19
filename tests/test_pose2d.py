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
from helpers import DEEPFLY3D_SKELETON_PATH, output_points_table

from deeperfly.config import Config
from deeperfly.pose2d import detector, inference
from deeperfly.pose2d.models import LoadedModel, ModelSpec

# -- heatmap decoding --------------------------------------------------------


def test_heatmap_to_points_argmax_and_conf():
    hm = np.zeros((1, 2, 64, 128), dtype=np.float32)
    hm[0, 0, 10, 20] = 5.0
    hm[0, 1, 30, 100] = 3.0
    # A lone spike has no neighbourhood mass, so every method returns its cell,
    # decoded at the cell centre (+0.5; see heatmap_to_points).
    for method in ("argmax", "weighted", "taylor"):
        points, conf = inference.heatmap_to_points(hm, method=method)
        np.testing.assert_allclose(points[0, 0], [(20 + 0.5) / 128, (10 + 0.5) / 64])
        np.testing.assert_allclose(points[0, 1], [(100 + 0.5) / 128, (30 + 0.5) / 64])
        np.testing.assert_allclose(conf[0], [5.0, 3.0])


@pytest.fixture
def module() -> "torch.nn.Module":
    """A minimal stand-in for a detector module.

    What the tests below exercise is the plumbing every class shares -- the recorded
    forward precision, `LoadedModel.prepare`'s resize-and-normalize, and the pathway
    fan-out -- none of which depends on an architecture. A real network here would make
    them slow and would tie them to whichever one happened to be shipping.
    """
    import torch.nn as nn

    from deeperfly.pose2d import inference as _inf

    class _Impl:
        """The `impl` seam a real class attaches at load time: prepare/forward/decode."""

        @staticmethod
        def predict_heatmaps(module, inputs):
            import numpy as _np

            x = (
                inputs
                if isinstance(inputs, torch.Tensor)
                else torch.as_tensor(_np.asarray(inputs))
            )
            with torch.inference_mode():
                return module(x.float())[-1].cpu().numpy()

        @classmethod
        def predict_points(cls, module, inputs, *, method="weighted", radius=2):
            hm = cls.predict_heatmaps(module, inputs)
            return _inf.heatmap_to_points(hm, method=method, radius=radius)

    class _Stub(nn.Module):
        impl = _Impl

        def forward(self, x):  # (..., C, H, W) -> one heatmap stack, stride 4
            lead = x.shape[:-3]
            h, w = x.shape[-2] // 4, x.shape[-1] // 4
            hm = torch.zeros(*lead, 19, h, w)
            hm[..., h // 2, w // 2] = 1.0  # one peak, so a decode has something to find
            return [hm]

    return _Stub().eval()


def test_set_precision_accepts_and_rejects(module):
    model = module
    for p in ("float32", "float16", "bfloat16"):
        detector.set_precision(model, p)  # all valid; autocast is a CUDA no-op here
    with pytest.raises(ValueError, match="unknown detector precision"):
        detector.set_precision(model, "int8")


def test_heatmap_to_points_subpixel_recovers_offgrid_gaussian():
    hh, ww = 64, 128
    ys, xs = np.mgrid[0:hh, 0:ww]
    true_r, true_c = 10.7, 20.3  # centre between cells
    hm = np.exp(-((ys - true_r) ** 2 + (xs - true_c) ** 2) / 2.0)[None, None]

    # Decoder returns cell-centre coordinates (+0.5), so a peak at heatmap index
    # p decodes to p + 0.5; compare against the shifted truth.
    def err(method):
        pts, _ = inference.heatmap_to_points(hm, method=method)
        return (
            abs(pts[0, 0, 0] * ww - (true_c + 0.5)),
            abs(pts[0, 0, 1] * hh - (true_r + 0.5)),
        )

    ax, ay = err("argmax")
    sx, sy = err("weighted")
    tx, ty = err("taylor")
    assert ax >= 0.3 and ay >= 0.3  # arg-max is quantized to the cell
    assert sx < 0.1 and sy < 0.1  # centroid lands well inside the cell
    assert tx < 1e-2 and ty < 1e-2  # Taylor is near-exact on a clean Gaussian


# -- model input preparation -------------------------------------------------


def _loaded(model, n_out_channels=19, input_size=(256, 512), mean=0.0):
    spec = ModelSpec(
        name="m",
        cls="hrnet",
        weights=None,
        input_size=input_size,
        mean=mean,
        n_out_channels=n_out_channels,
    )
    return LoadedModel(spec, model)


def test_model_prepare_shape_and_mean(module):
    model = module
    gray = np.full((200, 100, 3), 128, dtype=np.uint8)  # 128/255 ~ 0.502
    out = _loaded(model).prepare(gray)
    # ONE plane out: every shipped detector takes one, so nothing is replicated to three.
    assert tuple(out.shape) == (1, 256, 512)
    np.testing.assert_allclose(np.asarray(out), 128 / 255, atol=1e-4)


def test_model_prepare_accepts_on_device_tensor(module):
    model = module
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
            "class": "hrnet",
            "input_size": [256, 512],
            "n_out_channels": 19,
        }
    ]


def _fly38() -> dict:
    """The retired DeepFly3D skeleton table, spelled out.

    These plans are the SPARSE, mirrored ones, which only mean anything against a
    skeleton whose 38 points are two mirrored 19-point halves. That is the DeepFly3D set
    specifically, not "whatever the package ships" -- the packaged ``fly38`` is the
    midline one, where points ``i`` and ``i + 19`` are not each other's mirror image.
    """
    return Config.from_dict({"skeleton": {"file": str(DEEPFLY3D_SKELETON_PATH)}}).data[
        "skeleton"
    ]


def _mini_plan():
    """A 2-source / 2-pathway / 2-view plan: rh (plain) and lf (mirrored)."""
    skel = _fly38()
    point_names = skel["point_names"]
    data = {
        "sources": [{"name": "s0", "filename": "a"}, {"name": "s1", "filename": "b"}],
        "pose2d": {
            "preprocessors": [
                {"name": "plain", "ops": []},
                {"name": "mirror", "ops": [{"op": "fliplr"}]},
            ],
            "models": _model_list(),
            "pathways": [
                {"name": "rh_p", "source": "s0", "preprocessor": "plain", "model": "m"},
                {
                    "name": "lf_p",
                    "source": "s1",
                    "preprocessor": "mirror",
                    "model": "m",
                },
            ],
            "output_points": output_points_table(
                point_names,
                [
                    ("rh", "rh_p", list(range(19, 38))),
                    ("lf", "lf_p", list(range(0, 19))),
                ],
            ),
        },
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
    skel = _fly38()
    point_names = skel["point_names"]
    data = {
        "sources": [{"name": "fcam", "filename": "f"}],
        "pose2d": {
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
            "output_points": output_points_table(
                point_names,
                [
                    ("f", "f_plain", list(range(19, 38))),
                    ("f", "f_mirror", list(range(0, 19))),
                ],
            ),
        },
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


def test_detect_sequence_shapes_and_scatter(module):
    model = module
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


def test_detect_sequence_chunking_is_equivalent(module):
    model = module
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


def test_detect_sequence_batched_matches_per_frame(module, monkeypatch):
    model = module

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

    monkeypatch.setattr(model.impl, "predict_points", fake_predict_points)
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


def test_prepared_inputs_passed_in_match_preparing_them_inline(module):
    model = module
    # A streaming caller prepares the NEXT window while this one is in the network, so it
    # hands detect_sequence the result. Same arithmetic, only moved: passing `prepared` must
    # give exactly what preparing inline gives.
    plan = _mini_plan()
    models = _models(plan, model)
    rng = np.random.default_rng(7)
    windows = {
        s.name: rng.uniform(size=(3, 64, 128, 3)).astype(np.float32)
        for s in plan.sources
    }
    ref_pts, ref_conf = inference.detect_sequence(plan, models, windows)
    prepared = inference.prepare_pathways(plan, models, windows)
    got_pts, got_conf = inference.detect_sequence(
        plan, models, windows, prepared=prepared
    )
    np.testing.assert_array_equal(got_pts, ref_pts)
    np.testing.assert_array_equal(got_conf, ref_conf)


def test_prepare_pathways_is_order_stable_with_and_without_a_pool(module):
    model = module
    # The pathways are prepared concurrently, so the results must still line up with
    # plan.pathways -- a pool that returned them out of order would attach every view's
    # detections to the wrong camera and never raise.
    from concurrent.futures import ThreadPoolExecutor

    plan = _mini_plan()
    models = _models(plan, model)
    rng = np.random.default_rng(8)
    windows = {
        s.name: rng.uniform(size=(2, 64, 128, 3)).astype(np.float32)
        for s in plan.sources
    }
    with ThreadPoolExecutor(max_workers=4) as pool:
        pooled = inference.prepare_pathways(plan, models, windows, pool=pool)
    own = inference.prepare_pathways(plan, models, windows)
    assert len(pooled) == len(plan.pathways)
    for a, b in zip(pooled, own):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_a_host_preparing_model_is_not_handed_a_device_window(module):
    model = module
    # The upload a host-side preparation would immediately undo is skipped. Declared by the
    # model, so a plan mixing one with a device-side model still uploads the shared source.
    plan = _mini_plan()
    models = _models(plan, model)
    rng = np.random.default_rng(9)
    windows = {
        s.name: rng.uniform(size=(2, 64, 128, 3)).astype(np.float32)
        for s in plan.sources
    }
    # Nothing declares prepares_on_host -> every window is moved, as before.
    moved = inference._windows_to_device(plan, models, windows, "cpu")
    assert all(hasattr(w, "device") for w in moved.values())

    # Both flags, and that pairing is the point: the shared prepare is a torch resize, so
    # only a model that ALSO owns its preparation can be trusted to want host frames.
    for m in models.values():
        m.module.owns_prepare = True
        m.module.prepares_on_host = True
    try:
        assert all(m.prepares_on_host for m in models.values())
        kept = inference._windows_to_device(plan, models, windows, "cpu")
        for name, w in kept.items():
            assert w is windows[name], (
                f"{name} should have stayed on the host untouched"
            )
        # Dropping owns_prepare alone puts the upload back -- the two are read together.
        for m in models.values():
            m.module.owns_prepare = False
        assert not any(m.prepares_on_host for m in models.values())
    finally:
        for m in models.values():
            del m.module.prepares_on_host, m.module.owns_prepare


def test_front_source_two_pathways_bridge_both_sides(module):
    model = module
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


def test_detect_single_frame_matches_sequence(module):
    model = module
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


# -- the channel-order gate, at LOAD time -------------------------------------


def _plan_with_points(names):
    import types

    return types.SimpleNamespace(point_names=tuple(names))


def _model_with_points(names):
    import types

    return types.SimpleNamespace(module=types.SimpleNamespace(point_names=list(names)))


def test_load_refuses_a_model_trained_on_another_skeleton():
    """A dense detector's channels ARE a skeleton, and a COUNT check cannot compare two.

    `fly38` and `fly38b` are both 38 points and share 32 of them in a different order, so
    routing one through the other's config attaches six points to the wrong joints and
    shifts the rest. A config is written against one skeleton's order, but a
    generator never sees a `weights` path later repointed, a `[skeleton]` swapped
    underneath, or a hand-edited mapping -- so the comparison has to happen on every load.
    """
    from deeperfly.pose2d.stream import _check_channel_names

    fly38 = list(Config.default().data["skeleton"]["point_names"])
    fly38b = [n for n in fly38 if "abdomen" not in n] + [
        "neck",
        "abdomen0",
        "abdomen1",
        "abdomen2",
        "abdomen3",
        "abdomen4",
    ]
    with pytest.raises(SystemExit, match="different channel set"):
        _check_channel_names(
            "dense", _model_with_points(fly38b), _plan_with_points(fly38)
        )


def test_load_refuses_the_same_points_in_a_different_order():
    """The order is the mapping. Two configs with identical point SETS still disagree."""
    from deeperfly.pose2d.stream import _check_channel_names

    names = list(Config.default().data["skeleton"]["point_names"])
    swapped = names[:]
    swapped[0], swapped[5] = swapped[5], swapped[0]
    with pytest.raises(SystemExit, match="the ORDER differs"):
        _check_channel_names(
            "dense", _model_with_points(swapped), _plan_with_points(names)
        )


def test_load_accepts_the_matching_skeleton():
    from deeperfly.pose2d.stream import _check_channel_names

    names = list(Config.default().data["skeleton"]["point_names"])
    _check_channel_names("dense", _model_with_points(names), _plan_with_points(names))


def test_a_checkpoint_recording_no_channel_names_is_refused():
    """The permissive branch here existed for one retired network, and is now a hole.

    Every class this build ships records its point names, so a nameless artifact is either
    not one of ours or was stripped -- and skipping the check is skipping the one thing
    standing between a mis-stamped config and a fly with its limbs on the wrong joints. A
    count check cannot substitute: two 38-point skeletons in different orders load each
    other's files happily.
    """
    import types

    from deeperfly.pose2d.stream import _check_channel_names

    names = list(Config.default().data["skeleton"]["point_names"])
    nameless = types.SimpleNamespace(module=types.SimpleNamespace())
    with pytest.raises(SystemExit, match="records no channel names"):
        _check_channel_names("stripped", nameless, _plan_with_points(names))
