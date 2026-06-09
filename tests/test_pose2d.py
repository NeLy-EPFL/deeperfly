"""Tests for the 2D pose detector (PyTorch) and the shared orchestration.

Architecture/shape, the torch weight round-trip, heatmap decoding, preprocessing
and the single-side -> full-skeleton assembly are checked here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deeperfly.pose2d import backends, inference
from deeperfly.pose2d.backends.torch import HourglassNet, load_model


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


def test_default_is_sh8():
    # The shipped DeepFly2D checkpoint is "sh8": the default must be 8 stacks so
    # load_model builds a matching architecture for the published weights.
    assert HourglassNet().num_stacks == 8


def test_batched_inference(model):
    inputs = torch.randn(3, 3, 256, 512)
    hm = backends.predict_heatmaps(model, inputs)
    assert hm.shape == (3, 19, 64, 128)
    assert isinstance(hm, np.ndarray)  # backend always returns host NumPy


# -- weight I/O --------------------------------------------------------------


def test_infer_num_stacks_counts_score_heads(model):
    sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    assert backends.infer_num_stacks(sd) == 2


def test_infer_num_stacks_rejects_foreign_state_dict():
    with pytest.raises(KeyError, match="not a HourglassNet"):
        backends.infer_num_stacks({"conv.weight": np.zeros((1,))})


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
    hm = backends.predict_heatmaps(model, inputs)
    for method in ("argmax", "weighted", "taylor"):
        ref_pts, ref_conf = inference.heatmap_to_points(hm, method=method)
        pts, conf = backends.predict_points(model, inputs, method=method)
        np.testing.assert_allclose(pts, ref_pts, atol=1e-4)
        np.testing.assert_allclose(conf, ref_conf, atol=1e-4)


def test_set_precision_accepts_and_rejects(model):
    for p in ("float32", "float16", "bfloat16"):
        backends.set_precision(model, p)  # all valid; autocast is a CUDA no-op here
    with pytest.raises(ValueError, match="unknown detector precision"):
        backends.set_precision(model, "int8")


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


# -- preprocessing -----------------------------------------------------------


def test_preprocess_shape_and_mean():
    gray = np.full((200, 100, 3), 128, dtype=np.uint8)  # 128/255 ~ 0.502
    out = inference.preprocess(gray)
    assert tuple(out.shape) == (3, 256, 512)
    np.testing.assert_allclose(np.asarray(out), 128 / 255 - inference.MEAN, atol=1e-4)


def test_preprocess_flip_commutes_with_resize():
    rng = np.random.default_rng(0)
    img = rng.uniform(size=(200, 100, 3)).astype(np.float32)
    a = np.asarray(inference.preprocess(img, flip=True))
    b = np.asarray(inference.preprocess(img, flip=False))[:, :, ::-1]
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_preprocess_accepts_on_device_tensor():
    # A caller may hand frames in as a torch.Tensor (e.g. already on the GPU);
    # preprocess must keep them on the tensor's device and produce the same result
    # as the NumPy path. Exercised on the CPU here, so no GPU is required.
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
    from_numpy = np.asarray(inference.preprocess(img))
    from_tensor = np.asarray(inference.preprocess(torch.from_numpy(img)))
    np.testing.assert_array_equal(from_numpy, from_tensor)


def test_detect_accepts_tensor_frames(model):
    # The whole detect() path must accept torch.Tensor frames (not just NumPy) and
    # match the NumPy result -- verified on the CPU.
    sides, flips = inference.fly_camera_layout(["rh", "lf"])
    rng = np.random.default_rng(1)
    images = [rng.uniform(size=(96, 128, 3)).astype(np.float32) for _ in range(2)]
    pts_np, conf_np = inference.detect(model, images, sides, flips)
    pts_t, conf_t = inference.detect(
        model, [torch.from_numpy(im) for im in images], sides, flips
    )
    np.testing.assert_allclose(pts_np, pts_t, atol=1e-4, equal_nan=True)
    np.testing.assert_allclose(conf_np, conf_t, atol=1e-4)


def test_detect_sequence_chunking_is_equivalent(model):
    # Detection is per-frame independent, so processing a clip in windows and
    # concatenating along time must equal one full pass. This is the invariant the
    # CLI's streaming detector (bounded memory for long videos) relies on.
    sides, flips = inference.fly_camera_layout(["rh", "lf"])
    rng = np.random.default_rng(2)
    frames = [rng.uniform(size=(7, 64, 64, 3)).astype(np.float32) for _ in range(2)]
    full_pts, full_conf = inference.detect_sequence(model, frames, sides, flips)
    a = 4  # split time into windows [0:4) and [4:7)
    p0, c0 = inference.detect_sequence(model, [f[:a] for f in frames], sides, flips)
    p1, c1 = inference.detect_sequence(model, [f[a:] for f in frames], sides, flips)
    np.testing.assert_allclose(np.concatenate([p0, p1], axis=1), full_pts, atol=1e-5)
    np.testing.assert_allclose(np.concatenate([c0, c1], axis=1), full_conf, atol=1e-5)


def test_detect_sequence_batched_matches_per_frame(model, monkeypatch):
    # Batching the forward across (frame, pass) only regroups inputs, so it must
    # yield the same skeletons as the per-frame path -- even when a batch straddles
    # frame boundaries (batch_size not a multiple of the passes-per-frame). Stub the
    # fused forward+decode with a deterministic single peak keyed on each input's
    # content (identical however rows are batched). That isolates the batching
    # plumbing from any batch-size-dependent conv arithmetic, which on the real
    # *untrained* model perturbs near-flat heatmaps enough to flip arg-max peaks.
    def fake_predict_points(_model, inputs, *, method="weighted", radius=2):
        x = np.asarray(inputs.cpu() if hasattr(inputs, "cpu") else inputs)
        pts = np.zeros((x.shape[0], inference.N_SIDE_JOINTS, 2), np.float32)
        conf = np.ones((x.shape[0], inference.N_SIDE_JOINTS), np.float32)
        for i in range(x.shape[0]):
            r, c = divmod(int(abs(x[i]).sum() * 1e3) % (64 * 128), 128)
            pts[i, :, 0], pts[i, :, 1] = c / 128, r / 64  # decode of a lone spike
        return pts, conf

    monkeypatch.setattr(backends, "predict_points", fake_predict_points)
    sides, flips = inference.fly_camera_layout(["rh", "lf"])
    rng = np.random.default_rng(3)
    frames = [rng.uniform(size=(5, 64, 64, 3)).astype(np.float32) for _ in range(2)]
    ref_pts, ref_conf = inference.detect_sequence(model, frames, sides, flips)
    for bs in (1, 3, 64):  # < passes, straddling, and >> the whole window
        p, c = inference.detect_sequence(model, frames, sides, flips, batch_size=bs)
        np.testing.assert_array_equal(p, ref_pts)
        np.testing.assert_array_equal(c, ref_conf)


# -- single-side -> full skeleton --------------------------------------------


def test_assemble_skeleton_places_and_flips():
    points = np.stack([np.full((19, 2), [0.5, 0.25]), np.full((19, 2), [0.3, 0.4])])
    conf = np.stack([np.full(19, 0.9), np.full(19, 0.7)])
    pts, cout = inference.assemble_skeleton(
        points,
        conf,
        sides=["right", "left"],
        flips=[False, True],
        image_size=[(100, 200), (100, 200)],
    )
    assert pts.shape == (2, 38, 2)
    # right camera -> indices 19..37 in pixels, far side NaN
    np.testing.assert_allclose(pts[0, 19], [50.0, 50.0])
    assert np.isnan(pts[0, :19]).all()
    np.testing.assert_allclose(cout[0, 19:], 0.9)
    # left camera -> indices 0..18, x flip undone (1 - 0.3 = 0.7)
    np.testing.assert_allclose(pts[1, 0], [70.0, 80.0])
    assert np.isnan(pts[1, 19:]).all()
    np.testing.assert_allclose(cout[1, :19], 0.7)


def test_assemble_skeleton_front_camera_both_sides():
    # The front camera runs as two passes sharing physical view 1: a right pass
    # (indices 19..37) and a flipped left pass (0..18). Both must land on row 1.
    points = np.stack(
        [
            np.full((19, 2), [0.5, 0.25]),  # right side-camera (view 0)
            np.full((19, 2), [0.6, 0.1]),  # front, right pass (view 1)
            np.full((19, 2), [0.3, 0.4]),  # front, left pass (view 1, flipped)
        ]
    )
    conf = np.stack([np.full(19, 0.9), np.full(19, 0.8), np.full(19, 0.7)])
    pts, cout = inference.assemble_skeleton(
        points,
        conf,
        sides=["right", "right", "left"],
        flips=[False, False, True],
        image_size=[(100, 200), (100, 200)],
        views=[0, 1, 1],
        n_views=2,
    )
    assert pts.shape == (2, 38, 2)
    # Front camera (row 1) carries BOTH halves: right pass un-flipped ...
    np.testing.assert_allclose(pts[1, 19], [60.0, 20.0])
    # ... and left pass with the x flip undone (1 - 0.3 = 0.7).
    np.testing.assert_allclose(pts[1, 0], [70.0, 80.0])
    assert np.isfinite(pts[1]).all()  # no NaNs left on the bridging view
    np.testing.assert_allclose(cout[1, 19:], 0.8)
    np.testing.assert_allclose(cout[1, :19], 0.7)


def test_expand_passes_front_runs_twice():
    sides = ["right", "both", "left"]
    flips = [False, False, True]
    views, pass_sides, pass_flips = inference.expand_passes(sides, flips)
    assert views == [0, 1, 1, 2]  # the "both" view yields two passes
    assert pass_sides == ["right", "right", "left", "left"]
    assert pass_flips == [False, False, True, True]


def test_fly_camera_layout():
    names = ["rh", "rm", "rf", "f", "lf", "lm", "lh"]
    sides, flips = inference.fly_camera_layout(names)
    # The front camera ("f") is "both": expand_passes runs it un-flipped + flipped.
    assert sides == ["right", "right", "right", "both", "left", "left", "left"]
    assert flips == [False, False, False, False, True, True, True]


def test_detect_sequence_shapes_and_sides(model):
    names = ["rh", "rm", "rf", "f", "lf", "lm", "lh"]
    sides, flips = inference.fly_camera_layout(names)
    rng = np.random.default_rng(0)
    frames = rng.uniform(size=(7, 2, 64, 128, 3)).astype(np.float32)
    pts, conf = inference.detect_sequence(model, frames, sides, flips)
    assert pts.shape == (7, 2, 38, 2)
    assert conf.shape == (7, 2, 38)
    # Right cameras populate the right half (19..37), left cameras the left half (0..18).
    assert not np.isnan(pts[0, :, 19:]).any()
    assert np.isnan(pts[0, :, :19]).all()
    assert not np.isnan(pts[4, :, :19]).any()
    assert np.isnan(pts[4, :, 19:]).all()
    # The front camera (index 3) bridges: it fills BOTH halves.
    assert not np.isnan(pts[3, :, :19]).any()
    assert not np.isnan(pts[3, :, 19:]).any()
