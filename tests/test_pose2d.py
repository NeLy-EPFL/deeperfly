"""Tests for the JAX 2D pose detector backend (no torch required).

Architecture/shape, weight-conversion mechanics (via a torch-free round trip),
heatmap decoding and the single-side -> full-skeleton assembly are all checked
here. Numerical equivalence against the original PyTorch backend lives in
``test_pose2d_torch.py`` (skipped when torch is absent).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deeperfly.pose2d import backends, inference
from deeperfly.pose2d.backends.jax import HourglassNet, weights


@pytest.fixture
def model() -> HourglassNet:
    # A small 2-stack model keeps these mechanics tests fast; the published
    # config is 8 stacks (see test_deepfly2d_default_is_sh8).
    return HourglassNet.deepfly2d(key=jax.random.PRNGKey(0), num_stacks=2)


# -- model -------------------------------------------------------------------


def test_forward_shapes(model):
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 256, 512))
    outs = model(x)
    assert len(outs) == 2  # one heatmap stack per hourglass
    assert outs[-1].shape == (19, 64, 128)
    assert model.heatmaps(x).shape == (19, 64, 128)


def test_deepfly2d_default_is_sh8():
    # The shipped DeepFly2D checkpoint is "sh8": the default must be 8 stacks so
    # convert-weights / load_model build a matching architecture.
    assert HourglassNet.deepfly2d(key=jax.random.PRNGKey(0)).num_stacks == 8


def test_batched_inference(model):
    inputs = jax.random.normal(jax.random.PRNGKey(2), (3, 3, 256, 512))
    hm = backends.predict_heatmaps(model, inputs)
    assert hm.shape == (3, 19, 64, 128)


# -- weight conversion -------------------------------------------------------


def test_conversion_roundtrip_is_exact(model):
    sd = weights.export_state_dict(model)
    converted = weights.convert_state_dict(
        sd, HourglassNet.deepfly2d(key=jax.random.PRNGKey(9), num_stacks=2)
    )
    x = jax.random.normal(jax.random.PRNGKey(3), (3, 256, 512))
    np.testing.assert_array_equal(
        np.asarray(model.heatmaps(x)), np.asarray(converted.heatmaps(x))
    )


def test_auto_batch_size_fits_vram_and_clamps(monkeypatch):
    # No GPU -> the minimum batch.
    monkeypatch.setattr(backends, "gpu_memory_bytes", lambda device=None: None)
    assert backends.auto_batch_size(min_batch=3) == 3
    # Plenty of VRAM -> capped at max_batch.
    monkeypatch.setattr(backends, "gpu_memory_bytes", lambda device=None: 80 * 1024**3)
    assert backends.auto_batch_size((256, 512), max_batch=16) == 16
    # Fixed VRAM: larger images need more memory per image -> a smaller batch.
    monkeypatch.setattr(backends, "gpu_memory_bytes", lambda device=None: 2 * 1024**3)
    small_img = backends.auto_batch_size((128, 128), max_batch=64)
    big_img = backends.auto_batch_size((512, 512), max_batch=64)
    assert 1 <= big_img <= small_img


def test_infer_num_stacks_counts_score_heads(model):
    sd = weights.export_state_dict(model)
    assert backends.infer_num_stacks(sd) == 2
    # num_batches_tracked counters are ignored, not treated as unused keys.
    sd["bn1.num_batches_tracked"] = np.zeros((), dtype=np.int64)
    weights.convert_state_dict(
        sd, HourglassNet.deepfly2d(key=jax.random.PRNGKey(5), num_stacks=2)
    )


def test_conversion_missing_key_raises(model):
    sd = weights.export_state_dict(model)
    del sd["conv1.weight"]
    with pytest.raises(KeyError, match="missing weight 'conv1.weight'"):
        weights.convert_state_dict(sd, model)


def test_conversion_unused_key_raises(model):
    sd = weights.export_state_dict(model)
    sd["bogus.weight"] = np.zeros((1,))
    with pytest.raises(KeyError, match="unused state_dict keys"):
        weights.convert_state_dict(sd, model)


def test_checkpoint_save_load(model, tmp_path):
    path = tmp_path / "model.eqx"
    weights.save_checkpoint(model, path)
    loaded = weights.load_model(path, key=jax.random.PRNGKey(7), num_stacks=2)
    x = jax.random.normal(jax.random.PRNGKey(4), (3, 256, 512))
    np.testing.assert_array_equal(
        np.asarray(model.heatmaps(x)), np.asarray(loaded.heatmaps(x))
    )


# -- heatmap decoding --------------------------------------------------------


def test_heatmap_to_points_argmax_and_conf():
    hm = np.zeros((1, 2, 64, 128), dtype=np.float32)
    hm[0, 0, 10, 20] = 5.0
    hm[0, 1, 30, 100] = 3.0
    # A lone spike has no neighbourhood mass, so every method returns its cell.
    for method in ("argmax", "weighted", "taylor"):
        points, conf = inference.heatmap_to_points(jnp.asarray(hm), method=method)
        np.testing.assert_allclose(points[0, 0], [20 / 128, 10 / 64])
        np.testing.assert_allclose(points[0, 1], [100 / 128, 30 / 64])
        np.testing.assert_allclose(conf[0], [5.0, 3.0])


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


def test_refine_peaks_jax_matches_numpy():
    # The on-device decode (fused fast path) must equal the host refine_peaks.
    rng = np.random.default_rng(1)
    hh, ww, j = 64, 128, 19
    ys, xs = np.mgrid[0:hh, 0:ww]
    hm = np.zeros((j, hh, ww), np.float32)
    for jj in range(j):
        r, c = rng.uniform(3, hh - 3), rng.uniform(3, ww - 3)
        hm[jj] = np.exp(-((ys - r) ** 2 + (xs - c) ** 2) / 2.0)
    hm += 0.01 * rng.standard_normal(hm.shape).astype(np.float32)

    flat = hm.reshape(j, -1)
    idx = np.argmax(flat, axis=-1)
    row, col = idx // ww, idx % ww
    for method in ("argmax", "weighted", "taylor"):
        cx_np, cy_np = inference.refine_peaks(
            hm, row[:, None], col[:, None], method=method
        )
        cx_jx, cy_jx = inference.refine_peaks_jax(
            jnp.asarray(hm), jnp.asarray(row), jnp.asarray(col), method=method
        )
        np.testing.assert_allclose(np.asarray(cx_jx), cx_np[:, 0], atol=1e-4)
        np.testing.assert_allclose(np.asarray(cy_jx), cy_np[:, 0], atol=1e-4)


# -- preprocessing -----------------------------------------------------------


def test_preprocess_shape_and_mean():
    gray = np.full((200, 100, 3), 128, dtype=np.uint8)  # 128/255 ~ 0.502
    out = inference.preprocess(gray)
    assert out.shape == (3, 256, 512)
    np.testing.assert_allclose(np.asarray(out), 128 / 255 - inference.MEAN, atol=1e-4)


def test_preprocess_flip_commutes_with_resize():
    rng = np.random.default_rng(0)
    img = rng.uniform(size=(200, 100, 3)).astype(np.float32)
    a = np.asarray(inference.preprocess(img, flip=True))
    b = np.asarray(inference.preprocess(img, flip=False))[:, :, ::-1]
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_preprocess_accepts_on_device_tensor_via_dlpack():
    # A GPU-decoded frame arrives as a torch.Tensor; preprocess must bridge it to
    # JAX (DLPack) and produce the same result as the NumPy path. Exercised on the
    # CPU here (DLPack works host-side too), so no GPU is required.
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
    from_numpy = np.asarray(inference.preprocess(img))
    from_tensor = np.asarray(inference.preprocess(torch.from_numpy(img)))
    np.testing.assert_array_equal(from_numpy, from_tensor)


def test_detect_accepts_tensor_frames(model):
    # The whole detect() path must accept on-device (torch.Tensor) frames and match
    # the NumPy result -- the zero-copy GPU decode handoff, verified on the CPU.
    torch = pytest.importorskip("torch")
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
    np.testing.assert_array_equal(np.concatenate([p0, p1], axis=1), full_pts)
    np.testing.assert_array_equal(np.concatenate([c0, c1], axis=1), full_conf)


def test_detect_sequence_batched_matches_per_frame(model, monkeypatch):
    # Batching the forward across (frame, pass) only regroups inputs, so it must
    # yield the same skeletons as the per-frame path -- even when a batch straddles
    # frame boundaries (batch_size not a multiple of the passes-per-frame). Stub the
    # forward with a deterministic, sharply-peaked response keyed on each input's
    # content (identical however rows are batched). That isolates the batching
    # plumbing from XLA's batch-size-dependent conv arithmetic, which on the real
    # *untrained* model perturbs near-flat heatmaps enough to flip arg-max peaks.
    from deeperfly.pose2d import backends

    def fake_predict(_model, inputs):
        x = np.asarray(inputs)  # (N, 3, 256, 512) preprocessed passes
        hm = np.zeros((x.shape[0], inference.N_SIDE_JOINTS, 64, 128), np.float32)
        for i in range(x.shape[0]):
            r, c = divmod(int(abs(x[i]).sum() * 1e3) % (64 * 128), 128)
            hm[i, :, r, c] = 1.0  # one unambiguous peak -> arg-max is stable
        return hm

    monkeypatch.setattr(backends, "predict_heatmaps", fake_predict)
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
    # right camera -> indices 0..18 in pixels, far side NaN
    np.testing.assert_allclose(pts[0, 0], [50.0, 50.0])
    assert np.isnan(pts[0, 19:]).all()
    np.testing.assert_allclose(cout[0, :19], 0.9)
    # left camera -> indices 19..37, x flip undone (1 - 0.3 = 0.7)
    np.testing.assert_allclose(pts[1, 19], [70.0, 80.0])
    assert np.isnan(pts[1, :19]).all()
    np.testing.assert_allclose(cout[1, 19:], 0.7)


def test_assemble_skeleton_front_camera_both_sides():
    # The front camera runs as two passes sharing physical view 1: a right pass
    # (indices 0..18) and a flipped left pass (19..37). Both must land on row 1.
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
    np.testing.assert_allclose(pts[1, 0], [60.0, 20.0])
    # ... and left pass with the x flip undone (1 - 0.3 = 0.7).
    np.testing.assert_allclose(pts[1, 19], [70.0, 80.0])
    assert np.isfinite(pts[1]).all()  # no NaNs left on the bridging view
    np.testing.assert_allclose(cout[1, :19], 0.8)
    np.testing.assert_allclose(cout[1, 19:], 0.7)


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
    # Right cameras populate the right half (0..18), left cameras the left half.
    assert not np.isnan(pts[0, :, :19]).any()
    assert np.isnan(pts[0, :, 19:]).all()
    assert not np.isnan(pts[4, :, 19:]).any()
    assert np.isnan(pts[4, :, :19]).all()
    # The front camera (index 3) bridges: it fills BOTH halves.
    assert not np.isnan(pts[3, :, :19]).any()
    assert not np.isnan(pts[3, :, 19:]).any()
