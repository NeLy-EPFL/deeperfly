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
    points, conf = inference.heatmap_to_points(jnp.asarray(hm))
    np.testing.assert_allclose(points[0, 0], [20 / 128, 10 / 64])
    np.testing.assert_allclose(points[0, 1], [100 / 128, 30 / 64])
    np.testing.assert_allclose(conf[0], [5.0, 3.0])


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


def test_fly_camera_layout():
    names = ["rh", "rm", "rf", "f", "lf", "lm", "lh"]
    sides, flips = inference.fly_camera_layout(names)
    assert sides == ["right", "right", "right", "right", "left", "left", "left"]
    assert flips == [False, False, False, False, True, True, True]


def test_detect_sequence_shapes_and_sides(model):
    names = ["rh", "rm", "rf", "f", "lf", "lm", "lh"]
    sides, flips = inference.fly_camera_layout(names)
    rng = np.random.default_rng(0)
    frames = rng.uniform(size=(7, 2, 64, 128, 3)).astype(np.float32)
    pts, conf = inference.detect_sequence(model, frames, sides, flips)
    assert pts.shape == (7, 2, 38, 2)
    assert conf.shape == (7, 2, 38)
    # right cameras populate the right half (0..18), left cameras the left half.
    assert not np.isnan(pts[0, :, :19]).any()
    assert np.isnan(pts[0, :, 19:]).all()
    assert not np.isnan(pts[4, :, 19:]).any()
    assert np.isnan(pts[4, :, :19]).all()
