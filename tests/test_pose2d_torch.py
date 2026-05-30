"""Numerical equivalence of the JAX hourglass against the PyTorch backend.

Skipped unless ``torch`` is installed (the ``torch`` extra). The PyTorch
reference is ``deeperfly.pose2d.torch_backend`` (a faithful copy of DeepFly2D's
``df2d/model.py``); its random-initialised weights are converted with
:func:`deeperfly.pose2d.weights.convert_state_dict` and the two forward passes
must agree. This guards the architecture port and the conversion key-map, and
confirms :func:`deeperfly.pose2d.inference.detect` works with both backends.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deeperfly.pose2d import inference, torch_backend  # noqa: E402
from deeperfly.pose2d.model import HourglassNet  # noqa: E402
from deeperfly.pose2d.weights import convert_state_dict  # noqa: E402


def _matched_models(num_stacks: int = 2):
    # 2 stacks keeps the equivalence check fast; the conversion key-map is
    # identical at any depth (the published config is 8 stacks).
    torch.manual_seed(0)
    ref = torch_backend.HourglassNet(num_stacks=num_stacks).eval()
    state = {k: v.detach().numpy() for k, v in ref.state_dict().items()}
    model = convert_state_dict(
        state, HourglassNet.deepfly2d(key=jax.random.PRNGKey(0), num_stacks=num_stacks)
    )
    return ref, model


def test_jax_matches_pytorch():
    ref, model = _matched_models()
    x = np.random.default_rng(0).standard_normal((1, 3, 128, 256)).astype(np.float32)
    with torch.no_grad():
        ref_out = ref(torch.from_numpy(x))[-1].numpy()[0]
    jax_out = np.asarray(model.heatmaps(jnp.asarray(x[0])))
    # Random weights through a deep float32 net accumulate cross-framework
    # divergence (and torch runs on CPU vs JAX on GPU); ~1e-3 is the realistic
    # floor. Trained weights on real images produce sharp, robust peaks.
    np.testing.assert_allclose(jax_out, ref_out, atol=2e-3, rtol=1e-3)


def test_detect_dispatches_to_both_backends():
    """detect() yields the same skeleton points for the JAX and torch backends."""
    ref, model = _matched_models()
    sides, flips = inference.fly_camera_layout(["rh", "lf"])
    images = [
        np.random.default_rng(s).uniform(size=(96, 128, 3)).astype(np.float32)
        for s in (1, 2)
    ]

    pts_jax, conf_jax = inference.detect(model, images, sides, flips)
    pts_torch, conf_torch = inference.detect(ref, images, sides, flips)
    np.testing.assert_allclose(pts_jax, pts_torch, atol=1e-3, equal_nan=True)
    # float32 accumulation floor (see test_jax_matches_pytorch).
    np.testing.assert_allclose(conf_jax, conf_torch, atol=2e-3)
