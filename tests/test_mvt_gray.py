"""The multiview transformer accepts a FOLDED, one-plane artifact as well as an RGB one.

The corpus is monochrome, so a `deeperfly-mvt-1` artifact's patch embedding holds three
copies of one filter under three different (mean, std). `deeperfly-mvt-2` is the same
network with that stem folded onto a single plane -- exactly, see dfpose's
``scripts/gray_stem.py`` -- and the end-to-end equality of the two is checked against the
real 87 MB artifacts by ``dfpose/scripts/verify_gray_artifact.py`` (measured: max |d
heatmap| 5.3e-07, 0.0003% of range).

What is tested HERE is the part that would rot silently and needs no weights: the input
preparation must make exactly as many planes as the artifact's normalization describes.
Getting that wrong is not a crash -- three planes into a folded stem is a shape error, but
ONE plane normalized with ImageNet's first channel into a folded stem runs and is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mvt = pytest.importorskip("deeperfly.pose2d.mvt")


def _frames(n: int = 2, h: int = 32, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(n, h, w, 1), dtype=np.uint8)


def test_three_means_make_three_planes():
    x = mvt.prepare_images(
        _frames(), (32, 64), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), "cpu"
    )
    assert x.shape == (2, 3, 32, 64)
    # The three planes are the SAME image under three different affine maps, which is what
    # makes the fold exact. If they were ever genuinely different the fold would be wrong.
    a = x[:, 0] * 0.229 + 0.485
    b = x[:, 1] * 0.224 + 0.456
    assert torch.allclose(a, b, atol=1e-6)


def test_one_mean_makes_one_plane():
    x = mvt.prepare_images(_frames(), (32, 64), (0.0,), (1.0,), "cpu")
    assert x.shape == (2, 1, 32, 64)
    # mean 0 / std 1 is not decoration: a folded artifact carries the ImageNet constants
    # INSIDE its stem, so the pixels must arrive as gray/255 and nothing else.
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0


def test_the_two_preparations_agree_on_the_underlying_image():
    frames = _frames()
    rgb = mvt.prepare_images(
        frames, (32, 64), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), "cpu"
    )
    gray = mvt.prepare_images(frames, (32, 64), (0.0,), (1.0,), "cpu")
    # Undo the RGB normalization on any one plane and the same gray image must come back.
    assert torch.allclose(rgb[:, 0] * 0.229 + 0.485, gray[:, 0], atol=1e-6)


def test_both_artifact_formats_are_known():
    assert mvt.ARTIFACT_FORMAT in mvt.ARTIFACT_FORMATS
    assert "deeperfly-mvt-2" in mvt.ARTIFACT_FORMATS
