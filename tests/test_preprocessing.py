"""Tests for per-camera frame preprocessing (``[preprocess.*]`` -> FrameTransform).

The transform is applied once, at decode time, to a ``(T, H, W, C)`` batch; the
*transformed* frame is the canonical frame for the whole run. These tests pin the
flip/rot90 semantics (matching the NumPy functions), the device-preserving torch
path, the identity no-op, and the config parser's validation.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from deeperfly.config import Config
from deeperfly.preprocessing import FrameTransform, parse_frame_transforms


def _clip(rng, t=2, h=4, w=6):
    return rng.integers(0, 256, size=(t, h, w, 3), dtype=np.uint8)


def test_identity_is_strict_noop():
    rng = np.random.default_rng(0)
    frames = _clip(rng)
    idt = FrameTransform()
    assert idt.is_identity()
    assert idt.apply(frames) is frames  # untouched, no copy


def test_rot90_normalized_mod_4():
    assert FrameTransform(rot90=4).rot90 == 0
    assert FrameTransform(rot90=5).rot90 == 1
    assert FrameTransform(rot90=-1).rot90 == 3
    assert FrameTransform(rot90=4).is_identity()


@pytest.mark.parametrize(
    "fliplr,flipud,rot90",
    list(itertools.product((False, True), (False, True), range(4))),
)
def test_apply_matches_numpy_per_frame(fliplr, flipud, rot90):
    # apply() over the batch must equal the same fliplr/flipud/rot90 sequence run
    # per frame with NumPy -- pinning both the ops and their order.
    rng = np.random.default_rng(1)
    frames = _clip(rng)
    out = FrameTransform(fliplr=fliplr, flipud=flipud, rot90=rot90).apply(frames)
    for i in range(len(frames)):
        ref = frames[i]
        if fliplr:
            ref = np.fliplr(ref)
        if flipud:
            ref = np.flipud(ref)
        if rot90:
            ref = np.rot90(ref, rot90)
        assert np.array_equal(out[i], ref)


def test_rot90_swaps_height_and_width():
    rng = np.random.default_rng(2)
    frames = _clip(rng, h=4, w=6)
    assert FrameTransform(rot90=1).apply(frames).shape == (2, 6, 4, 3)
    assert FrameTransform(rot90=2).apply(frames).shape == (2, 4, 6, 3)


def test_apply_returns_contiguous_numpy():
    # cv2 (the visualization draw path) needs contiguous arrays; np.flip/np.rot90 yield views.
    rng = np.random.default_rng(3)
    out = FrameTransform(fliplr=True, rot90=1).apply(_clip(rng))
    assert out.flags["C_CONTIGUOUS"]


def test_torch_path_stays_torch_and_matches_numpy():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    frames_np = _clip(rng)
    frames_t = torch.from_numpy(frames_np)
    for t in (
        FrameTransform(fliplr=True),
        FrameTransform(flipud=True),
        FrameTransform(rot90=1),
        FrameTransform(fliplr=True, flipud=True, rot90=3),
    ):
        out = t.apply(frames_t)
        assert isinstance(out, torch.Tensor)  # device/type preserved, no host copy
        np.testing.assert_array_equal(out.numpy(), t.apply(frames_np))


# -- config parsing ----------------------------------------------------------


def test_parse_frame_transforms_reads_tables():
    cfg = {
        "cameras": {
            "rh": {"preprocess": {"fliplr": True, "rot90": 3}},
            "lf": {"preprocess": {"flipud": True}},
            "rm": {},  # a camera with no preprocess table
        }
    }
    d = parse_frame_transforms(Config.from_dict(cfg))
    assert d["rh"] == FrameTransform(fliplr=True, rot90=3)
    assert d["lf"] == FrameTransform(flipud=True)
    assert "rm" not in d  # cameras with no table are simply absent (-> identity)


def test_parse_frame_transforms_empty_when_section_missing():
    assert parse_frame_transforms(Config.from_dict({})) == {}


@pytest.mark.parametrize(
    "spec",
    [
        {"rotate": 90},  # unknown key (must be rot90)
        {"flip_lr": True},  # unknown key (must be fliplr)
        {"rot90": 1.5},  # non-integer
        {"rot90": True},  # bool is not a valid quarter-turn count
    ],
)
def test_parse_frame_transforms_rejects_bad_config(spec):
    with pytest.raises(ValueError):
        parse_frame_transforms(
            Config.from_dict({"cameras": {"rh": {"preprocess": spec}}})
        )
