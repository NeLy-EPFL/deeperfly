"""Tests for per-camera frame preprocessing (ordered op lists -> FrameTransform).

The transform is applied once, at decode time, to a ``(T, H, W, C)`` batch; the
*transformed* frame is the canonical frame for the whole run, and the composed
affine maps raw-frame intrinsics into it. These tests pin the per-op semantics
(matching the NumPy functions), the order sensitivity, the affine/pixel
agreement, the device-preserving torch path, the identity no-op, the intrinsics
mapping, and the config parser's validation.
"""

from __future__ import annotations

import tomllib

import numpy as np
import pytest

from deeperfly.config import Config
from deeperfly.preprocessing import (
    Crop,
    Fliplr,
    Flipud,
    FrameTransform,
    Resize,
    Rot90,
    parse_frame_transforms,
)


def _clip(rng, t=2, h=4, w=6):
    return rng.integers(0, 256, size=(t, h, w, 3), dtype=np.uint8)


def test_identity_is_strict_noop():
    rng = np.random.default_rng(0)
    frames = _clip(rng)
    idt = FrameTransform()
    assert idt.is_identity()
    assert idt.apply(frames) is frames  # untouched, no copy


def test_noop_ops_are_dropped():
    assert Rot90(k=5) == Rot90(k=1)
    assert Rot90(k=-1) == Rot90(k=3)
    assert FrameTransform((Rot90(k=4),)).is_identity()
    assert FrameTransform((Resize(scale=1.0),)).is_identity()
    # equivalent chains compare equal after normalization
    assert FrameTransform((Rot90(k=4), Fliplr())) == FrameTransform((Fliplr(),))


def test_adjacent_exact_compositions_fold():
    # Exact algebra only: consecutive rotations sum, double flips cancel -- so
    # equivalent chains share one fingerprint (and reuse each other's caches).
    assert FrameTransform((Rot90(k=2), Rot90(k=2))).is_identity()
    assert FrameTransform((Fliplr(), Fliplr())).is_identity()
    assert FrameTransform((Rot90(k=1), Rot90(k=1))) == FrameTransform((Rot90(k=2),))
    # cancellation exposes new exact neighbors
    assert FrameTransform((Rot90(k=1), Fliplr(), Fliplr(), Rot90(k=3))).is_identity()
    # different ops never fold
    assert not FrameTransform((Fliplr(), Flipud())).is_identity()
    assert len(FrameTransform((Fliplr(), Rot90(k=2), Fliplr())).ops) == 3


@pytest.mark.parametrize("rot90", range(4))
@pytest.mark.parametrize("flipud", (False, True))
@pytest.mark.parametrize("fliplr", (False, True))
def test_apply_matches_numpy_per_frame(fliplr, flipud, rot90):
    # A fliplr -> flipud -> rot90 chain over the batch must equal the same
    # sequence run per frame with NumPy -- pinning the ops and the in-order
    # application (this is the old fixed-order pipeline, now spelled out).
    rng = np.random.default_rng(1)
    frames = _clip(rng)
    ops = []
    if fliplr:
        ops.append(Fliplr())
    if flipud:
        ops.append(Flipud())
    if rot90:
        ops.append(Rot90(k=rot90))
    out = FrameTransform(tuple(ops)).apply(frames)
    for i in range(len(frames)):
        ref = frames[i]
        if fliplr:
            ref = np.fliplr(ref)
        if flipud:
            ref = np.flipud(ref)
        if rot90:
            ref = np.rot90(ref, rot90)
        assert np.array_equal(out[i], ref)


def test_order_matters():
    # fliplr then rot90 differs from rot90 then fliplr -- the list order is the
    # application order, there is no canonical reordering.
    rng = np.random.default_rng(5)
    frames = _clip(rng)
    a = FrameTransform((Fliplr(), Rot90(k=1))).apply(frames)
    b = FrameTransform((Rot90(k=1), Fliplr())).apply(frames)
    assert a.shape == b.shape
    assert not np.array_equal(a, b)


def test_rot90_swaps_height_and_width():
    rng = np.random.default_rng(2)
    frames = _clip(rng, h=4, w=6)
    assert FrameTransform((Rot90(k=1),)).apply(frames).shape == (2, 6, 4, 3)
    assert FrameTransform((Rot90(k=2),)).apply(frames).shape == (2, 4, 6, 3)


def test_crop_values_and_shape():
    rng = np.random.default_rng(6)
    frames = _clip(rng, h=4, w=6)
    out = FrameTransform((Crop(x=1, y=2, width=3, height=2),)).apply(frames)
    assert out.shape == (2, 2, 3, 3)
    assert np.array_equal(out, frames[:, 2:4, 1:4, :])


def test_crop_out_of_bounds_raises():
    rng = np.random.default_rng(7)
    frames = _clip(rng, h=4, w=6)
    crop = Crop(x=3, y=0, width=4, height=4)  # x + width = 7 > 6
    with pytest.raises(ValueError, match="exceeds"):
        FrameTransform((crop,)).apply(frames)
    with pytest.raises(ValueError, match="exceeds"):
        crop.output_size((4, 6))


def test_crop_validates_fields():
    with pytest.raises(ValueError):
        Crop(x=-1, y=0, width=3, height=3)
    with pytest.raises(ValueError):
        Crop(x=0, y=0, width=0, height=3)


def test_apply_returns_contiguous_numpy():
    # cv2 (the visualization draw path) needs contiguous arrays; np.flip/np.rot90 yield views.
    rng = np.random.default_rng(3)
    out = FrameTransform((Fliplr(), Rot90(k=1))).apply(_clip(rng))
    assert out.flags["C_CONTIGUOUS"]


# -- output_size / affine ------------------------------------------------------


def test_output_size_composes():
    t = FrameTransform((Rot90(k=1), Crop(x=0, y=1, width=3, height=2)))
    assert t.output_size((4, 6)) == (2, 3)  # rot90: (6, 4), then crop


def test_resize_output_size_rounds_half_away_from_zero():
    # cv2's rounding, not Python's banker's rounding (round(2.5) == 2).
    assert Resize(scale=0.5).output_size((5, 7)) == (3, 4)
    assert Resize(width=10, height=3).output_size((5, 7)) == (3, 10)


@pytest.mark.parametrize(
    "ops",
    [
        (Fliplr(),),
        (Flipud(),),
        (Rot90(k=1),),
        (Rot90(k=2),),
        (Rot90(k=3),),
        (Fliplr(), Rot90(k=1)),
        (Rot90(k=1), Fliplr()),
        (Crop(x=1, y=2, width=4, height=3),),
        (Rot90(k=1), Crop(x=0, y=1, width=3, height=4)),
        (Fliplr(), Crop(x=2, y=0, width=5, height=4), Rot90(k=3)),
        (Crop(x=1, y=1, width=5, height=3), Flipud()),
    ],
)
def test_affine_matches_pixel_movement(ops):
    # Property test: for lossless ops, every raw pixel that stays in frame must
    # land exactly where the composed affine says.
    h, w = 5, 7
    frame = np.arange(h * w, dtype=np.int64).reshape(h, w, 1)
    t = FrameTransform(ops)
    out = t.apply(frame)
    a = t.affine((h, w))
    oh, ow = t.output_size((h, w))
    assert out.shape[:2] == (oh, ow)
    checked = 0
    for y in range(h):
        for x in range(w):
            xp, yp, one = a @ (x, y, 1.0)
            assert one == 1.0
            if 0 <= xp < ow and 0 <= yp < oh:
                assert xp == int(xp) and yp == int(yp)  # integer pixel centers
                assert out[int(yp), int(xp), 0] == frame[y, x, 0]
                checked += 1
    assert checked > 0


def test_resize_affine_matches_centroid():
    # Half-pixel convention: a delta at x=4 upscaled 2x must center at
    # (4 + 0.5) * 2 - 0.5 = 8.5.
    h, w = 4, 6
    frame = np.zeros((h, w, 1), dtype=np.float32)
    frame[1, 4, 0] = 1.0
    t = FrameTransform((Resize(scale=2.0),))
    out = t.apply(frame)[..., 0]
    ys, xs = np.mgrid[0 : 2 * h, 0 : 2 * w]
    cx = (out * xs).sum() / out.sum()
    cy = (out * ys).sum() / out.sum()
    expect = t.affine((h, w)) @ (4.0, 1.0, 1.0)
    assert np.allclose([cx, cy], expect[:2], atol=1e-5)
    assert np.allclose(expect[:2], [8.5, 2.5])


# -- torch path ----------------------------------------------------------------


def test_torch_path_stays_torch_and_matches_numpy():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    frames_np = _clip(rng)
    frames_t = torch.from_numpy(frames_np)
    for t in (
        FrameTransform((Fliplr(),)),
        FrameTransform((Flipud(),)),
        FrameTransform((Rot90(k=1),)),
        FrameTransform((Crop(x=1, y=0, width=4, height=3),)),
        FrameTransform((Fliplr(), Flipud(), Rot90(k=3))),
        FrameTransform((Rot90(k=1), Crop(x=0, y=1, width=3, height=4), Fliplr())),
    ):
        out = t.apply(frames_t)
        assert isinstance(out, torch.Tensor)  # device/type preserved, no host copy
        np.testing.assert_array_equal(out.numpy(), t.apply(frames_np))


def test_torch_crop_out_of_bounds_raises():
    torch = pytest.importorskip("torch")
    frames = torch.zeros((2, 4, 6, 3), dtype=torch.uint8)
    with pytest.raises(ValueError, match="exceeds"):
        FrameTransform((Crop(x=0, y=2, width=6, height=3),)).apply(frames)


@pytest.mark.parametrize("resize", [Resize(scale=0.5), Resize(width=9, height=7)])
def test_torch_resize_bilinear_close_to_cv2(resize):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(8)
    frames_np = _clip(rng, h=10, w=14)
    t = FrameTransform((resize,))
    out_np = t.apply(frames_np)
    out_t = t.apply(torch.from_numpy(frames_np))
    assert isinstance(out_t, torch.Tensor)
    assert out_t.dtype == torch.uint8
    assert out_t.shape == out_np.shape
    # cv2 and torch bilinear share the half-pixel convention; measured max
    # difference is 1 LSB on uint8.
    diff = out_t.numpy().astype(int) - out_np.astype(int)
    assert np.abs(diff).max() <= 1


@pytest.mark.parametrize(
    "resize",
    [
        Resize(scale=0.5, interpolation="nearest"),
        Resize(width=9, height=7, interpolation="nearest"),
        Resize(scale=2.0, interpolation="nearest"),
    ],
)
def test_torch_resize_nearest_bitexact_with_numpy(resize):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(9)
    frames_np = _clip(rng, h=10, w=14)
    t = FrameTransform((resize,))
    out_np = t.apply(frames_np)
    out_t = t.apply(torch.from_numpy(frames_np))
    np.testing.assert_array_equal(out_t.numpy(), out_np)


# -- intrinsics mapping ---------------------------------------------------------


def test_map_intrinsics_crop_shifts_principal_point():
    # The acceptance example: 100x100 raw frame, principal point at the raw
    # center (49.5, 49.5), crop at (10, 10) -> (39.5, 39.5); focals untouched.
    t = FrameTransform((Crop(x=10, y=10, width=80, height=80),))
    intr = t.map_intrinsics([200.0, 210.0, 49.5, 49.5], np.array([]), (100, 100))
    assert np.allclose(intr, [200.0, 210.0, 39.5, 39.5])


def test_map_intrinsics_rot90_swaps_focals_and_maps_pp():
    # (x, y) -> (y, w-1-x) on a (h=100, w=200) frame.
    t = FrameTransform((Rot90(k=1),))
    intr = t.map_intrinsics([300.0, 400.0, 10.0, 20.0], np.array([]), (100, 200))
    assert np.allclose(intr, [400.0, 300.0, 20.0, 189.0])


def test_map_intrinsics_fliplr_reflects_pp_keeps_focals():
    t = FrameTransform((Fliplr(),))
    intr = t.map_intrinsics([300.0, 400.0, 10.0, 20.0], np.array([]), (50, 60))
    assert np.allclose(intr, [300.0, 400.0, 49.0, 20.0])


def test_map_intrinsics_resize_scales():
    t = FrameTransform((Resize(scale=0.5),))
    intr = t.map_intrinsics([300.0, 400.0, 49.5, 49.5], np.array([]), (100, 100))
    assert np.allclose(intr, [150.0, 200.0, 24.5, 24.5])


@pytest.mark.parametrize(
    "ops",
    [
        (Fliplr(),),
        (Flipud(),),
        (Rot90(k=1),),
        (Rot90(k=2),),
        (Fliplr(), Rot90(k=3)),
    ],
)
def test_raw_center_maps_to_canonical_center(ops):
    # Flip/rot90 chains keep the image center at the image center, so the
    # default principal point resolves bit-identically to the old behavior.
    h, w = 100, 200
    t = FrameTransform(ops)
    oh, ow = t.output_size((h, w))
    intr = t.map_intrinsics(
        [300.0, 300.0, (w - 1) / 2, (h - 1) / 2], np.array([]), (h, w)
    )
    assert np.allclose(intr[2:], [(ow - 1) / 2, (oh - 1) / 2])


def test_map_intrinsics_distortion_guard():
    radial_only = np.array([0.1, -0.05, 0.0, 0.0, 0.02])  # k1, k2, k3 nonzero
    tangential = np.array([0.1, -0.05, 0.01, 0.0])  # p1 nonzero
    intr = [300.0, 300.0, 50.0, 50.0]
    # radial terms survive mirrors/rotations
    FrameTransform((Fliplr(),)).map_intrinsics(intr, radial_only, (100, 100))
    # tangential terms survive crops and resizes (positive-diagonal maps)
    FrameTransform(
        (Crop(x=1, y=1, width=50, height=50), Resize(scale=2.0))
    ).map_intrinsics(intr, tangential, (100, 100))
    # ... but not mirrors or any rotation (180 deg included)
    with pytest.raises(ValueError, match="non-radial"):
        FrameTransform((Fliplr(),)).map_intrinsics(intr, tangential, (100, 100))
    with pytest.raises(ValueError, match="non-radial"):
        FrameTransform((Rot90(k=2),)).map_intrinsics(intr, tangential, (100, 100))


# -- config parsing ----------------------------------------------------------


def test_parse_frame_transforms_reads_lists():
    cfg = {
        "cameras": {
            "rh": {"preprocess": [{"op": "fliplr"}, {"op": "rot90", "k": 3}]},
            "lf": {"preprocess": [{"op": "flipud"}]},
            "rm": {},  # a camera with no preprocess list
        }
    }
    d = parse_frame_transforms(Config.from_dict(cfg))
    assert d["rh"] == FrameTransform((Fliplr(), Rot90(k=3)))
    assert d["lf"] == FrameTransform((Flipud(),))
    assert "rm" not in d  # cameras with no list are simply absent (-> identity)


def test_parse_frame_transforms_array_of_tables_form():
    text = """
    [[cameras.rh.preprocess]]
    op = "rot90"
    k = 1

    [[cameras.rh.preprocess]]
    op = "crop"
    x = 10
    y = 10
    width = 80
    height = 80

    [[cameras.rh.preprocess]]
    op = "resize"
    scale = 0.5
    """
    d = parse_frame_transforms(Config.from_dict(tomllib.loads(text)))
    assert d["rh"] == FrameTransform(
        (Rot90(k=1), Crop(x=10, y=10, width=80, height=80), Resize(scale=0.5))
    )


def test_parse_frame_transforms_empty_when_section_missing():
    assert parse_frame_transforms(Config.from_dict({})) == {}
    cfg = {"cameras": {"rh": {"preprocess": []}}}
    assert parse_frame_transforms(Config.from_dict(cfg)) == {}


def test_parse_frame_transforms_rejects_old_table_form():
    cfg = {"cameras": {"rh": {"preprocess": {"fliplr": True, "rot90": 3}}}}
    with pytest.raises(ValueError, match=r"ordered \*list\*"):
        parse_frame_transforms(Config.from_dict(cfg))


def test_parse_frame_transforms_rejects_defaults_preprocess():
    cfg = {"cameras": {"defaults": {"preprocess": [{"op": "fliplr"}]}, "rh": {}}}
    with pytest.raises(ValueError, match="defaults"):
        parse_frame_transforms(Config.from_dict(cfg))


@pytest.mark.parametrize(
    "steps",
    [
        "fliplr",  # not a list
        ["fliplr"],  # bare string step (must be a table)
        [{"op": "rotate"}],  # unknown op
        [{"op": "fliplr", "k": 1}],  # key not allowed for this op
        [{"op": "rot90", "k": 1.5}],  # non-integer k
        [{"op": "rot90", "k": True}],  # bool is not a quarter-turn count
        [{"op": "crop", "x": 0, "y": 0, "width": 10}],  # missing height
        [{"op": "crop", "x": -1, "y": 0, "width": 10, "height": 10}],
        [{"op": "crop", "x": 0, "y": 0, "width": 10, "height": True}],
        [{"op": "resize"}],  # neither scale nor width/height
        [{"op": "resize", "scale": 0.5, "width": 10, "height": 10}],  # both
        [{"op": "resize", "width": 10}],  # width without height
        [{"op": "resize", "scale": 0.0}],
        [{"op": "resize", "scale": 0.5, "interpolation": "bicubic"}],
    ],
)
def test_parse_frame_transforms_rejects_bad_config(steps):
    with pytest.raises(ValueError):
        parse_frame_transforms(
            Config.from_dict({"cameras": {"rh": {"preprocess": steps}}})
        )


def test_to_json_is_canonical():
    t = FrameTransform((Rot90(k=5), Resize(scale=0.5), Rot90(k=4)))
    assert t.to_json() == [
        {"op": "rot90", "k": 1},
        {"op": "resize", "scale": 0.5, "interpolation": "bilinear"},
    ]
