"""Tests for the frame geometry detection runs through: a WINDOW and a resize.

Two ops survive the op grammar's removal -- ``Crop`` (a detection window, from
``[pose2d.crops]``) and ``Resize`` (constructed internally to fit the detector's own
input) -- so what is pinned here is what remains true of them: the per-op semantics
against the NumPy functions, the affine/pixel agreement, the device-preserving torch
path, the identity no-op, and ``raw_window``, which is how a detection gets back out of a
crop into raw footage pixels.

Gone with ``fliplr`` / ``flipud`` / ``rot90``: every reorientation case, the op-parsing
section (there is no op grammar to parse) and the handedness section (with no reflection
possible, nothing reverses handedness).
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly.preprocessing import Crop, FrameTransform, PadToAspect, Resize


def _clip(rng, t=2, h=4, w=6):
    return rng.integers(0, 256, size=(t, h, w, 3), dtype=np.uint8)


def test_identity_is_strict_noop():
    rng = np.random.default_rng(0)
    frames = _clip(rng)
    idt = FrameTransform()
    assert idt.is_identity()
    assert idt.apply(frames) is frames  # untouched, no copy


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


# -- output_size / affine ------------------------------------------------------


def test_resize_output_size_rounds_half_away_from_zero():
    # cv2's rounding, not Python's banker's rounding (round(2.5) == 2).
    assert Resize(scale=0.5).output_size((5, 7)) == (3, 4)
    assert Resize(width=10, height=3).output_size((5, 7)) == (3, 10)


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


# -- raw_window ----------------------------------------------------------------
#
# The window is what a consumer that can only express a box -- a visualization panel --
# borrows from a chain instead of restating it. Every case below is one a config could
# hit, and getting any of them wrong shows the *wrong region* under a well-formed
# overlay, which reads as a calibration problem rather than a rendering one.


def test_raw_window_of_a_crop_is_the_crop():
    t = FrameTransform((Crop(x=12, y=7, width=40, height=20),))
    assert t.raw_window((96, 128)) == (12, 7, 40, 20)


def test_raw_window_of_the_identity_is_the_whole_frame():
    assert FrameTransform(()).raw_window((96, 128)) == (0, 0, 128, 96)


def test_a_resize_does_not_move_the_window():
    # The window is a region of the raw frame; resampling changes its resolution only.
    box = (12, 7, 40, 20)
    assert (
        FrameTransform(
            (Crop(x=12, y=7, width=40, height=20), Resize(scale=4.0))
        ).raw_window((96, 128))
        == box
    )
    assert (
        FrameTransform(
            (Crop(x=12, y=7, width=40, height=20), Resize(width=512, height=256))
        ).raw_window((96, 128))
        == box
    )


def test_a_crop_measured_in_resized_pixels_comes_back_in_raw_ones():
    # Halve the frame, then keep a 10x10 window of it: 20x20 raw pixels at raw (20, 20).
    t = FrameTransform((Resize(scale=0.5), Crop(x=10, y=10, width=10, height=10)))
    assert t.raw_window((96, 128)) == (20, 20, 20, 20)


def test_raw_window_rejects_a_crop_that_does_not_fit():
    # A stale crop plan against differently-sized footage: loud, not a truncated slice.
    t = FrameTransform((Crop(x=12, y=7, width=400, height=20),))
    with pytest.raises(ValueError, match="exceeds"):
        t.raw_window((96, 128))


# -- op parsing ----------------------------------------------------------------
#
# Every case here is one a config could hit. They go through `frame_transform_from_ops`,
# which is what `[[pose2d.preprocessors]] ops` is parsed by -- the per-camera
# `[cameras.*].preprocess` spelling these once covered was retired in 0.2 (the pathway's
# chain is the one whose transform is inverted on the way back).


# -- handedness ---------------------------------------------------------------


# -- PadToAspect: the op that makes a resize a pure scale ---------------------


def test_pad_to_aspect_widens_the_side_cameras_and_leaves_an_axial_window_alone():
    """The whole point, on the two geometries that exist.

    960x512 is 1.875:1 and takes a 6.7% vertical squeeze into a 2:1 network; the axial
    windows are already 2:1 and must come back untouched, or every recording that is
    already correct would move.
    """
    pad = PadToAspect(aspect=2.0)
    assert pad.output_size((512, 960)) == (512, 1024)  # 32 px of border, 16 a side
    assert pad.output_size((520, 1040)) == (520, 1040)  # already 2:1 -> nothing added
    # and a frame that is too WIDE gains rows rather than losing columns
    assert pad.output_size((256, 1024)) == (512, 1024)


def test_pad_to_aspect_makes_the_resize_into_the_model_uniform():
    """Why this exists at all: after the pad, x and y are scaled by the same number."""
    t = FrameTransform((PadToAspect(aspect=2.0), Resize(width=512, height=256)))
    assert t.output_size((512, 960)) == (256, 512)
    a = t.affine((512, 960))
    assert a[0, 0] == pytest.approx(
        a[1, 1]
    )  # sx == sy, which is what "no stretch" means


def test_pad_to_aspect_affine_agrees_with_what_apply_does():
    """The FrameOp pairing. Break it and points land wrong rather than raising."""
    pad = PadToAspect(aspect=2.0, value=7)
    frame = np.zeros((512, 960, 1), np.uint8)
    frame[50, 100] = 255  # a marker whose new position the affine must predict
    out = pad.apply(frame)
    assert out.shape == (512, 1024, 1)
    (y,), (x,), _ = np.where(out == 255)
    moved = pad.affine((512, 960)) @ np.array([100.0, 50.0, 1.0])
    assert (moved[0], moved[1]) == (x, y)
    assert out[0, 0, 0] == 7 and out[0, -1, 0] == 7  # the border took the fill value


def test_pad_to_aspect_round_trips_a_point_back_to_raw_pixels():
    """A detection through a padded chain has to come back where it started."""
    t = FrameTransform(
        (
            Crop(x=10, y=20, width=900, height=480),
            PadToAspect(aspect=2.0),
            Resize(width=512, height=256),
        )
    )
    pts = np.array([[100.0, 50.0], [800.0, 400.0]])
    fwd = (t.affine((512, 960)) @ np.c_[pts, np.ones(2)].T).T[:, :2]
    np.testing.assert_allclose(t.unmap_points(fwd, (512, 960)), pts, atol=1e-9)


def test_pad_to_aspect_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    pad = PadToAspect(aspect=2.0, value=3)
    frame = np.random.default_rng(0).integers(0, 255, (2, 512, 960, 1), dtype=np.uint8)
    np.testing.assert_array_equal(
        pad.apply(frame), pad.apply(torch.from_numpy(frame)).numpy()
    )


def test_pad_to_aspect_rejects_a_nonsense_aspect():
    with pytest.raises(ValueError, match="aspect must be positive"):
        PadToAspect(aspect=0.0)
