"""The heatmap numerics -- the contract that must not drift.

These three functions sit between the labels and every number anyone quotes. A change to any
of them shifts the reported pixel error of the whole pipeline *without changing the loss
curve*, so each property below is pinned rather than left to be noticed later.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly.training.heatmaps import (
    STRIDE,
    masked_heatmap_loss,
    refined_argmax,
    render_gaussian_targets,
)

HW = (64, 128)


# -- targets are rendered at the FRACTIONAL coordinate ---------------------------


@pytest.mark.parametrize("xy", [(40.3, 20.7), (40.0, 20.0), (63.5, 31.25), (1.5, 1.5)])
def test_a_fractional_label_trains_a_fractional_peak(xy):
    """Rounding here would bake in a half-cell = STRIDE/2 input-px bias, invisibly.

    This is the exact class of bug already found once in this package's own decode, so it is
    pinned at the sub-tenth-of-a-cell level.
    """
    target = render_gaussian_targets(np.array([xy]), np.array([1.0]), hw=HW)
    got, score = refined_argmax(target)
    assert np.abs(got[0] - np.asarray(xy)).max() < 0.1
    # ...and therefore well under one input pixel once scaled by the stride.
    assert np.abs(got[0] - np.asarray(xy)).max() * STRIDE < 0.5


def test_the_sampled_peak_falls_below_one_exactly_when_the_label_is_fractional():
    """The Gaussian's *continuous* peak is 1.0; the nearest *sampled* cell need not be.

    That deficit is not a defect -- it is where the sub-pixel information lives, and it is
    what the parabolic decode reads back out. An implementation that snapped the peak to a
    cell would report a clean 1.0 here and have thrown the information away.
    """
    on_cell = render_gaussian_targets(np.array([[40.0, 20.0]]), np.array([1.0]), hw=HW)
    off_cell = render_gaussian_targets(np.array([[40.5, 20.5]]), np.array([1.0]), hw=HW)
    assert on_cell.max() == pytest.approx(1.0, abs=1e-6)
    assert off_cell.max() < 1.0
    # Bounded below by a half-cell offset in both axes: exp(-(0.5^2+0.5^2)/(2*2^2)).
    assert off_cell.max() > 0.93


def test_an_invisible_keypoint_gets_an_all_zero_channel():
    """Better than a peak somewhere arbitrary, which would be supervised as truth."""
    target = render_gaussian_targets(
        np.array([[10.0, 10.0], [20.0, 20.0]]), np.array([1.0, 0.0]), hw=HW
    )
    assert target[0].max() == pytest.approx(1.0)
    assert (target[1] == 0).all()


def test_a_non_finite_coordinate_does_not_poison_the_target():
    """NaN through an exponential would spread across the whole channel."""
    target = render_gaussian_targets(
        np.array([[np.nan, 5.0], [np.inf, 1.0]]), np.array([1.0, 1.0]), hw=(16, 16)
    )
    assert np.isfinite(target).all()
    assert (target == 0).all()


def test_targets_batch():
    batched = render_gaussian_targets(
        np.array([[[10.0, 10.0]], [[20.0, 20.0]]]), np.array([[1.0], [1.0]]), hw=HW
    )
    assert batched.shape == (2, 1, *HW)
    single = render_gaussian_targets(np.array([[10.0, 10.0]]), np.array([1.0]), hw=HW)
    np.testing.assert_allclose(batched[0], single)


def test_sigma_controls_the_spread():
    tight = render_gaussian_targets(
        np.array([[30.0, 30.0]]), np.array([1.0]), hw=HW, sigma=1.0
    )
    wide = render_gaussian_targets(
        np.array([[30.0, 30.0]]), np.array([1.0]), hw=HW, sigma=4.0
    )
    assert wide.sum() > tight.sum()
    # Both peak at 1.0 -- sigma changes the spread, not the amplitude.
    assert tight.max() == pytest.approx(1.0, abs=1e-6)
    assert wide.max() == pytest.approx(1.0, abs=1e-6)


# -- the decode is refined, never the raw argmax --------------------------------


def test_the_refinement_beats_the_raw_argmax():
    """The whole reason this function exists: the argmax quantizes to STRIDE input px."""
    truth = np.array([[40.4, 20.6]])
    target = render_gaussian_targets(truth, np.array([1.0]), hw=HW)
    refined, _ = refined_argmax(target)

    flat = target.reshape(1, -1)
    idx = int(flat.argmax())
    raw = np.array([[idx % HW[1], idx // HW[1]]], dtype=float)

    assert np.abs(refined - truth).max() < np.abs(raw - truth).max()


def test_a_peak_on_the_border_is_not_refined_inward():
    """One neighbour does not exist there; inventing it would bias the peak."""
    target = np.zeros((1, 8, 8), dtype=np.float32)
    target[0, 0, 0] = 1.0
    xy, _ = refined_argmax(target)
    np.testing.assert_allclose(xy[0], [0.0, 0.0])


def test_a_flat_channel_decodes_without_nan():
    """A dead channel is normal (an unlabelled point), so it must not produce NaN."""
    xy, score = refined_argmax(np.zeros((1, 8, 8), dtype=np.float32))
    assert np.isfinite(xy).all()
    assert score[0] == 0.0


def test_the_decode_batches():
    pts = np.array([[[10.0, 10.0]], [[20.0, 30.0]]])
    target = render_gaussian_targets(pts, np.ones((2, 1)), hw=HW)
    xy, score = refined_argmax(target)
    assert xy.shape == (2, 1, 2)
    assert np.abs(xy - pts).max() < 0.1
    assert score.shape == (2, 1)


# -- the loss masks by weight mass ---------------------------------------------


def test_a_zero_weight_slot_is_equivalent_to_the_slot_not_existing():
    """The invariant. A frame labelling 6 of 38 points must not be penalized for the 32."""
    pred = np.zeros((1, 2, 8, 8))
    target = np.zeros((1, 2, 8, 8))
    target[0, 0, 4, 4] = 1.0

    with_dead = masked_heatmap_loss(pred, target, np.array([[1.0, 0.0]]))
    absent = masked_heatmap_loss(pred[:, :1], target[:, :1], np.array([[1.0]]))
    assert with_dead == pytest.approx(absent)


def test_a_labelled_but_empty_slot_does_count():
    """Distinct from the above: "nothing is here" is real supervision, and it dilutes.

    Confusing the two is how a masked loss silently stops masking.
    """
    pred = np.zeros((1, 2, 8, 8))
    target = np.zeros((1, 2, 8, 8))
    target[0, 0, 4, 4] = 1.0
    assert masked_heatmap_loss(
        pred, target, np.array([[1.0, 1.0]])
    ) < masked_heatmap_loss(pred, target, np.array([[1.0, 0.0]]))


def test_a_perfect_prediction_is_zero_loss():
    target = render_gaussian_targets(
        np.array([[10.0, 10.0]]), np.array([1.0]), hw=(16, 16)
    )
    assert masked_heatmap_loss(
        target[None], target[None], np.array([[1.0]])
    ) == pytest.approx(0.0)


def test_the_foreground_weight_penalizes_a_missed_peak_more():
    """A heatmap is ~99% background, so unweighted MSE is minimized by predicting zero."""
    target = np.zeros((1, 1, 8, 8))
    target[0, 0, 4, 4] = 1.0
    pred = np.zeros((1, 1, 8, 8))
    weight = np.array([[1.0]])
    assert masked_heatmap_loss(
        pred, target, weight, foreground=10.0
    ) > masked_heatmap_loss(pred, target, weight, foreground=1.0)


def test_an_all_zero_weight_batch_does_not_divide_by_zero():
    """A batch where nothing is labelled is unusual but must not produce NaN."""
    loss = masked_heatmap_loss(
        np.zeros((1, 2, 4, 4)), np.zeros((1, 2, 4, 4)), np.zeros((1, 2))
    )
    assert np.isfinite(loss)


def test_the_torch_and_numpy_paths_agree():
    """Training runs the torch branch; tests and eval often run the numpy one."""
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(0)
    pred = rng.uniform(0, 1, (2, 3, 8, 8))
    target = render_gaussian_targets(
        rng.uniform(1, 7, (2, 3, 2)), np.ones((2, 3)), hw=(8, 8)
    )
    weight = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

    numpy_loss = masked_heatmap_loss(pred, target, weight)
    torch_loss = masked_heatmap_loss(
        torch.tensor(pred),
        torch.tensor(target, dtype=torch.float64),
        torch.tensor(weight),
    )
    assert float(torch_loss) == pytest.approx(numpy_loss, rel=1e-9)


def test_the_torch_path_is_differentiable():
    """It is the training loss, so a detached or non-differentiable path would be silent."""
    torch = pytest.importorskip("torch")

    pred = torch.zeros((1, 1, 8, 8), requires_grad=True)
    target = torch.zeros((1, 1, 8, 8))
    target[0, 0, 4, 4] = 1.0
    masked_heatmap_loss(pred, target, torch.ones((1, 1))).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    # The gradient points at the peak, which is the whole point.
    assert pred.grad[0, 0, 4, 4].abs() > pred.grad[0, 0, 0, 0].abs()
