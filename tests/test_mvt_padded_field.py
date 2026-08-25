"""The multiview transformer's heatmap field can cover ground OUTSIDE the reported frame.

Through r27 the MVT's field was upsampled to exactly its input, so a joint the crop cut off
had nowhere to go: the soft-argmax is an expectation over in-image pixel coordinates, so it
saturated against the border and reported a confident point that was up to 200 px wrong.
``arch.hm_margin_px`` pads the network's INPUT instead, which moves the token grid and --
because the ViT interpolates its position embeddings and the head is convolutional -- grows
the field at unchanged stride 4. Every added cell is therefore computed from real tokens
that passed through every attention block, which is what a zero-pad bolted onto the head
cannot claim (its reach is one cell, measured).

The margin is deliberately INVISIBLE outside this module: ``input_hw`` stays the reported
frame, so ``ModelSpec.input_size``, the pathway's crop, the peak inversion and every config
are untouched. What changes is that a returned coordinate may now leave ``[0, 1]``, which
is the contract ``LoadedModel.padded_field`` already describes for the dense HRNet.

No weights are needed for any of this: the geometry, the guards and the coordinate mapping
are all properties of the module. End-to-end agreement with a real artifact is dfpose's
``scripts/verify_mvt_export.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mvt = pytest.importorskip("deeperfly.pose2d.mvt")

REPORTED = (256, 512)
MARGIN = 48  # 3 patches of 16; see test_margin_must_be_a_whole_number_of_patches


def _frames(n: int = 2, h: int = 300, w: int = 600) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(n, h, w, 1), dtype=np.uint8)


# -- preparation -------------------------------------------------------------------------


def test_margin_pads_the_input_and_leaves_the_animal_where_it_was():
    """The resize is unchanged; the margin is added AROUND it.

    This is what lets a padded checkpoint and an unpadded one share a coordinate system: the
    reported frame's pixels have to be bit-identical, or the margin would rescale the animal
    and every point would move.
    """
    kw = dict(mean=(0.0,), std=(1.0,), device="cpu")
    plain = mvt.prepare_images(_frames(), REPORTED, kw["mean"], kw["std"], "cpu")
    padded = mvt.prepare_images(
        _frames(), REPORTED, kw["mean"], kw["std"], "cpu", margin=MARGIN
    )
    assert plain.shape[-2:] == REPORTED
    assert padded.shape[-2:] == (REPORTED[0] + 2 * MARGIN, REPORTED[1] + 2 * MARGIN)
    inner = padded[:, :, MARGIN:-MARGIN, MARGIN:-MARGIN]
    assert torch.equal(inner, plain)


def test_the_margin_is_filled_with_the_recorded_constant_not_the_edge():
    """A flat fill, not ``BORDER_REPLICATE``.

    A replicated edge smears whatever touches the border, so a leg on its way out of frame
    grows a streaked copy of itself -- content a detector can learn to chase instead of
    learning to extrapolate. The fill is also a train/test contract, which is why the
    artifact records it and :func:`load_mvt` refuses a mismatch.
    """
    x = mvt.prepare_images(_frames(), REPORTED, (0.0,), (1.0,), "cpu", margin=MARGIN)
    band = x[:, :, :MARGIN, :]
    assert torch.allclose(band, torch.full_like(band, mvt.MARGIN_FILL_U8 / 255.0))


def test_a_negative_margin_is_refused_everywhere():
    with pytest.raises(ValueError, match="margin must be >= 0"):
        mvt.prepare_images(_frames(), REPORTED, (0.0,), (1.0,), "cpu", margin=-1)
    with pytest.raises(ValueError, match="margin must be >= 0"):
        mvt.decode_points(torch.zeros(1, 1, 64, 128), REPORTED, 2, margin=-1)


# -- decode ------------------------------------------------------------------------------


def _field(margin: int, downsample: int = 2):
    """An empty field of the size ``margin`` implies, at the shipped stride of 4."""
    del downsample  # the shipped stride is 4 regardless; see load_mvt
    h, w = REPORTED[0] + 2 * margin, REPORTED[1] + 2 * margin
    return torch.zeros(1, 1, h // 4, w // 4)


def test_the_padded_field_can_express_a_joint_outside_the_reported_frame():
    """A peak in the margin decodes to a coordinate outside ``[0, 1]``, not to the border.

    The whole point. Values outside the unit square are a location and must not be clipped;
    ``deeperfly.triangulation`` and the bundle adjustment both consume them as pixels.
    """
    hm = _field(MARGIN)
    # a sharp peak in the very first cell of the field == up and left of the reported frame
    hm[0, 0, 0, 0] = 50.0
    xy, conf = mvt.decode_points(hm, REPORTED, 2, margin=MARGIN)
    x, y = float(xy[0, 0, 0]), float(xy[0, 0, 1])
    assert x < 0.0 and y < 0.0, (
        f"peak in the margin decoded to {(x, y)}, inside the frame"
    )
    # And it lands exactly where the margin says. The margin is a pure TRANSLATION of the
    # coordinate system, so decoding the same field as if it were the whole reported frame
    # and then shifting by the margin has to give the same answer. Asserting the identity
    # rather than a literal keeps this honest about the decode's own half-cell conventions,
    # which are calibrated by the -1.5 grid offset and not by arithmetic here.
    fh, fw = REPORTED[0] + 2 * MARGIN, REPORTED[1] + 2 * MARGIN
    whole, _ = mvt.decode_points(hm, (fh, fw), 2, margin=0)
    assert x == pytest.approx(
        (float(whole[0, 0, 0]) * fw - MARGIN) / REPORTED[1], abs=1e-6
    )
    assert y == pytest.approx(
        (float(whole[0, 0, 1]) * fh - MARGIN) / REPORTED[0], abs=1e-6
    )


def test_without_a_margin_the_decode_is_exactly_what_r27_had():
    """``margin=0`` must be the shipped behavior, bit for bit.

    Every artifact through r27 declares no margin, so this path is the production one and a
    change to it would move every point of every recording ever processed.
    """
    hm = _field(0)
    hm[0, 0, 30, 64] = 50.0
    a, _ = mvt.decode_points(hm, REPORTED, 2)
    b, _ = mvt.decode_points(hm, REPORTED, 2, margin=0)
    assert torch.equal(a, b)
    # and it cannot leave the frame: expectation over in-image pixels, less the 1.5 offset
    lo_x, hi_x = -1.5 / REPORTED[1], (REPORTED[1] - 1 - 1.5) / REPORTED[1]
    assert lo_x <= float(a[0, 0, 0]) <= hi_x


def test_decoding_a_padded_field_as_unpadded_is_caught_not_shifted():
    """The one mistake that produces numbers rather than an exception.

    A 48 px margin read as 0 puts every point 48 px up and left -- an antenna's width -- and
    nothing downstream would flag it. The field-extent check is what makes it a crash.
    """
    hm = _field(MARGIN)
    with pytest.raises(RuntimeError, match="upsampled field"):
        mvt.decode_points(hm, REPORTED, 2, margin=0)
    with pytest.raises(RuntimeError, match="upsampled field"):
        mvt.decode_points(_field(0), REPORTED, 2, margin=MARGIN)


def test_the_confidence_window_is_still_read_before_the_margin_is_removed():
    """Order of operations inside the decode.

    ``_confidence_at`` indexes the padded tensor, so it must see the raw expectation. If the
    margin were subtracted first the 5x5 window would be centered `margin/1` cells away and
    the confidence would describe somewhere else.
    """
    hm = _field(MARGIN)
    hm[0, 0, 40, 70] = 50.0
    _, conf_padded = mvt.decode_points(hm, REPORTED, 2, margin=MARGIN)
    # the same peak, same field, decoded with the frame declared as the whole field: the
    # confidence must be identical because it is read in field coordinates either way
    whole = (REPORTED[0] + 2 * MARGIN, REPORTED[1] + 2 * MARGIN)
    _, conf_whole = mvt.decode_points(hm, whole, 2, margin=0)
    assert torch.allclose(conf_padded, conf_whole, atol=1e-6)


# -- the confidence gate -----------------------------------------------------------------


class _Stub(torch.nn.Module):
    """The smallest thing ``predict_points`` will run: a fixed field and the attributes it
    reads. Weights would add nothing -- the gate is arithmetic on the decode's output."""

    def __init__(self, *, margin: int, floor: float, k: int = 2) -> None:
        super().__init__()
        self.num_classes = k
        self.input_hw = REPORTED
        self.downsample_factor = 2
        self.hm_margin_px = margin
        self.conf_floor = floor
        self._p = torch.nn.Parameter(torch.zeros(1))
        h = (REPORTED[0] + 2 * margin) // 4
        w = (REPORTED[1] + 2 * margin) // 4
        self._hm = torch.zeros(1, k, h, w)
        self._hm[0, 0, h // 2, w // 2] = 50.0  # channel 0: a sharp, confident peak
        # channel 1 stays flat -- "I cannot localize this", the answer a checkpoint trained
        # with `off_frame_target=uniform` learns for a joint that left the frame

    def forward(self, x):  # noqa: D102
        return self._hm.expand(x.shape[0], -1, -1, -1)


def test_a_flat_map_is_far_less_confident_than_a_peak():
    """The separation the gate relies on, stated as a number.

    Softmax at temperature 1000 over the upsampled field: a peak concentrates its mass in
    the 5x5 window the confidence reads, a flat map spreads it over every cell. Two answers
    that are trivially separable by a threshold -- which is how confidence should be used
    here, as a cliff and not a gradient.
    """
    m = _Stub(margin=0, floor=0.0)
    _, conf = mvt.predict_points(m, torch.zeros(1, 1, 1, *REPORTED))
    peak, flat = float(conf[0, 0, 0]), float(conf[0, 0, 1])
    assert peak > 100 * flat, f"peak {peak} vs flat {flat} -- no cliff to threshold on"


def test_the_confidence_floor_reports_nan_rather_than_a_border_point():
    """Below the floor a point is ABSENT, not located.

    NaN is what ``deeperfly.triangulation`` already means by "this camera cannot see this
    point", so a gated joint is dropped from the fit instead of dragging it.
    """
    ungated = mvt.predict_points(
        _Stub(margin=0, floor=0.0), torch.zeros(1, 1, 1, *REPORTED)
    )
    gated = mvt.predict_points(
        _Stub(margin=0, floor=0.01), torch.zeros(1, 1, 1, *REPORTED)
    )
    assert np.isfinite(ungated[0]).all(), "nothing should be gated with the floor off"
    assert np.isfinite(gated[0][0, 0, 0]).all(), (
        "the confident peak must survive the floor"
    )
    assert np.isnan(gated[0][0, 0, 1]).all(), "the flat channel must come back NaN"
    # the confidence itself is reported unchanged -- the gate is about the COORDINATE, and a
    # caller wanting to know how sure the model was must still be able to ask
    assert np.allclose(ungated[1], gated[1])


def test_the_floor_defaults_off_so_r27_is_unaffected():
    """A checkpoint not trained to answer "not here" must not be gated.

    On such a model confidence does not track off-frame-ness (measured: a saturated border
    point comes back at 0.85-0.95), so a floor would drop good points on the strength of a
    number that does not mean what the gate needs it to mean.
    """
    m = _Stub(margin=0, floor=0.0)
    del m.conf_floor  # an artifact that never heard of a floor
    xy, _ = mvt.predict_points(m, torch.zeros(1, 1, 1, *REPORTED))
    assert np.isfinite(xy).all()
