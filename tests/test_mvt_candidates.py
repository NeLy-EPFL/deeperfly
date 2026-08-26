"""The multiview transformer's side of the top-K CANDIDATE path.

``decode_points`` reduces a whole channel to one soft-argmax, so it cannot serve a caller
that keeps several peaks per channel: :func:`deeperfly.pictorial.peak_candidates` holds
sub-pixel FIELD cells and has to place them itself. The shared ``(c + 0.5) / W_field``
convention it falls back to is right only when the field spans the reported frame, and this
model's does not -- it is the PADDED input's, so that convention is wrong by both the margin
and the ``(w + 2m) / w`` scale. At the shipped r28 geometry that is ~46 model px at the
frame's edge, three times the 15 px a candidate may sit from its hypothesis
(``pictorial.DEFAULT_INLIER_PX``), so every edge candidate would be silently discarded while
the run looked clean.

So the model states its own cell geometry (:func:`mvt.cells_to_input_normalized`), and this
module pins it against the production decode. Two further functions exist for the same path
and are pinned here too: :func:`mvt.points_from_heatmaps`, so the candidate path's arg-max is
the production arg-max rather than a second opinion decoded a different way, and
:func:`mvt.predict_points_and_heatmaps`, so getting both costs ONE forward.

No weights are needed: every claim here is a property of the module's geometry and decode.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mvt = pytest.importorskip("deeperfly.pose2d.mvt")

REPORTED = (256, 512)  # the r28 reported frame (h, w)
MARGIN = 48  # r28's field margin, in model px
STRIDE = 4  # 2 ** downsample_factor
FIELD = (
    (REPORTED[0] + 2 * MARGIN) // STRIDE,
    (REPORTED[1] + 2 * MARGIN) // STRIDE,
)  # (88, 152)


class _Geom:
    """Just the attributes :func:`mvt.cells_to_input_normalized` reads.

    It is a pure function of the module's declared geometry, so a real network would only
    make the test slower.
    """

    downsample_factor = 2
    hm_margin_px = MARGIN
    input_hw = REPORTED


def _to_model_px(normalized):
    """Input-normalized ``(x, y)`` -> model pixels, so a tolerance is a real distance."""
    return np.asarray(normalized) * np.array([REPORTED[1], REPORTED[0]], dtype=float)


def _decode_one_spike_per_channel(cells):
    """Decode a field carrying one unit spike per channel, at ``cells`` ``(N, 2)`` (cx, cy).

    One batched call rather than N, which is what makes sweeping every cell of an edge cheap
    enough to do exhaustively instead of at sampled points.
    """
    cells = np.asarray(cells, dtype=int)
    hm = torch.zeros(1, len(cells), *FIELD)
    for i, (cx, cy) in enumerate(cells):
        hm[0, i, cy, cx] = 1.0
    xy, _ = mvt.decode_points(hm, REPORTED, _Geom.downsample_factor, margin=MARGIN)
    return _to_model_px(xy[0].numpy())


# -- the cell -> input mapping, against the production decode ----------------------------


@pytest.mark.parametrize("axis", [0, 1], ids=["x", "y"])
def test_cell_geometry_agrees_with_the_production_decode_away_from_the_border(axis):
    """Every cell but the two at each edge maps to what ``decode_points`` reports.

    The whole justification for the candidate path having its own transform: it must be the
    SAME geometry the production readout uses, or turning ``pictorial_structures`` on would
    move every point. Swept exhaustively along a full row and a full column rather than at
    sampled cells, because a mapping wrong by a constant and one wrong by a scale look
    identical at any single point.
    """
    n = FIELD[1] if axis == 0 else FIELD[0]
    other = FIELD[0] // 2 if axis == 0 else FIELD[1] // 2
    idx = np.arange(n)
    fixed = np.full(n, other)
    cells = np.stack([idx, fixed] if axis == 0 else [fixed, idx], axis=-1)

    decoded = _decode_one_spike_per_channel(cells)
    ours = _to_model_px(mvt.cells_to_input_normalized(_Geom, cells.astype(float)))
    err = np.abs(decoded - ours).max(axis=-1)

    assert err[2:-2].max() < 1e-5, (
        f"the interior mapping disagrees with the decode by up to {err[2:-2].max()} "
        "model px; the candidate path would not be reporting production coordinates"
    )


@pytest.mark.parametrize("axis", [0, 1], ids=["x", "y"])
def test_only_the_border_cells_disagree_and_only_by_half_a_pixel(axis):
    """The bicubic upsample has nothing past the edge, so its expectation is pulled inward.

    Stated as a bound rather than left implicit, because the alternative reading -- that the
    mapping is subtly wrong everywhere and merely worst at the edge -- would be a real defect
    and looks the same if you only ever check one cell. Half a model pixel against the 15 px
    a candidate is allowed to sit from its hypothesis is not a threat to the election; a
    scale error would be.
    """
    n = FIELD[1] if axis == 0 else FIELD[0]
    other = FIELD[0] // 2 if axis == 0 else FIELD[1] // 2
    idx = np.arange(n)
    fixed = np.full(n, other)
    cells = np.stack([idx, fixed] if axis == 0 else [fixed, idx], axis=-1)

    decoded = _decode_one_spike_per_channel(cells)
    ours = _to_model_px(mvt.cells_to_input_normalized(_Geom, cells.astype(float)))
    err = np.abs(decoded - ours).max(axis=-1)

    assert sorted(np.flatnonzero(err > 1e-3)) == [0, 1, n - 2, n - 1], (
        "the disagreement is supposed to be confined to the two cells at each edge"
    )
    assert err.max() < 0.51, f"the border disagreement grew to {err.max()} model px"
    # Inward, never outward: the clamped expectation sits INSIDE the cell it belongs to.
    signed = (decoded - ours)[:, axis]
    assert signed[0] > 0 and signed[-1] < 0, (
        "the clamp should pull toward the field center"
    )


def test_a_cell_in_the_margin_maps_outside_the_reported_frame():
    """A candidate outside the frame is a location, and must not be clipped.

    The reason the padded field exists at all. If this mapping clamped to ``[0, 1]`` the
    padding would be inert for the candidate path -- every off-frame peak would pile onto the
    border, which is exactly the r27 failure the margin was added to fix.
    """
    corner = mvt.cells_to_input_normalized(_Geom, np.array([0.0, 0.0]))
    assert corner[0] < 0 and corner[1] < 0, "the first cell is up and left of the frame"
    assert _to_model_px(corner) == pytest.approx([-MARGIN, -MARGIN])

    far = mvt.cells_to_input_normalized(
        _Geom, np.array([float(FIELD[1] - 1), float(FIELD[0] - 1)])
    )
    assert far[0] > 1.0 and far[1] > 1.0, "the last cell is right of / below the frame"


def test_the_shared_convention_would_be_wrong_by_tens_of_model_pixels():
    """The size of the bug this dispatch exists to prevent, as a number.

    Not a redundant restatement of the test above: it is what makes the difference
    *actionable*, since a candidate more than ``DEFAULT_INLIER_PX`` from its hypothesis is
    dropped rather than merely misplaced. Silent, and worst exactly at the frame edge where
    an off-frame joint's candidate is the one worth having.
    """
    from deeperfly.pictorial import DEFAULT_INLIER_PX

    cell = np.array([float(FIELD[1] - 1), float(FIELD[0] - 1)])
    ours = _to_model_px(mvt.cells_to_input_normalized(_Geom, cell))
    shared = _to_model_px(
        [(cell[0] + 0.5) / FIELD[1], (cell[1] + 0.5) / FIELD[0]]
    )  # `(c + 0.5) / W_field`, correct only when the field spans the frame
    gap = np.abs(ours - shared)
    assert gap.max() > 2 * DEFAULT_INLIER_PX, (
        f"expected the shared convention to be off by well over {DEFAULT_INLIER_PX} px "
        f"(a candidate's whole budget); got {gap}"
    )


# -- the arg-max the candidate path reports ----------------------------------------------


class _Stub(torch.nn.Module):
    """The smallest module the candidate path's helpers will run.

    A fixed field plus the attributes the decode reads. Weights would add nothing: what is
    under test is that these helpers reproduce the production decode of a field, not what
    field a network produces.
    """

    def __init__(self, *, margin: int = MARGIN, floor: float = 0.0, k: int = 2) -> None:
        super().__init__()
        self.num_classes = k
        self.input_hw = REPORTED
        self.downsample_factor = 2
        self.hm_margin_px = margin
        self.conf_floor = floor
        self._p = torch.nn.Parameter(torch.zeros(1))
        h = (REPORTED[0] + 2 * margin) // STRIDE
        w = (REPORTED[1] + 2 * margin) // STRIDE
        self._hm = torch.zeros(1, k, h, w)
        self._hm[0, 0, h // 2, w // 2] = 50.0  # a sharp, confident peak
        # channel 1 stays flat: "I cannot localize this", what a checkpoint trained with
        # `off_frame_target=uniform` learns for a joint that left the frame
        self.forwards = 0

    def forward(self, x):  # noqa: D102
        self.forwards += 1
        return self._hm.expand(x.shape[0], -1, -1, -1)

    def heatmaps_by_view(self, images):  # noqa: D102
        hm = self.forward(images)
        k = self.num_classes
        return hm.reshape(hm.shape[0], hm.shape[1] // k, k, *hm.shape[-2:])


def test_points_from_heatmaps_reproduces_predict_points_exactly():
    """The candidate path's arg-max IS the production arg-max, not a second opinion.

    It genuinely would be a second opinion otherwise: the shared decode takes a windowed
    centroid over the RAW cells where this model's readout is a global soft-argmax over the
    16x-upsampled field. Before this, turning ``pictorial_structures`` on silently moved
    every point -- a change to a stage's INPUT dressed up as a change to its output.
    """
    m = _Stub()
    x = torch.zeros(1, 1, 1, REPORTED[0] + 2 * MARGIN, REPORTED[1] + 2 * MARGIN)
    want_xy, want_conf = mvt.predict_points(m, x)
    got_xy, got_conf = mvt.points_from_heatmaps(m, m.heatmaps_by_view(x))

    np.testing.assert_array_equal(got_xy, want_xy)
    np.testing.assert_array_equal(got_conf, want_conf)


def test_points_from_heatmaps_applies_the_confidence_floor():
    """Same decode AND same gate -- a floor skipped here would re-admit border points.

    The floor is not decoration: on a padded checkpoint a flat channel's soft-argmax lands
    near the field center with no confidence behind it, and admitting that as a detection is
    the failure mode ``conf_floor`` was added to stop.
    """
    x = torch.zeros(1, 1, 1, REPORTED[0] + 2 * MARGIN, REPORTED[1] + 2 * MARGIN)
    ungated, _ = mvt.points_from_heatmaps(
        _Stub(floor=0.0), _Stub(floor=0.0).heatmaps_by_view(x)
    )
    gated_m = _Stub(floor=0.01)
    gated, _ = mvt.points_from_heatmaps(gated_m, gated_m.heatmaps_by_view(x))

    assert np.isfinite(ungated).all(), "nothing should be gated with the floor off"
    assert np.isfinite(gated[0, 0, 0]).all(), (
        "the confident peak must survive the floor"
    )
    assert np.isnan(gated[0, 0, 1]).all(), "the flat channel must come back NaN"


def test_points_from_heatmaps_keeps_the_leading_axes():
    """``(B, V, K, Hm, Wm)`` in -> ``(B, V, K, 2)`` out, because the caller indexes by view.

    ``decode_points`` takes ``(B, C, Hm, Wm)``, so this has to flatten and restore. Getting
    it wrong would transpose batch against view, which for B=1 -- every call the candidate
    path makes -- is invisible.
    """
    m = _Stub(k=3)
    hm = m.heatmaps_by_view(torch.zeros(2, 1, 1, 8, 8))  # (2, 1, 3, Hf, Wf)
    xy, conf = mvt.points_from_heatmaps(m, hm)
    assert xy.shape == (2, 1, 3, 2)
    assert conf.shape == (2, 1, 3)


def test_predict_points_and_heatmaps_forwards_the_network_once():
    """One forward for both, which is the entire reason the fused function exists.

    ``predict_points`` then ``predict_heatmaps`` would run the network twice; on this
    architecture the forward is the run's dominant cost, so that is a straight 2x on the
    stage.
    """
    m = _Stub()
    x = torch.zeros(1, 1, 1, REPORTED[0] + 2 * MARGIN, REPORTED[1] + 2 * MARGIN)
    xy, conf, hm = mvt.predict_points_and_heatmaps(m, x)

    assert m.forwards == 1, f"the network was forwarded {m.forwards} times, not once"
    want_xy, want_conf = mvt.predict_points(m, x)
    np.testing.assert_array_equal(xy, want_xy)
    np.testing.assert_array_equal(conf, want_conf)
    assert hm.shape == (1, 1, m.num_classes, *FIELD)


# -- the dispatch: a model's own geometry, never another model's -------------------------


class _Loaded:
    """A :class:`~deeperfly.pose2d.models.LoadedModel` around a bare module.

    Built without the loader because loading needs an artifact; only ``self.module`` matters
    to the three methods under test.
    """

    def __new__(cls, module):
        from deeperfly.pose2d.models import LoadedModel

        obj = object.__new__(LoadedModel)
        obj.spec = None
        obj.module = module
        return obj


def test_a_model_with_impl_is_decoded_by_its_own_geometry_not_the_hrnet_fallback():
    """The trap this dispatch is one line away from: ``owns_decode`` also matches the MVT.

    ``LoadedModel.cells_to_normalized`` tries ``impl`` first and only then falls back to the
    dense HRNet's module-constant transform. The MVT sets BOTH ``impl`` and
    ``owns_decode = True``, so reversing that order -- or dropping ``impl`` from the artifact
    -- would decode this model with the other model's fixed stride, origin and input size.
    Every point would move, and nothing would raise.
    """
    import sys

    from deeperfly.pose2d import hrnet

    m = _Stub()
    m.impl = sys.modules[mvt.__name__]
    cells = np.array(
        [[0.0, 0.0], [40.0, 20.0], [float(FIELD[1] - 1), float(FIELD[0] - 1)]]
    )

    got = _Loaded(m).cells_to_normalized(cells, FIELD)
    np.testing.assert_allclose(got, mvt.cells_to_input_normalized(m, cells))
    # ... and it is NOT what the HRNet fallback would have said.
    assert not np.allclose(got, hrnet.cells_to_input_normalized_np(m, cells))


def test_a_model_without_impl_or_owns_decode_gets_the_shared_convention():
    """The documented fallback, which is correct exactly when the field spans the frame.

    Kept reachable rather than made an error: a detector class that has not claimed its
    field is padded is asserting the shared convention holds for it.
    """

    class _Plain:
        pass

    cells = np.array([[3.0, 5.0]])
    got = _Loaded(_Plain()).cells_to_normalized(cells, FIELD)
    want = [(3.0 + 0.5) / FIELD[1], (5.0 + 0.5) / FIELD[0]]
    assert got[0] == pytest.approx(want)


def test_predict_points_and_heatmaps_falls_back_to_two_calls():
    """A class not offering the fused form still works, just at two forwards.

    The fused version is an optimization, so its absence must degrade rather than fail --
    otherwise adding a detector class would mean implementing three functions to run one
    stage.
    """

    class _Plain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(1))
            self.owns_decode = False

    m = _Plain()
    loaded = _Loaded(m)
    calls = []
    loaded.predict_heatmaps = lambda inputs: (
        calls.append("hm"),
        np.zeros((1, 2, 4, 4)),
    )[1]
    loaded.points_from_heatmaps = lambda hm: (
        calls.append("pts"),
        (np.zeros((1, 2, 2)), np.ones((1, 2))),
    )[1]

    xy, conf, hm = loaded.predict_points_and_heatmaps(np.zeros((1, 1, 8, 8)))
    assert calls == ["hm", "pts"], "the fallback must decode the field it just computed"
    assert xy.shape == (1, 2, 2) and conf.shape == (1, 2) and hm.shape == (1, 2, 4, 4)


# -- the dense HRNet's NumPy counterpart -------------------------------------------------


def test_the_hrnet_numpy_transform_matches_its_torch_one():
    """Two implementations of one convention, so they are pinned to each other.

    The candidate path needs NumPy (it holds cells from ``peak_candidates``) where the
    production decode is torch. Duplicated arithmetic drifts; this is the gate that says so.
    """
    from deeperfly.pose2d import hrnet

    cells = np.array([[0.0, 0.0], [7.5, 3.25], [63.0, 31.0]])
    want = hrnet.cells_to_input_normalized(torch.as_tensor(cells)).numpy()
    got = hrnet.cells_to_input_normalized_np(None, cells)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)
