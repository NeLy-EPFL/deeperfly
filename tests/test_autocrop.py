"""Tests for the searched detector crop (``{ op = "crop", auto = true }``).

Three layers, because the failure modes are at three different heights:

* the **placeholder op** -- an unresolved automatic crop must be loud, and its JSON must be
  the declaration rather than the searched box (or a run recomputes detection forever);
* the **pieces** -- candidate geometry, frame sampling, the clipping veto, the compass
  search, the sidecar -- each with a property a bug would break;
* the **search itself**, against a stand-in detector whose optimum is known analytically, so
  "does it find the box" is a real assertion and not a smoke test.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from helpers import output_points_table

from deeperfly.config import AutoCropParams, Config
from deeperfly.pose2d import autocrop
from deeperfly.preprocessing import (
    AutoCrop,
    Crop,
    Fliplr,
    FrameTransform,
    UnresolvedAutoCrop,
    frame_transform_from_ops,
)

# -- the placeholder op -------------------------------------------------------------------


def test_an_unresolved_auto_crop_refuses_every_geometric_question():
    t = frame_transform_from_ops([{"op": "crop", "auto": True}], "p")
    assert t.needs_auto_crop
    for call in (
        lambda: t.output_size((64, 128)),
        lambda: t.affine((64, 128)),
        lambda: t.apply(np.zeros((2, 64, 128, 3), np.uint8)),
        lambda: t.raw_window((64, 128)),
        lambda: t.map_points(np.zeros((3, 2)), (64, 128)),
    ):
        with pytest.raises(UnresolvedAutoCrop, match="searched by the pose2d stage"):
            call()


def test_a_resolved_auto_crop_is_exactly_the_equivalent_crop():
    box = (12, 7, 40, 20)
    auto = frame_transform_from_ops(
        [{"op": "crop", "auto": True}], "p"
    ).resolve_auto_crop(box)
    plain = FrameTransform((Crop(*box),))
    size = (64, 128)
    frames = np.arange(2 * 64 * 128 * 3, dtype=np.uint8).reshape(2, 64, 128, 3)
    assert auto.output_size(size) == plain.output_size(size)
    assert np.array_equal(auto.affine(size), plain.affine(size))
    assert auto.raw_window(size) == plain.raw_window(size)
    assert np.array_equal(auto.apply(frames), plain.apply(frames))
    assert not auto.needs_auto_crop


def test_resolving_keeps_the_rest_of_the_chain_and_its_order():
    """A mirrored pathway crops THEN flips, which is what the detector was trained through."""
    t = frame_transform_from_ops(
        [{"op": "crop", "auto": True}, {"op": "fliplr"}], "p"
    ).resolve_auto_crop((1, 0, 4, 2))
    assert isinstance(t.ops[-1], Fliplr) and t.reverses_handedness
    frames = np.arange(6 * 8 * 1, dtype=np.uint8).reshape(1, 6, 8, 1)
    expected = np.flip(frames[:, 0:2, 1:5], axis=-2)
    assert np.array_equal(t.apply(frames), expected)


def test_the_json_is_the_declaration_not_the_searched_box():
    """Fingerprint stability: a searched box must not look like a config edit.

    The pose2d fingerprint is built from the plan's ops. If resolving changed them, the run
    that searched a box would invalidate the detections it just computed, and every later run
    would search and re-detect again.
    """
    t = frame_transform_from_ops(
        [{"op": "crop", "auto": True, "x": 1, "y": 2, "width": 8, "height": 4}], "p"
    )
    before = t.to_json()
    assert before == [
        {"op": "crop", "auto": True, "x": 1, "y": 2, "width": 8, "height": 4}
    ]
    assert t.resolve_auto_crop((90, 90, 10, 5)).to_json() == before


def test_a_partial_seed_is_refused():
    with pytest.raises(ValueError, match="partial seed box"):
        frame_transform_from_ops([{"op": "crop", "auto": True, "x": 1}], "p")


def test_a_crop_without_a_box_says_that_auto_exists():
    with pytest.raises(ValueError, match="auto = true"):
        frame_transform_from_ops([{"op": "crop"}], "p")


def test_auto_must_be_a_boolean():
    with pytest.raises(ValueError, match="auto must be true or false"):
        frame_transform_from_ops([{"op": "crop", "auto": "yes"}], "p")


def test_two_automatic_crops_in_one_chain_are_ambiguous():
    with pytest.raises(ValueError, match="at most one automatic crop"):
        frame_transform_from_ops(
            [{"op": "crop", "auto": True}, {"op": "crop", "auto": True}], "p"
        )


def test_the_pose2d_fingerprint_is_unchanged_by_resolving(tmp_path):
    from deeperfly.pipeline.fingerprint import stage_fingerprint

    config = _auto_config()
    enabled = config.stage_flags()
    before = stage_fingerprint("pose2d", config, enabled, None)
    config.auto_crops = {"crop_a": (2, 1, 20, 10)}
    assert stage_fingerprint("pose2d", config, enabled, None) == before


# -- candidate geometry -------------------------------------------------------------------


def test_fit_keeps_the_aspect_and_stays_inside_the_frame():
    frame = (100, 200)
    for cx, cy, w in [(100, 50, 80), (0, 0, 80), (199, 99, 80), (100, 50, 5000)]:
        x, y, bw, bh = autocrop._fit(cx, cy, w, 2.0, frame)
        assert 0 <= x and 0 <= y
        assert x + bw <= frame[1] and y + bh <= frame[0]
        assert abs(bw / bh - 2.0) < 0.05


def test_fit_translates_rather_than_shrinking_when_it_can():
    """A box that fits but is off-centre keeps its SIZE -- shrinking it would change the
    animal's apparent scale, which is the quantity being searched."""
    x, y, w, h = autocrop._fit(10, 10, 80, 2.0, (100, 200))
    assert (w, h) == (80, 40)
    assert (x, y) == (0, 0)


def test_the_blind_cover_spans_the_frame_and_a_range_of_scales():
    boxes = autocrop._cover_boxes((1008, 1600), 2.0)
    widths = {b[2] for b in boxes}
    centres_x = {b[0] + b[2] / 2 for b in boxes}
    assert max(widths) / min(widths) > 4  # scales differ by more than 4x
    assert min(centres_x) < 0.35 * 1600 < 0.65 * 1600 < max(centres_x)
    assert len(boxes) == len(set(boxes))  # deduplicated


def test_the_refine_step_shrinks_between_rounds():
    """Without decay the offsets are a fraction of a box that never shrinks, so the centre
    oscillates inside one step forever instead of converging."""
    assert autocrop.REFINE_DECAY < 1.0
    box, frame = (400, 200, 400, 200), (1008, 1600)
    spreads = []
    for rnd in range(1, 1 + autocrop.REFINE_ROUNDS):
        decay = autocrop.REFINE_DECAY ** (rnd - 1)
        offsets = tuple(o * decay for o in autocrop.REFINE_OFFSETS)
        cand = autocrop._local_boxes(box, frame, 2.0, (1.0,), offsets)
        spreads.append(max(b[0] for b in cand) - min(b[0] for b in cand))
    assert spreads == sorted(spreads, reverse=True)
    assert spreads[-1] < spreads[0] / 2


def test_the_incumbent_without_a_seed_is_the_widest_box_at_the_model_aspect():
    auto = AutoCrop()
    assert autocrop._incumbent_box(auto, (1008, 1600), 2.0) == (0, 104, 1600, 800)


def test_the_incumbent_with_a_seed_is_the_seed():
    auto = AutoCrop(seed=(1, 2, 30, 15))
    assert autocrop._incumbent_box(auto, (1008, 1600), 2.0) == (1, 2, 30, 15)


# -- frame sampling -----------------------------------------------------------------------


def test_frame_sets_are_spread_disjoint_and_deterministic():
    search, gate = autocrop.split_frame_indices(6000, 3, 8)
    assert not set(search) & set(gate)
    assert search == sorted(search) and gate == sorted(gate)
    assert autocrop.split_frame_indices(6000, 3, 8) == (search, gate)
    # Spread, not contiguous: the whole point. A block of adjacent frames at 100 fps is one
    # moment measured several times (and is what the sibling implementation silently did).
    assert min(search) < 0.1 * 6000 and max(search) > 0.9 * 6000
    assert max(np.diff(sorted(search + gate))) > 100


def test_frame_sets_degrade_gracefully_on_a_very_short_recording():
    search, gate = autocrop.split_frame_indices(2, 3, 8)
    assert search and not set(search) & set(gate)
    assert all(0 <= i < 2 for i in search + gate)


def test_no_gate_frames_are_requested_when_the_gate_is_off():
    search, gate = autocrop.split_frame_indices(100, 3, 0)
    assert len(search) == 3 and gate == []


# -- the clipping veto --------------------------------------------------------------------


class _Field:
    """Just enough model for :meth:`_Prober._clipped_fraction`."""

    def __init__(self, padded):
        self.input_size = (256, 512)
        self.padded_field = padded


def _clipped(padded, xy):
    prober = autocrop._Prober.__new__(autocrop._Prober)
    prober.model = _Field(padded)
    return prober._clipped_fraction(np.asarray(xy, float), [None])


def test_a_padded_field_counts_peaks_beyond_the_box():
    inside = [[[0.5, 0.5], [0.2, 0.9]]]
    assert _clipped(True, inside)[0] == 0.0
    assert _clipped(True, [[[0.5, 0.5], [-0.3, 0.5]]])[0] == 0.5


def test_an_unpadded_field_counts_peaks_PINNED_TO_the_border():
    """A soft-argmax over a field that stops at the input cannot leave the box, so a joint
    the crop cut off saturates against the edge -- the exact test would find nothing."""
    pinned = [[[0.5, 0.5], [0.0015, 0.5]]]  # ~0.8 model px from the left edge
    assert _clipped(True, pinned)[0] == 0.0
    assert _clipped(False, pinned)[0] == 0.5


def test_non_finite_peaks_do_not_count_as_clipped():
    assert _clipped(True, [[[0.5, 0.5], [np.nan, np.nan]]])[0] == 0.0


# -- the compass search -------------------------------------------------------------------


def test_the_compass_search_finds_a_known_optimum():
    """An analytic objective standing in for the agreement metric: distance to a target box."""
    frame, aspect = (1008, 1600), 2.0
    target = (600, 200, 500, 250)
    tcx, tcy = target[0] + target[2] / 2, target[1] + target[3] / 2

    def objective(box):
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        return abs(cx - tcx) + abs(cy - tcy) + abs(box[2] - target[2])

    start = autocrop._fit(tcx + 160, tcy - 90, target[2] * 1.9, aspect, frame)
    box, value, used, _ = autocrop._compass_refine(objective, start, aspect, frame, 60)
    assert value < objective(start) / 6
    assert used <= 60
    assert abs(box[2] - target[2]) < 0.15 * target[2]


def test_the_compass_search_respects_its_budget_and_never_worsens():
    frame, aspect = (100, 200), 2.0
    start = (40, 20, 80, 40)
    calls = []

    def objective(box):
        calls.append(box)
        return 1.0  # a flat objective: nothing ever improves

    box, value, used, _ = autocrop._compass_refine(objective, start, aspect, frame, 7)
    assert box == start and value == 1.0
    assert len(calls) <= 7


def test_an_infinite_objective_does_not_move_the_search():
    frame, aspect = (100, 200), 2.0
    start = (40, 20, 80, 40)
    box, _, _, _ = autocrop._compass_refine(
        lambda b: 0.0 if b == start else float("inf"), start, aspect, frame, 20
    )
    assert box == start


# -- the sidecar --------------------------------------------------------------------------


def test_the_sidecar_round_trips(tmp_path):
    res = autocrop.Resolution(
        preprocessor="crop_h",
        view_name="h",
        box=(1, 2, 30, 15),
        incumbent=(0, 0, 40, 20),
        seeded=False,
        accepted=True,
    )
    autocrop.write_sidecar(tmp_path, [res])
    assert autocrop.read_sidecar(tmp_path) == {"crop_h": (1, 2, 30, 15)}


def test_a_missing_or_foreign_sidecar_is_not_an_error(tmp_path):
    assert autocrop.read_sidecar(tmp_path) == {}
    (tmp_path / autocrop.SIDECAR_NAME).write_text(
        json.dumps({"version": 99, "boxes": {"crop_h": [1, 2, 3, 4]}})
    )
    assert autocrop.read_sidecar(tmp_path) == {}
    (tmp_path / autocrop.SIDECAR_NAME).write_text("not json")
    assert autocrop.read_sidecar(tmp_path) == {}


def test_a_malformed_box_is_dropped_and_the_rest_kept(tmp_path):
    (tmp_path / autocrop.SIDECAR_NAME).write_text(
        json.dumps(
            {
                "version": autocrop.SIDECAR_VERSION,
                "boxes": {"good": [1, 2, 3, 4], "bad": [1, 2]},
            }
        )
    )
    assert autocrop.read_sidecar(tmp_path) == {"good": (1, 2, 3, 4)}


# -- plan wiring --------------------------------------------------------------------------


def _auto_config(*, seed=None, extra_view=False, shared=False):
    """A two-camera config whose view ``a`` detects through an automatic crop."""
    skel = Config.default().data["skeleton"]
    names = skel["point_names"]
    op = {"op": "crop", "auto": True}
    if seed is not None:
        op |= {"x": seed[0], "y": seed[1], "width": seed[2], "height": seed[3]}
    pathways = [
        {"name": "a", "source": "s_a", "preprocessor": "crop_a", "model": "m"},
        {
            "name": "b",
            "source": "s_b",
            "preprocessor": "crop_a" if shared else None,
            "model": "m",
        },
    ]
    if extra_view:
        pathways.append(
            {"name": "c", "source": "s_c", "preprocessor": "crop_c", "model": "m"}
        )
    data = {
        "sources": [
            {"name": "s_a", "filename": "a*.mp4"},
            {"name": "s_b", "filename": "b*.mp4"},
            {"name": "s_c", "filename": "c*.mp4"},
        ],
        "pose2d": {
            "preprocessors": [{"name": "crop_a", "ops": [op]}]
            + ([{"name": "crop_c", "ops": [dict(op)]}] if extra_view else []),
            "models": [
                {
                    "name": "m",
                    "class": "hourglass",
                    "input_size": [32, 64],
                    "n_out_channels": len(names),
                }
            ],
            "pathways": pathways,
            "output_points": output_points_table(
                names,
                [(pw["name"], pw["name"], list(range(len(names)))) for pw in pathways],
            ),
        },
        "cameras": {
            v: {"azimuth_deg": az, "distance": 100, "focal_length_px": 500}
            for v, az in (("a", 0), ("b", 60), ("c", 120))
        },
        "skeleton": skel,
    }
    return Config.from_dict(data)


def test_targets_finds_the_unresolved_crops_and_their_view():
    plan = _auto_config().detection_plan()
    (target,) = autocrop.targets(plan)
    assert target.preprocessor == "crop_a"
    assert target.view_name == "a" and target.source == "s_a"
    assert target.pathways == ("a",)


def test_a_shared_automatic_crop_across_views_is_refused():
    with pytest.raises(ValueError, match="one searched window cannot be two"):
        autocrop.targets(_auto_config(shared=True).detection_plan())


def test_an_unused_automatic_crop_is_refused():
    config = _auto_config()
    config.data["pose2d"]["preprocessors"].append(
        {"name": "orphan", "ops": [{"op": "crop", "auto": True}]}
    )
    with pytest.raises(ValueError, match="no pathway uses it"):
        autocrop.targets(config.detection_plan())


def test_resolved_plan_reaches_the_pathways_too():
    plan = _auto_config().detection_plan()
    out = autocrop.resolved_plan(plan, {"crop_a": (3, 4, 20, 10)})
    assert not out.preprocessors["crop_a"].needs_auto_crop
    pathway = next(pw for pw in out.pathways if pw.name == "a")
    assert pathway.transform.raw_window((64, 128)) == (3, 4, 20, 10)
    assert autocrop.targets(out) == []


def test_resolved_plan_ignores_a_stale_name():
    plan = _auto_config().detection_plan()
    assert autocrop.resolved_plan(plan, {"renamed_away": (1, 2, 3, 4)}) is plan


def test_the_config_applies_recorded_boxes_to_every_later_plan():
    config = _auto_config()
    assert config.detection_plan().preprocessors["crop_a"].needs_auto_crop
    config.auto_crops = {"crop_a": (5, 6, 20, 10)}
    plan = config.detection_plan()
    assert plan.preprocessors["crop_a"].raw_window((64, 128)) == (5, 6, 20, 10)


def test_read_for_run_picks_up_a_recorded_box(tmp_path):
    (tmp_path / "config.toml").write_text(Config.default().text or "")
    autocrop.write_sidecar(
        tmp_path,
        [
            autocrop.Resolution(
                preprocessor="crop_x",
                view_name="x",
                box=(7, 8, 20, 10),
                incumbent=(0, 0, 1, 1),
                seeded=False,
            )
        ],
    )
    assert Config.read_for_run(None, tmp_path).auto_crops == {"crop_x": (7, 8, 20, 10)}


# -- the search, against a detector whose optimum is known --------------------------------


class BlobDetector:
    """A stand-in detector: most confident when the animal fills ``fill`` of the input width.

    Stands in for the real thing on exactly the property the search exploits -- confidence
    rises as the animal reaches its training scale and falls away on both sides -- so the
    optimum is analytic and "did the search find it" is a real assertion. The frames are
    black with one white rectangle (:func:`_blob_frames`).
    """

    input_size = (32, 64)
    padded_field = True
    peak_convention = "half-pixel"

    def __init__(self, *, fill: float = 0.5, joint_views: bool = False) -> None:
        self.fill = fill
        self.joint_views = joint_views
        self.forwards = 0

    def prepare(self, frames):
        import torch

        arr = np.asarray(frames)
        if arr.ndim == 3:
            arr = arr[None]
        h, w = self.input_size
        rows = np.minimum(
            (np.arange(h) * arr.shape[1] / h).astype(int), arr.shape[1] - 1
        )
        cols = np.minimum(
            (np.arange(w) * arr.shape[2] / w).astype(int), arr.shape[2] - 1
        )
        small = arr[:, rows][:, :, cols, 0].astype(np.float32) / 255.0
        return torch.from_numpy(small)[:, None].repeat(1, 3, 1, 1)

    def predict_points(self, inputs, **_):
        import torch

        x = inputs if isinstance(inputs, torch.Tensor) else torch.as_tensor(inputs)
        lead = tuple(x.shape[:-3])
        flat = x.reshape(-1, *x.shape[-3:]).numpy()
        self.forwards += 1
        n_out = 38
        conf = np.zeros((len(flat), n_out), np.float32)
        xy = np.full((len(flat), n_out, 2), 0.5, np.float32)
        h, w = self.input_size
        for i, img in enumerate(flat):
            mask = img[0] > 0.5
            if not mask.any():
                continue
            ys, xs = np.nonzero(mask)
            x0, x1 = xs.min() / w, (xs.max() + 1) / w
            y0, y1 = ys.min() / h, (ys.max() + 1) / h
            width = x1 - x0
            centre = abs((x0 + x1) / 2 - 0.5) + abs((y0 + y1) / 2 - 0.5)
            conf[i] = float(
                np.exp(-(((width - self.fill) / 0.18) ** 2)) * np.exp(-3 * centre)
            )
            corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
            xy[i] = np.resize(corners, (n_out, 2))
        return xy.reshape(*lead, n_out, 2), conf.reshape(*lead, n_out)

    def predict_points_for_views(self, inputs, views, **kw):
        """The fallback half of the real contract: decode everything, return the views asked.

        Kept here rather than narrowing, so the search's own tests exercise the path every
        model gets for free -- and so a search that silently depended on the narrowing would
        still be caught by comparing the two.
        """
        xy, conf = self.predict_points(inputs, **kw)
        idx = list(views)
        return xy[:, idx], conf[:, idx]


def _blob_frames(n, frame_hw, blob, jitter=0):
    """``n`` black frames with a white rectangle ``blob`` = ``(x, y, w, h)``."""
    frames = np.zeros((n, *frame_hw, 3), np.uint8)
    x, y, w, h = blob
    for t in range(n):
        dx = ((-1) ** t) * jitter
        frames[t, y : y + h, x + dx : x + w + dx] = 255
    return frames


def _ideal_box(blob, fill, aspect, frame_hw):
    """The crop whose resize puts ``blob`` at ``fill`` of the input width, centred."""
    x, y, w, h = blob
    return autocrop._fit(x + w / 2, y + h / 2, w / fill, aspect, frame_hw)


@pytest.mark.parametrize("joint", [False, True])
def test_the_blind_search_finds_the_animal(joint):
    """No seed, no rig: cover the frame, narrow, and land near the analytic optimum."""
    frame_hw, blob, fill = (300, 600, 3), (250, 120, 90, 45), 0.5
    frame_hw, blob = (300, 600), (250, 120, 90, 45)
    config = _auto_config()
    plan = config.detection_plan()
    model = BlobDetector(fill=fill, joint_views=joint)
    windows = {
        "s_a": _blob_frames(2, frame_hw, blob),
        "s_b": _blob_frames(2, frame_hw, blob),
        "s_c": _blob_frames(2, frame_hw, blob),
    }
    (res,) = autocrop.search(
        plan,
        {"m": model},
        search_windows=windows,
        params=AutoCropParams(search_frames=2, gate=False),
    )
    ideal = _ideal_box(blob, fill, 2.0, frame_hw)
    assert res.accepted
    got_cx = res.box[0] + res.box[2] / 2
    ideal_cx = ideal[0] + ideal[2] / 2
    assert abs(got_cx - ideal_cx) < 0.08 * ideal[2], (res.box, ideal)
    assert 0.7 < res.box[2] / ideal[2] < 1.45, (res.box, ideal)
    assert res.conf > res.conf_incumbent
    assert res.probes > 100  # it really searched


def test_a_seed_narrows_the_search_to_far_fewer_probes():
    frame_hw, blob, fill = (300, 600), (250, 120, 90, 45), 0.5
    ideal = _ideal_box(blob, fill, 2.0, frame_hw)
    seed = autocrop._fit(
        ideal[0] + ideal[2] / 2 + 20,
        ideal[1] + ideal[3] / 2,
        ideal[2] * 1.1,
        2.0,
        frame_hw,
    )
    windows = {n: _blob_frames(2, frame_hw, blob) for n in ("s_a", "s_b", "s_c")}
    params = AutoCropParams(search_frames=2, gate=False)
    blind = autocrop.search(
        _auto_config().detection_plan(),
        {"m": BlobDetector(fill=fill)},
        search_windows=windows,
        params=params,
    )[0]
    seeded = autocrop.search(
        _auto_config(seed=seed).detection_plan(),
        {"m": BlobDetector(fill=fill)},
        search_windows=windows,
        params=params,
    )[0]
    assert seeded.seeded and not blind.seeded
    assert seeded.probes < blind.probes
    for res in (blind, seeded):
        assert (
            abs((res.box[0] + res.box[2] / 2) - (ideal[0] + ideal[2] / 2))
            < 0.1 * ideal[2]
        )


def test_the_search_keeps_the_incumbent_when_it_cannot_be_beaten():
    """A seed already at the optimum: nothing to accept, and that is the right answer."""
    frame_hw, blob, fill = (300, 600), (250, 120, 90, 45), 0.5
    ideal = _ideal_box(blob, fill, 2.0, frame_hw)
    windows = {n: _blob_frames(2, frame_hw, blob) for n in ("s_a", "s_b", "s_c")}
    (res,) = autocrop.search(
        _auto_config(seed=ideal).detection_plan(),
        {"m": BlobDetector(fill=fill)},
        search_windows=windows,
        params=AutoCropParams(search_frames=2, gate=False),
    )
    assert (
        abs((res.box[0] + res.box[2] / 2) - (ideal[0] + ideal[2] / 2)) < 0.06 * ideal[2]
    )


def test_a_joint_view_model_sees_every_view_in_one_probe():
    """The V axis carries meaning for such a model, so a probe is a whole moment: the other
    views must be present, and the target must land in ITS slot."""
    frame_hw, blob = (300, 600), (250, 120, 90, 45)
    plan = _auto_config().detection_plan()
    seen: list[tuple] = []

    class Recorder(BlobDetector):
        def predict_points(self, inputs, **kw):
            seen.append(tuple(inputs.shape))
            return super().predict_points(inputs, **kw)

    autocrop.search(
        plan,
        {"m": Recorder(joint_views=True)},
        search_windows={
            n: _blob_frames(1, frame_hw, blob) for n in ("s_a", "s_b", "s_c")
        },
        params=AutoCropParams(search_frames=1, gate=False, probe_batch=4),
    )
    assert seen, "nothing was forwarded"
    assert all(len(shape) == 5 for shape in seen), seen
    assert all(shape[1] == len(plan.pathways) for shape in seen), seen


@pytest.mark.parametrize("narrows", [True, False])
def test_a_narrowed_decode_agrees_with_slicing_the_full_one(narrows):
    """The optimization must be EXACT, not merely close: same numbers, less work.

    Both halves of the contract: an implementation that can narrow its own decode, and the
    fallback for one that cannot (decode everything, slice afterwards). Which path ran is
    asserted too -- a silent fallback would still pass on the numbers alone.
    """
    import torch

    from deeperfly.pose2d.models import LoadedModel, ModelSpec

    detector = BlobDetector(joint_views=True)
    asked: list = []

    class Impl:
        """Stands in for a model module's ``impl`` (see ``LoadedModel._impl``)."""

        if narrows:

            def predict_points(
                self, model, inputs, *, method="weighted", radius=2, views=None
            ):
                asked.append(views)
                xy, conf = detector.predict_points(inputs)
                if views is None:
                    return xy, conf
                idx = list(views)
                return xy[:, idx], conf[:, idx]
        else:

            def predict_points(self, model, inputs, *, method="weighted", radius=2):
                asked.append("full")
                return detector.predict_points(inputs)

    module = torch.nn.Module()
    module.joint_views = True
    module.impl = Impl()
    loaded = LoadedModel(
        ModelSpec(name="m", cls="x", weights=None, input_size=(32, 64)), module
    )

    frames = _blob_frames(2, (300, 600), (250, 120, 90, 45))
    per_view = detector.prepare(frames)  # (2, 3, h, w)
    x = torch.stack(
        [torch.stack([per_view[t]] * 3) for t in range(2)]
    )  # (2, V=3, 3, h, w)

    full_xy, full_conf = detector.predict_points(x)
    got_xy, got_conf = loaded.predict_points_for_views(x, (1,))
    assert np.array_equal(got_xy, full_xy[:, [1]])
    assert np.array_equal(got_conf, full_conf[:, [1]])
    assert asked[-1] == ((1,) if narrows else "full")


def test_probe_batching_does_not_change_the_answer():
    frame_hw, blob = (300, 600), (250, 120, 90, 45)
    windows = {n: _blob_frames(2, frame_hw, blob) for n in ("s_a", "s_b", "s_c")}
    boxes = []
    for batch in (1, 4, 16):
        (res,) = autocrop.search(
            _auto_config().detection_plan(),
            {"m": BlobDetector()},
            search_windows=windows,
            params=AutoCropParams(search_frames=2, gate=False, probe_batch=batch),
        )
        boxes.append(res.box)
    assert len(set(boxes)) == 1, boxes


def test_a_clipping_candidate_is_refused_whatever_its_confidence():
    """The veto is on the candidate's OWN output, so a box that cuts the animal is refused
    even though cutting it raises the fill fraction and therefore the confidence."""
    frame_hw = (300, 600)
    prober = autocrop._Prober.__new__(autocrop._Prober)
    prober.model = _Field(True)
    # Half the peaks outside the box: over the 12% limit whichever way it is counted.
    xy = np.array([[[0.5, 0.5], [1.4, 0.5], [0.5, 0.5], [-0.2, 0.5]]])
    assert prober._clipped_fraction(xy, [None])[0] > autocrop.CLIP_FRACTION
    assert frame_hw  # (documents the geometry the numbers above stand for)


def test_resolved_boxes_reports_what_the_plan_carries():
    plan = _auto_config().detection_plan()
    assert autocrop.resolved_boxes(plan) == {}
    assert autocrop.resolved_boxes(
        autocrop.resolved_plan(plan, {"crop_a": (1, 2, 20, 10)})
    ) == {"crop_a": (1, 2, 20, 10)}


def test_a_recorded_box_is_reused_instead_of_searched(tmp_path, monkeypatch):
    """The caching contract: a run that already has a box neither re-searches nor moves it.

    Without this, every resume would re-search -- and worse, would detect through a box
    other than the one its cached 2D was computed with.
    """
    config = _auto_config()
    autocrop.write_sidecar(
        tmp_path,
        [
            autocrop.Resolution(
                preprocessor="crop_a",
                view_name="a",
                box=(9, 8, 40, 20),
                incumbent=(0, 0, 1, 1),
                seeded=False,
            )
        ],
    )
    config.auto_crops = autocrop.read_sidecar(tmp_path)

    def refuse(*a, **k):  # searching would need footage, a model and a rig
        raise AssertionError("the search ran despite a recorded box")

    monkeypatch.setattr(autocrop, "search", refuse)
    plan, searched = autocrop.ensure_resolved(
        config, config.detection_plan(), models={}, outdir=tmp_path
    )
    assert searched == []
    assert autocrop.resolved_boxes(plan) == {"crop_a": (9, 8, 40, 20)}


def test_a_carried_over_box_survives_rewriting_the_sidecar(tmp_path):
    """Re-searching one view must not delete another view's recorded box."""
    keep = autocrop.Resolution(
        preprocessor="crop_other",
        view_name="z",
        box=(1, 1, 10, 5),
        incumbent=(1, 1, 10, 5),
        seeded=False,
    )
    autocrop.write_sidecar(tmp_path, [keep])
    fresh = autocrop.Resolution(
        preprocessor="crop_a",
        view_name="a",
        box=(2, 2, 20, 10),
        incumbent=(0, 0, 1, 1),
        seeded=False,
    )
    carried = autocrop._recorded_resolutions(autocrop.read_sidecar(tmp_path), [fresh])
    autocrop.write_sidecar(tmp_path, [*carried, fresh])
    assert autocrop.read_sidecar(tmp_path) == {
        "crop_other": (1, 1, 10, 5),
        "crop_a": (2, 2, 20, 10),
    }


def test_a_panel_borrowing_an_unresolved_window_fails_loudly():
    """`crop = "pose2d"` in a run where no box was ever decided must not quietly render the
    whole frame -- the picture would look fine and disagree with the detector."""
    from deeperfly.visualization import compose

    config = _auto_config()
    plan = config.detection_plan()
    with pytest.raises(UnresolvedAutoCrop):
        compose.Sources(
            camera_group=config.camera_group(
                image_sizes={v: (300, 600) for v in ("a", "b", "c")}
            ),
            skeleton=config.skeleton(),
            frames={},
            pts2d=np.zeros((3, 1, 38, 2)),
            pts3d=np.zeros((1, 38, 3)),
        ).window("a", plan.view_transforms()["a"])


def test_search_without_frames_for_a_target_is_a_clear_failure():
    with pytest.raises(SystemExit, match="no frames for source"):
        autocrop.search(
            _auto_config().detection_plan(),
            {"m": BlobDetector()},
            search_windows={"s_b": _blob_frames(1, (60, 120), (10, 10, 20, 10))},
            params=AutoCropParams(gate=False),
        )
