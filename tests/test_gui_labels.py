"""Tests for the sparse ground-truth labels sidecar (``labels.h5``)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from deeperfly.gui.corrections import Corrections
from deeperfly.gui.labels import (
    LABELS_FORMAT_VERSION,
    Labels,
    Provenance,
    export_absent,
    export_gt,
    labels_identity,
    load_labels,
    migrate_from_corrections,
    resolve_point_names,
    save_labels,
)


def _identity(result):
    return labels_identity(
        point_names=list(result.skeleton.point_names),
        camera_names=list(result.cameras.names),
        n_frames=result.n_frames,
        image_sizes={name: (256, 256) for name in result.cameras.names},
        footage={name: {"rel": [f"{name}.mp4"]} for name in result.cameras.names},
    )


# -- in-memory model ----------------------------------------------------------


def test_empty_labels_are_blank(result):
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    assert lab.gt.shape == (*result.pts2d.shape[:3], 2)
    assert np.isnan(lab.gt).all()
    assert not lab.has_gt.any()
    assert not lab.occluded.any()
    assert not lab.any_labels
    assert not lab.dirty


def test_set_gt_records_pixel_and_provenance_and_clears_occluded(result):
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_occluded(0, 1, 4, True)
    lab.set_gt(0, 1, 4, (12.0, 34.0), provenance=Provenance.CONFIRMED_PREDICTION)
    assert lab.has_gt[0, 1, 4]
    assert np.allclose(lab.gt[0, 1, 4], [12.0, 34.0])
    assert lab.gt_provenance[0, 1, 4] == Provenance.CONFIRMED_PREDICTION
    assert not lab.occluded[0, 1, 4]  # placing GT clears occlusion
    assert lab.dirty


def test_set_occluded_clears_gt(result):
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(2, 0, 5, (1.0, 2.0))
    lab.set_occluded(2, 0, 5, True)
    assert lab.occluded[2, 0, 5]
    assert not lab.has_gt[2, 0, 5]
    assert lab.gt_provenance[2, 0, 5] == Provenance.NONE


def test_clear_helpers(result):
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(0, 0, 4, (1.0, 2.0))
    lab.set_occluded(1, 0, 4, True)
    lab.clear_point(0, 4)
    assert not lab.has_gt[:, 0, 4].any()
    assert not lab.occluded[:, 0, 4].any()


# -- persistence roundtrip ----------------------------------------------------


def test_roundtrip_sparse(tmp_path, result):
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(0, 1, 4, (5.0, 6.0), provenance=Provenance.DRAGGED)
    lab.set_gt(2, 0, 5, (7.0, 8.0), provenance=Provenance.CONFIRMED_PROJECTION)
    lab.set_occluded(3, 2, 6, True)
    lab.set_reviewed(1, True)  # a per-frame reviewed flag round-trips too

    path = tmp_path / "labels.h5"
    identity = _identity(result)
    save_labels(path, lab, identity=identity)
    assert not lab.dirty  # saving clears dirty

    loaded = load_labels(path, identity=identity)
    assert loaded is not None
    np.testing.assert_array_equal(loaded.has_gt, lab.has_gt)
    np.testing.assert_array_equal(loaded.occluded, lab.occluded)
    np.testing.assert_array_equal(loaded.gt_provenance, lab.gt_provenance)
    np.testing.assert_array_equal(np.nan_to_num(loaded.gt), np.nan_to_num(lab.gt))
    np.testing.assert_array_equal(loaded.reviewed, lab.reviewed)


def test_load_missing_returns_none(tmp_path, result):
    assert load_labels(tmp_path / "absent.h5", identity=_identity(result)) is None


def test_load_pre_v2_file_without_reviewed_group(tmp_path, result):
    # A pre-v2 sidecar has no `reviewed` group; loading must tolerate its absence and
    # yield no reviewed frames (the labels themselves still load).
    import h5py

    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(0, 0, 1, (5.0, 6.0))
    path = tmp_path / "labels.h5"
    identity = _identity(result)
    save_labels(path, lab, identity=identity)
    with h5py.File(path, "a") as f:
        del f["reviewed"]

    loaded = load_labels(path, identity=identity)
    assert loaded is not None
    assert not loaded.reviewed.any()
    assert loaded.has_gt[0, 0, 1]


def test_load_drops_out_of_range_reviewed(tmp_path, result):
    # A stale/oversized reviewed frame index is dropped, not indexed out of bounds.
    import h5py

    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_reviewed(0, True)
    path = tmp_path / "labels.h5"
    identity = _identity(result)
    save_labels(path, lab, identity=identity)
    with h5py.File(path, "a") as f:
        del f["reviewed/index"]
        f["reviewed"].create_dataset(
            "index", data=np.array([0, result.n_frames + 5], dtype="int32")
        )

    loaded = load_labels(path, identity=identity)
    assert loaded is not None
    assert loaded.reviewed[0] and int(loaded.reviewed.sum()) == 1


def test_prediction_only_rerun_keeps_labels(tmp_path, result):
    # The identity excludes predictions and created_utc, so a re-run of the same
    # recording (same names/sizes/footage) still loads its labels.
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(0, 0, 1, (1.0, 2.0))
    path = tmp_path / "labels.h5"
    save_labels(path, lab, identity=_identity(result))
    # a fresh identity built the same way (predictions changed but not the fingerprint)
    assert load_labels(path, identity=_identity(result)) is not None


def test_load_different_recording_refused(tmp_path, result):
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(0, 0, 1, (1.0, 2.0))
    path = tmp_path / "labels.h5"
    save_labels(path, lab, identity=_identity(result))

    other = labels_identity(
        point_names=list(result.skeleton.point_names),
        camera_names=list(result.cameras.names),
        n_frames=result.n_frames,
        image_sizes={name: (256, 256) for name in result.cameras.names},
        footage={name: {"rel": ["OTHER.mp4"]} for name in result.cameras.names},
    )
    with pytest.raises(ValueError, match="different recording"):
        load_labels(path, identity=other)


def test_load_different_domain_refused(tmp_path, result):
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(0, 0, 1, (1.0, 2.0))
    path = tmp_path / "labels.h5"
    save_labels(path, lab, identity=_identity(result))

    other = dict(_identity(result))
    other["n_frames"] = result.n_frames + 1
    with pytest.raises(ValueError, match="different result"):
        load_labels(path, identity=other)


def test_load_normalises_gt_occluded_collision(tmp_path, result):
    # Hand-write a file where one (view, frame, point) is BOTH gt and occluded; the
    # loader keeps the GT (it carries an authored pixel) and drops the occlusion.
    import h5py

    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(0, 0, 1, (5.0, 6.0))
    path = tmp_path / "labels.h5"
    identity = _identity(result)
    save_labels(path, lab, identity=identity)
    with h5py.File(path, "a") as f:
        del f["occluded/index"]
        f["occluded"].create_dataset("index", data=np.array([[0, 0, 1]], dtype="int32"))

    loaded = load_labels(path, identity=identity)
    assert loaded is not None
    assert loaded.has_gt[0, 0, 1]
    assert not loaded.occluded[0, 0, 1]


# -- migration from the legacy corrections.h5 ---------------------------------


def test_migration_maps_edits_and_confident_occlusions(result):
    corr = Corrections.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    # a dragged 2D edit on a view the detector saw -> GT
    corr.set_pts2d(0, 0, 4, (5.0, 6.0))
    # a "finalized" (fixed) pixel -> GT too
    corr.set_pts2d(1, 0, 4, (7.0, 8.0), fixed=True)
    # obscuring a view the detector DID see (finite prediction) -> confident occlusion
    corr.set_invisible(2, 0, 5, True)

    lab, report = migrate_from_corrections(corr, result.pts2d)
    assert lab.has_gt[0, 0, 4] and np.allclose(lab.gt[0, 0, 4], [5.0, 6.0])
    assert lab.has_gt[1, 0, 4] and np.allclose(lab.gt[1, 0, 4], [7.0, 8.0])
    assert lab.occluded[2, 0, 5]
    assert report["gt"] == 2
    assert report["occluded"] == 1


def test_migration_drops_ambiguous_occlusions_on_nan_predictions(result):
    # A view the detector MISSED (NaN prediction) that is invisible is ambiguous
    # (seed vs human), so it is dropped by default and re-derives as absent.
    result.pts2d[3, 0, 6] = np.nan
    corr = Corrections.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    corr.set_invisible(3, 0, 6, True)

    lab, report = migrate_from_corrections(corr, result.pts2d)
    assert not lab.occluded[3, 0, 6]
    assert report["dropped_ambiguous_occluded"] == 1

    lab_keep, _ = migrate_from_corrections(
        corr, result.pts2d, keep_ambiguous_occluded=True
    )
    assert lab_keep.occluded[3, 0, 6]


# -- export (training/eval seam) ----------------------------------------------


def test_export_gt_filters_projection_provenance(result):
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(0, 0, 5, (1.0, 2.0), provenance=Provenance.DRAGGED)
    lab.set_gt(1, 0, 5, (3.0, 4.0), provenance=Provenance.CONFIRMED_PROJECTION)
    lab.set_occluded(2, 0, 6, True)

    gt_xy, mask, occ = export_gt(lab)  # excludes projection-sourced GT by default
    assert mask[0, 0, 5] and not mask[1, 0, 5]
    assert np.allclose(gt_xy[0, 0, 5], [1.0, 2.0])
    assert not np.isfinite(gt_xy[1, 0, 5]).all()
    assert occ[2, 0, 6]

    _, mask_all, _ = export_gt(lab, include_projection=True)
    assert mask_all[1, 0, 5]  # projection GT kept when asked


def test_export_gt_never_exports_a_placeholder_seed(result):
    """``include_projection`` must not be able to turn a drag handle into a label.

    A placeholder seed is the coordinate ``EditorState._grabbable`` invents so a joint with
    no on-image position still has a dot to drag -- clamped to the image edge when the
    reprojection landed outside. Training on it would put a Gaussian at the frame border
    where the keypoint demonstrably is not. Before v5 it shared code 3 with genuine
    reprojected pixels, so "train on projections too" and "train on fabricated edge
    coordinates" were the same switch. They are not any more, and this pins that.
    """
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(0, 0, 5, (1.0, 2.0), provenance=Provenance.DRAGGED)
    lab.set_gt(1, 0, 5, (3.0, 4.0), provenance=Provenance.CONFIRMED_PROJECTION)
    lab.set_gt(2, 0, 5, (4.0, 387.9), provenance=Provenance.PLACEHOLDER_SEED)

    for include in (False, True):
        _, mask, _ = export_gt(lab, include_projection=include)
        assert mask[0, 0, 5], "a dragged pixel is always exported"
        assert bool(mask[1, 0, 5]) == include, "projection follows the flag"
        assert not mask[2, 0, 5], (
            f"placeholder seed exported with include_projection={include}"
        )


def test_a_dragged_placeholder_seed_becomes_real(result):
    """The seed is machinery until the operator moves it; then it is evidence."""
    from deeperfly.gui.state import EditorState

    state = EditorState.from_result(result)
    state.labels.set_gt(0, 0, 5, (4.0, 387.9), provenance=Provenance.PLACEHOLDER_SEED)
    _, mask, _ = export_gt(state.labels, include_projection=True)
    assert not mask[0, 0, 5]

    state.apply_2d_edit(0, 5, (123.0, 45.0), 0)
    assert state.labels.gt_provenance[0, 0, 5] == Provenance.DRAGGED
    _, mask, _ = export_gt(state.labels, include_projection=True)
    assert mask[0, 0, 5]


# -- absence: "this keypoint is not on this animal" (v3) ----------------------


def _absent_labels(result, points=(4,)):
    """An overlay with GT + an occlusion on ``points``, then ``points`` declared absent."""
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    for p in points:
        lab.set_gt(0, 0, p, (1.0, 2.0), provenance=Provenance.DRAGGED)
        lab.set_occluded(1, 0, p, True)
    lab.set_gt(0, 0, 7, (9.0, 9.0))  # a bystander point that must be untouched
    lab.set_occluded(1, 0, 7, True)
    lab.set_absent(list(points), True)
    return lab


def test_absent_defaults_to_nothing(result):
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    assert lab.absent.shape == (result.n_frames, result.pts2d.shape[2])
    assert not lab.absent.any()
    assert not lab.any_labels  # an all-False declaration is not an authored label


def test_absent_vetoes_derived_masks_but_keeps_authored_state(result):
    lab = _absent_labels(result, points=(4,))
    # The raw authored state is untouched ...
    assert lab.gt_authored[0, 0, 4]
    assert lab.occluded[1, 0, 4]
    assert np.allclose(lab.gt[0, 0, 4], [1.0, 2.0])
    # ... while every consumer mask vetoes it.
    assert not lab.has_gt[0, 0, 4]
    assert not lab.occluded_effective[1, 0, 4]
    # The bystander point is unaffected.
    assert lab.has_gt[0, 0, 7] and lab.occluded_effective[1, 0, 7]


def test_undeclaring_absence_restores_everything(result):
    lab = _absent_labels(result, points=(4,))
    lab.set_absent([4], False)
    assert lab.has_gt[0, 0, 4] and np.allclose(lab.gt[0, 0, 4], [1.0, 2.0])
    assert lab.occluded_effective[1, 0, 4]


def test_absence_alone_counts_as_an_authored_label(result):
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_absent([2], True)
    assert lab.any_labels  # so a declaration alone is worth saving
    assert lab.dirty


def test_absent_accessors_agree_on_a_whole_recording_declaration(result):
    lab = _absent_labels(result, points=(3, 4))
    np.testing.assert_array_equal(lab.absent_at(0), lab.absent_all_frames())
    np.testing.assert_array_equal(lab.absent_at(result.n_frames - 1), lab.absent[-1])
    np.testing.assert_array_equal(lab.absent_any_frame(), lab.absent_all_frames())


def test_absent_can_be_declared_for_a_single_frame(result):
    # Autotomy: the joint is there, then it is not. `absent_all_frames` is the stricter
    # question structural consumers ask, and it must stay False for a partial declaration.
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_absent([4], True, frames=1)
    assert lab.absent_at(1)[4]
    assert not lab.absent_at(0)[4]
    assert not lab.absent_all_frames()[4]
    assert lab.absent_any_frame()[4]


def test_absent_from_a_frame_onward(result):
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    onset = 1
    lab.set_absent([4], True, frames=range(onset, result.n_frames))
    assert not lab.absent_at(0)[4]
    assert all(lab.absent_at(t)[4] for t in range(onset, result.n_frames))
    assert not lab.absent_all_frames()[4]


def test_a_partial_declaration_vetoes_only_its_own_frames(result):
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(0, 0, 4, (1.0, 2.0))
    lab.set_gt(0, 1, 4, (3.0, 4.0))
    lab.set_absent([4], True, frames=1)
    assert lab.has_gt[0, 0, 4]  # frame 0 is untouched
    assert not lab.has_gt[0, 1, 4]  # frame 1 is vetoed
    assert lab.gt_authored[0, 1, 4]  # ... but not destroyed


def test_clear_frame_and_clear_point_do_not_undeclare_absence(result):
    # The everyday reset verbs must not silently un-declare an amputation.
    lab = _absent_labels(result, points=(4,))
    lab.clear_point(0, 4)
    assert lab.absent_all_frames()[4]
    lab.clear_frame(0)
    assert lab.absent_all_frames()[4]


def test_absent_roundtrip_quarantines_vetoed_rows(tmp_path, result):
    import h5py

    lab = _absent_labels(result, points=(4,))
    path = tmp_path / "labels.h5"
    identity = _identity(result)
    save_labels(path, lab, identity=identity, subject_id="Fly2")

    # The file itself is self-consistent: a naive reader of gt/index and
    # occluded/index sees only live rows, so it cannot train on a phantom leg.
    # v6 indices are [view, frame, instance, point], so the point is the LAST column.
    with h5py.File(path, "r") as f:
        live_gt = np.asarray(f["gt/index"][()]).reshape(-1, 4)
        live_occ = np.asarray(f["occluded/index"][()]).reshape(-1, 4)
        assert 4 not in live_gt[:, -1].tolist()
        assert 4 not in live_occ[:, -1].tolist()
        assert 7 in live_gt[:, -1].tolist()  # the bystander survives
        assert np.asarray(f["absent/index"][()]).tolist() == [4]
        assert np.asarray(f["absent/void_gt/index"][()]).reshape(-1, 4)[
            :, -1
        ].tolist() == [4]
        assert np.asarray(f["absent/void_occluded/index"][()]).reshape(-1, 4)[
            :, -1
        ].tolist() == [4]
        # Single-animal build: every row is instance 0 (the column is reserved, not used).
        assert set(live_gt[:, 2].tolist()) <= {0}

    # Nothing is lost: the quarantined rows come back, still vetoed.
    loaded = load_labels(path, identity=identity)
    assert loaded is not None
    assert loaded.subject_id == "Fly2"
    np.testing.assert_array_equal(loaded.absent, lab.absent)
    np.testing.assert_array_equal(loaded.gt_authored, lab.gt_authored)
    np.testing.assert_array_equal(loaded.occluded, lab.occluded)
    np.testing.assert_array_equal(loaded.has_gt, lab.has_gt)
    np.testing.assert_array_equal(np.nan_to_num(loaded.gt), np.nan_to_num(lab.gt))
    loaded.set_absent([4], False)
    assert loaded.has_gt[0, 0, 4] and loaded.occluded_effective[1, 0, 4]


def test_absent_roundtrip_is_idempotent(tmp_path, result):
    # Save -> load -> save must reach a fixed point. This is what the raw-mask tie
    # break at load time protects: reading `has_gt` there would let a vetoed
    # gt+occluded cell flip its resolution on the second pass.
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(0, 0, 4, (1.0, 2.0))
    lab.occluded[0, 0, 4] = True  # deliberately violate disjointness on a vetoed cell
    lab.set_absent([4], True)
    identity = _identity(result)
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    save_labels(a, lab, identity=identity)
    once = load_labels(a, identity=identity)
    assert once is not None
    save_labels(b, once, identity=identity)
    twice = load_labels(b, identity=identity)
    assert twice is not None
    np.testing.assert_array_equal(once.gt_authored, twice.gt_authored)
    np.testing.assert_array_equal(once.occluded, twice.occluded)
    np.testing.assert_array_equal(once.absent, twice.absent)


def test_load_v2_file_without_absent_group(tmp_path, result):
    # A v2 sidecar has no `absent` group: it loads with nothing declared absent.
    import h5py

    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(0, 0, 1, (5.0, 6.0))
    path = tmp_path / "labels.h5"
    identity = _identity(result)
    save_labels(path, lab, identity=identity)
    with h5py.File(path, "a") as f:
        del f["absent"]
        meta = json.loads(f.attrs["meta"])
        meta["deeperfly_labels_format_version"] = 2
        f.attrs["meta"] = json.dumps(meta)

    loaded = load_labels(path, identity=identity)
    assert loaded is not None
    assert not loaded.absent.any()
    assert loaded.has_gt[0, 0, 1]


def test_load_refuses_a_newer_format_version(tmp_path, result):
    import h5py

    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(0, 0, 1, (5.0, 6.0))
    path = tmp_path / "labels.h5"
    identity = _identity(result)
    save_labels(path, lab, identity=identity)
    with h5py.File(path, "a") as f:
        meta = json.loads(f.attrs["meta"])
        meta["deeperfly_labels_format_version"] = LABELS_FORMAT_VERSION + 1
        f.attrs["meta"] = json.dumps(meta)

    with pytest.raises(ValueError, match="newer deeperfly"):
        load_labels(path, identity=identity)


def test_load_drops_out_of_range_absent_index(tmp_path, result):
    # A v3 file carries only `absent/index`, meaning "absent in every frame".
    import h5py

    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_absent([1], True)
    path = tmp_path / "labels.h5"
    identity = _identity(result)
    save_labels(path, lab, identity=identity)
    n_points = result.pts2d.shape[2]
    with h5py.File(path, "a") as f:
        del f["absent/index"]
        del f["absent/spans"]  # force the v3 read path
        f["absent"].create_dataset(
            "index", data=np.array([1, n_points + 3], dtype="int32")
        )

    loaded = load_labels(path, identity=identity)
    assert loaded is not None
    assert loaded.absent_all_frames()[1]
    assert int(loaded.absent_all_frames().sum()) == 1


def test_load_v3_whole_recording_index_without_spans(tmp_path, result):
    import h5py

    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_absent([2], True)
    path = tmp_path / "labels.h5"
    identity = _identity(result)
    save_labels(path, lab, identity=identity)
    with h5py.File(path, "a") as f:
        del f["absent/spans"]
        meta = json.loads(f.attrs["meta"])
        meta["deeperfly_labels_format_version"] = 3
        f.attrs["meta"] = json.dumps(meta)

    loaded = load_labels(path, identity=identity)
    assert loaded is not None
    assert loaded.absent[:, 2].all()  # v3's index meant every frame
    assert loaded.absent_all_frames()[2]


def test_partial_absence_roundtrips_as_spans(tmp_path, result):
    import h5py

    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_absent([4], True, frames=[1, 2])
    lab.set_absent([5], True)  # whole recording
    path = tmp_path / "labels.h5"
    identity = _identity(result)
    save_labels(path, lab, identity=identity)

    with h5py.File(path, "r") as f:
        spans = np.asarray(f["absent/spans"][()]).reshape(-1, 3)
        # A whole-recording declaration costs ONE row however long the recording is.
        assert [list(r) for r in spans if r[0] == 5] == [[5, 0, result.n_frames]]
        # `index` stays the whole-recording subset, which is what a v3 reader understands.
        assert np.asarray(f["absent/index"][()]).tolist() == [5]

    loaded = load_labels(path, identity=identity)
    assert loaded is not None
    np.testing.assert_array_equal(loaded.absent, lab.absent)


def test_export_excludes_absent_from_both_masks(result):
    lab = _absent_labels(result, points=(4,))
    gt_xy, mask, occ = export_gt(lab)
    # An absent point is neither ground truth ...
    assert not mask[:, 0, 4].any()
    assert not np.isfinite(gt_xy[0, 0, 4]).all()
    # ... nor "occluded in every view", which would be a positive label teaching the
    # detector that the amputated joint exists but is hidden.
    assert not occ[:, 0, 4].any()
    # The bystander is still exported.
    assert mask[0, 0, 7] and occ[1, 0, 7]
    np.testing.assert_array_equal(export_absent(lab), lab.absent)
    assert export_absent(lab).shape == (result.n_frames, result.pts2d.shape[2])


def test_export_gt_is_byte_identical_without_a_declaration(result):
    # The golden no-op: with nothing absent, the export contract is unchanged.
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(0, 0, 5, (1.0, 2.0))
    lab.set_occluded(2, 0, 6, True)
    gt_xy, mask, occ = export_gt(lab)
    np.testing.assert_array_equal(mask, np.isfinite(lab.gt).all(axis=-1))
    np.testing.assert_array_equal(occ, lab.occluded)
    np.testing.assert_array_equal(np.nan_to_num(gt_xy), np.nan_to_num(lab.gt))
    assert not export_absent(lab).any()


def test_resolve_point_names_accepts_globs_and_rejects_typos(result):
    names = list(result.skeleton.point_names)
    assert resolve_point_names([names[2]], names) == [2]
    assert resolve_point_names(names[:3], names) == [0, 1, 2]
    with pytest.raises(ValueError, match="no skeleton point matches"):
        resolve_point_names(["definitely_not_a_point"], names)
