"""Tests for the sparse ground-truth labels sidecar (``labels.h5``)."""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly.gui.corrections import Corrections
from deeperfly.gui.labels import (
    Labels,
    Provenance,
    export_gt,
    labels_identity,
    load_labels,
    migrate_from_corrections,
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
