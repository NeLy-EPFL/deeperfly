"""Tests for the sparse ground-truth labels sidecar (``labels.h5``)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from deeperfly.gui.labels import (
    LABELS_FORMAT_VERSION,
    Labels,
    export_absent,
    export_gt,
    labels_identity,
    load_labels,
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


def test_gt_and_occlusion_are_independent(result):
    """Two orthogonal facts per cell: where it is, and whether a human can see it."""
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_occluded(0, 1, 4, True)
    lab.set_gt(0, 1, 4, (12.0, 34.0))
    assert lab.has_gt[0, 1, 4] and lab.occluded[0, 1, 4]
    assert np.allclose(lab.gt[0, 1, 4], [12.0, 34.0])
    assert lab.dirty

    lab.set_occluded(0, 1, 4, False)  # ... and each retracts on its own
    assert lab.has_gt[0, 1, 4] and not lab.occluded[0, 1, 4]
    lab.clear_gt(0, 1, 4)
    assert not lab.has_gt[0, 1, 4]


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
    lab.set_gt(0, 1, 4, (5.0, 6.0))
    lab.set_gt(2, 0, 5, (7.0, 8.0))
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


def test_a_reordered_camera_axis_is_remapped_by_name(tmp_path, result):
    """A from-scratch session names its views in the footage table's ALPHABETICAL order; a
    run names them in config order. Same cameras, different order -- refusing that threw
    away hand labels for a bookkeeping difference.
    """
    names = list(result.cameras.names)
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(0, 0, 1, (1.0, 2.0))  # authored on camera names[0]
    path = tmp_path / "labels.h5"
    save_labels(path, lab, identity=_identity(result))

    shuffled = names[::-1]
    other = labels_identity(
        point_names=list(result.skeleton.point_names),
        camera_names=shuffled,
        n_frames=result.n_frames,
        image_sizes={n: (256, 256) for n in shuffled},
        footage={n: {"rel": [f"{n}.mp4"]} for n in shuffled},
    )
    loaded = load_labels(path, identity=other)
    assert loaded is not None
    # The pixel followed its CAMERA: names[0] is last in the reversed order.
    assert loaded.gt_authored[len(names) - 1, 0, 1]
    np.testing.assert_allclose(loaded.gt[len(names) - 1, 0, 1], [1.0, 2.0])
    assert not loaded.gt_authored[0, 0, 1]


def test_renamed_cameras_are_remapped_by_their_footage(tmp_path, result):
    """The from-scratch break, exactly: ``camera_F`` before a run, ``f`` after it.

    The two names share no string, so name matching cannot bridge them -- but both are
    recorded against the same footage file, and a camera IS its footage. Without this, every
    label authored before the first run was refused afterwards, which is the whole
    label-first-then-calibrate workflow walking into a wall.
    """
    names = list(result.cameras.names)
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(2, 1, 3, (7.0, 8.0))
    path = tmp_path / "labels.h5"
    # Authored from a bare directory: views named after the FILE STEMS, alphabetically.
    stems = sorted(f"camera_{n}" for n in names)
    stem_of = {f"camera_{n}": n for n in names}
    save_labels(
        path,
        lab,
        identity=labels_identity(
            point_names=list(result.skeleton.point_names),
            camera_names=stems,
            n_frames=result.n_frames,
            image_sizes={s: (256, 256) for s in stems},
            footage={s: {"abs": [f"/data/{s}.mp4"]} for s in stems},
        ),
    )
    # After the run: view names, config order, same footage files.
    after = labels_identity(
        point_names=list(result.skeleton.point_names),
        camera_names=names,
        n_frames=result.n_frames,
        image_sizes={n: (256, 256) for n in names},
        footage={n: {"abs": [f"/data/camera_{n}.mp4"]} for n in names},
    )
    loaded = load_labels(path, identity=after)
    assert loaded is not None, "the labels were refused after the first run"
    # stems[2] is some camera_X; its pixel must land on view names.index(X).
    expected = names.index(stem_of[stems[2]])
    np.testing.assert_allclose(loaded.gt[expected, 1, 3], [7.0, 8.0])
    assert int(loaded.gt_authored.sum()) == 1


def test_cameras_that_cannot_be_put_in_correspondence_are_still_refused(
    tmp_path, result
):
    """The remap must not become a way to accept genuinely foreign labels."""
    names = list(result.cameras.names)
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(0, 0, 1, (1.0, 2.0))
    path = tmp_path / "labels.h5"
    save_labels(path, lab, identity=_identity(result))

    alien = [f"zzz{i}" for i in range(len(names))]
    other = labels_identity(
        point_names=list(result.skeleton.point_names),
        camera_names=alien,
        n_frames=result.n_frames,
        image_sizes={n: (256, 256) for n in alien},
        footage={n: {"rel": [f"{n}.mp4"]} for n in alien},  # different files too
    )
    with pytest.raises(ValueError, match="cannot be put in correspondence"):
        load_labels(path, identity=other)


def test_a_reordered_camera_axis_still_refuses_a_size_mismatch(tmp_path, result):
    """Through the correspondence, not around it: a same-camera size change stays fatal."""
    names = list(result.cameras.names)
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(0, 0, 1, (1.0, 2.0))
    path = tmp_path / "labels.h5"
    save_labels(path, lab, identity=_identity(result))

    shuffled = names[::-1]
    other = labels_identity(
        point_names=list(result.skeleton.point_names),
        camera_names=shuffled,
        n_frames=result.n_frames,
        image_sizes={n: (128, 128) for n in shuffled},  # a crop changed
        footage={n: {"rel": [f"{n}.mp4"]} for n in shuffled},
    )
    with pytest.raises(ValueError, match="different recording"):
        load_labels(path, identity=other)


def test_the_point_axis_is_never_silently_remapped(tmp_path, result):
    """Points stay exact. Reordering them is a project-wide migration with a dry run and a
    confirmation ('deeperfly project skeleton'); remapping here would bypass both.
    """
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(0, 0, 1, (1.0, 2.0))
    path = tmp_path / "labels.h5"
    save_labels(path, lab, identity=_identity(result))

    other = dict(_identity(result))
    other["point_names"] = list(reversed(other["point_names"]))
    with pytest.raises(ValueError, match="different result"):
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


def test_load_keeps_a_cell_that_is_both_gt_and_occluded(tmp_path, result):
    """It is no longer a collision to resolve -- it is a label to preserve.

    The loader used to drop the occlusion and keep the GT, because the two were exclusive.
    Under orthogonality that would silently discard the operator's "you cannot see this
    here", which is the half nothing else can reconstruct.
    """
    import h5py

    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(0, 0, 1, (5.0, 6.0))
    lab.set_occluded(0, 0, 1, True)
    path = tmp_path / "labels.h5"
    identity = _identity(result)
    save_labels(path, lab, identity=identity)
    with h5py.File(path, "r") as f:
        assert len(np.asarray(f["occluded/index"][()]).reshape(-1, 4)) == 1

    loaded = load_labels(path, identity=identity)
    assert loaded is not None
    assert loaded.has_gt[0, 0, 1] and loaded.occluded[0, 0, 1]
    np.testing.assert_allclose(loaded.gt[0, 0, 1], [5.0, 6.0])


# -- export (training/eval seam) ----------------------------------------------


def test_export_gt_yields_every_stored_pixel(result):
    """There is nothing to filter: a GT pixel is a pixel the operator created.

    v5/v6 tagged each row with the layer its pixel had been copied out of and the export
    dropped two of the four codes. v7 has one kind of ground truth, so the export is the
    stored rows -- which is also why the project's progress count and the export can no
    longer disagree. A cell with *no* GT is the consumer's business: it falls back in the
    editor's own precedence, GT -> detection -> projection.
    """
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(0, 0, 5, (1.0, 2.0))
    lab.set_gt(1, 0, 5, (3.0, 4.0))
    lab.set_occluded(2, 0, 6, True)

    gt_xy, mask, occ = export_gt(lab)
    assert mask[0, 0, 5] and mask[1, 0, 5]
    np.testing.assert_allclose(gt_xy[0, 0, 5], [1.0, 2.0])
    np.testing.assert_allclose(gt_xy[1, 0, 5], [3.0, 4.0])
    assert int(mask.sum()) == 2  # and nothing else
    assert occ[2, 0, 6]


def test_export_returns_the_hidden_flag_beside_the_gt_mask_not_folded_in(result):
    """The export contract the flag exists for: two masks, and their conjunction is the loss.

    A hidden cell that carries a pixel must still appear in ``gt_mask`` -- the label is
    there, and "there is a label here" and "train on it" are different questions. Folding the
    veto in would make a withheld pixel indistinguishable from one nobody ever placed, so a
    trainer could never put it back without re-reading the file.
    """
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(0, 0, 5, (1.0, 2.0))  # a pixel to train on
    lab.set_gt(1, 0, 5, (3.0, 4.0))  # ... and a pixel to withhold
    lab.set_occluded(1, 0, 5, True)

    gt_xy, mask, hidden = export_gt(lab)
    assert mask[0, 0, 5] and mask[1, 0, 5]  # both are labels
    np.testing.assert_allclose(gt_xy[1, 0, 5], [3.0, 4.0])  # the withheld pixel lives
    assert hidden[1, 0, 5] and not hidden[0, 0, 5]
    supervise = mask & ~hidden  # what a trainer's loss sees
    assert supervise[0, 0, 5] and not supervise[1, 0, 5]
    assert int(supervise.sum()) == 1


# -- absence: "this keypoint is not on this animal" (v3) ----------------------


def _absent_labels(result, points=(4,)):
    """An overlay with GT + an occlusion on ``points``, then ``points`` declared absent."""
    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    for p in points:
        lab.set_gt(0, 0, p, (1.0, 2.0))
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


def test_load_v6_file_keeps_real_gt_and_drops_the_invented_placeholder_rows(
    tmp_path, result
):
    """v5/v6 -> v7: every authored pixel survives; the editor's invented ones do not.

    A ``placeholder_seed`` row was a coordinate the editor made up (clamped to the image
    edge) so a joint with nothing on screen still had a dot to drag; ``export_gt`` dropped
    it unconditionally. v7 has no provenance column to keep telling it apart from a real
    label, so carrying it over would silently promote a fabricated edge pixel to ground
    truth. It has to be dropped at the boundary instead -- which is the one thing about
    this migration that is not reversible, so it is pinned here.
    """
    import h5py

    lab = Labels.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    lab.set_gt(0, 0, 1, (5.0, 6.0))  # a real pixel
    lab.set_gt(1, 0, 2, (7.0, 8.0))  # another
    lab.set_gt(2, 0, 3, (4.0, 387.9))  # the invented one
    path = tmp_path / "labels.h5"
    identity = _identity(result)
    save_labels(path, lab, identity=identity)
    # Re-add the v6 provenance column the current writer no longer emits.
    with h5py.File(path, "a") as f:
        rows = np.asarray(f["gt/index"][()])
        prov = np.where(
            (rows[:, 0] == 2) & (rows[:, -1] == 3),
            4,
            1,  # 4 = placeholder_seed
        ).astype(np.uint8)
        f["gt"].create_dataset("provenance", data=prov, dtype="uint8")
        meta = json.loads(f.attrs["meta"])
        meta["deeperfly_labels_format_version"] = 6
        f.attrs["meta"] = json.dumps(meta)

    loaded = load_labels(path, identity=identity)
    assert loaded is not None
    assert loaded.has_gt[0, 0, 1] and loaded.has_gt[1, 0, 2]
    np.testing.assert_allclose(loaded.gt[0, 0, 1], [5.0, 6.0])
    assert not loaded.has_gt[2, 0, 3], "an invented placeholder row became real GT"
    assert int(loaded.gt_authored.sum()) == 2

    # ... and re-saving writes v7, with no provenance column at all
    save_labels(path, loaded, identity=identity)
    with h5py.File(path, "r") as f:
        assert "gt/provenance" not in f
        assert json.loads(f.attrs["meta"])["deeperfly_labels_format_version"] == 8


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
