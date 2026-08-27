"""Importing a stray ``deeperfly_outputs/`` into the project that already indexes it.

The gap: one recording is one entry, so a second label set beside a second copy of the same
footage reads as zero. The tests here are weighted towards the three answers to *which
recording is this* -- especially the no-op, which is the likeliest invocation of all and the
one the older ``labels-merge`` called an error -- and towards the two things that must never
happen: a prediction becoming ground truth, and a label moving by index.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
from helpers import CAMERA_NAMES

from deeperfly.gui.labels import (
    Labels,
    labels_identity,
    load_labels,
    save_labels,
)
from deeperfly.import_outputs import find_outputs, identify, import_outputs
from deeperfly.project import Project
from deeperfly.results import StageStore

N_FRAMES = 6
SIZES = {name: (48, 64) for name in CAMERA_NAMES}


# -- fixtures ------------------------------------------------------------------


def _make_recording(root, *, seed=0):
    root.mkdir(parents=True, exist_ok=True)
    for i, camera in enumerate(CAMERA_NAMES):
        (root / f"camera_{camera}.mp4").write_bytes(b"\0" * (1000 + 7 * i + seed))
    return root


def _make_outputs(
    rec_root,
    cameras,
    fly,
    *,
    gt=(),
    occluded=(),
    seeds=(),
    reviewed=(),
    stem_keyed_footage=False,
):
    """A ``deeperfly_outputs/`` with a real results.h5 and a labels.h5 holding ``gt``.

    ``stem_keyed_footage`` records the footage keyed by FILE STEM instead of view name, which
    is what ``project add``'s own discovery uses -- so the content id agrees with the one
    adoption computed. The default (view names) is what a real ``deeperfly run`` writes, and
    the mismatch between the two is the live camera-namespace divergence.
    """
    outputs = rec_root / "deeperfly_outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pts2d = rng.uniform(0, 40, size=(len(CAMERA_NAMES), N_FRAMES, 38, 2))
    footage = {
        (f"camera_{c}" if stem_keyed_footage else c): [rec_root / f"camera_{c}.mp4"]
        for c in CAMERA_NAMES
    }
    StageStore(outputs / "results.h5").write_pose2d(
        cameras=cameras,
        skeleton=fly,
        pts2d=pts2d,
        conf=np.ones(pts2d.shape[:3]),
        image_sizes=SIZES,
        footage=footage,
    )
    labels = Labels.empty(len(CAMERA_NAMES), N_FRAMES, 38)
    for (v, t, p), xy in dict(gt).items():
        labels.set_gt(v, t, p, xy)
    for v, t, p in occluded:
        labels.set_occluded(v, t, p, True)
    for (v, t, p), xy in dict(seeds).items():
        labels.seeds[v, t, p] = xy
    for t in reviewed:
        labels.set_reviewed(t, True)
    save_labels(
        outputs / "labels.h5",
        labels,
        identity=labels_identity(
            point_names=list(fly.point_names),
            camera_names=list(CAMERA_NAMES),
            n_frames=N_FRAMES,
            image_sizes=SIZES,
            footage={
                c: {"abs": [str(rec_root / f"camera_{c}.mp4")]} for c in CAMERA_NAMES
            },
        ),
    )
    return outputs


@pytest.fixture
def project(tmp_path, cameras, fly):
    """A project with one symlink-adopted recording carrying two GT cells."""
    proj = Project.create(tmp_path / "proj", skeleton="fly38")
    rec = _make_recording(tmp_path / "flyA")
    _make_outputs(rec, cameras, fly, gt={(0, 1, 0): (10.0, 20.0)})
    proj.add_recording(rec)
    return proj


def _dest_gt(project, slug="flyA"):
    entry = project.recording(slug)
    ident = json.loads(h5py.File(project.labels_path(entry), "r").attrs["meta"])[
        "identity"
    ]
    return load_labels(project.labels_path(entry), identity=ident)


# -- which recording is this ----------------------------------------------------


def test_an_adopted_recording_reports_a_no_op_rather_than_a_self_merge(project):
    """The likeliest invocation of all: the project already READS that file.

    ``project add`` symlinks by default, so pointing this command at the original outputs
    directory is the normal state, not an error. ``labels-merge`` calls it "the source and
    destination are the same file" and exits non-zero, which trains people to distrust the
    tool.
    """
    entry = project.recording("flyA")
    original = project.labels_path(entry).resolve().parent
    (plan,) = import_outputs(project, find_outputs(original))
    assert plan.outcome == "noop"
    assert "already reads that very file" in plan.reason
    assert plan.entry is not None and plan.entry.slug == "flyA"


def test_the_same_footage_in_another_tree_is_identified_and_merged(
    tmp_path, project, cameras, fly
):
    """A second copy of the recording, labeled separately -- the whole point of the feature."""
    other = _make_recording(tmp_path / "copies" / "flyA", seed=0)  # same byte sizes
    outputs = _make_outputs(other, cameras, fly, gt={(0, 2, 5): (33.0, 44.0)})
    (plan,) = import_outputs(project, find_outputs(outputs))
    assert plan.outcome == "merged", plan.reason
    assert plan.entry.slug == "flyA"
    assert "content id" in plan.matched_by
    assert plan.merge.taken_from_source == 1


def test_a_view_keyed_results_still_matches_a_stem_keyed_adoption(
    tmp_path, project, cameras, fly
):
    """The camera KEY is inside the fingerprint, so one recording has several legitimate ids.

    ``project add``'s discovery keys footage by FILE STEM; a ``deeperfly run`` records it by
    VIEW name. Same seven files, same bytes, two ids -- so an outputs directory produced by a
    real run would never match a recording adopted from a directory. Every keying is tried,
    and each candidate is still a full basename+byte-size fingerprint, so this widens the
    match without weakening it.
    """
    other = _make_recording(tmp_path / "copies" / "flyA", seed=0)
    view_keyed = _make_outputs(
        other, cameras, fly, gt={(0, 2, 5): (33.0, 44.0)}, stem_keyed_footage=False
    )
    (source,) = find_outputs(view_keyed)
    assert len(source.rec_id_candidates) > 1
    (plan,) = import_outputs(project, [source])
    assert plan.outcome == "merged", plan.reason
    assert "re-keyed camera naming" in plan.matched_by


def test_a_stem_keyed_results_matches_exactly_without_the_rekey_note(
    tmp_path, project, cameras, fly
):
    other = _make_recording(tmp_path / "copies" / "flyA", seed=0)
    outputs = _make_outputs(
        other, cameras, fly, gt={(0, 2, 5): (33.0, 44.0)}, stem_keyed_footage=True
    )
    (plan,) = import_outputs(project, find_outputs(outputs))
    assert plan.outcome == "merged", plan.reason
    assert "re-keyed" not in plan.matched_by


def test_a_recording_the_project_does_not_index_names_project_add_instead(
    tmp_path, project, cameras, fly
):
    """Adopting is strictly better: it symlinks, so there is never a second copy to merge."""
    stranger = _make_recording(tmp_path / "flyZ", seed=500)  # different byte sizes
    outputs = _make_outputs(stranger, cameras, fly, gt={(0, 0, 0): (1.0, 1.0)})
    (plan,) = import_outputs(project, find_outputs(outputs))
    assert plan.outcome == "unindexed"
    assert "deeperfly project add" in plan.reason


def test_an_explicit_recording_is_the_escape_hatch(tmp_path, project, cameras, fly):
    """For an archived recording whose content id cannot be derived at all."""
    stranger = _make_recording(tmp_path / "flyZ", seed=500)
    outputs = _make_outputs(stranger, cameras, fly, gt={(0, 0, 0): (1.0, 1.0)})
    (plan,) = import_outputs(project, find_outputs(outputs), recording="flyA")
    assert plan.outcome == "merged", plan.reason
    assert plan.matched_by.startswith("named")


def test_an_unknown_recording_name_is_reported_not_raised(
    project, tmp_path, cameras, fly
):
    rec = _make_recording(tmp_path / "flyQ", seed=900)
    outputs = _make_outputs(rec, cameras, fly, gt={(0, 0, 0): (1.0, 1.0)})
    (plan,) = import_outputs(project, find_outputs(outputs), recording="nope")
    assert plan.outcome == "fatal"
    assert "no recording" in plan.reason


# -- what travels ---------------------------------------------------------------


def test_importing_never_creates_ground_truth_from_seeds(
    tmp_path, project, cameras, fly
):
    """A seed is the MODEL's proposal. Promoting one would be indistinguishable from hand
    work forever, in the one file that cannot be regenerated -- and v7 deleted the column
    that would have recorded the difference.
    """
    other = _make_recording(tmp_path / "copies" / "flyA", seed=0)
    outputs = _make_outputs(
        other, cameras, fly, seeds={(0, 3, 7): (5.0, 6.0), (1, 3, 8): (7.0, 8.0)}
    )
    (plan,) = import_outputs(project, find_outputs(outputs), apply=True)
    assert plan.outcome == "merged", plan.reason
    assert plan.merge.taken_from_source == 0  # no GT invented
    assert plan.merge.seeds_taken == 2  # they arrive AS SEEDS
    dest = _dest_gt(project)
    assert not dest.gt_authored[0, 3, 7]
    np.testing.assert_allclose(dest.seeds[0, 3, 7], [5.0, 6.0])


def test_a_source_with_no_gt_but_occlusions_is_not_treated_as_empty(
    tmp_path, project, cameras, fly
):
    """Occlusion has no producer but the operator: 400 occlusions is 400 pieces of work."""
    other = _make_recording(tmp_path / "copies" / "flyA", seed=0)
    outputs = _make_outputs(other, cameras, fly, occluded=[(0, 0, 3), (1, 0, 3)])
    (plan,) = import_outputs(project, find_outputs(outputs))
    assert plan.outcome == "merged", plan.reason
    assert plan.merge.occluded_taken == 2


def test_a_source_with_no_labels_refuses_and_does_not_offer_predictions(
    tmp_path, project, cameras, fly
):
    other = _make_recording(tmp_path / "copies" / "flyA", seed=0)
    outputs = other / "deeperfly_outputs"
    outputs.mkdir(parents=True)
    rng = np.random.default_rng(0)
    pts2d = rng.uniform(0, 40, size=(len(CAMERA_NAMES), N_FRAMES, 38, 2))
    StageStore(outputs / "results.h5").write_pose2d(
        cameras=cameras,
        skeleton=fly,
        pts2d=pts2d,
        conf=np.ones(pts2d.shape[:3]),
        image_sizes=SIZES,
    )
    (plan,) = import_outputs(project, find_outputs(outputs))
    assert plan.outcome == "empty"
    assert "only predictions" in plan.reason


def test_an_imported_gt_frame_gains_an_annotation_skeleton(
    tmp_path, project, cameras, fly
):
    """Otherwise the frame drops to the pre-v8 display layer and looks unlabeled."""
    other = _make_recording(tmp_path / "copies" / "flyA", seed=0)
    outputs = _make_outputs(other, cameras, fly, gt={(0, 4, 9): (1.0, 2.0)})
    (plan,) = import_outputs(project, find_outputs(outputs), apply=True)
    assert plan.merge.instances_added >= 1
    assert _dest_gt(project).instance[4]


def test_a_reordered_source_skeleton_is_remapped_by_name(
    tmp_path, project, cameras, fly
):
    """The nightmare case. An index copy here would transpose every point silently."""
    other = _make_recording(tmp_path / "copies" / "flyA", seed=0)
    outputs = _make_outputs(other, cameras, fly, gt={(0, 2, 0): (11.0, 22.0)})
    # Restamp the source's identity with the point order REVERSED, leaving its rows put.
    names = list(fly.point_names)
    with h5py.File(outputs / "labels.h5", "r+") as f:
        meta = json.loads(f.attrs["meta"])
        meta["identity"]["point_names"] = list(reversed(names))
        f.attrs["meta"] = json.dumps(meta)

    (plan,) = import_outputs(project, find_outputs(outputs), apply=True)
    assert plan.outcome == "merged", plan.reason
    assert plan.merge.points.reordered
    dest = _dest_gt(project)
    # Point index 0 in the source is the LAST name, so it lands at index 37.
    np.testing.assert_allclose(dest.gt[0, 2, 37], [11.0, 22.0])
    assert not dest.gt_authored[0, 2, 0]


# -- safety ---------------------------------------------------------------------


def test_a_dry_run_writes_nothing_and_still_reports_everything(
    tmp_path, project, cameras, fly
):
    other = _make_recording(tmp_path / "copies" / "flyA", seed=0)
    outputs = _make_outputs(other, cameras, fly, gt={(0, 2, 5): (33.0, 44.0)})
    dest_path = project.labels_path(project.recording("flyA"))
    before = dest_path.read_bytes()
    (plan,) = import_outputs(project, find_outputs(outputs), apply=False)
    assert plan.merge.taken_from_source == 1
    assert dest_path.read_bytes() == before
    assert plan.snapshot is None
    assert not list(dest_path.parent.glob("*.preimport-*.h5"))


def test_apply_snapshots_the_destination_before_writing(
    tmp_path, project, cameras, fly
):
    other = _make_recording(tmp_path / "copies" / "flyA", seed=0)
    outputs = _make_outputs(other, cameras, fly, gt={(0, 2, 5): (33.0, 44.0)})
    dest_path = project.labels_path(project.recording("flyA"))
    before = dest_path.read_bytes()
    (plan,) = import_outputs(project, find_outputs(outputs), apply=True)
    assert plan.snapshot is not None and plan.snapshot.exists()
    assert plan.snapshot.read_bytes() == before
    assert dest_path.read_bytes() != before


def test_the_source_is_never_written(tmp_path, project, cameras, fly):
    other = _make_recording(tmp_path / "copies" / "flyA", seed=0)
    outputs = _make_outputs(other, cameras, fly, gt={(0, 2, 5): (33.0, 44.0)})
    before = (outputs / "labels.h5").read_bytes()
    import_outputs(project, find_outputs(outputs), apply=True)
    assert (outputs / "labels.h5").read_bytes() == before


def test_results_h5_is_never_written(tmp_path, project, cameras, fly):
    other = _make_recording(tmp_path / "copies" / "flyA", seed=0)
    outputs = _make_outputs(other, cameras, fly, gt={(0, 2, 5): (33.0, 44.0)})
    dest_results = project.results_path(project.recording("flyA"))
    before_dest = dest_results.read_bytes()
    before_src = (outputs / "results.h5").read_bytes()
    import_outputs(project, find_outputs(outputs), apply=True)
    assert dest_results.read_bytes() == before_dest
    assert (outputs / "results.h5").read_bytes() == before_src


def test_no_ground_truth_is_ever_lost(tmp_path, project, cameras, fly):
    """The invariant the whole command exists to preserve."""
    dest_before = _dest_gt(project)
    kept = {tuple(map(int, c)) for c in np.argwhere(dest_before.gt_authored)}
    other = _make_recording(tmp_path / "copies" / "flyA", seed=0)
    outputs = _make_outputs(other, cameras, fly, gt={(0, 2, 5): (33.0, 44.0)})
    import_outputs(project, find_outputs(outputs), apply=True)
    after = _dest_gt(project)
    for v, t, p in kept:
        assert after.gt_authored[v, t, p]
    np.testing.assert_allclose(after.gt[0, 1, 0], [10.0, 20.0])
    np.testing.assert_allclose(after.gt[0, 2, 5], [33.0, 44.0])


def test_an_unknown_absent_policy_is_refused(project):
    with pytest.raises(ValueError, match="on_absent must be"):
        import_outputs(project, [], on_absent="whatever")


def test_the_iteration_is_bumped_once_per_applied_invocation(
    tmp_path, project, cameras, fly
):
    before = project.iteration
    other = _make_recording(tmp_path / "copies" / "flyA", seed=0)
    outputs = _make_outputs(other, cameras, fly, gt={(0, 2, 5): (33.0, 44.0)})
    import_outputs(project, find_outputs(outputs), apply=True)
    assert project.iteration == before + 1


def test_a_dry_run_does_not_bump_the_iteration(tmp_path, project, cameras, fly):
    before = project.iteration
    other = _make_recording(tmp_path / "copies" / "flyA", seed=0)
    outputs = _make_outputs(other, cameras, fly, gt={(0, 2, 5): (33.0, 44.0)})
    import_outputs(project, find_outputs(outputs), apply=False)
    assert project.iteration == before


# -- discovery ------------------------------------------------------------------


def test_a_tree_of_outputs_dirs_is_found_whole(tmp_path, cameras, fly):
    """The measured problem is 21 label files across three trees; one path is not enough."""
    root = tmp_path / "scattered"
    for i, name in enumerate(("a", "b", "c")):
        rec = _make_recording(root / name, seed=100 * (i + 1))
        _make_outputs(rec, cameras, fly, gt={(0, 0, i): (1.0, 1.0)})
    found = find_outputs(root)
    assert len(found) == 3
    assert all(s.labels is not None for s in found)


def test_a_labels_path_resolves_to_its_outputs_dir(tmp_path, cameras, fly):
    rec = _make_recording(tmp_path / "flyA")
    outputs = _make_outputs(rec, cameras, fly, gt={(0, 0, 0): (1.0, 1.0)})
    (found,) = find_outputs(outputs / "labels.h5")
    assert found.path == outputs


def test_a_recording_dir_resolves_to_its_outputs_dir(tmp_path, cameras, fly):
    rec = _make_recording(tmp_path / "flyA")
    outputs = _make_outputs(rec, cameras, fly, gt={(0, 0, 0): (1.0, 1.0)})
    (found,) = find_outputs(rec)
    assert found.path == outputs


def test_a_missing_path_is_an_error_not_an_empty_list(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_outputs(tmp_path / "nope")


def test_an_unidentifiable_recording_asks_for_recording_rather_than_guessing(
    tmp_path, project, cameras, fly
):
    """Basenames must NOT be used as a fallback identity.

    Every recording on this rig has files named ``camera_RH.mp4`` ... ``camera_LH.mp4``, so
    matching on them would compare two unrelated flies equal and merge one's ground truth
    into the other -- unrecoverable in practice. Adding ``image_sizes`` does not help (one
    rig, one resolution). So when nothing recording-specific survives, this asks.
    """
    other = _make_recording(tmp_path / "copies" / "flyA", seed=0)
    outputs = _make_outputs(other, cameras, fly, gt={(0, 0, 0): (1.0, 1.0)})
    for f in other.glob("*.mp4"):
        f.unlink()
    with h5py.File(outputs / "results.h5", "r+") as h:
        del h["pose2d"].attrs["footage"]

    (source,) = find_outputs(outputs)
    assert source.rec_id is None  # nothing recording-specific to fingerprint
    entry, how = identify(project, source)
    assert entry is None and how == ""

    (plan,) = import_outputs(project, [source])
    assert plan.outcome == "unindexed"
    assert "--recording" in plan.reason
    # And it still imports once told which recording it is.
    (named,) = import_outputs(project, [source], recording="flyA", apply=True)
    assert named.outcome == "merged", named.reason


def test_two_different_flies_are_never_merged_by_their_identical_filenames(
    tmp_path, project, cameras, fly
):
    """The regression guard for the fallback that was removed."""
    stranger = _make_recording(tmp_path / "flyZ", seed=500)
    outputs = _make_outputs(stranger, cameras, fly, gt={(0, 0, 0): (99.0, 99.0)})
    # Same basenames, same image sizes, same frame count as flyA -- only the bytes differ.
    (plan,) = import_outputs(project, find_outputs(outputs), apply=True)
    assert plan.outcome == "unindexed"
    dest = _dest_gt(project)
    assert not dest.gt_authored[0, 0, 0]  # flyZ's pixel did not land in flyA
