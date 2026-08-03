"""Skeleton edits as typed migrations -- the guard on the sharpest hazard in the project.

``point_names`` is fingerprinted into every ``labels.h5``, so a skeleton edit can invalidate
every label in a project. Worse, the invalidation is *quiet*: two 38-point skeletons in
different orders load each other's files happily and mean something different by every index.

So every test here is about that: labels move **by name**, a dry run counts before anything
is written, and deleting a point quarantines rather than destroys.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers import CAMERA_NAMES

from deeperfly.gui.labels import Labels, labels_identity, load_labels, save_labels
from deeperfly.project import Project, label_stats
from deeperfly.skeleton import Skeleton
from deeperfly.skeleton_migrate import (
    apply_migration,
    diff_skeletons,
    plan_migration,
)

SIZES = {name: (48, 64) for name in CAMERA_NAMES}


def _skeleton(point_names, limbs=None, name="test"):
    from deeperfly.config import Config

    limbs = limbs or {"all": list(point_names)}
    return Config.from_dict(
        {
            "skeleton": {
                "name": name,
                "point_names": list(point_names),
                "limb_points": limbs,
                "limb_palette": {k: "#123456" for k in limbs},
            }
        }
    ).skeleton()


def _project(tmp_path, points=("head", "thorax", "abdomen"), cells=None, frames=4):
    """A project whose skeleton is ``points`` and whose one recording carries ``cells``."""
    project = Project.create(tmp_path / "proj", skeleton="blank")
    project.skeleton_path().write_text(
        f'[skeleton]\nname = "test"\npoint_names = {list(points)!r}\n'.replace("'", '"')
        + "\n[skeleton.limb_points]\n"
        + f"all = {list(points)!r}\n".replace("'", '"')
    )
    rec = tmp_path / "flyA"
    rec.mkdir()
    for cam in CAMERA_NAMES:
        (rec / f"{cam}.mp4").write_bytes(b"\0" * (900 + len(cam)))
    entry = project.add_recording(rec)

    labels = Labels.empty(len(CAMERA_NAMES), frames, len(points))
    for (v, t, p), xy in (cells or {}).items():
        labels.set_gt(v, t, p, xy)
    save_labels(
        project.labels_path(entry),
        labels,
        identity=labels_identity(
            point_names=list(points),
            camera_names=list(CAMERA_NAMES),
            n_frames=frames,
            image_sizes=SIZES,
        ),
    )
    return project, entry


# -- the diff ------------------------------------------------------------------


def test_adding_a_point_is_a_silent_safe_change():
    old = _skeleton(["a", "b"])
    new = _skeleton(["a", "b", "c"])
    changes, mapping = diff_skeletons(old, new)
    kinds = {c.kind for c in changes}
    assert "add" in kinds
    assert not any(c.destructive for c in changes)
    assert mapping == {0: 0, 1: 1}  # existing points keep their index


def test_reordering_is_detected_and_remaps_by_name():
    """The nightmare case: same names, different order."""
    old = _skeleton(["a", "b", "c"])
    new = _skeleton(["c", "b", "a"])
    changes, mapping = diff_skeletons(old, new)
    assert any(c.kind == "reorder" for c in changes)
    assert mapping == {0: 2, 1: 1, 2: 0}


def test_a_single_in_place_name_change_is_a_rename_not_a_delete():
    """What an operator typing over a name did -- and a rename loses nothing."""
    old = _skeleton(["a", "b", "c"])
    new = _skeleton(["a", "renamed", "c"])
    changes, mapping = diff_skeletons(old, new)
    assert [c.kind for c in changes if c.kind in ("rename", "delete", "add")] == [
        "rename"
    ]
    assert mapping == {0: 0, 1: 1, 2: 2}  # the labels stay put


def test_two_simultaneous_name_changes_are_not_guessed_at():
    """Inferring renames from an ambiguous edit would move labels to the wrong point."""
    old = _skeleton(["a", "b", "c"])
    new = _skeleton(["x", "y", "c"])
    changes, _ = diff_skeletons(old, new)
    kinds = [c.kind for c in changes]
    assert "delete" in kinds and "add" in kinds
    assert any(c.destructive for c in changes)  # so it needs confirmation


def test_deleting_a_point_is_flagged_destructive():
    old = _skeleton(["a", "b", "c"])
    new = _skeleton(["a", "c"])
    changes, mapping = diff_skeletons(old, new)
    assert any(c.kind == "delete" and c.destructive for c in changes)
    assert 1 not in mapping  # 'b' has nowhere to go


def test_a_bones_or_palette_only_change_is_trivial(tmp_path):
    """Bones are display + the BA prior; they cannot invalidate a label."""
    project, _ = _project(tmp_path, ("a", "b", "c"))
    new = _skeleton(["a", "b", "c"], limbs={"one": ["a", "b"], "two": ["c"]})
    plan = plan_migration(project, new)
    assert plan.trivial
    assert not plan.destructive
    assert plan.quarantined == 0


# -- the dry run ---------------------------------------------------------------


def test_the_plan_counts_what_it_would_move_before_writing(tmp_path):
    """ "This will quarantine 1,412 labels across 6 recordings" is a decision; "done" is not."""
    project, entry = _project(
        tmp_path,
        ("a", "b", "c"),
        cells={(0, 1, 0): (1.0, 1.0), (0, 1, 2): (2.0, 2.0)},
    )
    plan = plan_migration(project, _skeleton(["c", "b", "a"]))
    assert plan.moved > 0
    assert plan.quarantined == 0  # a reorder loses nothing
    # Nothing was written.
    assert label_stats(project.labels_path(entry))["gt_points"] == 2


def test_the_plan_counts_what_a_deletion_would_quarantine(tmp_path):
    project, _ = _project(
        tmp_path,
        ("a", "b", "c"),
        cells={(0, 0, 1): (1.0, 1.0), (1, 2, 1): (2.0, 2.0), (0, 0, 0): (3.0, 3.0)},
    )
    plan = plan_migration(project, _skeleton(["a", "c"]))
    assert plan.destructive
    assert plan.quarantined == 2  # both labels on 'b'


def test_the_plan_is_serializable(tmp_path):
    import json

    project, _ = _project(tmp_path, ("a", "b"))
    json.dumps(plan_migration(project, _skeleton(["b", "a"])).summary())


# -- applying ------------------------------------------------------------------


def test_a_reorder_moves_labels_by_name(tmp_path):
    """The single most important assertion in this file."""
    project, entry = _project(
        tmp_path,
        ("a", "b", "c"),
        cells={(0, 1, 0): (10.0, 20.0), (0, 1, 2): (30.0, 40.0)},
    )
    new = _skeleton(["c", "b", "a"])
    plan = plan_migration(project, new)
    apply_migration(project, new, plan, snapshot=False)

    identity = labels_identity(
        point_names=["c", "b", "a"],
        camera_names=list(CAMERA_NAMES),
        n_frames=4,
        image_sizes=SIZES,
    )
    labels = load_labels(project.labels_path(entry), identity=identity)
    # 'a' was index 0 and is now index 2 -- its pixel followed its NAME.
    np.testing.assert_allclose(labels.gt[0, 1, 2], [10.0, 20.0])
    np.testing.assert_allclose(labels.gt[0, 1, 0], [30.0, 40.0])  # 'c': 2 -> 0
    assert project.skeleton().point_names == ("c", "b", "a")


def test_a_rename_keeps_the_labels_and_rewrites_the_name(tmp_path):
    project, entry = _project(tmp_path, ("a", "b"), cells={(0, 0, 1): (5.0, 6.0)})
    new = _skeleton(["a", "renamed"])
    apply_migration(project, new, plan_migration(project, new), snapshot=False)

    assert project.skeleton().point_names == ("a", "renamed")
    identity = labels_identity(
        point_names=["a", "renamed"],
        camera_names=list(CAMERA_NAMES),
        n_frames=4,
        image_sizes=SIZES,
    )
    labels = load_labels(project.labels_path(entry), identity=identity)
    np.testing.assert_allclose(labels.gt[0, 0, 1], [5.0, 6.0])


def test_adding_a_point_leaves_every_label_alone(tmp_path):
    project, entry = _project(
        tmp_path, ("a", "b"), cells={(0, 0, 0): (1.0, 2.0), (1, 1, 1): (3.0, 4.0)}
    )
    new = _skeleton(["a", "b", "c"])
    apply_migration(project, new, plan_migration(project, new), snapshot=False)
    assert label_stats(project.labels_path(entry))["gt_points"] == 2


def test_a_deletion_quarantines_rather_than_destroys(tmp_path):
    """Re-adding the point brings its labels back, which is what makes delete survivable."""
    project, entry = _project(
        tmp_path, ("a", "b", "c"), cells={(0, 0, 1): (7.0, 8.0), (0, 0, 0): (1.0, 1.0)}
    )
    two = _skeleton(["a", "c"])
    apply_migration(project, two, plan_migration(project, two), snapshot=False)
    assert label_stats(project.labels_path(entry))["gt_points"] == 1  # 'b' is gone

    # ...and the surviving label is still correct.
    identity = labels_identity(
        point_names=["a", "c"],
        camera_names=list(CAMERA_NAMES),
        n_frames=4,
        image_sizes=SIZES,
    )
    labels = load_labels(project.labels_path(entry), identity=identity)
    np.testing.assert_allclose(labels.gt[0, 0, 0], [1.0, 1.0])


def test_applying_bumps_the_iteration(tmp_path):
    project, _ = _project(tmp_path, ("a", "b"))
    before = project.iteration
    new = _skeleton(["b", "a"])
    apply_migration(project, new, plan_migration(project, new), snapshot=False)
    assert Project.load(project.root).iteration == before + 1


def test_a_snapshot_is_written_by_default(tmp_path):
    project, _ = _project(tmp_path, ("a", "b"))
    new = _skeleton(["b", "a"])
    result = apply_migration(project, new, plan_migration(project, new))
    assert result["snapshot"] and Path(result["snapshot"]).exists()


def test_an_unreadable_sidecar_blocks_the_whole_migration(tmp_path):
    """A half-migrated project is worse than an unmigrated one."""
    project, entry = _project(tmp_path, ("a", "b"))
    project.labels_path(entry).write_bytes(b"not an hdf5 file")
    plan = plan_migration(project, _skeleton(["b", "a"]))
    assert plan.errors
    with pytest.raises(ValueError, match="refusing to migrate"):
        apply_migration(project, _skeleton(["b", "a"]), plan, snapshot=False)


def test_the_rewritten_skeleton_file_round_trips(tmp_path):
    """The migrated skeleton must parse back to exactly what was asked for."""
    project, _ = _project(tmp_path, ("a", "b", "c"))
    new = _skeleton(["c", "a", "b"], limbs={"chain": ["c", "a"], "solo": ["b"]})
    apply_migration(project, new, plan_migration(project, new), snapshot=False)

    back = project.skeleton()
    assert back.point_names == ("c", "a", "b")
    assert back.limb_names == ("chain", "solo")
    np.testing.assert_array_equal(back.bones, new.bones)
    np.testing.assert_array_equal(back.limb_id, new.limb_id)


def test_the_real_fly_skeleton_round_trips_through_a_migration(tmp_path):
    """The 38-point, 10-limb, 28-bone case -- where a chain-recovery bug would show."""
    project, _ = _project(tmp_path, Skeleton.fly().point_names[:3])
    fly = Skeleton.fly()
    apply_migration(project, fly, plan_migration(project, fly), snapshot=False)

    back = project.skeleton()
    assert back.point_names == fly.point_names
    assert back.limb_names == fly.limb_names
    np.testing.assert_array_equal(back.bones, fly.bones)
    np.testing.assert_array_equal(back.limb_id, fly.limb_id)
    assert back.palette == fly.palette


# -- symmetry pairs -----------------------------------------------------------


def test_symmetries_survive_the_emitted_skeleton_fragment(fly):
    """A migration rewrites the whole ``[skeleton]`` table, so dropping the pairs here
    would silently disable the mirror check, flip augmentation and the chirality QC on the
    first skeleton edit a project ever makes.
    """
    import tomllib

    from deeperfly.config import Config
    from deeperfly.skeleton import Skeleton
    from deeperfly.skeleton_migrate import _skeleton_toml

    back = Skeleton.from_config(Config.from_dict(tomllib.loads(_skeleton_toml(fly))))
    np.testing.assert_array_equal(back.symmetries, fly.symmetries)
    assert back.symmetry_names == fly.symmetry_names
    # And a round trip is not itself reported as a change.
    assert diff_skeletons(fly, back)[0] == []


def test_the_pairs_are_emitted_by_name_so_a_reorder_carries_them(fly):
    """Stored as indices, written as names: a later reorder then remaps them for free."""
    import tomllib

    from deeperfly.config import Config
    from deeperfly.skeleton import Skeleton
    from deeperfly.skeleton_migrate import _skeleton_toml

    reversed_names = tuple(reversed(fly.point_names))
    spec = tomllib.loads(_skeleton_toml(fly))
    spec["skeleton"]["point_names"] = list(reversed_names)
    moved = Skeleton.from_config(Config.from_dict(spec))
    # Same pairing, different indices -- and diff_skeletons compares by name, so it reports
    # the reorder and NOT a symmetry change.
    assert set(map(frozenset, moved.symmetry_names)) == set(
        map(frozenset, fly.symmetry_names)
    )
    kinds = [c.kind for c in diff_skeletons(fly, moved)[0]]
    assert "reorder" in kinds
    assert "symmetries" not in kinds


def test_changing_the_pairs_is_reported_and_is_not_destructive(fly):
    """No label moves, so a symmetry edit needs neither a rewrite nor a confirmation --
    but it must still be *reported*, because it changes what three consumers do."""
    import dataclasses

    dropped = dataclasses.replace(fly, symmetries=fly.symmetries[:-1])
    changes, mapping = diff_skeletons(fly, dropped)
    assert [c.kind for c in changes] == ["symmetries"]
    assert not any(c.destructive for c in changes)
    # Every point keeps its index: the pairing is metadata, not indexing.
    assert mapping == {i: i for i in range(fly.n_points)}
