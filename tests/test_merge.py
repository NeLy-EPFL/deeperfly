"""Merging label sets -- the reconciliation that unlocks labels stranded in two places.

The single most dangerous operation in this codebase is copying labels between two
38-point skeletons whose point *order* differs: every cell would still look plausible
afterwards and the whole set would be silently wrong. So the tests here are weighted
towards the refusals and the by-name remap, not towards the happy path.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly.gui.labels import Labels
from deeperfly.merge import map_by_name, merge_labels, remap_labels

POINTS = ["head", "thorax", "abdomen"]
CAMS = ["left", "right"]


def _labels(cells, *, n_views=2, n_frames=3, n_points=3, reviewed=(), absent=()):
    """An overlay with ``cells`` = ``{(v, t, p): xy}``."""
    labels = Labels.empty(n_views, n_frames, n_points)
    for (v, t, p), xy in cells.items():
        labels.set_gt(v, t, p, xy)
    for t in reviewed:
        labels.set_reviewed(t, True)
    for p in absent:
        labels.set_absent([p], True)
    labels.dirty = False
    return labels


def _merge(dest, source, **kwargs):
    opts = dict(
        point_names_dest=POINTS,
        point_names_source=POINTS,
        camera_names_dest=CAMS,
        camera_names_source=CAMS,
    )
    opts.update(kwargs)
    return merge_labels(dest, source, **opts)


# -- mapping by name -----------------------------------------------------------


def test_reordering_is_detected_and_remapped_by_name():
    """The nightmare case: same names, different order. An index copy would transpose."""
    mapping = map_by_name(["a", "b", "c"], ["c", "b", "a"])
    assert mapping.reordered
    assert mapping.source_to_dest == {0: 2, 1: 1, 2: 0}
    assert mapping.complete


def test_names_only_on_one_side_are_reported():
    mapping = map_by_name(["a", "b", "x"], ["a", "b", "y"])
    assert mapping.only_source == ["x"]
    assert mapping.only_dest == ["y"]
    assert not mapping.complete


def test_remapping_moves_labels_to_the_right_columns():
    source = _labels({(0, 1, 0): (10.0, 20.0)})
    points = map_by_name(POINTS, list(reversed(POINTS)))  # head -> index 2
    cameras = map_by_name(CAMS, CAMS)
    out = remap_labels(
        source, points=points, cameras=cameras, n_views=2, n_frames=3, n_points=3
    )
    assert out.gt_authored[0, 1, 2]
    assert not out.gt_authored[0, 1, 0]
    np.testing.assert_allclose(out.gt[0, 1, 2], [10.0, 20.0])


def test_a_source_only_point_is_dropped_and_counted():
    source = _labels({(0, 0, 2): (1.0, 2.0)})
    report = _merge(
        _labels({}),
        source,
        point_names_source=["head", "thorax", "extra"],
    )
    assert report.points.only_source == ["extra"]
    assert report.dropped_cells == 1
    assert any("dropped" in n for n in report.notes)


# -- the fatal case ------------------------------------------------------------


def test_mismatched_image_sizes_are_fatal():
    """GT is stored in footage pixels, so merging across resolutions reinterprets it."""
    report = _merge(
        _labels({}),
        _labels({(0, 0, 0): (1.0, 2.0)}),
        image_sizes_dest={"left": (512, 960)},
        image_sizes_source={"left": (1008, 1600)},
    )
    assert not report.ok
    assert "footage pixels" in report.fatal[0]
    assert report.taken_from_source == 0  # nothing was touched


# -- cell resolution -----------------------------------------------------------


def test_a_cell_only_the_source_has_is_taken():
    dest = _labels({})
    report = _merge(dest, _labels({(1, 2, 0): (5.0, 6.0)}))
    assert report.taken_from_source == 1
    assert dest.gt_authored[1, 2, 0]
    np.testing.assert_allclose(dest.gt[1, 2, 0], [5.0, 6.0])


def test_an_identical_cell_is_not_a_conflict():
    cells = {(0, 0, 0): (3.0, 4.0)}
    report = _merge(_labels(cells), _labels(cells))
    assert report.identical == 1
    assert report.conflicts == []


def test_two_labels_that_disagree_go_to_review():
    """There is one kind of GT, so neither side outranks the other: the default asks.

    Nothing in the data can settle this -- it is two operators disagreeing about where a
    keypoint is -- which is why the merge has an explicit policy instead of a rule.
    """
    dest = _labels({(0, 0, 0): (1.0, 1.0)})
    report = _merge(dest, _labels({(0, 0, 0): (20.0, 1.0)}))
    assert len(report.unresolved) == 1
    decision = report.unresolved[0]
    assert "19.0 px apart" in decision.reason
    np.testing.assert_allclose(dest.gt[0, 0, 0], [1.0, 1.0])  # untouched


@pytest.mark.parametrize(
    "policy,expected", [("ours", [1.0, 1.0]), ("theirs", [20.0, 1.0])]
)
def test_an_explicit_policy_settles_a_conflict(policy, expected):
    dest = _labels({(0, 0, 0): (1.0, 1.0)})
    report = _merge(
        dest,
        _labels({(0, 0, 0): (20.0, 1.0)}),
        on_conflict=policy,
    )
    assert report.unresolved == []
    np.testing.assert_allclose(dest.gt[0, 0, 0], expected)


def test_newest_uses_the_declared_direction():
    dest = _labels({(0, 0, 0): (1.0, 1.0)})
    _merge(
        dest,
        _labels({(0, 0, 0): (20.0, 1.0)}),
        on_conflict="newest",
        source_is_newer=False,
    )
    np.testing.assert_allclose(dest.gt[0, 0, 0], [1.0, 1.0])


def test_an_unknown_policy_is_refused():
    with pytest.raises(ValueError, match="on_conflict must be"):
        _merge(_labels({}), _labels({}), on_conflict="vibes")


# -- the other authored state ---------------------------------------------------


def test_an_occlusion_and_a_hand_placed_pixel_now_coexist():
    """v8 made the two facts ORTHOGONAL, so an occlusion no longer has to yield to a pixel.

    A joint can be hand-placed *through* an occluder from the geometry of the other views,
    and recording both is exactly right. The old rule -- "an occlusion never displaces a
    pixel" -- silently discarded a source occlusion on every cell the destination had a
    pixel for, losing a training signal nothing else can supply. Nothing is displaced here
    either: ``gt`` and ``occluded`` are separate arrays.
    """
    dest = _labels({(0, 0, 0): (1.0, 1.0)})
    source = Labels.empty(2, 3, 3)
    source.set_occluded(0, 0, 0, True)
    source.set_occluded(1, 0, 0, True)
    report = _merge(dest, source)
    assert report.occluded_taken == 2  # both, including the one over a GT pixel
    assert dest.gt_authored[0, 0, 0]  # the pixel is untouched
    assert tuple(dest.gt[0, 0, 0]) == (1.0, 1.0)
    assert dest.occluded[0, 0, 0] and dest.occluded[1, 0, 0]


def test_an_occlusion_the_destination_already_has_is_not_double_counted():
    dest = _labels({})
    dest.set_occluded(0, 0, 0, True)
    source = Labels.empty(2, 3, 3)
    source.set_occluded(0, 0, 0, True)
    assert _merge(dest, source).occluded_taken == 0


def test_seeds_are_carried_across_by_name():
    """A seed is authored state -- the array persisted so a re-run cannot re-solve it."""
    dest = _labels({})
    source = Labels.empty(2, 3, 3)
    source.seeds[0, 1, 2] = (7.0, 8.0)
    report = _merge(dest, source)
    assert report.seeds_taken == 1
    np.testing.assert_allclose(dest.seeds[0, 1, 2], [7.0, 8.0])


def test_a_seed_is_never_overwritten_because_reseeding_is_a_deliberate_gesture():
    dest = _labels({})
    dest.seeds[0, 1, 2] = (1.0, 2.0)
    source = Labels.empty(2, 3, 3)
    source.seeds[0, 1, 2] = (9.0, 9.0)
    report = _merge(dest, source)
    assert report.seeds_taken == 0
    np.testing.assert_allclose(dest.seeds[0, 1, 2], [1.0, 2.0])


def test_seeds_are_remapped_by_name_not_by_index():
    """The same hazard as GT: a reordered source must not transpose its seeds."""
    source = Labels.empty(2, 3, 3)
    source.seeds[0, 0, 0] = (5.0, 5.0)  # 'abdomen' in the reversed order
    dest = _labels({})
    _merge(dest, source, point_names_source=list(reversed(POINTS)))
    np.testing.assert_allclose(dest.seeds[0, 0, 2], [5.0, 5.0])  # 'abdomen' is index 2
    assert not np.isfinite(dest.seeds[0, 0, 0]).all()


def test_an_imported_gt_frame_gains_an_annotation_skeleton():
    """Otherwise the editor treats the frame as un-annotated and drops to the pre-v8 layer.

    The source here carries NO instance flag of its own -- which is exactly what a pre-v8
    label set looks like -- so a plain union would leave the frame flagless.
    """
    dest = _labels({})
    source = Labels.empty(2, 3, 3)
    source.set_gt(0, 1, 0, (3.0, 4.0))
    assert not source.instance.any()
    report = _merge(dest, source)
    assert report.taken_from_source == 1
    assert report.instances_added == 1
    assert dest.instance[1]
    assert not dest.instance[0] and not dest.instance[2]


def test_a_frame_that_only_gained_a_seed_also_gains_the_flag():
    dest = _labels({})
    source = Labels.empty(2, 3, 3)
    source.seeds[0, 2, 1] = (1.0, 1.0)
    report = _merge(dest, source)
    assert report.instances_added == 1
    assert dest.instance[2]


def test_the_sources_own_instance_flags_are_unioned_in():
    dest = _labels({})
    source = Labels.empty(2, 3, 3)
    source.instance[0] = True  # a created instance with nothing authored on it yet
    report = _merge(dest, source)
    assert report.instances_added == 1
    assert dest.instance[0]


def test_a_dry_run_reports_seeds_and_instances_without_writing_them():
    dest = _labels({})
    source = Labels.empty(2, 3, 3)
    source.seeds[0, 1, 2] = (7.0, 8.0)
    source.set_gt(0, 1, 0, (3.0, 4.0))
    report = _merge(dest, source, apply=False)
    assert report.seeds_taken == 1 and report.instances_added == 1
    assert not np.isfinite(dest.seeds).any()
    assert not dest.instance.any()
    assert not dest.gt_authored.any()


def test_absence_is_unioned_because_declaring_it_destroys_nothing():
    dest = _labels({}, absent=(0,))
    report = _merge(dest, _labels({}, absent=(1,)))
    assert report.absent_union > 0
    assert dest.absent_all_frames()[0] and dest.absent_all_frames()[1]


def test_reviewed_flags_are_ored():
    dest = _labels({}, reviewed=(0,))
    report = _merge(dest, _labels({}, reviewed=(2,)))
    assert report.reviewed_added == 1
    assert dest.reviewed[0] and dest.reviewed[2]


# -- dry run -------------------------------------------------------------------


def test_a_dry_run_changes_nothing_but_reports_everything():
    dest = _labels({(0, 0, 0): (1.0, 1.0)})
    before = dest.gt.copy()
    report = _merge(
        dest,
        _labels(
            {
                (0, 0, 0): (9.0, 9.0),
                (1, 1, 1): (4.0, 4.0),
            },
            reviewed=(1,),
        ),
        apply=False,
    )
    np.testing.assert_array_equal(dest.gt, before)
    assert not dest.dirty
    # ...but the report is complete enough to decide on.
    # Only the cell the destination did not have: the disagreeing one is left for review,
    # because no rule in the data can pick a winner between two authored pixels.
    assert report.taken_from_source == 1
    assert report.reviewed_added == 1
    assert len(report.conflicts) == 1 and len(report.unresolved) == 1


def test_the_summary_is_serializable(tmp_path):
    """It gets written beside the project, so it must be plain data."""
    import json

    report = _merge(
        _labels({(0, 0, 0): (1.0, 1.0)}),
        _labels({(0, 0, 0): (2.0, 2.0)}),
    )
    json.dumps(report.summary())
    assert report.summary()["unresolved"] == 1


# -- the CLI -------------------------------------------------------------------


def _project_with(tmp_path, dest_cells, source_cells, *, dest_labels=True):
    """A project whose recording has ``dest_cells``, plus a separate source labels.h5."""
    from helpers import CAMERA_NAMES

    from deeperfly.config import Config
    from deeperfly.gui.labels import labels_identity, save_labels
    from deeperfly.project import Project
    from deeperfly.results import StageStore
    from deeperfly.skeleton import Skeleton

    n_views, n_frames = len(CAMERA_NAMES), 4
    sizes = {n: (64, 80) for n in CAMERA_NAMES}
    fly = Skeleton.fly()
    cameras = Config.default().camera_group(image_sizes=sizes)

    project = Project.create(tmp_path / "proj", skeleton="fly38")
    rec = tmp_path / "flyA"
    (rec / "deeperfly_outputs").mkdir(parents=True)
    for name in CAMERA_NAMES:
        (rec / f"{name}.mp4").write_bytes(b"\0" * (900 + len(name)))
    StageStore(rec / "deeperfly_outputs" / "results.h5").write_pose2d(
        cameras=cameras,
        skeleton=fly,
        pts2d=np.full((n_views, n_frames, 38, 2), np.nan),
        conf=None,
        image_sizes=sizes,
    )
    entry = project.add_recording(rec)

    identity = labels_identity(
        point_names=list(fly.point_names),
        camera_names=list(CAMERA_NAMES),
        n_frames=n_frames,
        image_sizes=sizes,
    )
    if dest_labels:
        save_labels(
            project.labels_path(entry),
            _labels(dest_cells, n_views=n_views, n_frames=n_frames, n_points=38),
            identity=identity,
        )
    source_path = tmp_path / "other_labels.h5"
    save_labels(
        source_path,
        _labels(source_cells, n_views=n_views, n_frames=n_frames, n_points=38),
        identity=identity,
    )
    return project, entry, source_path


def test_cli_dry_run_reports_without_writing(tmp_path, capsys):
    from deeperfly import cli
    from deeperfly.project import label_stats

    project, entry, source = _project_with(tmp_path, {}, {(0, 1, 5): (3.0, 4.0)})
    cli.main(
        [
            "labels-merge",
            entry.slug,
            str(source),
            str(project.root),
            "--log-level",
            "error",
        ]
    )
    out = capsys.readouterr().out
    assert "would merge" in out
    assert "dry run" in out
    assert label_stats(project.labels_path(entry))["gt_points"] == 0


def test_cli_apply_merges_and_snapshots(tmp_path, capsys):
    from deeperfly import cli
    from deeperfly.project import Project, label_stats

    project, entry, source = _project_with(
        tmp_path,
        {(0, 0, 1): (1.0, 1.0)},
        {(0, 1, 5): (3.0, 4.0)},
    )
    cli.main(
        [
            "labels-merge",
            entry.slug,
            str(source),
            str(project.root),
            "--apply",
            "--log-level",
            "error",
        ]
    )
    assert label_stats(project.labels_path(entry))["gt_points"] == 2
    # The pre-merge state is recoverable, which is what makes --apply safe to try.
    snapshots = list(project.outputs_dir(entry).glob("labels.premerge-*.h5"))
    assert len(snapshots) == 1
    assert label_stats(snapshots[0])["gt_points"] == 1
    # A merge is a project-state change, so the iteration counter moves.
    assert Project.load(project.root).iteration == 1


def test_cli_merges_into_a_recording_with_no_labels_yet(tmp_path, capsys):
    """The common stranded case: the indexed copy was never labeled."""
    from deeperfly import cli
    from deeperfly.project import label_stats

    project, entry, source = _project_with(
        tmp_path,
        {},
        {(0, 1, 5): (3.0, 4.0)},
        dest_labels=False,
    )
    cli.main(
        [
            "labels-merge",
            entry.slug,
            str(source),
            str(project.root),
            "--apply",
            "--log-level",
            "error",
        ]
    )
    out = capsys.readouterr().out
    assert "no labels yet" in out
    assert "reconciled by name" in out  # not a blind copy
    assert label_stats(project.labels_path(entry))["gt_points"] == 1


def test_cli_writes_a_conflict_queue(tmp_path, capsys):
    from deeperfly import cli

    project, entry, source = _project_with(
        tmp_path,
        {(0, 0, 1): (1.0, 1.0)},
        {(0, 0, 1): (30.0, 1.0)},
    )
    cli.main(
        [
            "labels-merge",
            entry.slug,
            str(source),
            str(project.root),
            "--apply",
            "--log-level",
            "error",
        ]
    )
    queue = project.root / "exports" / f"merge_conflicts_{entry.slug}.json"
    assert queue.exists()
    import json

    data = json.loads(queue.read_text())
    assert len(data["conflicts"]) == 1
    assert data["conflicts"][0]["theirs_xy"] == [30.0, 1.0]


def test_cli_refuses_to_merge_a_file_into_itself(tmp_path):
    from deeperfly import cli

    project, entry, _ = _project_with(tmp_path, {(0, 0, 1): (1.0, 1.0)}, {})
    with pytest.raises(SystemExit, match="same file"):
        cli.main(
            [
                "labels-merge",
                entry.slug,
                str(project.labels_path(entry)),
                str(project.root),
                "--log-level",
                "error",
            ]
        )


def test_cli_refuses_a_missing_source(tmp_path):
    from deeperfly import cli

    project, entry, _ = _project_with(tmp_path, {}, {})
    with pytest.raises(SystemExit, match="no labels.h5"):
        cli.main(
            [
                "labels-merge",
                entry.slug,
                str(tmp_path / "nope.h5"),
                str(project.root),
                "--log-level",
                "error",
            ]
        )
