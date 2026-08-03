"""Tests for :mod:`deeperfly.project` -- the project layer and its CLI.

The project's whole promise is that it *indexes* rather than re-homes: adopting a
recording must not move, copy or rewrite a single label. Most of what is tested here is
therefore about identity and non-destructiveness --

- a recording is identified by **content**, so the same footage adopted from two
  different paths is one recording and re-adopting is a no-op;
- adopted outputs are **symlinks**, so the file the editor writes is the file a training
  set reads;
- dropping a recording, or a listing hitting a corrupt sidecar, must not lose data.

``caplog.at_level`` is always given ``logger="deeperfly"`` here. The CLI tests in this
file call ``_configure_logging``, which does ``log.setLevel(...)`` on the ``deeperfly``
logger -- so a later bare ``at_level("WARNING")``, which only raises the *root* logger,
captures nothing and the assertion fails for a reason that has nothing to do with the
code under test.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
from helpers import CAMERA_NAMES, HEIGHT, WIDTH

from deeperfly import cli
from deeperfly.gui.labels import Labels, labels_identity, save_labels
from deeperfly.project import (
    PROJECT_FILENAME,
    Project,
    label_stats,
    recording_fingerprint,
    recording_id,
)
from deeperfly.results import StageStore
from deeperfly.skeleton import Skeleton

SIZES = {name: (HEIGHT, WIDTH) for name in CAMERA_NAMES}


# -- fixtures ------------------------------------------------------------------


def _make_recording(root, *, cameras=CAMERA_NAMES, sizes=None, seed=0):
    """A recording directory of per-camera video files (bytes, not real video).

    Byte sizes are what the footage fingerprint reads, so they are made distinct per
    camera and controllable per recording -- that is the whole variable under test.
    """
    root.mkdir(parents=True, exist_ok=True)
    sizes = sizes or {c: 1000 + 7 * i + seed for i, c in enumerate(cameras)}
    for camera in cameras:
        (root / f"camera_{camera}.mp4").write_bytes(b"\0" * sizes[camera])
    return root


def _make_outputs(rec_root, cameras, fly, *, n_frames=6, gt_cells=0, reviewed=0):
    """A ``deeperfly_outputs/`` with a real results.h5 and (optionally) a labels.h5."""
    outputs = rec_root / "deeperfly_outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pts2d = rng.uniform(0, 100, size=(len(CAMERA_NAMES), n_frames, 38, 2))
    store = StageStore(outputs / "results.h5")
    store.write_pose2d(
        cameras=cameras,
        skeleton=fly,
        pts2d=pts2d,
        conf=np.ones(pts2d.shape[:3]),
        image_sizes=SIZES,
        footage={c: [rec_root / f"camera_{c}.mp4"] for c in CAMERA_NAMES},
    )
    if gt_cells or reviewed:
        labels = Labels.empty(len(CAMERA_NAMES), n_frames, 38)
        for i in range(gt_cells):
            labels.set_gt(
                i % len(CAMERA_NAMES), i % n_frames, i % 38, (1.0 * i, 2.0 * i)
            )
        for t in range(reviewed):
            labels.set_reviewed(t, True)
        save_labels(
            outputs / "labels.h5",
            labels,
            identity=labels_identity(
                point_names=list(fly.point_names),
                camera_names=list(CAMERA_NAMES),
                n_frames=n_frames,
            ),
        )
    return outputs


@pytest.fixture
def project(tmp_path) -> Project:
    return Project.create(tmp_path / "proj", name="testproj")


# -- creation ------------------------------------------------------------------


def test_the_fly38_preset_round_trips_to_the_packaged_skeleton(project):
    """The preset is lifted from the packaged config, so it must parse to the same thing.

    If it drifts, every project seeded from it silently tracks a different skeleton than
    the pipeline's default -- and the labels would still load, because both are 38 points.
    """
    got, want = project.skeleton(), Skeleton.fly()
    assert got.point_names == want.point_names
    assert got.limb_names == want.limb_names
    assert got.palette == want.palette
    np.testing.assert_array_equal(got.bones, want.bones)
    np.testing.assert_array_equal(got.limb_id, want.limb_id)


def test_the_fly38_preset_keeps_its_comments(project):
    """The prose explaining the limb ordering and palette travels with the project."""
    text = project.skeleton_path().read_text()
    assert "kinematic-chain order" in text
    assert "#" in text


def test_a_blank_skeleton_is_valid_and_empty(tmp_path):
    project = Project.create(tmp_path / "blank", skeleton="blank")
    assert project.skeleton().n_points == 0


def test_a_skeleton_path_keeps_only_the_skeleton_section(tmp_path):
    """Seeding from a whole run config must not drag its rig into skeleton.toml.

    Two definitions of the rig in one project, with nothing saying which wins, is how a
    project ends up projecting with the wrong cameras.
    """
    from deeperfly.config import DEFAULT_CONFIG_PATH

    project = Project.create(tmp_path / "p", skeleton=str(DEFAULT_CONFIG_PATH))
    text = project.skeleton_path().read_text()
    assert "[skeleton]" in text
    assert "[cameras" not in text
    assert "[pose2d" not in text
    assert project.skeleton().n_points == 38


def test_creating_over_an_existing_project_is_refused(tmp_path):
    Project.create(tmp_path / "p")
    with pytest.raises(FileExistsError, match="already exists"):
        Project.create(tmp_path / "p")


def test_an_unknown_skeleton_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="unknown skeleton"):
        Project.create(tmp_path / "p", skeleton="octopus")


# -- manifest round trip -------------------------------------------------------


def test_the_manifest_round_trips(tmp_path, project):
    _make_recording(tmp_path / "flyA")
    entry = project.add_recording(tmp_path / "flyA", subject="specimen-1")
    reloaded = Project.load(project.root)

    assert reloaded.name == "testproj"
    assert reloaded.id == project.id
    assert [e.id for e in reloaded.recordings] == [entry.id]
    got = reloaded.recordings[0]
    assert (got.slug, got.subject, got.id_basis) == ("flyA", "specimen-1", "footage")
    assert got.origin["kind"] == "adopted"
    assert got.origin["from"] == str((tmp_path / "flyA").resolve())


def test_a_newer_format_version_is_refused(project):
    manifest = project.root / PROJECT_FILENAME
    manifest.write_text(
        manifest.read_text().replace("format_version = 1", "format_version = 2")
    )
    with pytest.raises(ValueError, match="newer deeperfly"):
        Project.load(project.root)


def test_a_malformed_recording_row_is_skipped_not_fatal(project, caplog):
    manifest = project.root / PROJECT_FILENAME
    manifest.write_text(manifest.read_text() + '\n[[recordings]]\nslug = "no-id"\n')
    with caplog.at_level("WARNING", logger="deeperfly"):
        assert Project.load(project.root).recordings == []
    assert "malformed" in caplog.text


def test_find_walks_upward(project, monkeypatch):
    nested = project.root / "recordings" / "deep" / "deeper"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert Project.find(".") == project.root.resolve()


def test_find_returns_none_outside_a_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Project.find(".") is None


# -- identity ------------------------------------------------------------------


def test_the_same_footage_at_a_different_path_is_the_same_recording(tmp_path):
    """The point of a content id: a moved or copied recording is not a new one.

    ``~/fly-pose-data`` already holds 20+ backup copies of the same recordings, and a
    merge that treated each as distinct would double-count every label in them.
    """
    a = _make_recording(tmp_path / "here" / "flyA")
    b = _make_recording(tmp_path / "elsewhere" / "renamed")
    from deeperfly.project import discover_footage

    id_a = recording_id(recording_fingerprint(discover_footage(a))[0])
    id_b = recording_id(recording_fingerprint(discover_footage(b))[0])
    assert id_a == id_b


def test_different_footage_bytes_are_different_recordings(tmp_path):
    """Ground truth is in footage pixels, so re-encoded footage is a different recording."""
    from deeperfly.project import discover_footage

    a = _make_recording(tmp_path / "a", seed=0)
    b = _make_recording(tmp_path / "b", seed=1)
    assert (
        recording_fingerprint(discover_footage(a))[0]
        != recording_fingerprint(discover_footage(b))[0]
    )


def test_re_adopting_a_recording_is_a_no_op(tmp_path, project, caplog):
    _make_recording(tmp_path / "flyA")
    first = project.add_recording(tmp_path / "flyA")
    with caplog.at_level("INFO", logger="deeperfly"):
        again = project.add_recording(tmp_path / "flyA")
    assert again.id == first.id
    assert len(project.recordings) == 1
    assert "already in this project" in caplog.text


def test_the_result_basis_identifies_a_recording_whose_footage_is_gone(
    cameras, fly, tmp_path
):
    """An archived recording must still be adoptable -- its labels are why."""
    rec = _make_recording(tmp_path / "flyA")
    _make_outputs(rec, cameras, fly, n_frames=5)
    for video in rec.glob("*.mp4"):
        video.unlink()

    project = Project.create(tmp_path / "proj")
    entry = project.add_recording(rec)
    assert entry.id_basis == "result"
    assert entry.n_frames == 5


def test_a_recording_with_neither_footage_nor_a_result_cannot_be_identified(
    tmp_path, project
):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="cannot identify"):
        project.add_recording(empty)


def test_an_explicit_id_is_recorded_as_manual(tmp_path, project):
    _make_recording(tmp_path / "flyA")
    entry = project.add_recording(tmp_path / "flyA", rec_id="rec_deadbeef")
    assert (entry.id, entry.id_basis) == ("rec_deadbeef", "manual")


# -- adoption ------------------------------------------------------------------


def test_adopting_links_the_outputs_rather_than_copying_them(
    cameras, fly, tmp_path, project
):
    """The file the editor writes must be the file a training set reads."""
    rec = _make_recording(tmp_path / "flyA")
    outputs = _make_outputs(rec, cameras, fly, gt_cells=3)
    entry = project.add_recording(rec)

    linked = project.outputs_dir(entry)
    assert linked.is_symlink()
    assert linked.resolve() == outputs.resolve()
    # A label authored "in the original" is visible through the project, and vice versa.
    assert project.labels_path(entry).resolve() == (outputs / "labels.h5").resolve()
    assert project.status()[0]["gt_points"] == 3


def test_copy_makes_a_real_directory(cameras, fly, tmp_path, project):
    rec = _make_recording(tmp_path / "flyA")
    _make_outputs(rec, cameras, fly, gt_cells=2)
    entry = project.add_recording(rec, link=False)

    outputs = project.outputs_dir(entry)
    assert outputs.is_dir() and not outputs.is_symlink()
    assert (outputs / "results.h5").exists()


def test_a_recording_with_no_outputs_is_still_adoptable(tmp_path, project):
    """Seven videos and nothing else -- the from-scratch starting point."""
    rec = _make_recording(tmp_path / "fresh")
    entry = project.add_recording(rec)

    assert entry.n_frames is None
    row = project.status()[0]
    assert row["outputs_missing"] and not row["has_results"]


def test_adoption_reads_the_frame_count_and_subject_from_the_result(
    cameras, fly, tmp_path, project
):
    rec = _make_recording(tmp_path / "flyA")
    _make_outputs(rec, cameras, fly, n_frames=9)
    StageStore(rec / "deeperfly_outputs" / "results.h5").write_animal(
        absent=np.zeros(38, dtype=bool), subject_id="specimen-7"
    )
    entry = project.add_recording(rec)
    assert entry.n_frames == 9
    assert entry.subject == "specimen-7"


@pytest.mark.parametrize("kind", ["outputs_dir", "results_file"])
def test_a_results_path_or_outputs_dir_is_accepted_as_the_source(
    cameras, fly, tmp_path, project, kind
):
    """Users have a path to whichever of the three; all must work."""
    rec = _make_recording(tmp_path / "flyA")
    outputs = _make_outputs(rec, cameras, fly)
    source = outputs if kind == "outputs_dir" else outputs / "results.h5"
    entry = project.add_recording(source)
    assert project.outputs_dir(entry).resolve() == outputs.resolve()


def test_a_slug_collision_is_disambiguated_by_content_id(tmp_path, project):
    """Two different recordings both named `fly1` (a real pattern) must coexist."""
    a = _make_recording(tmp_path / "day1" / "fly1", seed=0)
    b = _make_recording(tmp_path / "day2" / "fly1", seed=99)
    first = project.add_recording(a)
    second = project.add_recording(b)

    assert first.slug == "fly1"
    assert second.slug.startswith("fly1-") and second.slug != first.slug
    assert len({e.slug for e in project.recordings}) == 2


def test_footage_discovery_names_a_camera_after_its_file(tmp_path):
    from deeperfly.project import discover_footage

    rec = _make_recording(tmp_path / "flyA", cameras=["RH", "F"])
    assert sorted(discover_footage(rec)) == ["camera_F", "camera_RH"]


# -- lookup and removal --------------------------------------------------------


def test_a_recording_resolves_by_slug_id_or_prefix(tmp_path, project):
    _make_recording(tmp_path / "flyA")
    entry = project.add_recording(tmp_path / "flyA")
    assert project.recording("flyA").id == entry.id
    assert project.recording(entry.id).id == entry.id
    assert project.recording(entry.id[:12]).id == entry.id


def test_an_unknown_recording_is_a_clear_error(project):
    with pytest.raises(KeyError, match="no recording"):
        project.recording("nope")


def test_an_ambiguous_prefix_is_refused(tmp_path, project):
    _make_recording(tmp_path / "a")
    _make_recording(tmp_path / "b", seed=5)
    project.add_recording(tmp_path / "a")
    project.add_recording(tmp_path / "b")
    with pytest.raises(KeyError, match="ambiguous"):
        project.recording("rec_")


def test_removing_a_recording_leaves_the_originals_alone(
    cameras, fly, tmp_path, project
):
    """A symlinked project must not be able to delete the originals."""
    rec = _make_recording(tmp_path / "flyA")
    outputs = _make_outputs(rec, cameras, fly, gt_cells=4)
    entry = project.add_recording(rec)

    project.remove_recording(entry.slug, delete=True)
    assert project.recordings == []
    assert (outputs / "results.h5").exists()
    assert (outputs / "labels.h5").exists()
    assert not project.recording_dir(entry).exists()


# -- label statistics ----------------------------------------------------------


def test_label_stats_counts_the_live_rows(cameras, fly, tmp_path):
    rec = _make_recording(tmp_path / "flyA")
    outputs = _make_outputs(rec, cameras, fly, n_frames=6, gt_cells=5, reviewed=2)
    stats = label_stats(outputs / "labels.h5")
    assert stats["gt_points"] == 5
    assert stats["reviewed_frames"] == 2
    assert stats["labeled_frames"] == len({i % 6 for i in range(5)})
    assert stats["format_version"] is not None


def test_label_stats_excludes_labels_an_absence_declaration_quarantines(
    cameras, fly, tmp_path
):
    """An absent keypoint is not ground truth, so it must not inflate the progress count.

    The quarantined rows are still in the file (under ``absent/void_gt``) -- the point is
    that a status listing reports what an export would actually yield.
    """
    n_frames = 4
    labels = Labels.empty(len(CAMERA_NAMES), n_frames, 38)
    labels.set_gt(0, 0, 0, (1.0, 2.0))
    labels.set_gt(0, 0, 1, (3.0, 4.0))
    labels.set_absent([0], True)  # whole recording

    path = tmp_path / "labels.h5"
    save_labels(
        path,
        labels,
        identity=labels_identity(
            point_names=list(fly.point_names),
            camera_names=list(CAMERA_NAMES),
            n_frames=n_frames,
        ),
    )
    stats = label_stats(path)
    assert stats["gt_points"] == 1  # point 1 only; point 0 is quarantined
    assert stats["absent_points"] == 1


def test_label_stats_of_a_missing_file_is_zero(tmp_path):
    assert label_stats(tmp_path / "nope.h5")["gt_points"] == 0


def test_a_corrupt_sidecar_is_reported_not_fatal(tmp_path, caplog):
    """A listing must survive one bad file: that is when it is needed most."""
    bad = tmp_path / "labels.h5"
    bad.write_bytes(b"not an hdf5 file")
    with caplog.at_level("WARNING", logger="deeperfly"):
        assert label_stats(bad)["gt_points"] == 0
    assert "could not read" in caplog.text


def test_status_and_totals_aggregate_across_recordings(cameras, fly, tmp_path, project):
    for i, name in enumerate(("flyA", "flyB")):
        rec = _make_recording(tmp_path / name, seed=i * 13)
        _make_outputs(rec, cameras, fly, gt_cells=3 + i, reviewed=1)
        project.add_recording(rec)

    totals = project.totals()
    assert totals["recordings"] == 2
    assert totals["gt_points"] == 7  # 3 + 4
    assert totals["reviewed_frames"] == 2
    assert totals["with_labels"] == 2


def test_status_reports_a_missing_share_rather_than_raising(
    cameras, fly, tmp_path, project
):
    """An unmounted share is a normal Monday, not an error."""
    rec = _make_recording(tmp_path / "flyA")
    outputs = _make_outputs(rec, cameras, fly, gt_cells=1)
    entry = project.add_recording(rec)
    # Break the link the way an unmounted share does: the target disappears.
    import shutil

    shutil.rmtree(outputs)

    row = project.status()[0]
    assert row["outputs_missing"] and not row["has_results"]
    assert row["gt_points"] == 0
    assert project.recording(entry.slug).slug == entry.slug


# -- the CLI -------------------------------------------------------------------


def test_cli_new_add_ls_status_rm(cameras, fly, tmp_path, capsys):
    root = tmp_path / "proj"
    cli.main(
        ["project", "new", str(root), "--name", "cli-proj", "--log-level", "error"]
    )
    assert (root / PROJECT_FILENAME).exists()

    rec = _make_recording(tmp_path / "flyA")
    _make_outputs(rec, cameras, fly, gt_cells=6, reviewed=2)
    cli.main(["project", "add", str(root), str(rec), "--log-level", "error"])

    capsys.readouterr()
    cli.main(["project", "ls", str(root), "--log-level", "error"])
    assert "flyA" in capsys.readouterr().out

    cli.main(["project", "status", str(root), "--log-level", "error"])
    out = capsys.readouterr().out
    assert "cli-proj" in out
    assert "uncalibrated" in out  # a fresh project has no rig, and says so

    cli.main(["project", "rm", "flyA", str(root), "--log-level", "error"])
    assert Project.load(root).recordings == []
    # The originals survive dropping the entry.
    assert (rec / "deeperfly_outputs" / "labels.h5").exists()


def test_cli_add_skips_a_bad_source_and_keeps_going(tmp_path, capsys, caplog):
    root = tmp_path / "proj"
    cli.main(["project", "new", str(root), "--log-level", "error"])
    good = _make_recording(tmp_path / "flyA")
    empty = tmp_path / "empty"
    empty.mkdir()

    with caplog.at_level("WARNING", logger="deeperfly"):
        cli.main(
            [
                "project",
                "add",
                str(root),
                str(empty),
                str(good),
                "--log-level",
                "warning",
            ]
        )
    assert "skipping" in caplog.text
    assert [e.slug for e in Project.load(root).recordings] == ["flyA"]


def test_cli_status_without_a_path_uses_the_enclosing_project(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "proj"
    cli.main(["project", "new", str(root), "--name", "walkup", "--log-level", "error"])
    monkeypatch.chdir(root / "recordings")
    capsys.readouterr()
    cli.main(["project", "status", "--log-level", "error"])
    assert "walkup" in capsys.readouterr().out


def test_cli_status_outside_a_project_says_how_to_make_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="project new"):
        cli.main(["project", "status", "--log-level", "error"])


def test_cli_rm_refuses_to_delete_the_only_copy_of_the_labels(
    cameras, fly, tmp_path, capsys
):
    """--delete on a COPIED recording would destroy labels that exist nowhere else."""
    root = tmp_path / "proj"
    cli.main(["project", "new", str(root), "--log-level", "error"])
    rec = _make_recording(tmp_path / "flyA")
    _make_outputs(rec, cameras, fly, gt_cells=2)
    cli.main(["project", "add", str(root), str(rec), "--copy", "--log-level", "error"])

    with pytest.raises(SystemExit):
        cli.main(
            ["project", "rm", "flyA", str(root), "--delete", "--log-level", "error"]
        )
    assert "refusing" in capsys.readouterr().out
    assert Project.load(root).recordings  # still indexed


def test_cli_new_reports_a_blank_skeleton_as_empty(tmp_path, capsys):
    cli.main(
        [
            "project",
            "new",
            str(tmp_path / "p"),
            "--skeleton",
            "blank",
            "--log-level",
            "error",
        ]
    )
    assert "empty" in capsys.readouterr().out


# -- the h5 shape label_stats depends on ---------------------------------------


def test_label_stats_reads_the_schema_labels_actually_writes(cameras, fly, tmp_path):
    """Pin the group/dataset names, since label_stats bypasses load_labels.

    Reading HDF5 directly is what keeps a status listing from needing a result to
    validate against -- but it also means a schema change would silently zero every
    count instead of raising.
    """
    rec = _make_recording(tmp_path / "flyA")
    outputs = _make_outputs(rec, cameras, fly, gt_cells=2, reviewed=1)
    with h5py.File(outputs / "labels.h5", "r") as f:
        # v6: [view, frame, instance, point]. label_stats reads HDF5 directly, so a
        # width change would silently zero every count rather than raise.
        assert f["gt/index"].shape[1] == 4
        assert "occluded/index" in f
        assert "reviewed/index" in f
        assert "absent/index" in f
        assert json.loads(f.attrs["meta"])["deeperfly_labels_format_version"] >= 2


def test_extract_section_leaves_the_next_sections_banner_behind():
    """A comment block above the next header documents THAT section, not this one.

    Without this, a project's skeleton.toml ships with the packaged config's paragraph
    about camera rigs attached to the end of its skeleton.
    """
    from deeperfly._toml import extract_section

    text = "\n".join(
        [
            "[a]",
            "x = 1",
            "",
            "[a.sub]",
            "y = 2",
            "",
            "# ===========",
            "# All about b",
            "# ===========",
            "[b]",
            "z = 3",
        ]
    )
    got = extract_section(text, "a")
    assert got == "[a]\nx = 1\n\n[a.sub]\ny = 2\n"
    assert "about b" not in got


def test_extract_section_takes_the_last_section_to_the_end_of_file():
    from deeperfly._toml import extract_section

    assert extract_section("[a]\nx = 1\n\n[b]\ny = 2\n", "b") == "[b]\ny = 2\n"


def test_extract_section_ignores_a_header_named_inside_a_comment():
    """The packaged config discusses [cameras.defaults] before declaring it."""
    from deeperfly._toml import extract_section

    got = extract_section("# see [b] below\n[a]\nx = 1\n[b]\ny = 2\n", "a")
    assert got == "[a]\nx = 1\n"


def test_extract_section_of_a_missing_table_is_a_clear_error():
    from deeperfly._toml import extract_section

    with pytest.raises(ValueError, match=r"no \[nope\] table"):
        extract_section("[a]\nx = 1\n", "nope")


# -- provenance: labeled is not the same as trainable --------------------------


def test_label_stats_separates_trainable_points_from_stored_ones(
    cameras, fly, tmp_path
):
    """The status count must match what `labels-export` actually yields.

    ``export_gt`` drops ``confirmed_projection`` (a bulk-accepted triangulation guess)
    and ``placeholder_seed`` (a drag handle the editor invented at the image edge). A
    progress number that counted those would overstate the training set by however much
    geometry got bulk-confirmed into it.
    """
    from deeperfly.gui.labels import Provenance

    n_frames = 4
    labels = Labels.empty(len(CAMERA_NAMES), n_frames, 38)
    labels.set_gt(0, 0, 0, (1.0, 1.0), provenance=Provenance.DRAGGED)
    labels.set_gt(0, 0, 1, (2.0, 2.0), provenance=Provenance.CONFIRMED_PREDICTION)
    labels.set_gt(0, 0, 2, (3.0, 3.0), provenance=Provenance.CONFIRMED_PROJECTION)
    labels.set_gt(0, 0, 3, (4.0, 4.0), provenance=Provenance.PLACEHOLDER_SEED)

    path = tmp_path / "labels.h5"
    save_labels(
        path,
        labels,
        identity=labels_identity(
            point_names=list(fly.point_names),
            camera_names=list(CAMERA_NAMES),
            n_frames=n_frames,
        ),
    )
    stats = label_stats(path)
    assert stats["gt_points"] == 4
    assert stats["gt_trainable"] == 2  # dragged + confirmed_prediction
    assert stats["provenance"] == {
        "dragged": 1,
        "confirmed_prediction": 1,
        "confirmed_projection": 1,
        "placeholder_seed": 1,
    }


def test_gt_trainable_agrees_with_export_gt(cameras, fly, tmp_path):
    """Pin the agreement rather than trusting two copies of the same rule.

    ``label_stats`` counts provenance codes out of HDF5; ``export_gt`` masks a dense
    array. They must produce the same number, or the status table is quietly lying.
    """
    from deeperfly.gui.labels import Provenance, export_gt, load_labels

    n_frames = 5
    labels = Labels.empty(len(CAMERA_NAMES), n_frames, 38)
    for i, prov in enumerate(
        [
            Provenance.DRAGGED,
            Provenance.DRAGGED,
            Provenance.CONFIRMED_PREDICTION,
            Provenance.CONFIRMED_PROJECTION,
            Provenance.PLACEHOLDER_SEED,
            Provenance.PLACEHOLDER_SEED,
        ]
    ):
        labels.set_gt(i % 3, i % n_frames, i, (1.0 * i, 2.0 * i), provenance=prov)

    identity = labels_identity(
        point_names=list(fly.point_names),
        camera_names=list(CAMERA_NAMES),
        n_frames=n_frames,
    )
    path = tmp_path / "labels.h5"
    save_labels(path, labels, identity=identity)

    _, mask, _ = export_gt(load_labels(path, identity=identity))
    assert label_stats(path)["gt_trainable"] == int(mask.sum())


def test_status_names_the_untrainable_rows(cameras, fly, tmp_path, capsys):
    """A "dropped" column with no explanation would just be a mystery number."""
    from deeperfly.gui.labels import Provenance

    rec = _make_recording(tmp_path / "flyA")
    outputs = _make_outputs(rec, cameras, fly, n_frames=4)
    labels = Labels.empty(len(CAMERA_NAMES), 4, 38)
    labels.set_gt(0, 0, 0, (1.0, 1.0), provenance=Provenance.DRAGGED)
    labels.set_gt(0, 0, 1, (2.0, 2.0), provenance=Provenance.CONFIRMED_PROJECTION)
    save_labels(
        outputs / "labels.h5",
        labels,
        identity=labels_identity(
            point_names=list(fly.point_names),
            camera_names=list(CAMERA_NAMES),
            n_frames=4,
        ),
    )
    root = tmp_path / "proj2"
    cli.main(["project", "new", str(root), "--log-level", "error"])
    cli.main(["project", "add", str(root), str(rec), "--log-level", "error"])
    capsys.readouterr()
    cli.main(["project", "status", str(root), "--log-level", "error"])

    out = capsys.readouterr().out
    assert "not trainable" in out
    assert "confirmed_projection" in out


def test_re_adopting_warns_when_the_duplicate_carries_labels_the_index_cannot_see(
    cameras, fly, tmp_path, project, caplog
):
    """One recording is one entry -- but its second label set must not read as zero.

    ``~/fly-pose-data`` holds the same recordings under both ``recordings/`` and
    ``predicted/<rid>_label/``, and only one of the two usually has the hand labels.
    Adopting the unlabeled one first would silently report 0, which looks exactly like
    "the labels are gone".
    """
    # Same footage bytes in two places -> the same content id.
    bare = _make_recording(tmp_path / "plain" / "flyA")
    labeled = _make_recording(tmp_path / "labeled" / "flyA_label")
    _make_outputs(bare, cameras, fly, gt_cells=0)
    _make_outputs(labeled, cameras, fly, gt_cells=40)

    first = project.add_recording(bare)
    with caplog.at_level("WARNING", logger="deeperfly"):
        again = project.add_recording(labeled)

    assert again.id == first.id  # correctly deduplicated
    assert "will NOT count" in caplog.text
    assert "40 ground-truth point" in caplog.text


def test_re_adopting_a_duplicate_with_no_extra_labels_is_quiet(
    cameras, fly, tmp_path, project, caplog
):
    """The warning must not fire on the ordinary "adopt the same thing twice" case."""
    rec = _make_recording(tmp_path / "flyA")
    _make_outputs(rec, cameras, fly, gt_cells=3)
    project.add_recording(rec)
    with caplog.at_level("WARNING", logger="deeperfly"):
        project.add_recording(rec)
    assert "will NOT count" not in caplog.text
