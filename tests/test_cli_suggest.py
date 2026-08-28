"""Tests for ``deeperfly labels-suggest`` -- the CLI half of the acquisition feature.

Beyond the usual argument plumbing, two properties are asserted here because a real
recording directory holds irreplaceable work: the command never writes to
``results.h5`` or ``labels.h5`` (verified by md5 across the run), and the ranked
list it prints is complete enough to act on without the GUI.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from deeperfly import cli
from deeperfly.acquisition import file_md5, read_suggestions
from deeperfly.gui.labels import Labels, labels_identity, save_labels
from deeperfly.results import StageStore
from deeperfly.triangulation import reprojection_error

# -- a synthetic recording directory ------------------------------------------


@pytest.fixture
def outputs(tmp_path, cameras, fly, rng):
    """A ``deeperfly_outputs``-shaped directory with three deliberately hard frames."""
    pts3d = rng.uniform(-1.5, 1.5, size=(600, 38, 3))
    pts2d = np.array(cameras.project(pts3d))
    for t, shift in ((150, 400.0), (400, 300.0), (401, 300.0)):
        pts2d[:4, t, 3:10] += shift
    outdir = tmp_path / "rec" / "deeperfly_outputs"
    outdir.mkdir(parents=True)
    store = StageStore(outdir / "results.h5")
    store.write_pose2d(
        cameras=cameras,
        skeleton=fly,
        pts2d=pts2d,
        conf=np.full(pts2d.shape[:3], 0.9),
        image_sizes={name: (512, 1024) for name in cameras.names},
    )
    store.write_cameras("bundle_adjustment", cameras)
    store.write_points(
        "triangulation",
        pts2d=pts2d,
        pts3d=pts3d,
        reproj_error=reprojection_error(cameras, pts3d, pts2d),
    )
    return outdir


def _add_labels(outdir, cameras, fly, frames=(150,), n_frames=600):
    identity = labels_identity(
        point_names=list(fly.point_names),
        camera_names=list(cameras.names),
        n_frames=n_frames,
        image_sizes={name: (512, 1024) for name in cameras.names},
    )
    labels = Labels.empty(len(cameras.names), n_frames, len(fly.point_names))
    for t in frames:
        labels.set_gt(0, t, 3, (10.0, 20.0))
    save_labels(outdir / "labels.h5", labels, identity=identity)
    return outdir / "labels.h5"


# -- the happy paths ----------------------------------------------------------


def test_writes_the_sidecar_and_ranks_the_hard_frames(outputs, capsys):
    cli.main(["labels-suggest", str(outputs), "-n", "5", "--min-gap-s", "1.0"])
    doc = read_suggestions(outputs / "labels_suggest.json")
    assert doc is not None
    frames = [e["frame"] for e in doc["frames"]]
    assert frames[0] == 150  # the worst frame ranks first
    # 400/401 are the same hard moment: exactly one of them may be offered.
    assert len({400, 401} & set(frames)) == 1
    out = capsys.readouterr().out
    assert "pose2d/points" in out
    assert "most-wrong" in out
    assert "labels_suggest.json" in out


def test_works_with_no_labels_yet(outputs, capsys):
    """A first round: no labels.h5 beside results.h5 is normal, not an error."""
    assert not (outputs / "labels.h5").exists()
    cli.main(["labels-suggest", str(outputs), "-n", "3"])
    doc = read_suggestions(outputs / "labels_suggest.json")
    assert doc["labels"]["exists"] is False
    assert doc["excluded"]["labeled"] == []
    assert "none yet" in " ".join(capsys.readouterr().out.split())


def test_excludes_labeled_frames_and_their_neighbourhood(outputs, cameras, fly):
    _add_labels(outputs, cameras, fly, frames=(150,))
    cli.main(["labels-suggest", str(outputs), "-n", "5", "--min-gap-s", "1.0"])
    doc = read_suggestions(outputs / "labels_suggest.json")
    frames = [e["frame"] for e in doc["frames"]]
    assert 150 not in frames
    assert all(abs(t - 150) >= 100 for t in frames)
    assert doc["labels"]["labeled_frames"] == [150]
    assert doc["excluded"]["labeled"] == [150]
    # ...unless the operator opts out.
    cli.main(
        [
            "labels-suggest",
            str(outputs),
            "-n",
            "5",
            "--min-gap-s",
            "1.0",
            "--no-exclude-labeled",
        ]
    )
    doc = read_suggestions(outputs / "labels_suggest.json")
    assert 150 in [e["frame"] for e in doc["frames"]]


def test_reports_a_reason_for_every_pick(outputs):
    cli.main(["labels-suggest", str(outputs), "-n", "4", "--min-gap-s", "1.0"])
    doc = read_suggestions(outputs / "labels_suggest.json")
    for entry in doc["frames"]:
        assert entry["kind"] in ("most-wrong", "diversity")
        assert entry["reason"]["summary"]
        assert 0.0 <= entry["percentile"] <= 100.0
        if entry["kind"] == "most-wrong":
            driver = entry["reason"]["drivers"][0]
            assert driver["point_name"]
            assert driver["worst_camera"] in [
                *json.loads(json.dumps(doc["source"]["camera_names"]))
            ]
            assert driver["relation"] in ("far", "near")
        else:
            assert entry["reason"]["grid_slot"]


def test_dry_run_writes_nothing(outputs, capsys):
    cli.main(["labels-suggest", str(outputs), "--dry-run"])
    assert not (outputs / "labels_suggest.json").exists()
    assert "nothing written" in capsys.readouterr().out


def test_output_path_is_honoured(outputs, tmp_path):
    dest = tmp_path / "elsewhere" / "queue.json"
    cli.main(["labels-suggest", str(outputs), "-o", str(dest)])
    assert read_suggestions(dest) is not None
    assert not (outputs / "labels_suggest.json").exists()
    # The sidecar keeps a path back to the file it scored, relative to itself.
    assert read_suggestions(dest)["source"]["results"].endswith("results.h5")


def test_accepts_a_recording_dir_a_results_dir_or_the_file(outputs, tmp_path):
    for path in (
        outputs,  # the outputs dir
        outputs.parent,  # the recording dir
        outputs / "results.h5",  # the file
    ):
        (outputs / "labels_suggest.json").unlink(missing_ok=True)
        cli.main(["labels-suggest", str(path), "-n", "2"])
        assert read_suggestions(outputs / "labels_suggest.json") is not None


def test_deterministic_across_runs(outputs):
    def run():
        cli.main(["labels-suggest", str(outputs), "-n", "8", "--min-gap-s", "0.5"])
        doc = read_suggestions(outputs / "labels_suggest.json")
        return [
            (e["rank"], e["frame"], e["score"], e["kind"], e["reason"]["summary"])
            for e in doc["frames"]
        ]

    assert run() == run()


def test_reports_the_shortfall(outputs, capsys):
    cli.main(["labels-suggest", str(outputs), "-n", "40", "--min-gap-s", "2.0"])
    doc = read_suggestions(outputs / "labels_suggest.json")
    assert doc["shortfall"]["requested"] == 40
    assert doc["shortfall"]["selected"] < 40
    assert doc["shortfall"]["reason"]
    assert "only" in " ".join(capsys.readouterr().out.split())


def test_point_and_camera_globs(outputs, capsys):
    cli.main(
        [
            "labels-suggest",
            str(outputs),
            "-n",
            "3",
            "--points",
            "l?_pretarsus",
            "--cameras",
            "l*",
        ]
    )
    doc = read_suggestions(outputs / "labels_suggest.json")
    assert doc["params"]["points"] == ["l?_pretarsus"]
    assert doc["params"]["cameras"] == ["l*"]
    assert "l?_pretarsus" in " ".join(capsys.readouterr().out.split())


def test_a_glob_matching_nothing_is_an_error(outputs):
    with pytest.raises(SystemExit, match="none of"):
        cli.main(["labels-suggest", str(outputs), "--points", "wing*"])


def test_missing_results_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="no results.h5"):
        cli.main(["labels-suggest", str(tmp_path)])


def test_labels_from_another_recording_is_refused(outputs, cameras, fly):
    _add_labels(outputs, cameras, fly, frames=(150,), n_frames=599)  # wrong T
    with pytest.raises(SystemExit, match="different result"):
        cli.main(["labels-suggest", str(outputs)])


# -- the safety contract ------------------------------------------------------


def test_never_writes_to_results_or_labels(outputs, cameras, fly):
    """``results.h5`` may hold the only copy of a rig; ``labels.h5`` of the GT."""
    labels_path = _add_labels(outputs, cameras, fly, frames=(150, 300))
    results_path = outputs / "results.h5"
    before = (file_md5(results_path), file_md5(labels_path))
    cli.main(["labels-suggest", str(outputs), "-n", "6", "--min-gap-s", "0.5"])
    after = (file_md5(results_path), file_md5(labels_path))
    assert before == after
    # The only new file is the JSON sidecar.
    assert sorted(p.name for p in outputs.iterdir()) == [
        "labels.h5",
        "labels_suggest.json",
        "results.h5",
    ]


def test_scores_pose2d_even_when_the_stored_layer_is_degenerate(outputs, capsys):
    """The headline trap: a reseeded directory must produce the *same* queue.

    ``triangulation/points`` is overwritten with the exact reprojections (so the
    stored ``reproj_error`` is 0 everywhere) and stamped as a ``dfpose.predict``
    output. If the command ever read that layer, the ranking would collapse.
    """
    cli.main(["labels-suggest", str(outputs), "-n", "5", "--min-gap-s", "1.0"])
    pristine = read_suggestions(outputs / "labels_suggest.json")

    with h5py.File(outputs / "results.h5", "a") as f:
        pts3d = np.asarray(f["triangulation/points3d"][()])
        proj = np.asarray(f["triangulation/points"][()])
        f["triangulation/points"][...] = np.zeros_like(proj)
        f["triangulation/reproj_error"][...] = np.zeros(proj.shape[:3])
        meta = json.loads(f.attrs["meta"])
        meta["dfpose_predict"] = {"model": "hrnet_w18_small_v2"}
        f.attrs["meta"] = json.dumps(meta)
        del pts3d

    cli.main(["labels-suggest", str(outputs), "-n", "5", "--min-gap-s", "1.0"])
    reseeded = read_suggestions(outputs / "labels_suggest.json")

    assert [e["frame"] for e in reseeded["frames"]] == [
        e["frame"] for e in pristine["frames"]
    ]
    assert [e["score"] for e in reseeded["frames"]] == [
        e["score"] for e in pristine["frames"]
    ]
    assert reseeded["source"]["reseeded"] is True
    assert reseeded["source"]["scored_array"] == "pose2d/points"
    # ...and the trap is printed, so it cannot be silently reintroduced.
    out = " ".join(capsys.readouterr().out.split())
    assert "the trap:" in out
    assert "STORED reproj_error says median 0 px" in out


def test_warns_when_the_global_residual_exceeds_the_threshold(outputs, capsys):
    """The signature of ranking calibration error rather than model mistakes.

    When *every* frame's residual already sits above the gate, the ranking is
    driven by whichever poses amplify a rig/decode error, and the operator is being
    sent to a pose rather than to a mistake.
    """
    rng = np.random.default_rng(3)
    with h5py.File(outputs / "results.h5", "a") as f:
        pts = np.asarray(f["pose2d/points"][()])
        f["pose2d/points"][...] = pts + rng.normal(scale=30.0, size=pts.shape)
    cli.main(["labels-suggest", str(outputs), "-n", "2", "--threshold", "5.0"])
    out = " ".join(capsys.readouterr().out.split())
    assert "already exceeds" in out
    assert "bundle_adjustment/cameras" in out


# -- absence: keypoints that are not on this animal ---------------------------


def test_labels_absent_declares_and_clears(tmp_path, result, capsys):
    """The batch CLI: declare, then un-declare, without losing anything."""
    import argparse

    from deeperfly.cli.gui import _cmd_labels_absent
    from deeperfly.gui.labels import labels_identity, load_labels

    outdir = tmp_path / "deeperfly_outputs"
    outdir.mkdir()
    result.save(outdir / "results.h5")
    identity = labels_identity(
        point_names=list(result.skeleton.point_names),
        camera_names=list(result.cameras.names),
        n_frames=result.n_frames,
    )
    names = list(result.skeleton.point_names)
    want = names[2:4]

    _cmd_labels_absent(
        argparse.Namespace(
            paths=[str(outdir)],
            points=",".join(want),
            subject="Fly2",
            clear=False,
            frames=None,
        )
    )
    lab = load_labels(outdir / "labels.h5", identity=identity)
    assert lab is not None
    assert [names[i] for i in np.nonzero(lab.absent_all_frames())[0]] == want
    assert lab.subject_id == "Fly2"

    _cmd_labels_absent(
        argparse.Namespace(
            paths=[str(outdir)],
            points=",".join(want),
            subject=None,
            clear=True,
            frames=None,
        )
    )
    lab = load_labels(outdir / "labels.h5", identity=identity)
    assert lab is not None and not lab.absent.any()


def test_labels_absent_accepts_a_frame_range(tmp_path, result):
    """`--frames 1:` is the autotomy case: absent from frame 1 to the end."""
    import argparse

    from deeperfly.cli.gui import _cmd_labels_absent
    from deeperfly.gui.labels import labels_identity, load_labels

    outdir = tmp_path / "deeperfly_outputs"
    outdir.mkdir()
    result.save(outdir / "results.h5")
    name = list(result.skeleton.point_names)[4]
    _cmd_labels_absent(
        argparse.Namespace(
            paths=[str(outdir)], points=name, subject=None, clear=False, frames="1:"
        )
    )
    lab = load_labels(
        outdir / "labels.h5",
        identity=labels_identity(
            point_names=list(result.skeleton.point_names),
            camera_names=list(result.cameras.names),
            n_frames=result.n_frames,
        ),
    )
    assert lab is not None
    assert not lab.absent_at(0)[4]
    assert all(lab.absent_at(t)[4] for t in range(1, result.n_frames))
    assert not lab.absent_all_frames()[4]  # structural consumers must not act on it


def test_labels_absent_rejects_an_unknown_point(tmp_path, result):
    import argparse

    import pytest as _pytest

    from deeperfly.cli.gui import _cmd_labels_absent

    outdir = tmp_path / "deeperfly_outputs"
    outdir.mkdir()
    result.save(outdir / "results.h5")
    with _pytest.raises(SystemExit, match="no skeleton point matches"):
        _cmd_labels_absent(
            argparse.Namespace(
                paths=[str(outdir)],
                points="not_a_keypoint",
                subject=None,
                clear=False,
                frames=None,
            )
        )
