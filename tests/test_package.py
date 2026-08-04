"""``.dfpkg`` -- a project as one shareable file.

Two properties carry the weight. The labels are stored as **the original bytes**, so a
round trip cannot be corrupted by a schema mistake in the packager -- ground truth is the
one irreplaceable thing here. And only the **labeled** frames are embedded, which is what
makes a package small enough to be worth having.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import CAMERA_NAMES

from deeperfly.gui.labels import Labels, labels_identity, save_labels
from deeperfly.package import (
    describe_package,
    export_package,
    import_package,
)
from deeperfly.project import Project
from deeperfly.skeleton import Skeleton

SIZES = {name: (48, 64) for name in CAMERA_NAMES}


def _project(tmp_path, *, labelled=True, frames=6, with_video=False):
    """A project with one recording, optionally labelled and with real tiny videos."""
    project = Project.create(tmp_path / "proj", name="pkg", skeleton="fly38")
    rec = tmp_path / "flyA"
    rec.mkdir()
    if with_video:
        import av

        for i, cam in enumerate(CAMERA_NAMES):
            with av.open(str(rec / f"{cam}.mp4"), mode="w") as c:
                st = c.add_stream("libx264", rate=10)
                st.width, st.height, st.pix_fmt = 64, 48, "yuv420p"
                for t in range(frames):
                    img = np.full((48, 64, 3), 20 + 8 * i, dtype=np.uint8)
                    img[5 + t : 9 + t, 20:30] = 220
                    c.mux(st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")))
                c.mux(st.encode())
    else:
        for cam in CAMERA_NAMES:
            (rec / f"{cam}.mp4").write_bytes(b"\0" * (900 + len(cam)))
    entry = project.add_recording(rec)

    if labelled:
        labels = Labels.empty(len(CAMERA_NAMES), frames, 38)
        labels.set_gt(0, 1, 5, (10.0, 20.0))
        labels.set_gt(1, 3, 7, (30.0, 40.0))
        labels.set_reviewed(1, True)
        save_labels(
            project.labels_path(entry),
            labels,
            identity=labels_identity(
                point_names=list(Skeleton.fly().point_names),
                camera_names=list(CAMERA_NAMES),
                n_frames=frames,
                image_sizes=SIZES,
            ),
        )
    return project, entry


# -- export --------------------------------------------------------------------


def test_a_package_carries_what_the_project_owns(tmp_path):
    project, _ = _project(tmp_path)
    project.write_rig()
    report = export_package(project, tmp_path / "out.dfpkg", embed="none")

    assert report.recordings == 1
    assert report.label_files == 1
    described = describe_package(tmp_path / "out.dfpkg")
    assert described["has_rig"]
    assert described["profiles"] == ["default"]
    assert described["recordings"][0]["has_labels"]


def test_the_suffix_is_added_when_missing(tmp_path):
    project, _ = _project(tmp_path)
    export_package(project, tmp_path / "bare", embed="none")
    assert (tmp_path / "bare.dfpkg").exists()


def test_an_unknown_embed_policy_is_refused(tmp_path):
    project, _ = _project(tmp_path)
    with pytest.raises(ValueError, match="embed must be"):
        export_package(project, tmp_path / "x.dfpkg", embed="everything")


def test_only_the_labelled_frames_are_embedded(tmp_path):
    """The whole reason a package is affordable."""
    pytest.importorskip("av")
    project, _ = _project(tmp_path, frames=8, with_video=True)
    report = export_package(project, tmp_path / "out.dfpkg", embed="user")

    # Frames 1 and 3 carry work (two GT cells plus a reviewed flag), so 2 frames x 7 views.
    assert report.embedded_frames == 2 * len(CAMERA_NAMES)
    described = describe_package(tmp_path / "out.dfpkg")
    assert described["recordings"][0]["embedded_frames"] == 2 * len(CAMERA_NAMES)


def test_unresolvable_footage_is_reported_not_silently_dropped(tmp_path):
    """A package quietly missing its frames looks exactly like one that never had them."""
    project, _ = _project(tmp_path)  # fake .mp4 bytes: not decodable
    report = export_package(project, tmp_path / "out.dfpkg", embed="user")
    assert report.embedded_frames == 0
    assert report.label_files == 1  # the labels still travelled


def test_a_recording_with_no_labels_is_noted(tmp_path):
    project, _ = _project(tmp_path, labelled=False)
    report = export_package(project, tmp_path / "out.dfpkg", embed="none")
    assert report.label_files == 0
    assert any("no labels.h5" in n for n in report.notes)


# -- round trip ----------------------------------------------------------------


def test_the_labels_survive_byte_for_byte(tmp_path):
    """Stored as the original bytes, so a schema mistake here cannot corrupt them."""
    project, entry = _project(tmp_path)
    original = project.labels_path(entry).read_bytes()
    export_package(project, tmp_path / "out.dfpkg", embed="none")
    import_package(tmp_path / "out.dfpkg", tmp_path / "back")

    restored = (
        tmp_path
        / "back"
        / "recordings"
        / entry.slug
        / "deeperfly_outputs"
        / "labels.h5"
    )
    assert restored.read_bytes() == original


def test_a_round_trip_reopens_as_a_project(tmp_path):
    project, entry = _project(tmp_path)
    project.write_rig()
    export_package(project, tmp_path / "out.dfpkg", embed="none")
    import_package(tmp_path / "out.dfpkg", tmp_path / "back")

    back = Project.load(tmp_path / "back")
    assert back.name == "pkg"
    assert back.id == project.id  # the manifest travelled verbatim
    assert [e.slug for e in back.recordings] == [entry.slug]
    assert back.skeleton().n_points == 38
    assert back.rig_path().exists()
    # ...and its composed config is still valid on the far side.
    import tomllib

    from deeperfly.config import Config

    assert (
        Config.from_dict(tomllib.loads(back.compose_config())).skeleton().n_points == 38
    )


def test_an_imported_recording_can_actually_be_re_pointed_at_its_footage(tmp_path):
    """A package used to arrive with no recording.toml at all, which made it unopenable.

    The editor reads a project recording's footage from that file. With none, it fell back
    to ``origin.from`` -- the *exporter's* absolute path on another machine -- and found
    nothing, so an imported package could not be opened even with the frames embedded.

    The exporter's ``abs``/``rel`` are fictions here, so only ``names`` and ``bytes``
    travel: enough for ``--footage-dir``, and enough to re-derive the content id.
    """
    import tomllib

    from deeperfly.footage import resolve

    project, entry = _project(tmp_path)
    export_package(project, tmp_path / "out.dfpkg", embed="none")
    import_package(tmp_path / "out.dfpkg", tmp_path / "back")

    rec_toml = tmp_path / "back" / "recordings" / entry.slug / "recording.toml"
    assert rec_toml.exists(), "an imported recording has no footage pointer at all"
    footage = tomllib.loads(rec_toml.read_text())["recording"]["footage"]
    assert footage, "the pointer is empty"
    for spec in footage.values():
        assert spec["names"], "the file names did not travel"
        # The exporter's machine-specific paths deliberately did NOT travel.
        assert "abs" not in spec and "rel" not in spec
    # And --footage-dir finds them, which is the whole point.
    a_camera = sorted(footage)[0]
    assert resolve(footage[a_camera], rec_toml.parent, tmp_path / "flyA") is not None


def test_the_package_records_the_footage_names_and_sizes_it_promised(tmp_path):
    """The module docstring always claimed "footage basenames, sizes"; nothing wrote them."""
    import json

    import h5py

    project, entry = _project(tmp_path)
    out = export_package(project, tmp_path / "out.dfpkg", embed="none")
    assert out.recordings == 1
    with h5py.File(tmp_path / "out.dfpkg", "r") as f:
        pointer = json.loads(f[f"recordings/{entry.slug}"].attrs["footage"])
    assert pointer
    for spec in pointer.values():
        assert spec["names"] and spec["bytes"]


def test_a_calibration_travels(cameras, tmp_path):
    project, _ = _project(tmp_path)
    cameras.to_calibration(name="rig", image_sizes=SIZES).save(
        project.root / "calibrations" / "rig.toml"
    )
    project.calibration = "calibrations/rig.toml"
    project.save()
    export_package(project, tmp_path / "out.dfpkg", embed="none")
    import_package(tmp_path / "out.dfpkg", tmp_path / "back")

    from deeperfly.calibration import Calibration

    back = Project.load(tmp_path / "back")
    assert back.calibration == "calibrations/rig.toml"
    restored = Calibration.load(back.calibration_path())
    np.testing.assert_allclose(restored.cameras.tvecs, cameras.tvecs)


def test_embedded_frames_land_as_files_named_by_source_frame(tmp_path):
    """The index is in the FILENAME, so a frame is traceable without opening anything."""
    pytest.importorskip("av")
    project, entry = _project(tmp_path, frames=8, with_video=True)
    export_package(project, tmp_path / "out.dfpkg", embed="user")
    import_package(tmp_path / "out.dfpkg", tmp_path / "back")

    folder = tmp_path / "back" / "recordings" / entry.slug / "frames" / CAMERA_NAMES[0]
    names = sorted(p.name for p in folder.glob("*.jpg"))
    assert names == ["000001.jpg", "000003.jpg"]
    # Real JPEGs, not placeholders.
    import cv2

    img = cv2.imread(str(folder / "000001.jpg"))
    assert img is not None and img.shape[:2] == (48, 64)


# -- guards --------------------------------------------------------------------


def test_importing_into_an_existing_project_is_refused(tmp_path):
    """That is a merge, and overwriting instead would look like it had worked."""
    project, _ = _project(tmp_path)
    export_package(project, tmp_path / "out.dfpkg", embed="none")
    with pytest.raises(SystemExit, match="merge, not an unpack"):
        import_package(tmp_path / "out.dfpkg", project.root)


def test_a_dry_run_writes_nothing(tmp_path):
    project, _ = _project(tmp_path)
    export_package(project, tmp_path / "out.dfpkg", embed="none")
    report = import_package(tmp_path / "out.dfpkg", tmp_path / "back", apply=False)
    assert report.recordings == 1
    assert not (tmp_path / "back").exists()


def test_a_newer_package_is_refused(tmp_path):
    import h5py

    project, _ = _project(tmp_path)
    path = tmp_path / "out.dfpkg"
    export_package(project, path, embed="none")
    with h5py.File(path, "a") as f:
        f["meta"].attrs["package_format_version"] = 99
    with pytest.raises(ValueError, match="newer deeperfly"):
        describe_package(path)


def test_a_missing_package_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no package at"):
        describe_package(tmp_path / "nope.dfpkg")


# -- the CLI -------------------------------------------------------------------


def test_cli_export_then_import(tmp_path, capsys):
    from deeperfly import cli

    project, _ = _project(tmp_path)
    pkg = tmp_path / "p.dfpkg"
    cli.main(
        [
            "project",
            "export",
            str(pkg),
            str(project.root),
            "--embed",
            "none",
            "--log-level",
            "error",
        ]
    )
    assert pkg.exists()

    capsys.readouterr()
    cli.main(
        ["project", "import", str(pkg), str(tmp_path / "back"), "--log-level", "error"]
    )
    out = capsys.readouterr().out
    assert "dry run" in out
    assert not (tmp_path / "back").exists()

    cli.main(
        [
            "project",
            "import",
            str(pkg),
            str(tmp_path / "back"),
            "--apply",
            "--log-level",
            "error",
        ]
    )
    assert Project.load(tmp_path / "back").name == "pkg"
    # The warning that footage did not travel is the one thing a new user must be told.
    assert "footage is NOT in the package" in capsys.readouterr().out
