"""Tests for :mod:`deeperfly.calibration` -- the solved-rig artifact.

The artifact exists because a config camera is an *orbit* and a solver's output is
raw ``(rvec, tvec)``, which :func:`deeperfly.cameras.resolve_extrinsics` refuses. So the
tests here are mostly about the two things that make a stored rig safe to reuse: that it
round-trips exactly (a rig quietly rounded on every rewrite is worse than no rig) and
that it *refuses* to be applied to footage or a camera set it does not describe.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import (
    CAMERA_NAMES,
    HEIGHT,
    WIDTH,
    seven_camera_default,
    seven_camera_default_text,
)

from deeperfly.calibration import (
    CALIBRATION_FILENAME,
    CALIBRATION_FORMAT_VERSION,
    Calibration,
    quality_from_errors,
)
from deeperfly.cameras import CameraGroup
from deeperfly.config import Config

SIZES = {name: (HEIGHT, WIDTH) for name in CAMERA_NAMES}


@pytest.fixture
def calibration(cameras) -> Calibration:
    """The reference rig wrapped as an artifact, with provenance and quality."""
    return cameras.to_calibration(
        name="reference",
        image_sizes=SIZES,
        units="config",
        scale_source="board",
        provenance={
            "method": "labels_ba",
            "frames": 42,
            "recordings": ["rec_a91f4e07"],
            "intrinsics": "optics",
            "solver": {"loss": "cauchy", "f_scale": 4.0, "nfev": 431},
        },
        quality={"rms_reproj_px": 1.84, "per_camera_rms_px": {"rh": 1.62}},
    )


# -- round trip ----------------------------------------------------------------


def test_round_trip_preserves_every_camera_parameter_exactly(calibration, tmp_path):
    """Floats must survive save -> load bit for bit, not merely closely.

    A calibration is re-read and re-written whenever a run resumes, so an
    approximate round trip would let the rig drift with each rewrite.
    """
    path = calibration.save(tmp_path)
    loaded = Calibration.load(path)

    assert loaded.camera_names == calibration.camera_names
    for name in calibration.camera_names:
        for attr in ("rvec", "tvec", "intr", "dist"):
            np.testing.assert_array_equal(
                getattr(loaded.cameras[name], attr),
                getattr(calibration.cameras[name], attr),
            )


def test_round_trip_preserves_the_metadata_blocks(calibration, tmp_path):
    loaded = Calibration.load(calibration.save(tmp_path))
    assert loaded.name == "reference"
    assert loaded.units == "config"
    assert loaded.scale_source == "board"
    assert loaded.provenance["method"] == "labels_ba"
    assert loaded.provenance["solver"]["f_scale"] == 4.0
    assert loaded.provenance["recordings"] == ["rec_a91f4e07"]
    assert loaded.quality["per_camera_rms_px"]["rh"] == 1.62
    assert loaded.image_sizes["rh"] == (HEIGHT, WIDTH)


def test_writing_a_loaded_calibration_is_byte_stable(calibration, tmp_path):
    """save -> load -> save reproduces the file, so a resume leaves no diff."""
    first = calibration.save(tmp_path / "a.toml")
    second = Calibration.load(first).save(tmp_path / "b.toml")
    assert second.read_text() == first.read_text()


def test_save_accepts_a_directory_and_load_finds_the_file(calibration, tmp_path):
    written = calibration.save(tmp_path)
    assert written.name == CALIBRATION_FILENAME
    assert Calibration.load(tmp_path).name == "reference"


def test_a_scalar_key_cannot_be_reparented_by_a_following_sub_table(cameras, tmp_path):
    """A nested metadata block must not swallow its siblings.

    TOML binds a bare ``key = value`` to the most recent ``[table]`` header, so a writer
    that emitted ``solver`` (a sub-table) before ``method`` (a scalar) would silently
    move ``method`` into ``[...solver]``.
    """
    cal = cameras.to_calibration(
        # Deliberately dict-ordered sub-table first, scalar second.
        provenance={"solver": {"loss": "huber"}, "method": "labels_ba", "frames": 7},
    )
    loaded = Calibration.load(cal.save(tmp_path))
    assert loaded.provenance["method"] == "labels_ba"
    assert loaded.provenance["frames"] == 7
    assert loaded.provenance["solver"] == {"loss": "huber"}


# -- guards --------------------------------------------------------------------


def test_a_newer_format_version_is_refused(calibration, tmp_path):
    path = calibration.save(tmp_path)
    path.write_text(
        path.read_text().replace(
            f"format_version = {CALIBRATION_FORMAT_VERSION}",
            f"format_version = {CALIBRATION_FORMAT_VERSION + 1}",
        )
    )
    with pytest.raises(ValueError, match="newer deeperfly"):
        Calibration.load(path)


def test_differently_sized_footage_is_refused(calibration):
    """Intrinsics are pixel quantities; rescaled footage must fail, not be rescaled."""
    with pytest.raises(ValueError, match="different footage"):
        calibration.check_image_sizes({"rh": (1008, 1600)})


def test_matching_footage_and_unknown_cameras_pass_the_size_check(calibration):
    calibration.check_image_sizes(SIZES)  # exact match
    calibration.check_image_sizes({"nobody": (1, 1)})  # not in the calibration


def test_a_partly_covering_calibration_narrows_the_rig_and_names_what_it_dropped(
    cameras, tmp_path, caplog
):
    """A view the rig never measured cannot be placed, so it is dropped like a view with
    no footage -- one config routinely describes more rig than one solve covers.

    Named in the warning rather than counted: "6 of 7" is not actionable and "dropped:
    ['f']" is.
    """
    partial = CameraGroup({k: v for k, v in cameras.cameras.items() if k != "f"})
    path = partial.to_calibration(name="partial").save(tmp_path)
    with caplog.at_level("WARNING", logger="deeperfly"):
        rig = CameraGroup.from_calibration(path, names=CAMERA_NAMES)
    assert rig.names == [n for n in CAMERA_NAMES if n != "f"]
    assert "'f'" in caplog.text and "partial" in caplog.text


def test_a_calibration_covering_no_requested_camera_is_refused(cameras, tmp_path):
    """A subset is a narrower rig; NO overlap is the wrong rig, and cannot be narrowed to
    anything."""
    renamed = CameraGroup({f"other_{k}": v for k, v in cameras.cameras.items()})
    path = renamed.to_calibration(name="elsewhere").save(tmp_path)
    with pytest.raises(ValueError, match="covers none of the cameras"):
        CameraGroup.from_calibration(path, names=CAMERA_NAMES)


def test_a_calibration_with_no_cameras_is_refused(tmp_path):
    path = tmp_path / "empty.toml"
    path.write_text("[calibration]\nformat_version = 1\n")
    with pytest.raises(ValueError, match="defines no cameras"):
        Calibration.load(path)


def test_a_camera_missing_a_vector_is_refused(calibration, tmp_path):
    path = calibration.save(tmp_path)
    path.write_text(
        "\n".join(
            line
            for line in path.read_text().splitlines()
            if not line.startswith("tvec")
        )
    )
    with pytest.raises(ValueError, match="missing 'tvec'"):
        Calibration.load(path)


def test_a_wrong_length_vector_is_refused(calibration, tmp_path):
    path = calibration.save(tmp_path)
    path.write_text(path.read_text().replace("intr = [", "intr = [1.0, ", 1))
    with pytest.raises(ValueError, match="must have 4 entries"):
        Calibration.load(path)


@pytest.mark.parametrize("kwargs", [{"units": "furlongs"}, {"scale_source": "vibes"}])
def test_an_unknown_units_or_scale_source_is_refused(cameras, kwargs):
    with pytest.raises(ValueError, match="must be one of"):
        cameras.to_calibration(**kwargs)


def test_a_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no calibration at"):
        Calibration.load(tmp_path / "nope.toml")


# -- ordering ------------------------------------------------------------------


def test_cameras_are_returned_in_the_requested_order(cameras, tmp_path):
    """The ``V`` axis is positional, so a differently-ordered file must be reordered.

    Reinterpreting it in file order instead would silently transpose which camera's
    parameters project which view's points.
    """
    reversed_group = CameraGroup(dict(reversed(list(cameras.cameras.items()))))
    path = reversed_group.to_calibration().save(tmp_path)

    assert Calibration.load(path).camera_names == CAMERA_NAMES[::-1]
    loaded = CameraGroup.from_calibration(path, names=CAMERA_NAMES)
    assert loaded.names == CAMERA_NAMES
    np.testing.assert_array_equal(loaded.tvecs, cameras.tvecs)


# -- quality -------------------------------------------------------------------


def test_quality_ignores_unobserved_cells(cameras):
    """NaN must not count as zero error, or a sparse rig would look better than a dense one."""
    err = np.full((len(CAMERA_NAMES), 4, 5), np.nan)
    err[0] = 3.0
    q = quality_from_errors(err, CAMERA_NAMES)

    assert q["n_observations"] == 20
    assert q["rms_reproj_px"] == pytest.approx(3.0)
    assert q["max_reproj_px"] == pytest.approx(3.0)
    assert set(q["per_camera_rms_px"]) == {"rh"}


def test_quality_of_nothing_observed_is_empty_not_zero(cameras):
    """An empty block is honest; zeros would read as a perfect fit."""
    assert quality_from_errors(np.full((7, 2, 3), np.nan), CAMERA_NAMES) == {}


def test_quality_reports_per_camera_rms(cameras):
    err = np.zeros((len(CAMERA_NAMES), 2, 2))
    err[1] = 4.0
    q = quality_from_errors(err, CAMERA_NAMES)
    assert q["per_camera_rms_px"]["rh"] == pytest.approx(0.0)
    assert q["per_camera_rms_px"]["rm"] == pytest.approx(4.0)


def test_summary_names_the_scale_when_there_is_none(cameras):
    text = cameras.to_calibration(name="bare").summary()
    assert "nothing fixed the scale" in text
    assert "not recorded" in text  # no quality block


# -- config integration --------------------------------------------------------


def _config_with_calibration(tmp_path, cameras, *, sizes=SIZES) -> Config:
    """A default config pointed at a calibration whose rig is *moved* from the orbit.

    The offset is what proves which of the two the loader actually used.
    """
    moved = CameraGroup.from_arrays(
        cameras.names, cameras.rvecs, cameras.tvecs + 1.25, cameras.intrs, cameras.dists
    )
    moved.to_calibration(name="solved", image_sizes=sizes).save(tmp_path)
    # PREPENDED, not spliced in beside a marker: the packaged config discusses
    # `[calibration]` and `[default_camera]` in prose before declaring either, so a
    # first-occurrence replace lands inside a comment. A top-level table is
    # order-independent in TOML, so the front of the file is a fine place for it.
    text = (
        f'[calibration]\npath = "{CALIBRATION_FILENAME}"\n\n'
        + seven_camera_default_text()
    )
    (tmp_path / "config.toml").write_text(text)
    return Config.from_toml(tmp_path / "config.toml")


def test_a_config_calibration_wins_over_the_orbit_spec(cameras, tmp_path):
    config = _config_with_calibration(tmp_path, cameras)
    rig = config.camera_group(image_sizes=SIZES)

    assert rig.names == CAMERA_NAMES  # the config's order, not the file's
    np.testing.assert_allclose(rig.tvecs, cameras.tvecs + 1.25)


def test_a_relative_calibration_path_resolves_beside_the_config(cameras, tmp_path):
    config = _config_with_calibration(tmp_path, cameras)
    assert config.calibration_path() == tmp_path / CALIBRATION_FILENAME


def test_a_config_without_a_calibration_still_uses_the_orbit(cameras):
    config = seven_camera_default()
    assert config.calibration_path() is None
    assert config.camera_group(image_sizes=SIZES).names == CAMERA_NAMES


def test_the_config_calibration_key_is_not_read_as_a_camera(cameras, tmp_path):
    """``calibration`` sits under ``[cameras]`` beside the view tables, so the splitter
    must drop it -- otherwise it reaches ``Camera.from_spec`` as a spec."""
    config = _config_with_calibration(tmp_path, cameras)
    _, specs = config.camera_table()
    assert "calibration" not in specs
    assert list(specs) == CAMERA_NAMES


def test_a_config_calibration_for_other_footage_is_refused(cameras, tmp_path):
    config = _config_with_calibration(tmp_path, cameras)
    with pytest.raises(ValueError, match="different footage"):
        config.camera_group(image_sizes={name: (1008, 1600) for name in CAMERA_NAMES})


# -- the error that sends people here -----------------------------------------


def test_raw_extrinsics_in_a_camera_spec_point_at_the_calibration_file(cameras):
    """The orbit parser still refuses rvec/tvec -- but now it says where they go."""
    config = Config.from_dict(
        {"cameras": {"cam": {"focal_length_px": 700.0, "rvec": [0, 0, 0]}}}
    )
    with pytest.raises(ValueError, match="calibration"):
        config.camera_group(image_sizes={"cam": (100, 100)})


# -- the CLI -------------------------------------------------------------------


def _result_with(cameras, fly, tmp_path, *, refined=None):
    """A results.h5 whose pose2d rig is ``cameras``, optionally with a BA rig."""
    from deeperfly.results import StageStore

    rng = np.random.default_rng(0)
    pts3d = rng.uniform(-1.5, 1.5, size=(4, 38, 3))
    pts2d = np.asarray(cameras.project(pts3d))
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    store = StageStore(out / "results.h5")
    store.write_pose2d(
        cameras=cameras,
        skeleton=fly,
        pts2d=pts2d,
        conf=np.ones(pts2d.shape[:3]),
        image_sizes=SIZES,
    )
    if refined is not None:
        store.write_cameras("bundle_adjustment", refined)
    return out


def test_export_prefers_the_bundle_adjusted_rig(cameras, fly, tmp_path):
    from deeperfly import cli

    refined = CameraGroup.from_arrays(
        cameras.names, cameras.rvecs, cameras.tvecs + 2.0, cameras.intrs, cameras.dists
    )
    out = _result_with(cameras, fly, tmp_path, refined=refined)
    cli.main(["calibration", "export", str(out), "--log-level", "error"])

    cal = Calibration.load(out / CALIBRATION_FILENAME)
    np.testing.assert_allclose(cal.cameras.tvecs, refined.tvecs)
    assert cal.provenance["method"] == "labels_ba"
    assert cal.image_sizes["rh"] == (HEIGHT, WIDTH)


def test_export_without_bundle_adjustment_says_so_rather_than_claiming_a_solve(
    cameras, fly, tmp_path, caplog
):
    """An un-refined rig is still worth exporting -- but not as `labels_ba`."""
    from deeperfly import cli

    out = _result_with(cameras, fly, tmp_path)
    with caplog.at_level("WARNING"):
        cli.main(["calibration", "export", str(out), "--log-level", "warning"])

    assert "no bundle_adjustment/cameras" in caplog.text
    cal = Calibration.load(out / CALIBRATION_FILENAME)
    assert cal.provenance["method"] == "orbit_prior"
    np.testing.assert_allclose(cal.cameras.tvecs, cameras.tvecs)


def test_export_measures_the_rig_it_is_exporting(cameras, fly, tmp_path):
    """Residuals come from re-measuring, not from the stored reproj_error.

    The stored array can describe a different rig (or a substituted 2D layer) than the
    one going into the file, which is how a bad rig ships with someone else's good
    numbers attached.
    """
    from deeperfly import cli

    out = _result_with(cameras, fly, tmp_path)
    cli.main(["calibration", "export", str(out), "--log-level", "error"])
    quality = Calibration.load(out / CALIBRATION_FILENAME).quality

    # The 2D was projected from this very rig, so it reprojects onto itself.
    assert quality["rms_reproj_px"] == pytest.approx(0.0, abs=1e-6)
    assert quality["n_observations"] > 0


def test_export_honors_an_explicit_output_path(cameras, fly, tmp_path):
    from deeperfly import cli

    out = _result_with(cameras, fly, tmp_path)
    dest = tmp_path / "rigs" / "mine.toml"
    cli.main(
        ["calibration", "export", str(out), "-o", str(dest), "--log-level", "error"]
    )
    assert dest.exists()
    assert not (out / CALIBRATION_FILENAME).exists()


def test_show_prints_the_summary(calibration, tmp_path, capsys):
    from deeperfly import cli

    path = calibration.save(tmp_path)
    cli.main(["calibration", "show", str(path), "--log-level", "error"])
    out = capsys.readouterr().out
    assert "calibration: reference" in out
    assert "rms" in out


def test_export_of_a_missing_result_is_a_clear_error(tmp_path):
    from deeperfly import cli

    with pytest.raises(SystemExit, match="does not exist"):
        cli.main(["calibration", "export", str(tmp_path / "nope.h5")])
