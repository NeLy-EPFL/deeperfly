"""Tests for narrowing a run to the footage it actually has.

One config routinely describes more rig than one recording holds: the packaged default
declares the eight-camera rig, and a seven-camera recording under it is not malformed. So a
camera whose ``video`` pattern matched nothing leaves the rig, and everything keyed on it
follows -- rather than the whole recording being refused.

Under v2 that is one deletion rather than three: a camera IS a source IS a pathway IS a
view, so dropping the camera table entry drops all of them. What still needs pinning is
that the ``V`` axis really shortens (a camera left in place with no footage gives a view
whose 2D is all-NaN, which reads as a camera that detected nothing rather than one that
was never there) and that everything keyed on the camera -- the video grids -- follows.
"""

from __future__ import annotations

import pytest

from deeperfly.config import MIN_VIEWS_FOR_3D, Config


def _have(cfg: Config, *without: str) -> dict[str, list[str]]:
    """The config's cameras, all with footage except ``without``."""
    return {
        name: ["frame.mp4"] for name in cfg.source_patterns() if name not in without
    }


def test_nothing_missing_returns_the_very_same_config():
    """The common case allocates nothing and cannot perturb anything."""
    cfg = Config.default()
    assert cfg.narrowed_to_sources(_have(cfg)) is cfg


def test_a_camera_with_no_footage_leaves_the_rig_entirely():
    """One deletion covers all four, because they are one name."""
    cfg = Config.default()
    view = list(cfg.camera_table()[1])[-1]

    narrowed = cfg.narrowed_to_sources(_have(cfg, view))

    assert view not in narrowed.source_patterns()
    assert view not in narrowed.camera_table()[1]
    plan = narrowed.detection_plan()
    assert view not in plan.view_names
    assert view not in [pw.name for pw in plan.pathways]


def test_the_view_axis_actually_shortens():
    """The point of dropping the CAMERA table entry rather than only the pathway.

    ``view_names`` comes from the camera table, so removing a pathway alone leaves a view
    whose 2D is all-NaN -- which reads as a camera that was detected and found nothing,
    not one that was never there. Here the axis is genuinely shorter and every surviving
    view is observed.
    """
    cfg = Config.default()
    view = list(cfg.camera_table()[1])[-1]
    plan = cfg.narrowed_to_sources(_have(cfg, view)).detection_plan()

    assert plan.n_views == len(cfg.camera_table()[1]) - 1
    mask = plan.visibility_mask()
    assert mask.shape[0] == plan.n_views
    assert mask.all(), "every view left in the plan is one a pathway feeds"


def test_the_original_config_is_not_mutated():
    """Narrowing returns a copy; the caller's config still describes the whole rig."""
    cfg = Config.default()
    before = list(cfg.camera_table()[1])
    cfg.narrowed_to_sources(_have(cfg, before[-1]))
    assert list(cfg.camera_table()[1]) == before


def test_the_snapshot_text_still_describes_the_configured_rig():
    """`data` narrows, `text` does not -- the same split a skeleton preset reference uses.

    The snapshot records what was *asked for*; the narrowed plan is what reaches the
    fingerprints. That is what keeps the cache honest AND keeps the snapshot honest: a
    seven-view run records a seven-view fingerprint, while the file still says which rig the
    operator configured.
    """
    cfg = Config.default()
    view = list(cfg.camera_table()[1])[-1]
    narrowed = cfg.narrowed_to_sources(_have(cfg, view))
    assert narrowed.snapshot_text() == cfg.snapshot_text()
    assert f"[cameras.{view}]" in narrowed.snapshot_text()


def test_a_dropped_view_is_blanked_out_of_every_video_grid():
    """Blanked, not removed: a grid's shape is a layout.

    The montage reads as the animal from above, so closing the gap would slide every
    remaining camera into a neighbour's place. `""` is already the config's own spelling
    for "leave a gap here".
    """
    cfg = Config.default()
    view = list(cfg.camera_table()[1])[-1]

    narrowed = cfg.narrowed_to_sources(_have(cfg, view))

    for before, after in zip(
        cfg.data["visualization"]["videos"], narrowed.data["visualization"]["videos"]
    ):
        assert [len(r) for r in after["grid"]] == [len(r) for r in before["grid"]]
        assert view not in [cell for row in after["grid"] for cell in row]
    # ...and the surviving cameras did not move.
    survivor = list(narrowed.camera_table()[1])[0]
    first_before = cfg.data["visualization"]["videos"][0]["grid"]
    first_after = narrowed.data["visualization"]["videos"][0]["grid"]
    assert [
        (r, c)
        for r, row in enumerate(first_after)
        for c, cell in enumerate(row)
        if cell == survivor
    ] == [
        (r, c)
        for r, row in enumerate(first_before)
        for c, cell in enumerate(row)
        if cell == survivor
    ]


def test_a_dropped_cameras_crop_goes_with_it():
    """Left behind, a crop naming a camera that is gone REFUSES the whole plan.

    Which is the right answer for a typo and the wrong one here: narrowing exists exactly
    so that a config describing more rig than this recording has still runs. So both the
    fixed box and the searched name are narrowed too.
    """
    cfg = Config.from_dict(
        {
            **Config.default().data,
            "pose2d": {
                **Config.default().data["pose2d"],
                "crops": {"lh": [0, 0, 8, 8]},
                "auto_crops": ["f", "h"],
            },
        }
    )
    narrowed = cfg.narrowed_to_sources(_have(cfg, "h", "lh"))
    pose2d = narrowed.data["pose2d"]
    assert "lh" not in pose2d["crops"]
    assert pose2d["auto_crops"] == ["f"]
    narrowed.detection_plan()  # would refuse if either name had survived


def test_narrowing_below_two_views_refuses():
    """The floor, and why it is a refusal rather than one more degradation.

    One view fails *silently* everywhere downstream: triangulation returns all-NaN without
    raising, RANSAC gives a lone observation zero inliers and then erases it, and bundle
    adjustment reports success at a cost near zero. Nothing outside the smoother has a
    "too few views" diagnostic, so a one-view run produces a confident-looking nothing.
    """
    cfg = Config.default()
    keep = list(cfg.source_patterns())[0]
    with pytest.raises(SystemExit, match=r"only 1 view\(s\) have footage"):
        cfg.narrowed_to_sources({keep: ["frame.mp4"]})


def test_exactly_the_floor_is_allowed():
    cfg = Config.default()
    two = list(cfg.source_patterns())[:MIN_VIEWS_FOR_3D]
    narrowed = cfg.narrowed_to_sources({n: ["frame.mp4"] for n in two})
    assert narrowed.detection_plan().n_views == MIN_VIEWS_FOR_3D


def test_an_empty_file_list_counts_as_absent():
    """The established encoding: `camera_files` returns `[]` for a source with nothing."""
    cfg = Config.default()
    source = list(cfg.source_patterns())[-1]
    have = _have(cfg)
    have[source] = []
    assert source not in cfg.narrowed_to_sources(have).source_patterns()


def test_the_warning_names_the_camera_it_dropped(caplog):
    """A narrowed run has to be legible in a log read after the fact."""
    cfg = Config.default()
    view = list(cfg.camera_table()[1])[-1]
    with caplog.at_level("WARNING", logger="deeperfly"):
        cfg.narrowed_to_sources(_have(cfg, view))
    text = caplog.text
    assert view in text and "narrowing" in text


# -- the other way a view becomes unusable: no measured camera -----------------


def _config_with_partial_calibration(tmp_path, drop="h"):
    """The packaged config pointed at a calibration that covers every view but ``drop``."""
    from deeperfly.cameras import CameraGroup

    cfg = Config.default()
    sizes = {n: (512, 1024) for n in cfg.camera_table()[1]}
    full = cfg.camera_group(image_sizes=sizes)
    partial = CameraGroup({k: v for k, v in full.cameras.items() if k != drop})
    cal = tmp_path / "rig.toml"
    partial.to_calibration(
        name="rig", image_sizes={k: (512, 1024) for k in partial.names}
    ).save(cal)
    # Prepended: the packaged config discusses `[calibration]` in prose before declaring
    # it, so a first-occurrence replace would land inside a comment.
    text = f'[calibration]\npath = "{cal}"\n\n' + cfg.snapshot_text()
    path = tmp_path / "config.toml"
    path.write_text(text)
    return Config.from_toml(path), sizes


def test_a_rig_that_covers_fewer_views_narrows_the_plan_too(tmp_path):
    """The rig and the ``V`` axis have to shorten TOGETHER.

    A calibration covering fewer cameras than the config declares already subsetted the
    rig -- but the detection plan kept every view, so ``pts2d`` came out with more view rows
    than the rig had cameras. The first thing to notice was an einsum shape error naming no
    camera at all, several stages downstream of the cause.
    """
    from deeperfly.pipeline.run import _rig_coverage

    cfg, sizes = _config_with_partial_calibration(tmp_path)
    assert cfg.detection_plan().n_views == 8, "the un-narrowed config declares eight"

    narrowed = cfg.narrowed_to_covered_views(_rig_coverage(cfg))
    rig = narrowed.camera_group(image_sizes=sizes)
    plan = narrowed.detection_plan()

    assert plan.n_views == len(rig.names)
    assert "h" not in plan.view_names and "h" not in rig.names


def test_a_rig_covering_every_view_narrows_nothing(tmp_path):
    """An orbit is not a measurement, so there is nothing for it to fail to cover."""
    from deeperfly.pipeline.run import _rig_coverage

    cfg = Config.default()
    assert cfg.narrowed_to_covered_views(_rig_coverage(cfg)) is cfg


def test_an_unreadable_calibration_is_left_to_the_stage_that_needs_it(tmp_path):
    """Not this narrowing's error to report -- and it must not silently drop every view."""
    from deeperfly.pipeline.run import _rig_coverage

    cal = tmp_path / "rig.toml"
    cal.write_text("this is not toml {{{")
    text = f'[calibration]\npath = "{cal}"\n\n' + Config.default().snapshot_text()
    path = tmp_path / "config.toml"
    path.write_text(text)
    cfg = Config.from_toml(path)
    assert cfg.narrowed_to_covered_views(_rig_coverage(cfg)) is cfg
