"""Tests for the four ways a config is allowed to say less than it used to.

Each of these replaces a block of the packaged config with a *reference* to something
that already knows the answer, so each test's real job is to pin that the short form and
the long form mean exactly the same thing:

* no ``[skeleton]`` table at all -- the packaged skeleton, resolved for you;
  ``include = "fly38"`` names another.
* ``[[pose2d.models]]`` with only ``class``/``weights`` -- the rest from the class.
* ``weights = "x.pth"`` -- found on ``$DEEPERFLY_MODELS`` instead of a machine's path.
* ``grid = [[...]]``  -- a montage instead of hand-computed panel offsets.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest
from helpers import AZIMUTHS_DEG, CAMERA_NAMES

from deeperfly.config import DEFAULT_CONFIG_PATH, Config, skeleton_presets
from deeperfly.pose2d import download
from deeperfly.pose2d.models import CLASS_ALIASES, class_defaults

# -- skeleton presets ---------------------------------------------------------


def _skeleton_signature(s):
    """Everything a skeleton IS -- points, edges, symmetries, colors."""
    return (
        s.name,
        tuple(s.point_names),
        tuple(map(tuple, s.bones)),
        s.symmetries.tolist(),
        tuple(s.point_colors),
    )


def test_packaged_presets_are_discoverable():
    assert set(skeleton_presets()) == {"fly38"}


def test_no_skeleton_table_resolves_to_the_packaged_one():
    """The norm: a run config says NOTHING about its skeleton.

    Which points exist is a property of the detector, not of the recording, so the
    default is allowed to be "the one packaged skeleton" -- and what makes that safe is
    the check at the other end, where a checkpoint's recorded point names are compared
    against these (`deeperfly.pose2d.stream._check_channel_names`).
    """
    assert _skeleton_signature(Config.from_dict({}).skeleton()) == _skeleton_signature(
        Config.from_dict({"skeleton": {"include": "fly38"}}).skeleton()
    )


def test_a_retired_skeleton_name_still_resolves():
    """``fly38b`` was renamed to ``fly38``; the old spelling has to keep loading.

    Every config and every output-directory snapshot written before the rename holds the
    old reference, and a reference that no longer resolves is not a rename, it is a file
    that cannot be opened. The alias resolves to the same table -- and the config's own
    ``name`` still wins over the file's, which is why such a run goes on *calling* its
    skeleton fly38b while computing exactly what it always did.
    """
    aliased = Config.from_dict({"skeleton": {"include": "fly38b"}}).skeleton()
    current = Config.from_dict({"skeleton": {"include": "fly38"}}).skeleton()
    assert list(aliased.point_names) == list(current.point_names)
    assert aliased.bones.tolist() == current.bones.tolist()


@pytest.mark.parametrize("preset", ["fly38"])
def test_include_matches_the_same_table_written_out(preset):
    """``include`` resolves to exactly the table it replaces -- the whole premise."""
    spelled = tomllib.loads(skeleton_presets()[preset].read_text())["skeleton"]
    by_name = Config.from_dict({"skeleton": {"include": preset}}).skeleton()
    written = Config.from_dict({"skeleton": spelled}).skeleton()
    assert _skeleton_signature(by_name) == _skeleton_signature(written)
    assert by_name.n_points == 38


def test_included_keys_are_overridden_wholesale():
    cfg = Config.from_dict(
        {"skeleton": {"include": "fly38", "colors": {"neck": "#ffffff"}}}
    )
    colors = dict(zip(cfg.skeleton().point_names, cfg.skeleton().point_colors))
    assert colors["neck"] == "#ffffff"
    # Wholesale, not merged: the override replaces the included colors entirely, so
    # every other point falls back to the colormap.
    assert colors["lf_claw"] != "#0f7399"


def test_a_self_contained_table_is_left_alone():
    """A config that spells out its points keeps its meaning -- `name` stays free text."""
    cfg = Config.from_dict({"skeleton": {"name": "mine", "points": ["a", "b", "c"]}})
    assert cfg.skeleton().n_points == 3


def test_include_resolves_next_to_its_config(tmp_path):
    (tmp_path / "sk").mkdir()
    (tmp_path / "sk" / "mine.toml").write_text(
        '[skeleton]\nname = "mine"\npoints = ["a", "b"]\n'
    )
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[skeleton]\ninclude = "sk/mine.toml"\n')
    assert Config.from_toml(cfg_path).skeleton().name == "mine"


def test_an_unknown_include_names_the_skeletons_that_exist():
    with pytest.raises(ValueError, match=r"neither a packaged skeleton.*fly38"):
        Config.from_dict({"skeleton": {"include": "fly99"}})


def test_a_missing_include_says_where_it_looked(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[skeleton]\ninclude = "nope.toml"\n')
    with pytest.raises(ValueError, match="readable file"):
        Config.from_toml(cfg_path)


def test_a_file_without_a_skeleton_table_is_refused(tmp_path):
    (tmp_path / "notaskeleton.toml").write_text("[cameras]\n")
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[skeleton]\ninclude = "notaskeleton.toml"\n')
    with pytest.raises(ValueError, match="carries no .skeleton. table"):
        Config.from_toml(cfg_path)


@pytest.mark.parametrize(
    "key,advice",
    [
        ("point_names", "points"),
        ("limb_points", "edges"),
        ("limb_palette", r"\[skeleton.colors\]"),
        ("file", "include"),
    ],
)
def test_a_v1_skeleton_key_is_refused_by_name(key, advice):
    """Named, never ignored: silence is what a reader would read as "still honored"."""
    with pytest.raises(ValueError, match=advice):
        Config.from_dict({"skeleton": {key: "whatever"}})


def test_the_resolved_points_reach_the_fingerprint():
    """``include`` is a reference in the snapshot but the RESOLVED names are cached on.

    Otherwise a skeleton file edited between releases would leave every cached run
    believing a skeleton it no longer routes.
    """
    from deeperfly.pipeline.fingerprint import _skeleton_digest

    cfg = Config.from_dict({"skeleton": {"include": "fly38"}})
    assert _skeleton_digest(cfg)["point_names"][:2] == [
        "lf_thorax_coxa",
        "lf_coxa_trochanter",
    ]


# -- model class defaults -----------------------------------------------------


@pytest.mark.parametrize("alias", sorted(CLASS_ALIASES))
def test_every_registered_class_has_defaults(alias):
    d = class_defaults(alias, 38)
    assert d["input_size"] == (256, 512)
    assert d["n_out_channels"] in (19, 38)


def test_dense_classes_take_their_channel_count_from_the_skeleton():
    """Every shipped class is dense, so its channel count IS the skeleton's point count."""
    assert class_defaults("hrnet", 38)["n_out_channels"] == 38
    assert class_defaults("mvt", 12)["n_out_channels"] == 12
    # An alias resolves to the same defaults as its canonical name.
    assert class_defaults("multiview_transformer", 38) == class_defaults("mvt", 38)


def test_an_unknown_class_is_refused_and_lists_the_real_ones():
    """Refused here rather than left to a fallback.

    A fallback's defaults are some other network's: a typo'd class used to inherit
    DeepFly2D's 19 channels and 0.22 mean, then fail at load with a channel-count
    mismatch -- which says nothing about the word that was actually wrong.
    """
    with pytest.raises(ValueError, match="unknown detector class 'hrnett'"):
        class_defaults("hrnett", 38)
    with pytest.raises(ValueError, match="mvt"):
        class_defaults("hourglass", 38)  # retired in 0.2


def test_mvt_pins_float32_without_the_config_saying_so():
    assert class_defaults("mvt")["precision"] == "float32"
    assert class_defaults("hrnet")["precision"] is None


def _dense_config(**pose2d):
    return Config.from_dict(
        {
            "skeleton": {"include": "fly38"},
            "cameras": {
                v: {"azimuth_deg": az, "distance": 100.0, "focal_length_px": 1.0}
                for v, az in zip(CAMERA_NAMES, AZIMUTHS_DEG)
            },
            "pose2d": pose2d,
        }
    )


def test_two_keys_are_a_complete_dense_plan():
    """`class` and `weights`, and the whole detection plan follows from the cameras."""
    plan = _dense_config(**{"class": "mvt", "weights": "x.pth"}).detection_plan()
    spec = plan.models["mvt"]
    assert (spec.input_size, spec.mean, spec.n_out_channels, spec.precision) == (
        (256, 512),
        0.0,
        38,
        "float32",
    )
    assert [pw.name for pw in plan.pathways] == CAMERA_NAMES
    assert plan.visibility_mask().all(), "a dense plan leaves no cell unobserved"


def test_an_explicit_key_still_wins_over_the_class():
    plan = _dense_config(
        **{"class": "mvt", "weights": "x.pth", "precision": "float16"}
    ).detection_plan()
    assert plan.models["mvt"].precision == "float16"


# -- weights resolution -------------------------------------------------------


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    d = tmp_path / "models"
    d.mkdir()
    (d / "detector.pth").write_bytes(b"weights")
    monkeypatch.setenv(download.MODELS_ENV, str(d))
    return d


def test_a_bare_filename_is_found_on_the_search_path(models_dir):
    got = download.resolve_weights("detector.pth", cls="mvt", model_name="m")
    assert got == models_dir / "detector.pth"


def test_a_path_is_used_as_written(models_dir):
    explicit = models_dir / "detector.pth"
    assert (
        download.resolve_weights(str(explicit), cls="mvt", model_name="m") == explicit
    )


def test_a_relative_path_is_not_searched(tmp_path, models_dir, monkeypatch):
    """`./detector.pth` means *here*, not "look on the path" -- the separator decides."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="no detector checkpoint at"):
        download.resolve_weights("./detector.pth", cls="mvt", model_name="m")


def test_an_empty_value_defers_to_the_class(models_dir):
    assert download.resolve_weights("", cls="mvt", model_name="m") is None
    assert download.resolve_weights(None, cls="hourglass", model_name="m") is None


def test_a_missing_bare_name_lists_every_directory_searched(models_dir):
    with pytest.raises(SystemExit) as e:
        download.resolve_weights("absent.pth", cls="mvt", model_name="dense38mv")
    message = str(e.value)
    assert str(models_dir) in message
    assert str(download.cache_dir()) in message
    assert "dense38mv" in message and download.MODELS_ENV in message


def test_multiple_search_directories_are_honored_in_order(tmp_path, monkeypatch):
    first, second = tmp_path / "a", tmp_path / "b"
    for d in (first, second):
        d.mkdir()
    (second / "only_in_b.pth").write_bytes(b"w")
    (first / "both.pth").write_bytes(b"w")
    (second / "both.pth").write_bytes(b"w")
    monkeypatch.setenv(download.MODELS_ENV, os.pathsep.join([str(first), str(second)]))
    assert download.resolve_weights("only_in_b.pth", cls="mvt", model_name="m") == (
        second / "only_in_b.pth"
    )
    assert download.resolve_weights("both.pth", cls="mvt", model_name="m") == (
        first / "both.pth"
    )


def test_a_class_with_no_weights_explains_the_setup():
    """Nothing auto-provisions, so this message IS the first-run experience.

    It has to name the environment variable, the model whose table is short, and where the
    released checkpoints are documented -- "no weights" alone leaves a new user with
    nothing to do next.
    """
    message = str(download.missing_weights("mvt", "dense38mv"))
    assert download.MODELS_ENV in message
    assert "dense38mv" in message
    assert "trained per project" in message
    assert "configuration.md" in message


# -- the visualization grid ---------------------------------------------------


def _grid_config(video, cameras=CAMERA_NAMES, *, cell=(480, 240)):
    azimuths = dict(zip(CAMERA_NAMES, AZIMUTHS_DEG))
    return Config.from_dict(
        {
            "skeleton": {"include": "fly38"},
            "cameras": {
                v: {
                    "azimuth_deg": azimuths.get(v, 0.0),
                    "distance": 100.0,
                    "focal_length_px": 1.0,
                    "video": rf"{v}\.mp4",
                }
                for v in cameras
            },
            "pose2d": {"class": "hrnet", "weights": "x.pth"},
            "visualization": {
                "default_video": {"cell": list(cell)},
                "videos": {"v": video},
            },
        }
    )


def _panel_signature(spec):
    return [(p.plot, p.view, p.x0, p.y0, p.stage) for p in spec.panels]


def test_a_grid_is_footage_plus_one_panel_per_layer_per_cell():
    """What the grid replaces: fourteen hand-written panels with hand-computed offsets."""
    spec = _grid_config(
        {
            "grid": [["rf", "f", "lf"], ["rm", "bird", "lm"]],
            "layers": [{"draw": "skeleton_3d", "stage": "triangulation"}],
        }
    ).videos[0]
    assert _panel_signature(spec) == [
        ("imshow", "rf", 0, 0, None),
        ("skeleton_3d", "rf", 0, 0, "triangulation"),
        ("imshow", "f", 480, 0, None),
        ("skeleton_3d", "f", 480, 0, "triangulation"),
        ("imshow", "lf", 960, 0, None),
        ("skeleton_3d", "lf", 960, 0, "triangulation"),
        ("imshow", "rm", 0, 240, None),
        ("skeleton_3d", "rm", 0, 240, "triangulation"),
        # `bird` is no camera, so it gets no footage under its overlay.
        ("skeleton_3d", "bird", 480, 240, "triangulation"),
        ("imshow", "lm", 960, 240, None),
        ("skeleton_3d", "lm", 960, 240, "triangulation"),
    ]


def test_layer_order_is_draw_order():
    """The whole point of layers: a dashed reference UNDER a solid fit, in one video."""
    spec = _grid_config(
        {
            "grid": [["rf"]],
            "layers": [
                {"draw": "skeleton_3d", "stage": "triangulation", "line_dash": [4, 9]},
                {"draw": "skeleton_3d", "stage": "postprocess"},
            ],
        }
    ).videos[0]
    assert [(p.plot, p.stage) for p in spec.panels] == [
        ("imshow", None),
        ("skeleton_3d", "triangulation"),
        ("skeleton_3d", "postprocess"),
    ]
    # Footage is laid down ONCE, under the first layer -- again per layer it would paint
    # over the layer beneath.
    assert sum(p.plot == "imshow" for p in spec.panels) == 1
    assert spec.panels[1].options["line_dash"] == [4, 9]
    assert "line_dash" not in spec.panels[2].options


def test_footage_false_drops_the_picture_under_every_cell():
    spec = _grid_config(
        {"grid": [["rf"]], "layers": [{"draw": "skeleton_3d"}], "footage": False}
    ).videos[0]
    assert [p.plot for p in spec.panels] == ["skeleton_3d"]


def test_a_non_camera_cell_is_skipped_entirely_in_a_2d_video():
    """There is no picture to draw 2D detections on, so the tile is left empty."""
    spec = _grid_config(
        {"grid": [["rf", "bird"]], "layers": [{"draw": "skeleton_2d"}]}
    ).videos[0]
    assert {p.view for p in spec.panels} == {"rf"}


@pytest.mark.parametrize("gap", ["", "-", "."])
def test_a_gap_cell_leaves_its_tile_empty(gap):
    spec = _grid_config(
        {"grid": [["rf", gap, "lf"]], "layers": [{"draw": "skeleton_3d"}]}
    ).videos[0]
    assert [(p.view, p.x0) for p in spec.panels if p.plot == "skeleton_3d"] == [
        ("rf", 0),
        ("lf", 960),
    ]


def test_the_cell_size_can_be_stated_per_video():
    cfg = _grid_config(
        {
            "cell": [100, 50],
            "grid": [["rf", "lf"], ["rm", "lm"]],
            "layers": [{"draw": "skeleton_3d"}],
        }
    )
    offsets = [(p.x0, p.y0) for p in cfg.videos[0].panels if p.plot == "skeleton_3d"]
    assert offsets == [(0, 0), (100, 0), (0, 50), (100, 50)]


def test_a_grid_with_an_unknown_draw_op_is_an_error():
    with pytest.raises(ValueError, match="unknown draw op"):
        _grid_config({"grid": [["rf"]], "layers": [{"draw": "nope"}]}).videos


def test_a_grid_with_no_cell_size_says_so():
    cfg = _grid_config({"grid": [["rf"]], "layers": [{"draw": "skeleton_3d"}]})
    del cfg.data["visualization"]["default_video"]
    with pytest.raises(ValueError, match="cell size"):
        cfg.videos


# -- the packaged config it all adds up to ------------------------------------


def test_the_packaged_config_is_a_dense_plan_over_the_whole_skeleton():
    cfg = Config.default()
    assert cfg.skeleton().name == "fly38"
    plan = cfg.detection_plan()
    # One pathway per camera: no mirrored twins, so no [pose2d.output_points] table.
    assert [p.name for p in plan.pathways] == plan.view_names
    assert "output_points" not in cfg.data["pose2d"]
    assert plan.visibility_mask().all()


def test_the_packaged_config_states_only_what_it_changes():
    """Every stage key it writes must differ from that stage's dataclass default.

    This is the property that keeps the file short as defaults move: a key that drifts
    back to its default shows up here as a line to delete, rather than as one more line
    nobody reads.
    """
    from deeperfly import config_schema as cs

    cfg = Config.default()
    redundant = {}
    for section in cs.SECTIONS:
        defaults = {f.name: f.default for f in cs.describe(section).fields}
        for key, (value, is_default) in cs.effective(cfg, section).items():
            if not is_default and value == defaults[key]:
                redundant.setdefault(section, []).append(key)
    assert not redundant, f"these keys restate their own default: {redundant}"


def test_the_packaged_config_still_snapshots_byte_exactly(tmp_path):
    cfg = Config.from_toml(DEFAULT_CONFIG_PATH)
    cfg.save_snapshot(tmp_path)
    assert (tmp_path / "config.toml").read_text() == Path(
        DEFAULT_CONFIG_PATH
    ).read_text()


def test_the_packaged_config_names_its_checkpoint_portably(tmp_path, monkeypatch):
    """The shipped `weights` is a bare FILENAME, never a path.

    There is no dense model to auto-download, so the config has to name one -- but naming
    `/mnt/.../mvt_r28_pad48_gray_fly38.pth` would make the packaged default a fact about one
    machine's mount. A bare name is a fact about which model to use, resolved per machine
    against $DEEPERFLY_MODELS, and the failure when it is not there is the actionable one.

    The search path is EMPTIED first. What is under test is a property of the config and of
    the failure message, not of whether this machine happens to have the checkpoint: without
    this the test passes only for a developer who has never installed the default model, and
    fails the moment someone does.
    """
    monkeypatch.setenv(download.MODELS_ENV, str(tmp_path / "nowhere"))
    monkeypatch.setattr(download, "cache_dir", lambda: tmp_path / "empty-cache")
    (tmp_path / "empty-cache").mkdir()

    spec = Config.default().detection_plan().models["mvt"]
    assert spec.weights == "mvt_r28_pad48_gray_fly38.pth"
    assert "/" not in spec.weights and not Path(spec.weights).is_absolute()
    with pytest.raises(SystemExit) as e:
        download.resolve_weights(spec.weights, cls=spec.cls, model_name=spec.name)
    assert download.MODELS_ENV in str(e.value)


# -- the pathway's model, hoisted one level ----------------------------------


def _plan_config(pose2d: dict) -> Config:
    """A minimal two-view config wrapped around a ``[pose2d]`` table."""
    return Config.from_dict(
        {
            "sources": [{"name": "vid_a"}, {"name": "vid_b"}],
            "skeleton": {"include": "fly38"},
            "cameras": {
                "a": {"azimuth_deg": 0, "focal_length_px": 1.0, "distance": 1.0},
                "b": {"azimuth_deg": 90, "focal_length_px": 1.0, "distance": 1.0},
            },
            "pose2d": pose2d,
        }
    )


DENSE = {"name": "m", "class": "mvt", "weights": "w.pth"}
OTHER = {"name": "m2", "class": "hrnet", "weights": "w2.pth"}
BARE = [{"name": "a", "source": "vid_a"}, {"name": "b", "source": "vid_b"}]


# -- every config in the repo, held to the same shape -------------------------

REPO = Path(__file__).resolve().parents[1]
#: Every config a user is meant to read or run, packaged and staged alike.
ALL_CONFIGS = [DEFAULT_CONFIG_PATH, *sorted(REPO.glob("examples/*/config.toml"))]


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: Path(p).parent.name)
def test_every_config_states_only_what_it_changes(path):
    """The rule that keeps them short applies to the staged examples too.

    Six recordings each carrying its own copy of `[triangulation]`, `[eks]` and
    `[pictorial_structures]` at their defaults is how they reached 600 lines, and it is
    also how a default change silently stops reaching them.
    """
    from deeperfly import config_schema as cs

    cfg = Config.from_toml(path)
    redundant = {}
    for section in cs.SECTIONS:
        defaults = {f.name: f.default for f in cs.describe(section).fields}
        for key, (value, is_default) in cs.effective(cfg, section).items():
            if not is_default and value == defaults[key]:
                redundant.setdefault(section, []).append(key)
    assert not redundant, f"these keys restate their own default: {redundant}"


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: Path(p).parent.name)
def test_every_config_is_dense_over_the_whole_skeleton(path):
    cfg = Config.from_toml(path)
    plan = cfg.detection_plan()
    assert [p.name for p in plan.pathways] == plan.view_names
    assert plan.visibility_mask().all()
    assert cfg.skeleton().name == "fly38"
    # One detector for the whole run, named by its class.
    assert len(plan.models) == 1
