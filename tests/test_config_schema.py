"""The config described from the code -- the guard being that it cannot drift from it.

The value of deriving the schema (rather than writing one) is that a new ``*Params`` field
shows up everywhere with no second place to update. So the tests that matter are the ones
that would fail if someone *did* introduce a second source of truth: every describable
section reflects, every field is accounted for, and validation is still ``Config``'s.
"""

from __future__ import annotations

import json

import pytest

from deeperfly.config import Config, TriangulationParams
from deeperfly.config_schema import (
    SECTIONS,
    describe,
    effective,
    sections,
    stage_flags_spec,
)


def test_every_section_reflects_and_covers_its_dataclass():
    """No field may be missing: a form that silently omits a knob is worse than no form."""
    import dataclasses

    for name, cls in SECTIONS.items():
        spec = describe(name)
        assert spec.name == name
        assert {f.name for f in spec.fields} == {
            f.name for f in dataclasses.fields(cls)
        }


def test_defaults_come_from_the_dataclass_not_a_copy():
    spec = describe("triangulation")
    got = {f.name: f.default for f in spec.fields}
    want = {
        f.name: getattr(TriangulationParams(), f.name)
        for f in __import__("dataclasses").fields(TriangulationParams)
    }
    assert got == want


def test_the_prose_is_lifted_from_the_docstring():
    """The docstrings are the good part; re-authoring them as help text would be a loss."""
    spec = describe("inverse_kinematics")
    docs = {f.name: f.doc for f in spec.fields}
    # InverseKinematicsParams explains *why* damping defaults high -- the abdomen's five
    # near-collinear hinges. That reasoning is what a form's help text should carry.
    assert "abdomen" in docs["damping"]
    assert "collinear" in docs["damping"]


def test_rst_markup_is_stripped_from_the_prose():
    """Double backticks and :role:`targets` are noise in a table or an HTML field."""
    for name in sections():
        spec = describe(name)
        assert "``" not in spec.doc
        for f in spec.fields:
            assert "``" not in f.doc
            assert ":func:" not in f.doc and ":class:" not in f.doc


def test_an_unknown_section_says_which_ones_exist():
    with pytest.raises(KeyError, match="describable sections are"):
        describe("cameras")


def test_effective_separates_what_was_set_from_what_defaulted():
    """The question a config file cannot answer."""
    config = Config.from_dict({"triangulation": {"method": "dlt"}})
    values = effective(config, "triangulation")
    assert values["method"] == ("dlt", False)  # set
    assert values["min_inliers"][1] is True  # default
    assert values["min_inliers"][0] == TriangulationParams().min_inliers


def test_effective_reports_an_unknown_key_through_configs_own_validator():
    """Validation is not reimplemented, so a typo reads the same however it arrived."""
    config = Config.from_dict({"triangulation": {"methd": "dlt"}})
    with pytest.raises(ValueError, match="unknown key"):
        effective(config, "triangulation")


def test_the_packaged_default_describes_cleanly():
    """Every section of the shipped config must reflect without raising."""
    config = Config.default()
    for name in sections():
        assert effective(config, name)


def test_the_stage_flags_are_described_even_though_they_are_generated():
    """'Which stages run' is the most-changed thing in the config; it must not be missing."""
    spec = stage_flags_spec()
    names = {f.name for f in spec.fields}
    assert "do_pose2d" in names and "do_triangulation" in names
    assert all(f.type == "bool" for f in spec.fields)


def test_a_spec_is_json_serializable():
    """The GUI forms are built on this, so it has to cross the wire as plain data."""
    for name in sections():
        json.dumps(describe(name).as_dict())
    json.dumps(stage_flags_spec().as_dict())


# -- the CLI -------------------------------------------------------------------


def test_cli_show_marks_set_keys(tmp_path, capsys):
    from deeperfly import cli

    cfg = tmp_path / "c.toml"
    cfg.write_text('[triangulation]\nmethod = "dlt"\n')
    cli.main(["config", "show", "triangulation", "-c", str(cfg)])
    out = capsys.readouterr().out
    assert "dlt" in out
    assert "set" in out


def test_cli_show_all_sections_by_default(capsys):
    from deeperfly import cli

    cli.main(["config", "show"])
    out = capsys.readouterr().out
    for name in ("pipeline", "triangulation", "annotation"):
        assert name in out


def test_cli_show_an_undescribable_section_says_so(capsys):
    from deeperfly import cli

    with pytest.raises(SystemExit, match="describable sections"):
        cli.main(["config", "show", "cameras"])


def test_cli_set_writes_and_validates(tmp_path, capsys):
    from deeperfly import cli

    cfg = tmp_path / "c.toml"
    cfg.write_text("# my config\n")
    cli.main(
        [
            "config",
            "set",
            "triangulation.method",
            "dlt",
            "-c",
            str(cfg),
            "--log-level",
            "error",
        ]
    )
    text = cfg.read_text()
    assert "# my config" in text  # comments survive: it appends, never rewrites
    assert Config.from_toml(cfg).triangulation.method == "dlt"


def test_cli_set_coerces_types(tmp_path):
    from deeperfly import cli

    cfg = tmp_path / "c.toml"
    cfg.write_text("")
    cli.main(
        [
            "config",
            "set",
            "triangulation.min_inliers",
            "3",
            "-c",
            str(cfg),
            "--log-level",
            "error",
        ]
    )
    assert Config.from_toml(cfg).triangulation.min_inliers == 3

    cfg2 = tmp_path / "d.toml"
    cfg2.write_text("")
    cli.main(
        [
            "config",
            "set",
            "annotation.undistort_before_solve",
            "true",
            "-c",
            str(cfg2),
            "--log-level",
            "error",
        ]
    )
    assert Config.from_toml(cfg2).annotation.undistort_before_solve is True


def test_cli_set_refuses_an_unknown_key(tmp_path):
    from deeperfly import cli

    cfg = tmp_path / "c.toml"
    cfg.write_text("")
    with pytest.raises(SystemExit, match="has no key"):
        cli.main(
            [
                "config",
                "set",
                "triangulation.methd",
                "dlt",
                "-c",
                str(cfg),
                "--log-level",
                "error",
            ]
        )
    assert cfg.read_text() == ""  # nothing written


def test_cli_set_will_not_edit_the_packaged_default():
    """``-c`` is required, so the packaged config cannot be edited in place.

    Typer refuses with a usage error before the worker runs, which is the better place for
    it -- the worker keeps its own guard for library callers.
    """
    import argparse

    from deeperfly import cli
    from deeperfly.cli.config import _cmd_config_set

    with pytest.raises(SystemExit) as usage:
        cli.main(
            ["config", "set", "triangulation.method", "dlt", "--log-level", "error"]
        )
    assert usage.value.code == 2  # a usage error, not a traceback

    with pytest.raises(SystemExit, match="refusing to edit the packaged default"):
        _cmd_config_set(
            argparse.Namespace(key="triangulation.method", value="dlt", config=None)
        )


def test_cli_set_refuses_when_the_table_already_exists(tmp_path):
    """Appending a bare key after an existing table would reparent it silently."""
    from deeperfly import cli

    cfg = tmp_path / "c.toml"
    cfg.write_text('[triangulation]\nmethod = "ransac"\n')
    with pytest.raises(SystemExit, match="already exists"):
        cli.main(
            [
                "config",
                "set",
                "triangulation.method",
                "dlt",
                "-c",
                str(cfg),
                "--log-level",
                "error",
            ]
        )


def test_cli_set_needs_a_dotted_key(tmp_path):
    from deeperfly import cli

    cfg = tmp_path / "c.toml"
    cfg.write_text("")
    with pytest.raises(SystemExit, match="SECTION.KEY"):
        cli.main(
            ["config", "set", "method", "dlt", "-c", str(cfg), "--log-level", "error"]
        )


# -- the GUI's foundation -------------------------------------------------------


def test_the_schema_endpoint_serves_every_section(result, tmp_path):
    """The GUI forms are generated from this, so it must cover what it claims to."""
    from fastapi.testclient import TestClient

    from deeperfly.gui.readers import FrameSource
    from deeperfly.gui.server import create_app
    from deeperfly.gui.session import Session
    from deeperfly.gui.state import EditorState

    session = Session.build(
        EditorState.from_result(result),
        FrameSource({}),
        results_path=str(tmp_path / "results.h5"),
        labels_path=tmp_path / "labels.h5",
    )
    client = TestClient(create_app(session))

    payload = client.get("/api/schema").json()
    names = [s["name"] for s in payload["sections"]]
    assert "pipeline" in names
    for name in sections():
        assert name in names
    # The open-ended sections are NAMED, so a form builder can say "this needs the file"
    # rather than render nothing and look broken.
    assert "cameras" in payload["undescribable"]
    assert "pose2d.output_points" in payload["undescribable"]

    one = client.get("/api/schema", params={"section": "triangulation"}).json()
    assert one["name"] == "triangulation"
    assert any(f["name"] == "method" for f in one["fields"])

    missing = client.get("/api/schema", params={"section": "cameras"})
    assert missing.status_code == 404
    assert "describable sections" in missing.json()["detail"]
