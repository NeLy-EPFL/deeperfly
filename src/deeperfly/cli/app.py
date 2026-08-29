"""The Typer application: the command list and the ``main`` entry point.

Every command lives with its implementation, in the module named for its topic
(:mod:`deeperfly.cli.run`, :mod:`deeperfly.cli.project`, ...). What is here is the
assembly: the root app, the order the commands appear in ``--help``, and ``main``.

Commands used to be declared twice -- a Typer signature here that repacked its options
into an ``argparse.Namespace``, and a ``_cmd_*`` worker there that unpacked them again by
name. That adapter outlived the argparse parser it was written for, so the options are
now declared once, in the function that uses them.
"""

from __future__ import annotations

import sys

import typer

from ..config import STAGES
from ..pipeline import _OVERWRITE_ALL
from . import calibrate, gui, merge, report, suggest
from . import run as run_mod
from .autocrop import auto_crop
from .bind import ik_app
from .calibration import calibration_app
from .config import config_app
from .project import project_app

# The CLI is built with Typer: typed signatures over click, with usage and --help
# rendered through rich. Constrained options are `str`-valued Enums (see
# `deeperfly.cli.console.LogLevel`), so a command receives a plain string from `.value`.
app = typer.Typer(
    add_completion=False,  # no shell-completion options; keep the surface minimal
    rich_markup_mode="rich",  # rich-rendered (boxed, colored) usage and --help
    no_args_is_help=True,  # bare 'deeperfly' prints help instead of a usage error
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Markerless 3D pose estimation of tethered Drosophila from a multi-camera "
    "rig. 'deeperfly init' writes a config to edit; 'deeperfly run' detects 2D "
    "pose, reconstructs 3D and renders a video; 'deeperfly inspect' summarizes a "
    "result file; 'deeperfly doctor' reports the installation/runtime.",
)

# The order below is the order `deeperfly --help` lists them in, so it is grouped for a
# reader of that help -- the whole-recording verbs first, then the label verbs, then the
# rig -- rather than by which module each comes from.
app.command(name="auto-crop")(auto_crop)
app.command()(report.init)
app.command()(run_mod.run)
app.command()(report.inspect)
app.command()(report.repack)
app.command()(report.doctor)
app.command()(gui.gui)
app.command(name="labels-export")(gui.labels_export)
app.command(name="labels-absent")(gui.labels_absent)
app.command(name="labels-suggest")(suggest.labels_suggest)
app.command()(calibrate.calibrate)
app.command(name="labels-merge")(merge.labels_merge)

# Sub-apps own their own commands; each is declared in its module.
app.add_typer(ik_app, name="ik")
app.add_typer(project_app, name="project")
app.add_typer(config_app, name="config")
app.add_typer(calibration_app, name="calibration")


def _normalize_overwrite_argv(argv: list[str]) -> list[str]:
    """Let ``run``'s ``--overwrite`` take zero or more space-separated stage names.

    click options can't be variadic, so rewrite a bare ``--overwrite`` into
    ``--overwrite <_OVERWRITE_ALL>`` and ``--overwrite a b`` into the repeated
    ``--overwrite a --overwrite b`` the ``multiple=True`` option accepts. Only known
    stage names (:data:`STAGES`) are consumed after the flag, leaving the positional
    argument and later options untouched; ``--overwrite=...`` passes through as-is.

    Parameters
    ----------
    argv
        The raw argument vector (the subcommand first).

    Returns
    -------
    list of str
        ``argv`` with ``run``'s variadic ``--overwrite`` rewritten (unchanged for
        other subcommands).
    """
    if not (argv and argv[0] == "run"):
        return argv
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok != "--overwrite":
            out.append(tok)
            i += 1
            continue
        j = i + 1
        picked: list[str] = []
        while j < len(argv) and argv[j] in STAGES:
            picked.append(argv[j])
            j += 1
        for stage in picked or [_OVERWRITE_ALL]:
            out += ["--overwrite", stage]
        i = j
    return out


def main(argv: list[str] | None = None) -> None:
    """Entry point: parse ``argv`` (default ``sys.argv``) and dispatch a subcommand.

    Runs the Typer app in standalone mode so usage errors and ``--help`` render
    through rich, but swallows the ``SystemExit(0)`` of a clean exit so
    ``main([...])`` returns normally as a library / from the tests. Real failures
    still propagate. ``argv`` is normalized first
    (:func:`_normalize_overwrite_argv`).

    Parameters
    ----------
    argv
        The argument vector; defaults to ``sys.argv[1:]``.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    argv = _normalize_overwrite_argv(argv)
    command = typer.main.get_command(app)
    try:
        command(args=argv, prog_name="deeperfly")
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
