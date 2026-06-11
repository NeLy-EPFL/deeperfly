"""The Typer application: command definitions and the ``main`` entry point."""

from __future__ import annotations

import argparse
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from ..config import STAGES
from ..pipeline import _OVERWRITE_ALL
from .console import _configure_logging
from .gui import _cmd_gui
from .report import _cmd_doctor, _cmd_init, _cmd_inspect
from .run import _cmd_run

# -- typer app ---------------------------------------------------------------
#
# The CLI is built with Typer: typed signatures over click, with usage/--help
# rendered through rich. Each command declares its options, configures logging,
# then hands an argparse-style namespace to the matching ``_cmd_*`` worker; the
# workers stay namespace-driven so they remain callable as a library and from the
# tests. Constrained options are ``str``-valued Enums; commands pass their
# ``.value``, so workers keep receiving plain strings.


class LogLevel(str, Enum):
    """``--log-level`` choices, shared by every subcommand. A ``str`` enum, so each
    member's ``.value`` is the name :func:`_configure_logging` expects."""

    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"
    critical = "critical"


#: The shared ``--log-level`` option, declared once as a reusable parameter
#: annotation and spread across every command. ``case_sensitive=False`` accepts
#: INFO/Info/info.
LogLevelOption = Annotated[
    LogLevel,
    typer.Option(
        case_sensitive=False,
        help="logging verbosity; 'warning' or higher hides the per-stage logs and "
        "the progress bar",
    ),
]

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


@app.command()
def init(
    output: Annotated[
        str, typer.Argument(help="destination (defaults to config.toml)")
    ] = "config.toml",
    force: Annotated[
        bool, typer.Option("--force", help="overwrite an existing file")
    ] = False,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Write a default config.toml to edit (destination defaults to config.toml)."""
    _configure_logging(log_level.value)
    _cmd_init(argparse.Namespace(output=output, force=force))


@app.command()
def run(
    inputs: Annotated[
        list[Path],
        typer.Argument(
            metavar="INPUT...",
            help="one or more recording dirs or wildcard patterns (per-camera videos "
            "or image folders); several inputs / a wildcard run as a batch",
        ),
    ],
    recursive: Annotated[
        bool,
        typer.Option(
            "-r",
            "--recursive",
            help="treat each INPUT as a parent directory and run every recording "
            "nested under it (each subdirectory holding the configured per-camera "
            "footage)",
        ),
    ] = False,
    config: Annotated[
        str | None,
        typer.Option(
            "-c",
            "--config",
            help="merged config TOML (from 'deeperfly init'); "
            "defaults to the packaged default config",
        ),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option(
            "-o",
            "--output-dir",
            help="output directory (default: <input>/deeperfly_outputs; created if "
            "missing). For a batch of several recordings: end it with '/' to "
            "collect one subdirectory per recording under it (colliding names "
            "fall back to mirroring the input tree, after confirmation); a "
            "relative name without '/' creates that directory inside each "
            "recording.",
        ),
    ] = None,
    overwrite: Annotated[
        list[str] | None,
        typer.Option(
            "--overwrite",
            help="force stages to recompute even though their config is unchanged "
            "(config changes are detected automatically). A bare --overwrite "
            "recomputes everything; name stages to recompute only those (e.g. "
            "--overwrite pose2d visualization). Recomputing a stage also "
            "refreshes the stages after it.",
        ),
    ] = None,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """detect 2D -> reconstruct 3D -> visualization (the enabled stages, reusing cache).

    INPUT is one or more recording directories (per-camera videos or image folders)
    and/or wildcards matching several (e.g. 'fly*' -> fly1/, fly2/, ...), each run
    in turn. Several inputs or a wildcard run as a batch, keeping only the valid
    recordings. With -r/--recursive, each INPUT is a parent directory and every
    recording nested under it is run in turn.

    A stage already in the output dir is reused when its config is unchanged, so
    re-running a finished recording is a cheap no-op -- and editing the config
    recomputes exactly the affected stages (tweak the triangulation or the videos
    and re-run; the slow 2D detection is reused). Pass --overwrite to force a
    recompute anyway: bare redoes every stage, or name stages to redo only those
    (plus the stages after them).

    Everything else is set in the config: the do_<stage> toggles choose which stages
    run, alongside fps, background and each stage's parameters. -c wins when given;
    with no -c, a run reuses the config.toml already in the output dir, else the
    packaged default.
    """
    _configure_logging(log_level.value)
    _cmd_run(
        argparse.Namespace(
            inputs=inputs,
            recursive=recursive,
            config=config,
            output=output,
            overwrite=overwrite,
            log_level=log_level.value,
        )
    )


@app.command()
def inspect(
    input: Annotated[str, typer.Argument(help="path to a result .h5 file")],
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Print a summary of a result .h5 file."""
    _configure_logging(log_level.value)
    _cmd_inspect(argparse.Namespace(input=input))


@app.command()
def doctor(log_level: LogLevelOption = LogLevel.info) -> None:
    """Report installation/runtime: accelerators, frame I/O, weights."""
    _configure_logging(log_level.value)
    _cmd_doctor(argparse.Namespace())


@app.command()
def gui(
    path: Annotated[
        str,
        typer.Argument(
            help="a results.h5 file, or a directory containing one "
            "(e.g. <recording>/deeperfly_outputs)"
        ),
    ],
    footage_dir: Annotated[
        str | None,
        typer.Option(
            "--footage-dir",
            help="directory to search for the footage if the paths recorded in "
            "results.h5 no longer resolve",
        ),
    ] = None,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Open the interactive viewer/corrector on a result (needs the 'gui' extra).

    View every camera with its 2D skeleton overlay, drag keypoints to correct
    the 2D pose, or switch to 3D mode to drag a reprojected 3D point (the other
    views update live). Corrections are written to a corrections.h5 sidecar and
    never modify results.h5. Install the viewer with 'pip install deeperfly[gui]'.
    """
    _configure_logging(log_level.value)
    _cmd_gui(argparse.Namespace(path=path, footage_dir=footage_dir))


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
