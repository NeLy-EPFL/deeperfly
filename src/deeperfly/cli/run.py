"""The ``run`` command worker: resolve recordings and drive the pipeline per run."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.text import Text

from ..config import Config
from ..pipeline import run_recording
from ..recordings import (
    Recording,
    find_recording,
    plan_outdirs,
    resolve_recordings,
)
from .console import (
    LogLevel,
    LogLevelOption,
    _configure_logging,
    _rich_progress,
    console,
    log,
)


def _footage_for_run(
    root: Path,
    discovered: dict[str, list[Path]],
    outdir: Path,
    cli_config: str | None,
) -> dict[str, list[Path]]:
    """Re-resolve a recording's footage against the config its run will actually use.

    Discovery runs before output directories are known, so it has to recognize recording
    directories with *some* config -- the packaged default when no ``-c`` is given. But
    the run then picks its own (``-c``, else the output dir's snapshot), and the two can
    declare different cameras. The packaged default is a seven-camera rig, so an
    eight-camera recording resolved through it came back missing its hind view, and the
    run went on to use that short map as its footage.

    That is invisible until something downstream needs a frame. With a cached ``pose2d``
    nothing opens the videos, so the loss surfaces only in the visualization stage, as a
    "the run resolved no footage" error advising you to pass the recording as the input --
    which you did. Re-resolving here means the run's footage always matches the run's
    config.

    Falls back to the discovered map when the run config recognizes nothing at ``root``
    (already warned about by :func:`~deeperfly.recordings.find_recording`), so a recording
    kept for its cached results is still kept.
    """
    try:
        run_config = Config.read_for_run(cli_config, outdir)
    except (OSError, ValueError):  # unreadable/invalid snapshot: keep what we found
        return discovered
    resolved = find_recording(root, run_config)
    return resolved if resolved else discovered


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
    log_level = log_level.value
    if not inputs:
        raise SystemExit("give at least one recording directory (or wildcard) to run")
    # Only used to RECOGNIZE recording directories while resolving the inputs; each run
    # then resolves its own config against its output dir (Config.read_for_run), and its
    # footage is re-resolved against that config below.
    discovery_config = Config.from_toml(config) if config else Config.default()
    found = resolve_recordings(inputs, recursive=recursive, config=discovery_config)
    plan = plan_outdirs([d for d, _ in found], output)
    if plan.mirror_confirm:
        if not sys.stdin.isatty():
            raise SystemExit(
                plan.mirror_confirm + "\nre-run interactively to confirm this "
                "layout, or pick distinct recording names / a different -o"
            )
        typer.confirm(plan.mirror_confirm + "\nproceed?", abort=True)
    recordings = [
        Recording(_footage_for_run(root, src, outdir, config), outdir)
        for (root, src), outdir in zip(found, plan.outdirs)
    ]
    batch = len(recordings) > 1
    if batch:
        log.info(
            "matched %d recordings (output dirs): %s",
            len(recordings),
            [str(r.outdir) for r in recordings],
        )

    failures: list[Path] = []
    for i, rec in enumerate(recordings, 1):
        if batch:
            console.rule(
                Text(f"{rec.outdir}  ({i}/{len(recordings)})", style="bold cyan")
            )
        try:
            run_recording(
                config,
                rec.outdir,
                sources=rec.sources,  # footage resolved up front by discovery
                overwrite=overwrite,
                progress=_rich_progress,
            )
        except (Exception, SystemExit) as exc:  # noqa: BLE001
            if not batch:
                raise  # a single recording fails fast (unchanged behavior)
            log.error("recording %s failed: %s", rec.outdir, exc)
            failures.append(rec.outdir)

    if failures:
        raise SystemExit(
            f"{len(failures)}/{len(recordings)} recordings failed: "
            + ", ".join(str(r) for r in failures)
        )
