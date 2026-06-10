"""The ``run`` command worker: resolve recordings and drive the pipeline per run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import typer
from rich.text import Text

from ..config import Config
from ..pipeline import run_recording
from ..recordings import Recording, plan_outdirs, resolve_recordings
from .console import _rich_progress, console, log


def _cmd_run(args: argparse.Namespace) -> None:
    """Run the pipeline for each recording the inputs resolve to.

    ``args.inputs`` is one or more recording directories and/or wildcard/recursive
    patterns (see :func:`deeperfly.recordings.resolve_recordings`); each recording's
    output directory comes from :func:`deeperfly.recordings.plan_outdirs` (a
    name-collision fallback is confirmed with the user up front, before any run
    starts). Each resolved recording is handed to
    :func:`deeperfly.pipeline.run_recording` with the Rich-backed progress factory.
    In a batch each run is independent and a failure is logged and skipped; a
    single recording fails fast.

    Parameters
    ----------
    args
        The ``run`` namespace (``inputs``, ``recursive``, ``config``, ``output``,
        ``overwrite``).

    Raises
    ------
    SystemExit
        If no inputs are given, the collision fallback cannot be confirmed
        non-interactively, or (in a batch) if any recording failed.
    """
    if not args.inputs:
        raise SystemExit("give at least one recording directory (or wildcard) to run")
    # Only used to recognize recording directories while resolving the inputs; each
    # run then resolves its own config against its output dir (Config.read_for_run).
    discovery_config = (
        Config.from_toml(args.config) if args.config else Config.default()
    )
    found = resolve_recordings(
        args.inputs, recursive=args.recursive, config=discovery_config
    )
    plan = plan_outdirs([d for d, _ in found], args.output)
    if plan.mirror_confirm:
        if not sys.stdin.isatty():
            raise SystemExit(
                plan.mirror_confirm + "\nre-run interactively to confirm this "
                "layout, or pick distinct recording names / a different -o"
            )
        typer.confirm(plan.mirror_confirm + "\nproceed?", abort=True)
    recordings = [
        Recording(src, outdir) for (_, src), outdir in zip(found, plan.outdirs)
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
                args.config,
                rec.outdir,
                sources=rec.sources,  # footage resolved up front by discovery
                overwrite=getattr(args, "overwrite", None),
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
