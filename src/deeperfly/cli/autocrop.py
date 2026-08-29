"""The ``auto-crop`` command worker: search a recording's detector crops and report them.

The same search the ``pose2d`` stage runs (:mod:`deeperfly.pose2d.autocrop`), reachable on
its own so a box can be measured, *looked at*, and only then trusted -- or frozen into the
config as an explicit window, which is what a recording that is about to be labelled wants:
a number someone can read, not a search that reruns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ..config import Config
from ..recordings import Recording, plan_outdirs, resolve_recordings
from .console import LogLevel, LogLevelOption, _configure_logging, console, log


def auto_crop(
    inputs: Annotated[
        list[Path],
        typer.Argument(
            metavar="INPUT...",
            help="one or more recording dirs or wildcard patterns",
        ),
    ],
    config: Annotated[
        str | None,
        typer.Option(
            "-c", "--config", help="config TOML declaring the automatic crop(s)"
        ),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option("-o", "--output-dir", help="output directory (as for 'run')"),
    ] = None,
    recursive: Annotated[
        bool,
        typer.Option(
            "-r", "--recursive", help="run every recording nested under INPUT"
        ),
    ] = False,
    write: Annotated[
        bool,
        typer.Option(
            "--write/--no-write",
            help="record the searched box in <outdir>/autocrop.json, so the next run "
            "detects through it instead of searching again",
        ),
    ] = True,
    no_gate: Annotated[
        bool,
        typer.Option(
            "--no-gate",
            help="accept whatever confidence proposed, without checking it against the "
            "other cameras' 3D. Measurably unsafe on its own -- a box can get more "
            "confident and less accurate -- so only for a rig with no usable calibration",
        ),
    ] = False,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Search the detector crop for each view whose config says { op = "crop", auto = true }."""
    _configure_logging(log_level.value)
    from ..pose2d import autocrop
    from ..pose2d.stream import load_models
    from ..recordings import source_image_sizes

    if not inputs:
        raise SystemExit("give at least one recording directory (or wildcard)")
    discovery = Config.from_toml(config) if config else Config.default()
    found = resolve_recordings(inputs, recursive=recursive, config=discovery)
    outdirs = plan_outdirs([d for d, _ in found], output)
    recordings = [
        Recording(src, outdir) for (_, src), outdir in zip(found, outdirs.outdirs)
    ]

    for rec in recordings:
        console.rule(str(rec.outdir))
        cfg = Config.read_for_run(config, rec.outdir)
        # Searching is the point of this command, so an already-recorded box is ignored
        # rather than reused (`ensure_resolved(force=True)` below does the same).
        cfg.auto_crops = {}
        if no_gate:
            cfg.data.setdefault("pose2d", {}).setdefault("autocrop", {})["gate"] = False
        plan = cfg.detection_plan()
        todo = autocrop.targets(plan)
        if not todo:
            raise SystemExit(
                f"{rec.outdir}: this config declares no automatic crop. Add "
                '`{ op = "crop", auto = true }` to a [[pose2d.preprocessors]] the view '
                "needs one for (usually the front and hind cameras), then run this again."
            )
        source_sizes = source_image_sizes(cfg, sources=rec.sources)
        view_sources = plan.view_sources()
        cameras = cfg.camera_group(
            image_sizes={
                v: source_sizes[s] for v, s in view_sources.items() if s in source_sizes
            }
        )
        models = load_models(plan)
        for name, model in models.items():
            model.set_precision(model.spec.precision or cfg.pose2d.precision)
            log.info("model %r on %s", name, model.device())
        _, resolutions = autocrop.ensure_resolved(
            cfg,
            plan,
            models=models,
            cameras=cameras,
            sources=rec.sources,
            outdir=rec.outdir if write else None,
            force=True,
        )
        _report(resolutions, rec.outdir if write else None)


def _report(resolutions, outdir: Path | None) -> None:
    """Print each searched box, the evidence, and the TOML that would freeze it."""
    from rich.table import Table

    table = Table(title="auto-crop", header_style="bold")
    for col in (
        "preprocessor",
        "view",
        "incumbent",
        "searched",
        "conf",
        "agree px",
        "",
    ):
        table.add_column(col)
    for r in resolutions:
        conf = f"{r.conf_incumbent:.3f} -> {r.conf:.3f}"
        agree = (
            f"{r.agreement_incumbent_px:.1f} -> {r.agreement_px:.1f}"
            if r.gated
            else "not gated"
        )
        table.add_row(
            r.preprocessor,
            r.view_name,
            str(tuple(r.incumbent)),
            str(tuple(r.box)),
            conf,
            agree,
            "[green]accepted[/]" if r.accepted else "[yellow]kept[/]",
        )
    console.print(table)
    for r in resolutions:
        for note in r.notes:
            console.print(f"  [yellow]{r.view_name or r.preprocessor}[/]: {note}")
    if outdir is not None:
        console.print(
            f"\nrecorded in {outdir / 'autocrop.json'} -- the next run reuses it"
        )
    console.print(
        "\nTo freeze a searched window as an explicit box (so nothing re-searches), "
        "replace that preprocessor's op with:"
    )
    for r in resolutions:
        x, y, w, h = r.box
        # markup=False: the TOML is full of square brackets, which rich would read as tags.
        console.print(
            f'  [[pose2d.preprocessors]]  name = "{r.preprocessor}"\n'
            f'  ops = [{{ op = "crop", x = {x}, y = {y}, width = {w}, height = {h} }}]',
            markup=False,
            highlight=False,
        )
