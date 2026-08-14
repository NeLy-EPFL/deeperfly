"""The ``auto-crop`` command worker: search a recording's detector crops and report them.

The same search the ``pose2d`` stage runs (:mod:`deeperfly.pose2d.autocrop`), reachable on
its own so a box can be measured, *looked at*, and only then trusted -- or frozen into the
config as an explicit window, which is what a recording that is about to be labelled wants:
a number someone can read, not a search that reruns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import Config
from ..recordings import Recording, plan_outdirs, resolve_recordings
from .console import console, log


def _cmd_auto_crop(args: argparse.Namespace) -> None:
    """Search every automatic crop of each resolved recording and print the outcome.

    Parameters
    ----------
    args
        The ``auto-crop`` namespace (``inputs``, ``config``, ``output``, ``recursive``,
        ``write``, ``no_gate``).

    Raises
    ------
    SystemExit
        If no inputs are given, or the config declares no automatic crop to search.
    """
    from ..pose2d import autocrop
    from ..pose2d.stream import load_models
    from ..recordings import source_image_sizes

    if not args.inputs:
        raise SystemExit("give at least one recording directory (or wildcard)")
    discovery = Config.from_toml(args.config) if args.config else Config.default()
    found = resolve_recordings(args.inputs, recursive=args.recursive, config=discovery)
    outdirs = plan_outdirs([d for d, _ in found], args.output)
    recordings = [
        Recording(src, outdir) for (_, src), outdir in zip(found, outdirs.outdirs)
    ]

    for rec in recordings:
        console.rule(str(rec.outdir))
        config = Config.read_for_run(args.config, rec.outdir)
        # Searching is the point of this command, so an already-recorded box is ignored
        # rather than reused (`ensure_resolved(force=True)` below does the same).
        config.auto_crops = {}
        if args.no_gate:
            config.data.setdefault("pose2d", {}).setdefault("autocrop", {})["gate"] = (
                False
            )
        plan = config.detection_plan()
        todo = autocrop.targets(plan)
        if not todo:
            raise SystemExit(
                f"{rec.outdir}: this config declares no automatic crop. Add "
                '`{ op = "crop", auto = true }` to a [[pose2d.preprocessors]] the view '
                "needs one for (usually the front and hind cameras), then run this again."
            )
        source_sizes = source_image_sizes(config, sources=rec.sources)
        view_sources = plan.view_sources()
        cameras = config.camera_group(
            image_sizes={
                v: source_sizes[s] for v, s in view_sources.items() if s in source_sizes
            }
        )
        models = load_models(plan)
        for name, model in models.items():
            model.set_precision(model.spec.precision or config.pose2d.precision)
            log.info("model %r on %s", name, model.device())
        _, resolutions = autocrop.ensure_resolved(
            config,
            plan,
            models=models,
            cameras=cameras,
            sources=rec.sources,
            outdir=rec.outdir if args.write else None,
            force=True,
        )
        _report(resolutions, rec.outdir if args.write else None)


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
