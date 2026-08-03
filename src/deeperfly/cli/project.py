"""``deeperfly project`` -- create projects and adopt recordings into them.

The workers behind the ``project`` command group. Each takes an argparse-style namespace
so it stays callable as a library and from the tests, matching the other ``_cmd_*``
workers.

The listing commands (``ls`` / ``status``) are deliberately tolerant: a project with one
unreadable ``labels.h5`` or one recording whose share is unmounted must still print every
other row. A status listing that dies on a bad row is useless exactly when it is needed.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.table import Table

from ..project import PROJECT_FILENAME, Project
from .console import _info_line, console

log = logging.getLogger("deeperfly")


def _open(path: str | None) -> Project:
    """The project at ``path``, or the nearest one enclosing the cwd.

    Walking upward (like ``git``) means ``deeperfly project status`` works from inside a
    project without retyping its root.

    Raises
    ------
    SystemExit
        If no project can be found, with the command that would create one.
    """
    if path:
        try:
            return Project.load(path)
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from None
    found = Project.find(".")
    if found is None:
        raise SystemExit(
            f"no {PROJECT_FILENAME} here or in any parent directory -- pass a project "
            "path, or create one with 'deeperfly project new <dir>'"
        )
    return Project.load(found)


def _cmd_project_new(args: argparse.Namespace) -> None:
    """Create a project directory (``deeperfly project new``)."""
    try:
        project = Project.create(
            args.root,
            name=args.name,
            skeleton=args.skeleton,
            description=args.description or "",
        )
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    skeleton = project.skeleton()
    console.print(f"[green]created project[/green] {project.root}")
    _info_line("name:     ", project.name)
    _info_line("id:       ", project.id)
    _info_line(
        "skeleton: ",
        f"{skeleton.name}  ({skeleton.n_points} points, {skeleton.n_limbs} limbs)"
        if skeleton.n_points
        else f"{skeleton.name}  (empty -- edit {project.skeleton_file})",
    )
    console.print(
        f"next: 'deeperfly project add {project.root} <recording>' to adopt a recording "
        "(its results.h5 / labels.h5 are linked, never copied)",
        highlight=False,
    )


def _cmd_project_add(args: argparse.Namespace) -> None:
    """Adopt one or more recordings (``deeperfly project add``).

    Each source is adopted independently: one that cannot be identified is reported and
    skipped rather than aborting the batch, so adopting a directory of twenty recordings
    is not defeated by one whose footage has moved.
    """
    project = _open(args.project)
    config = None
    if args.config:
        from ..config import Config

        config = Config.from_toml(args.config)

    added, failed = [], []
    for source in args.sources:
        try:
            before = {e.id for e in project.recordings}
            entry = project.add_recording(
                source,
                link=not args.copy,
                slug=args.slug if len(args.sources) == 1 else None,
                subject=args.subject,
                config=config,
            )
        except (FileNotFoundError, ValueError, FileExistsError) as exc:
            log.warning("skipping %s: %s", source, exc)
            failed.append(str(source))
            continue
        if entry.id in before:
            console.print(
                f"[yellow]already present[/yellow] {entry.slug}  ({entry.id})"
            )
            continue
        added.append(entry)
        outputs = project.outputs_dir(entry)
        how = (
            "linked"
            if outputs.is_symlink()
            else ("copied" if outputs.exists() else "no outputs yet")
        )
        console.print(
            f"[green]added[/green] {entry.slug}  ({entry.id}, {entry.id_basis}; {how})"
        )
    if failed:
        console.print(f"[red]{len(failed)} source(s) could not be adopted[/red]")
    if added:
        console.print(
            f"{len(added)} recording(s) added; {len(project.recordings)} in the project"
        )


def _cmd_project_ls(args: argparse.Namespace) -> None:
    """List a project's recordings (``deeperfly project ls``)."""
    project = _open(args.project)
    if not project.recordings:
        console.print(
            f"{project.name} has no recordings yet -- "
            f"'deeperfly project add {project.root} <recording>'"
        )
        return
    table = Table(title=f"{project.name}  ({len(project.recordings)} recordings)")
    table.add_column("slug", style="bold")
    table.add_column("id", style="dim")
    table.add_column("frames", justify="right")
    table.add_column("subject")
    table.add_column("outputs")
    for row in project.status():
        entry = row["entry"]
        table.add_row(
            entry.slug,
            entry.id.removeprefix("rec_")[:10],
            "?" if entry.n_frames is None else f"{entry.n_frames:,}",
            entry.subject or "—",
            _outputs_note(row),
        )
    console.print(table)


def _outputs_note(row: dict) -> str:
    """How a recording's outputs are attached, in one short phrase."""
    if row["outputs_missing"]:
        return "[red]missing[/red]"
    kind = "link" if row["linked"] else "dir"
    if not row["has_results"]:
        return f"{kind}, [yellow]no results[/yellow]"
    return kind


def _cmd_project_status(args: argparse.Namespace) -> None:
    """Report labeling progress across a project (``deeperfly project status``).

    The per-recording counts come from each ``labels.h5``'s sparse indices, so the
    numbers are the *live* rows -- a keypoint declared absent is not counted as ground
    truth, matching what an export and a training set will see.
    """
    project = _open(args.project)
    rows = project.status()
    totals = project.totals(rows)

    _info_line("project:  ", f"{project.name}  ({project.root})")
    _info_line("id:       ", project.id)
    _info_line("iteration:", project.iteration)
    try:
        skeleton = project.skeleton()
        _info_line("skeleton: ", f"{skeleton.name}  ({skeleton.n_points} points)")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        _info_line("skeleton: ", f"[unreadable: {exc}]")
    _info_line(
        "calib:    ",
        project.calibration or "none (uncalibrated -- no 3D until a rig is solved)",
    )

    if not rows:
        console.print(
            f"\nno recordings yet -- 'deeperfly project add {project.root} <recording>'"
        )
        return

    table = Table(show_footer=True)
    table.add_column("slug", style="bold", footer="total")
    table.add_column("frames", justify="right")
    table.add_column("labeled", justify="right", footer=f"{totals['labeled_frames']:,}")
    table.add_column(
        "reviewed", justify="right", footer=f"{totals['reviewed_frames']:,}"
    )
    # "trainable" rather than "GT pts": it is what an export would actually yield, with
    # bulk-confirmed reprojections and invented drag handles excluded, exactly as
    # export_gt excludes them. A progress number that disagrees with the export is worse
    # than no progress number.
    table.add_column("trainable", justify="right", footer=f"{totals['gt_trainable']:,}")
    table.add_column("dropped", justify="right", footer=f"{totals['gt_untrainable']:,}")
    table.add_column("occl", justify="right", footer=f"{totals['occluded']:,}")
    table.add_column("absent", justify="right")
    table.add_column("state")
    for row in rows:
        entry = row["entry"]
        table.add_row(
            entry.slug,
            "?" if entry.n_frames is None else f"{entry.n_frames:,}",
            f"{row['labeled_frames']:,}",
            f"{row['reviewed_frames']:,}",
            f"{row['gt_trainable']:,}",
            f"{row['gt_points'] - row['gt_trainable']:,}"
            if row["gt_points"] != row["gt_trainable"]
            else "—",
            f"{row['occluded']:,}",
            f"{row['absent_points']}" if row["absent_points"] else "—",
            _outputs_note(row),
        )
    console.print(table)
    console.print(
        f"{totals['with_labels']} of {totals['recordings']} recording(s) carry labels",
        highlight=False,
    )
    if totals["gt_untrainable"]:
        # Name the provenance mix, since "dropped" alone does not say why.
        mix = {}
        for row in rows:
            for name, count in row["provenance"].items():
                if name in ("confirmed_projection", "placeholder_seed"):
                    mix[name] = mix.get(name, 0) + count
        console.print(
            f"[yellow]{totals['gt_untrainable']:,} stored point(s) are not trainable[/yellow] "
            "(" + ", ".join(f"{n}={c:,}" for n, c in sorted(mix.items())) + ") -- "
            "'deeperfly labels-export' drops them by default: a bulk-confirmed "
            "reprojection is the model's own guess, and a placeholder seed is a drag "
            "handle the editor invented at the image edge.",
            highlight=False,
        )


def _cmd_project_rm(args: argparse.Namespace) -> None:
    """Drop a recording from the index (``deeperfly project rm``).

    The adopted outputs are left alone by default -- a symlinked project must not be able
    to delete the originals by accident. ``--delete`` removes the project's own directory
    for the recording, which for a linked one removes only the link.
    """
    project = _open(args.project)
    try:
        entry = project.recording(args.recording)
    except KeyError as exc:
        raise SystemExit(str(exc).strip("'")) from None
    outputs = project.outputs_dir(entry)
    if args.delete and outputs.exists() and not outputs.is_symlink():
        console.print(
            f"[red]refusing[/red] to --delete {entry.slug}: its outputs at {outputs} are "
            "a real directory, not a link, so deleting would destroy the only copy of "
            "its labels. Move them out first, or drop the entry without --delete."
        )
        raise SystemExit(1)
    project.remove_recording(entry.id, delete=args.delete)
    console.print(
        f"[green]removed[/green] {entry.slug} from the index"
        + (" (and its project directory)" if args.delete else "")
    )
    if not args.delete:
        console.print(
            f"its files are untouched at {Path(entry.origin.get('from', outputs))}",
            highlight=False,
        )
