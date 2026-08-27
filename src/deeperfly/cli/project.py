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
        f"{skeleton.name}  ({skeleton.n_points} points, {skeleton.n_bones} bones)"
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
    # `overflow="fold"` rather than rich's default ellipsis: the slug is not decoration,
    # it is the name the operator types back into '--recording' / 'project rm', and a
    # project with a couple of dozen long recording names is exactly when a narrow
    # terminal starts truncating. A wrapped slug is readable; an ellipsized one is not
    # usable at all.
    table.add_column("slug", style="bold", footer="total", overflow="fold")
    table.add_column("frames", justify="right")
    table.add_column("labeled", justify="right", footer=f"{totals['labeled_frames']:,}")
    table.add_column(
        "reviewed", justify="right", footer=f"{totals['reviewed_frames']:,}"
    )
    table.add_column("GT pts", justify="right", footer=f"{totals['gt_points']:,}")
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
            f"{row['gt_points']:,}",
            f"{row['occluded']:,}",
            f"{row['absent_points']}" if row["absent_points"] else "—",
            _outputs_note(row),
        )
    console.print(table)
    console.print(
        f"{totals['with_labels']} of {totals['recordings']} recording(s) carry labels",
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


def _cmd_project_config(args: argparse.Namespace) -> None:
    """Print or write the project's resolved run config (``deeperfly project config``).

    The composition is skeleton + rig + calibration + profile + whatever the base config
    still supplies. A run consumes the *result*, so layering never reaches the pipeline as
    ambiguity -- what gets snapshotted and fingerprinted is one resolved text, exactly as
    before.
    """
    project = _open(args.project)
    try:
        text = project.compose_config(profile=args.profile, base=args.base)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from None

    # Validate before handing it over: a composed config that does not load is worse than
    # no composition, and the strict loader is the same one a run uses.
    import tomllib

    try:
        from ..config import Config

        parsed = tomllib.loads(text)
        config = Config.from_dict(parsed)
        config.skeleton()
        config.detection_plan()
    except Exception as exc:
        raise SystemExit(
            f"the composed config is not valid: {exc}\nThis is a project-file problem -- "
            f"check {project.rig_path().name}, {project.skeleton_file} and the profile"
        ) from None

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        console.print(f"[green]wrote[/green] {out}  ({len(text.splitlines())} lines)")
        console.print(f"next: deeperfly run RECORDING -c {out}", highlight=False)
    else:
        print(text, end="")


def _cmd_project_rig(args: argparse.Namespace) -> None:
    """Lift the rig tables out of a config into the project (``deeperfly project rig``)."""
    project = _open(args.project)
    try:
        path = project.write_rig(args.source)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from None
    console.print(f"[green]wrote[/green] {path}")
    console.print(
        "the rig is now the project's: every recording composes against it, and no "
        "recording carries its own copy",
        highlight=False,
    )


def _cmd_project_export(args: argparse.Namespace) -> None:
    """Package a project into one shareable file (``deeperfly project export``)."""
    from ..package import export_package

    project = _open(args.project)
    try:
        report = export_package(project, args.output, embed=args.embed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    _info_line("wrote:    ", str(args.output))
    _info_line("size:     ", f"{report.bytes_written / 1e6:.1f} MB")
    _info_line("recordings:", f"{report.recordings} ({report.label_files} with labels)")
    _info_line("frames:   ", f"{report.embedded_frames} embedded ({args.embed})")
    for note in report.notes:
        console.print(f"[yellow]note:[/yellow] {note}", highlight=False)
    for skipped in report.skipped:
        console.print(f"[yellow]skipped:[/yellow] {skipped}", highlight=False)


def _cmd_project_import(args: argparse.Namespace) -> None:
    """Unpack a ``.dfpkg`` into a new project (``deeperfly project import``)."""
    from ..package import describe_package, import_package

    try:
        described = describe_package(args.package)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    _info_line("package:  ", str(args.package))
    _info_line("created:  ", described["created_utc"] or "unknown")
    _info_line("embed:    ", described["embed"] or "none")
    table = Table(title="contents")
    table.add_column("recording", style="bold")
    table.add_column("frames", justify="right")
    table.add_column("labels")
    table.add_column("embedded", justify="right")
    for row in described["recordings"]:
        table.add_row(
            row["slug"],
            "?" if row["n_frames"] < 0 else f"{row['n_frames']:,}",
            "yes" if row["has_labels"] else "[yellow]no[/yellow]",
            f"{row['embedded_frames']:,}",
        )
    console.print(table)

    if not args.apply:
        console.print(
            f"dry run -- nothing written. Re-run with --apply to unpack into {args.dest}",
            highlight=False,
        )
        return
    report = import_package(args.package, args.dest, apply=True)
    console.print(f"[green]imported[/green] into {args.dest}")
    for note in report.notes:
        console.print(f"[yellow]note:[/yellow] {note}", highlight=False)
    console.print(
        "the recordings' footage is NOT in the package -- re-point it, or re-add the "
        "recordings locally, before the editor can show frames",
        highlight=False,
    )


def _cmd_project_import_outputs(args: argparse.Namespace) -> None:
    """Merge stray ``deeperfly_outputs/`` corrections in (``deeperfly project import-outputs``)."""
    from ..import_outputs import find_outputs, import_outputs

    project = _open(args.project)
    sources = []
    for raw in args.sources:
        try:
            found = find_outputs(raw)
        except FileNotFoundError as exc:
            console.print(f"[red]skipped[/red] {raw}: {exc}", highlight=False)
            continue
        if not found:
            console.print(
                f"[yellow]skipped[/yellow] {raw}: no deeperfly_outputs/ there",
                highlight=False,
            )
            continue
        sources += found

    if not sources:
        raise SystemExit("no outputs directories found in what you passed")

    _info_line("project:  ", f"{project.name}  ({project.root})")
    _info_line("iteration:", str(project.iteration))
    _info_line("sources:  ", str(len(sources)))

    try:
        plans = import_outputs(
            project,
            sources,
            recording=args.recording,
            on_conflict=args.on_conflict,
            on_absent=args.on_absent,
            apply=False,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    for plan in plans:
        _print_import(plan)

    counts: dict[str, int] = {}
    for plan in plans:
        counts[plan.outcome] = counts.get(plan.outcome, 0) + 1
    console.print(
        "\n"
        + ", ".join(f"{n} {name}" for name, n in sorted(counts.items()))
        + f"  (of {len(plans)} source(s))",
        highlight=False,
    )

    blocked = [p for p in plans if p.outcome == "fatal"]
    mergeable = [p for p in plans if p.outcome == "merged"]
    if blocked:
        raise SystemExit("refusing to import: see the fatal source(s) above")

    # A quarantine cost must be chosen, not absorbed: the project's live GT total would drop.
    cost = [p for p in mergeable if p.quarantined_dest_gt]
    if cost and args.on_absent == "union" and not args.absent_explicit:
        total = sum(p.quarantined_dest_gt for p in cost)
        raise SystemExit(
            f"the incoming absence declarations would quarantine {total} ground-truth "
            f"row(s) this project currently counts, across {len(cost)} recording(s). They "
            "are recoverable -- un-declaring the point restores them -- but the project's "
            "GT total drops. Pass --on-absent union to accept, or --on-absent ours to "
            "ignore the source's declarations"
        )

    if not args.apply:
        if mergeable:
            console.print(
                "dry run -- nothing written. Re-run with --apply to import "
                "(a snapshot of every destination labels.h5 is taken first)",
                highlight=False,
            )
        return
    if not mergeable:
        console.print("nothing to import.", highlight=False)
        return

    applied = import_outputs(
        project,
        [p.source for p in mergeable],
        recording=args.recording,
        on_conflict=args.on_conflict,
        on_absent=args.on_absent,
        apply=True,
    )
    for plan in applied:
        if plan.outcome != "merged":
            console.print(
                f"[red]failed[/red] {plan.slug}: {plan.reason}", highlight=False
            )
            continue
        console.print(f"[green]imported[/green] into {project.labels_path(plan.entry)}")
        if plan.snapshot is not None:
            console.print(f"  pre-import snapshot: {plan.snapshot}", highlight=False)
        if plan.merge is not None and plan.merge.unresolved:
            console.print(
                f"  {len(plan.merge.unresolved)} cell(s) need a human decision -- they "
                "were left as the destination had them",
                highlight=False,
            )
    console.print(f"project iteration is now {project.iteration}", highlight=False)


def _print_import(plan) -> None:
    """One source's section: the verdict first, then the numbers."""
    console.print(f"\n[bold]── {plan.slug}[/bold]", highlight=False)
    _info_line("source:   ", str(plan.source.path))
    if plan.matched_by:
        _info_line("matched:  ", plan.matched_by)
    if plan.source.format_version is not None:
        _info_line("labels:   ", f"v{plan.source.format_version}")

    if plan.outcome == "noop":
        console.print(f"[green]nothing to do:[/green] {plan.reason}", highlight=False)
        return
    if plan.outcome == "unindexed":
        console.print(
            f"[yellow]not in this project:[/yellow] {plan.reason}", highlight=False
        )
        return
    if plan.outcome == "empty":
        console.print(f"[yellow]skipped:[/yellow] {plan.reason}", highlight=False)
        return
    if plan.outcome == "fatal":
        console.print(f"[red]fatal:[/red] {plan.reason}", highlight=False)
        return

    report = plan.merge
    for note in report.notes:
        console.print(f"[yellow]note:[/yellow] {note}", highlight=False)
    table = Table(title="would import")
    table.add_column("what", style="bold")
    table.add_column("n", justify="right")
    for label, count in (
        ("points matched by name", len(report.points.matched)),
        ("cameras matched by name", len(report.cameras.matched)),
        ("ground-truth pixels taken (human-placed)", report.taken_from_source),
        ("ground-truth pixels kept (the destination's)", report.kept_from_dest),
        ("ground-truth pixels already identical", report.identical),
        ("instance seeds taken", report.seeds_taken),
        ("frames newly carrying an annotation skeleton", report.instances_added),
        ("occlusions taken", report.occluded_taken),
        ("absence declarations added", report.absent_union),
        ("frames newly marked reviewed", report.reviewed_added),
        ("landmark observations taken", plan.landmarks_taken),
        ("cells dropped (no destination point/camera)", report.dropped_cells),
        ("conflicts needing a human", len(report.unresolved)),
    ):
        table.add_row(label, f"{count:,}")
    console.print(table)

    for decision in report.unresolved[:3]:
        console.print(
            f"  conflict view={decision.view} frame={decision.frame} "
            f"point={decision.point}: {decision.reason}",
            highlight=False,
        )
    if len(report.unresolved) > 3:
        console.print(f"  ... and {len(report.unresolved) - 3} more", highlight=False)
    if plan.landmarks_only_source:
        console.print(
            f"[yellow]note:[/yellow] landmark(s) {plan.landmarks_only_source} are in the "
            "source but not declared in this project's landmarks.toml -- dropped",
            highlight=False,
        )
    if plan.subject_id_taken:
        console.print(
            f"  subject id {plan.subject_id_taken!r} taken from the source "
            "(the destination had none)",
            highlight=False,
        )
    if plan.quarantined_dest_gt:
        console.print(
            f"[yellow]warning:[/yellow] the incoming absence declarations would "
            f"QUARANTINE {plan.quarantined_dest_gt} ground-truth row(s) this project "
            "currently counts. Recoverable -- un-declaring restores them -- but the GT "
            "total drops.",
            highlight=False,
        )


def _cmd_project_skeleton(args: argparse.Namespace) -> None:
    """Change a project's skeleton as a migration (``deeperfly project skeleton``).

    A skeleton edit can invalidate every label in the project, and quietly: two same-sized
    skeletons in different orders load each other's files happily and mean something
    different by every index. So this always reports first, moves labels **by name**, and
    refuses a destructive change without ``--apply``.
    """
    from ..config import Config
    from ..skeleton_migrate import apply_migration, plan_migration

    project = _open(args.project)
    try:
        new = Config.from_toml(args.source).skeleton()
    except Exception as exc:
        raise SystemExit(
            f"could not read a skeleton from {args.source}: {exc}"
        ) from None

    plan = plan_migration(project, new)
    _info_line("project:  ", f"{project.name}  ({project.root})")
    _info_line("points:   ", f"{len(plan.old_names)} -> {len(plan.new_names)}")
    if not plan.changes:
        console.print("no change -- the skeletons are identical")
        return

    table = Table(title="skeleton changes")
    table.add_column("", width=2)
    table.add_column("change", style="bold")
    table.add_column("detail")
    for change in plan.changes:
        table.add_row(
            "[red]![/red]" if change.destructive else " ", change.kind, change.detail
        )
    console.print(table)
    _info_line("labels moved:      ", f"{plan.moved:,} (remapped by name)")
    _info_line("labels quarantined:", f"{plan.quarantined:,}")
    for err in plan.errors:
        console.print(f"[red]blocked:[/red] {err}", highlight=False)

    if plan.quarantined:
        console.print(
            f"[yellow]{plan.quarantined:,} label(s) belong to point(s) the new skeleton "
            "does not have.[/yellow] They are QUARANTINED, not deleted -- re-adding the "
            "point restores them -- but nothing downstream will see them meanwhile.",
            highlight=False,
        )
    if not args.apply:
        console.print(
            "dry run -- nothing written. Re-run with --apply to migrate "
            "(a pre-migration .dfpkg snapshot is written first)",
            highlight=False,
        )
        return
    try:
        result = apply_migration(project, new, plan)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    console.print(
        f"[green]migrated[/green] {len(result['migrated'])} label file(s): "
        f"{result['moved']:,} moved, {result['quarantined']:,} quarantined"
    )
    if result["snapshot"]:
        console.print(f"snapshot: {result['snapshot']}", highlight=False)
