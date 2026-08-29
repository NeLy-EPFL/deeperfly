"""``deeperfly project`` -- create projects and adopt recordings into them.

Every command of the ``project`` group, declared on :data:`project_app` and mounted by
:mod:`deeperfly.cli.app`. Each is an ordinary function with ordinary parameters, so it is
callable as a library and from the tests as readily as from the command line.

The listing commands (``ls`` / ``status``) are deliberately tolerant: a project with one
unreadable ``labels.h5`` or one recording whose share is unmounted must still print every
other row. A status listing that dies on a bad row is useless exactly when it is needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from ..project import PROJECT_FILENAME, Project
from .console import (
    LogLevel,
    LogLevelOption,
    ProjectArg,
    _configure_logging,
    _info_line,
    console,
)

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


project_app = typer.Typer(
    no_args_is_help=True,
    help="Group related recordings into a project: one skeleton, shared camera rigs, "
    "and one place to see what is labeled. A project INDEXES recordings -- their "
    "results.h5 / labels.h5 stay where they are and are adopted by symlink, so no "
    "label is ever copied or moved to create one.",
)


@project_app.command("new")
def project_new(
    root: Annotated[str, typer.Argument(help="directory to create the project in")],
    name: Annotated[
        str | None,
        typer.Option("--name", help="project name (default: the directory's name)"),
    ] = None,
    skeleton: Annotated[
        str,
        typer.Option(
            "--skeleton",
            help="'fly38' (the packaged 38-point Drosophila skeleton), 'blank' (define "
            "your own), or a path to a TOML file with a [skeleton] table",
        ),
    ] = "fly38",
    description: Annotated[
        str | None, typer.Option("--description", help="free-text description")
    ] = None,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Create a project: a skeleton, a place for rigs, and an empty recording index.

    Writes a project.toml (the index) and a skeleton.toml (what is tracked). Nothing
    else -- recordings are adopted afterwards with 'deeperfly project add', and a camera
    rig is either solved later or pointed at with a calibration file.

    Start from 'blank' for a new animal or rig: you then label with no calibration at
    all and solve the rig from those labels once there are enough correspondences.
    """
    _configure_logging(log_level.value)
    try:
        project = Project.create(
            root,
            name=name,
            skeleton=skeleton,
            description=description or "",
        )
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    skeleton = project.skeleton()
    console.print(f"[green]created project[/green] {project.root}")
    _info_line("name:     ", project.name)
    _info_line("id:       ", project.id)
    _info_line(
        "skeleton: ",
        f"{skeleton.label}  ({skeleton.n_points} points, {skeleton.n_edges} edges)"
        if skeleton.n_points
        else f"{skeleton.name}  (empty -- edit {project.skeleton_file})",
    )
    console.print(
        f"next: 'deeperfly project add {project.root} <recording>' to adopt a recording "
        "(its results.h5 / labels.h5 are linked, never copied)",
        highlight=False,
    )


@project_app.command("add")
def project_add(
    project: Annotated[str, typer.Argument(help="the project directory to adopt into")],
    sources: Annotated[
        list[str],
        typer.Argument(
            metavar="RECORDING...",
            help="one or more recordings: a recording directory, its "
            "deeperfly_outputs/, or a results.h5",
        ),
    ],
    copy: Annotated[
        bool,
        typer.Option(
            "--copy",
            help="copy each recording's outputs into the project instead of linking "
            "them. The copy is a SNAPSHOT: labels authored in the original will not "
            "appear in the project, and vice versa",
        ),
    ] = False,
    slug: Annotated[
        str | None,
        typer.Option(
            "--slug",
            help="name for the recording inside the project (single source only; "
            "default: the recording directory's name)",
        ),
    ] = None,
    subject: Annotated[
        str | None,
        typer.Option(
            "--subject",
            help="animal identifier, so one specimen's several clips group together "
            "(read from results.h5 when it records one)",
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            "-c",
            "--config",
            help="config supplying the per-camera footage globs. Without it, each "
            "video file in the recording directory becomes a camera named after the "
            "file (which is what a from-scratch recording wants)",
        ),
    ] = None,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Adopt recordings into a project, by reference.

    Each recording's deeperfly_outputs/ is SYMLINKED into the project, so the labels.h5
    the editor writes is the very file a training set reads -- adopting copies nothing
    and can lose nothing. A recording with no outputs yet (just videos) is adopted too;
    that is the from-scratch starting point.

    Recordings are identified by content, not path, so adopting the same one twice is a
    no-op and a backup copy is recognized as the same recording.
    """
    _configure_logging(log_level.value)
    project = _open(project)
    config = None
    if config:
        from ..config import Config

        config = Config.from_toml(config)

    added, failed = [], []
    for source in sources:
        try:
            before = {e.id for e in project.recordings}
            entry = project.add_recording(
                source,
                link=not copy,
                slug=slug if len(sources) == 1 else None,
                subject=subject,
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


@project_app.command("ls")
def project_ls(
    project: ProjectArg = None,
    log_level: LogLevelOption = LogLevel.warning,
) -> None:
    """List a project's recordings."""
    _configure_logging(log_level.value)
    project = _open(project)
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


@project_app.command("status")
def project_status(
    project: ProjectArg = None,
    log_level: LogLevelOption = LogLevel.warning,
) -> None:
    """Report labeling progress across a project.

    Per recording: frames, frames carrying labels, frames marked reviewed, ground-truth
    points, occlusion marks, and whether its outputs are present. The counts are the
    LIVE rows of each labels.h5 -- a keypoint declared absent is not counted as ground
    truth, matching what an export and a training set will see.
    """
    _configure_logging(log_level.value)
    project = _open(project)
    rows = project.status()
    totals = project.totals(rows)

    _info_line("project:  ", f"{project.name}  ({project.root})")
    _info_line("id:       ", project.id)
    _info_line("iteration:", project.iteration)
    try:
        skeleton = project.skeleton()
        _info_line("skeleton: ", f"{skeleton.label}  ({skeleton.n_points} points)")
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


@project_app.command("rm")
def project_rm(
    recording: Annotated[
        str, typer.Argument(help="a recording's slug, id, or unambiguous id prefix")
    ],
    project: ProjectArg = None,
    delete: Annotated[
        bool,
        typer.Option(
            "--delete",
            help="also remove the project's own directory for the recording. For a "
            "linked recording that removes only the link; it refuses when the outputs "
            "are a real directory, since that would be the only copy of the labels",
        ),
    ] = False,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Drop a recording from the project index (its files are left alone)."""
    _configure_logging(log_level.value)
    project = _open(project)
    try:
        entry = project.recording(recording)
    except KeyError as exc:
        raise SystemExit(str(exc).strip("'")) from None
    outputs = project.outputs_dir(entry)
    if delete and outputs.exists() and not outputs.is_symlink():
        console.print(
            f"[red]refusing[/red] to --delete {entry.slug}: its outputs at {outputs} are "
            "a real directory, not a link, so deleting would destroy the only copy of "
            "its labels. Move them out first, or drop the entry without --delete."
        )
        raise SystemExit(1)
    project.remove_recording(entry.id, delete=delete)
    console.print(
        f"[green]removed[/green] {entry.slug} from the index"
        + (" (and its project directory)" if delete else "")
    )
    if not delete:
        console.print(
            f"its files are untouched at {Path(entry.origin.get('from', outputs))}",
            highlight=False,
        )


@project_app.command("config")
def project_config(
    project: ProjectArg = None,
    output: Annotated[
        str | None,
        typer.Option("-o", "--output", help="write to this file instead of stdout"),
    ] = None,
    profile: Annotated[
        str | None,
        typer.Option(
            "--profile", help="profile filename under profiles/ (default: default.toml)"
        ),
    ] = None,
    base: Annotated[
        str | None,
        typer.Option(
            "--base",
            help="config to take the remaining tables from -- the detection plan and "
            "visualization, which are open-ended (default: the packaged config)",
        ),
    ] = None,
    log_level: LogLevelOption = LogLevel.warning,
) -> None:
    """Compose the project's resolved run config from its parts.

    A project keeps its skeleton, its rig and its algorithm deltas in separate files; a run
    consumes one config. This combines them, validates the result through the same strict
    loader a run uses, and prints or writes it.

    So layering is an AUTHORING convenience only: what a run snapshots and fingerprints is a
    single resolved text, exactly as before. Change one triangulation knob in the profile
    without restating 132 detector channel mappings.
    """
    _configure_logging(log_level.value)
    project = _open(project)
    try:
        text = project.compose_config(profile=profile, base=base)
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

    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        console.print(f"[green]wrote[/green] {out}  ({len(text.splitlines())} lines)")
        console.print(f"next: deeperfly run RECORDING -c {out}", highlight=False)
    else:
        print(text, end="")


@project_app.command("rig")
def project_rig(
    project: ProjectArg = None,
    source: Annotated[
        str | None,
        typer.Option(
            "--from",
            help="config to lift the rig out of (default: the packaged config)",
        ),
    ] = None,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Lift a config's camera rig into the project as rig.toml.

    The rig -- footage sources, camera topology, per-camera preprocessing -- is a property
    of the SETUP, shared by every recording on it. Making it the project's stops each
    recording carrying its own copy, and is what lets one calibration serve all of them.
    """
    _configure_logging(log_level.value)
    project = _open(project)
    try:
        path = project.write_rig(source)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from None
    console.print(f"[green]wrote[/green] {path}")
    console.print(
        "the rig is now the project's: every recording composes against it, and no "
        "recording carries its own copy",
        highlight=False,
    )


@project_app.command("export")
def project_export(
    output: Annotated[str, typer.Argument(help="destination .dfpkg")],
    project: ProjectArg = None,
    embed: Annotated[
        str,
        typer.Option(
            "--embed",
            help="which frames' PIXELS to include: 'user' (default -- the frames carrying "
            "human labels), 'all' (also the suggested ones), or 'none' (index only, for a "
            "collaborator who shares the filesystem)",
        ),
    ] = "user",
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Package a project into one shareable file.

    Carries everything the project OWNS -- skeleton, rig, calibrations, landmarks, manifest
    -- plus each recording's labels.h5 byte for byte and, by default, the frames those
    labels annotate. That last part is affordable because only the labeled frames matter:
    on this project's own corpus, 50 frames out of 4,073.

    Footage is never packaged. A package makes the LABELS portable, not the videos.
    """
    _configure_logging(log_level.value)
    from ..project.package import export_package

    project = _open(project)
    try:
        report = export_package(project, output, embed=embed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    _info_line("wrote:    ", str(output))
    _info_line("size:     ", f"{report.bytes_written / 1e6:.1f} MB")
    _info_line("recordings:", f"{report.recordings} ({report.label_files} with labels)")
    _info_line("frames:   ", f"{report.embedded_frames} embedded ({embed})")
    for note in report.notes:
        console.print(f"[yellow]note:[/yellow] {note}", highlight=False)
    for skipped in report.skipped:
        console.print(f"[yellow]skipped:[/yellow] {skipped}", highlight=False)


@project_app.command("import")
def project_import(
    package: Annotated[str, typer.Argument(help="the .dfpkg to unpack")],
    dest: Annotated[str, typer.Argument(help="directory to create the project in")],
    apply: Annotated[
        bool,
        typer.Option(
            "--apply", help="actually write (otherwise this lists the contents)"
        ),
    ] = False,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Unpack a .dfpkg into a NEW project directory.

    Only into a new or empty directory. Importing into an existing project is a merge --
    with skeleton reconciliation, content dedup and a conflict policy ('deeperfly
    labels-merge') -- and overwriting files instead would be the destructive shortcut that
    looks like it worked.

    Lists the contents and writes nothing without --apply.
    """
    _configure_logging(log_level.value)
    from ..project.package import describe_package, import_package

    try:
        described = describe_package(package)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None

    _info_line("package:  ", str(package))
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

    if not apply:
        console.print(
            f"dry run -- nothing written. Re-run with --apply to unpack into {dest}",
            highlight=False,
        )
        return
    report = import_package(package, dest, apply=True)
    console.print(f"[green]imported[/green] into {dest}")
    for note in report.notes:
        console.print(f"[yellow]note:[/yellow] {note}", highlight=False)
    console.print(
        "the recordings' footage is NOT in the package -- re-point it, or re-add the "
        "recordings locally, before the editor can show frames",
        highlight=False,
    )


@project_app.command("import-outputs")
def project_import_outputs(
    project: Annotated[
        str, typer.Argument(help="the project to import the corrections INTO")
    ],
    sources: Annotated[
        list[str],
        typer.Argument(
            metavar="SOURCE...",
            help="one or more deeperfly_outputs/ dirs, recording dirs, or a tree "
            "containing them (every deeperfly_outputs/ underneath is found)",
        ),
    ],
    recording: Annotated[
        str | None,
        typer.Option(
            "--recording",
            help="force every source onto this recording (slug, id, or id prefix). The "
            "escape hatch for an archived recording whose content id cannot be derived",
        ),
    ] = None,
    on_conflict: Annotated[
        str,
        typer.Option(
            "--on-conflict",
            help="a cell BOTH sides authored differently: manual (queue it, the "
            "default), ours, theirs, newest",
        ),
    ] = "manual",
    on_absent: Annotated[
        str | None,
        typer.Option(
            "--on-absent",
            help="an incoming absence declaration that would quarantine ground truth "
            "this project counts: union (accept) or ours (ignore the source's). Must be "
            "chosen explicitly when there is a cost",
        ),
    ] = None,
    apply: Annotated[
        bool,
        typer.Option("--apply", help="actually import (otherwise this only reports)"),
    ] = False,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Add corrections from a stray deeperfly_outputs/ to the recording it belongs to.

    For labels made in the standalone editor, outside the project -- the gap 'project add'
    reports and cannot fix, because one recording is one entry, so a second label set beside
    a second copy of the footage reads as zero.

    Which recording is answered from CONTENT identity, not from a name you type. Three
    answers: the project already reads that very file (nothing to do -- the normal state for
    a symlink-adopted recording); the same recording in a different file (merge); or a
    recording this project does not index, which is 'deeperfly project add' instead -- and
    better, because adding SYMLINKS the outputs and needs no merge at all.

    Labels move by NAME, never by index. Predictions are never promoted to ground truth: a
    source's seeds arrive as seeds. Dry-run by default, and every destination labels.h5 is
    snapshotted before the first write.
    """
    _configure_logging(log_level.value)
    on_absent = on_absent or "union"
    absent_explicit = on_absent is not None
    from ..project.import_outputs import find_outputs, import_outputs

    project = _open(project)
    sources = []
    for raw in sources:
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
            recording=recording,
            on_conflict=on_conflict,
            on_absent=on_absent,
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
    if cost and on_absent == "union" and not absent_explicit:
        total = sum(p.quarantined_dest_gt for p in cost)
        raise SystemExit(
            f"the incoming absence declarations would quarantine {total} ground-truth "
            f"row(s) this project currently counts, across {len(cost)} recording(s). They "
            "are recoverable -- un-declaring the point restores them -- but the project's "
            "GT total drops. Pass --on-absent union to accept, or --on-absent ours to "
            "ignore the source's declarations"
        )

    if not apply:
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
        recording=recording,
        on_conflict=on_conflict,
        on_absent=on_absent,
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


@project_app.command("skeleton")
def project_skeleton(
    source: Annotated[
        str,
        typer.Argument(
            help="a TOML file with a [skeleton] table (a config, or a skeleton.toml)"
        ),
    ],
    project: ProjectArg = None,
    rename: Annotated[
        list[str] | None,
        typer.Option(
            "--rename",
            help="OLD=NEW: a point that only changed name, so its labels move with it "
            "(repeatable; '*' on both sides renames a family, e.g. '*_claw=*_pretarsus')",
        ),
    ] = None,
    apply: Annotated[
        bool,
        typer.Option("--apply", help="actually migrate (otherwise this only reports)"),
    ] = False,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Change the project's skeleton, migrating every label onto the new point order.

    A skeleton edit can invalidate every label in the project -- and quietly, because two
    same-sized skeletons in different orders load each other's files happily and mean
    something different by every index. So labels move BY NAME, never by index; the change
    is reported and counted before anything is written; and a deleted point's labels are
    QUARANTINED rather than destroyed, so re-adding the point brings them back.

    A rename is the one edit the two files cannot describe: "the claw point is now called
    pretarsus" and "claw is gone, pretarsus is new" are the same diff, and the second
    quarantines every label on it. One in place is inferred; declare the rest with
    --rename OLD=NEW.

    Reports and writes nothing without --apply. Applying snapshots the project to a .dfpkg
    first.
    """
    _configure_logging(log_level.value)
    from ..config import Config
    from ..project.migrate import apply_migration, expand_renames, plan_migration

    project = _open(project)
    try:
        new = Config.from_toml(source).skeleton()
    except Exception as exc:
        raise SystemExit(f"could not read a skeleton from {source}: {exc}") from None

    try:
        renames = expand_renames(
            list(rename or ()),
            project.skeleton().point_names,
            new.point_names,
        )
        plan = plan_migration(project, new, renames=renames)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
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
    if not apply:
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
