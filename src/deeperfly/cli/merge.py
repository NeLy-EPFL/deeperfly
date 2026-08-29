"""``deeperfly labels-merge`` -- reconcile a second label set into a project recording.

The concrete job: ``~/fly-pose-data`` holds the same recordings under both ``recordings/``
and ``predicted/<rid>_label/``, usually with the hand labels in only one of the two. A
project indexes a recording once (by content), so the other copy's labels are invisible.
This is how they get in.

Dry-run by default, and a snapshot is written before anything is applied -- the destination
is a ``labels.h5`` full of irreplaceable hand work, so "I'll just try it" has to be safe.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from ..labels import Labels, load_labels, save_labels
from ..labels.merge import merge_labels
from .console import (
    LogLevel,
    LogLevelOption,
    ProjectArg,
    _configure_logging,
    _info_line,
    console,
)

log = logging.getLogger("deeperfly")


def _identity(path: Path) -> dict:
    """The identity a ``labels.h5`` was written against.

    Raises
    ------
    SystemExit
        If the file is unreadable or carries none -- there is then no way to know which
        skeleton or camera order its indices refer to, and guessing is exactly the
        corruption this command exists to avoid.
    """
    import h5py

    try:
        with h5py.File(path, "r") as f:
            identity = json.loads(f.attrs["meta"])["identity"]
    except Exception as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from None
    if not identity.get("camera_names") or not identity.get("point_names"):
        raise SystemExit(
            f"{path} records no skeleton/camera names, so its indices cannot be "
            "interpreted -- and guessing is the corruption this command exists to prevent"
        )
    return identity


def _derived_identity(project, entry) -> dict | None:
    """A labels identity for a recording with no ``labels.h5``, from its ``results.h5``.

    Built exactly the way :func:`deeperfly.gui.build_session` builds it, so the identity a
    merge writes is the identity the editor will later validate against. Returns ``None``
    when there is no result either.
    """
    from ..labels import labels_identity
    from ..results import PoseResult, StageStore

    results_path = project.results_path(entry)
    if not results_path.exists():
        return None
    try:
        result = PoseResult.load(results_path)
        store = StageStore(results_path)
        return labels_identity(
            point_names=list(result.skeleton.point_names),
            camera_names=list(result.cameras.names),
            n_frames=result.n_frames,
            image_sizes=store.read_image_sizes(),
            footage=store.read_footage(),
        )
    except Exception as exc:
        log.warning("could not derive an identity from %s: %s", results_path, exc)
        return None


def labels_merge(
    recording: Annotated[
        str, typer.Argument(help="the project recording to merge INTO (slug or id)")
    ],
    source: Annotated[
        str,
        typer.Argument(
            metavar="SOURCE",
            help="the labels.h5 to merge FROM, or a directory containing one",
        ),
    ],
    project: ProjectArg = None,
    on_conflict: Annotated[
        str,
        typer.Option(
            "--on-conflict",
            help="how to settle a cell both sides authored differently: 'manual' "
            "(default -- leave it and queue it for review), 'ours', 'theirs', or "
            "'newest'. Two disagreeing labels are two operators disagreeing, and nothing "
            "in the data ranks one above the other, so the default is to ask",
        ),
    ] = "manual",
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="actually write. Without it this is a dry run that changes nothing. "
            "Applying always snapshots the destination labels.h5 first",
        ),
    ] = False,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Merge a second label set into a project recording's labels.

    For ground truth that ended up in two places -- an earlier round, another annotator's
    directory, the same recording under a different tree. A project indexes a recording
    once (by content), so the other copy's labels are otherwise invisible.

    Points and cameras are matched BY NAME, never by index: two same-sized skeletons in
    different orders are the one case where an index-based copy would silently corrupt
    every label while leaving each cell looking plausible. A same-named camera whose
    footage size differs is refused outright -- ground truth is stored in footage pixels.

    Dry run by default.
    """
    _configure_logging(log_level.value)
    from .project import _open

    proj = _open(project)
    entry = proj.recording(recording)
    dest_path = proj.labels_path(entry)
    source_path = Path(source)
    if source_path.is_dir():
        for candidate in (
            source_path / "labels.h5",
            source_path / "deeperfly_outputs" / "labels.h5",
        ):
            if candidate.exists():
                source_path = candidate
                break
    if not source_path.exists():
        raise SystemExit(f"no labels.h5 at {source}")
    if source_path.resolve() == dest_path.resolve():
        raise SystemExit("the source and destination are the same file")

    source_identity = _identity(source_path)
    source = load_labels(source_path, identity=source_identity)
    if source is None:
        raise SystemExit(f"{source_path} holds no labels")

    if dest_path.exists():
        dest_identity = _identity(dest_path)
        dest = load_labels(dest_path, identity=dest_identity)
        if dest is None:
            raise SystemExit(f"{dest_path} holds no labels")
    else:
        # The common case for a stranded label set: the indexed copy of the recording has
        # never been labeled, and the labels live beside a *different* copy of it. Merging
        # into an empty overlay is the right move -- and it still goes through the same
        # name-based reconciliation, which a plain `cp` would have skipped entirely.
        dest_identity = _derived_identity(proj, entry)
        if dest_identity is None:
            raise SystemExit(
                f"{entry.slug} has no labels.h5 and no results.h5, so there is nothing "
                "to establish its skeleton, camera order or frame count against. Open it "
                "once in 'deeperfly gui' (which records them), then merge"
            )
        dest = Labels.empty(
            len(dest_identity["camera_names"]),
            int(dest_identity["n_frames"]),
            len(dest_identity["point_names"]),
        )
        console.print(
            f"{entry.slug} has no labels yet -- merging into an empty set "
            "(still reconciled by name, unlike a plain copy)",
            highlight=False,
        )

    report = merge_labels(
        dest,
        source,
        point_names_dest=list(dest_identity["point_names"]),
        point_names_source=list(source_identity["point_names"]),
        camera_names_dest=list(dest_identity["camera_names"]),
        camera_names_source=list(source_identity["camera_names"]),
        image_sizes_dest=dest_identity.get("image_sizes"),
        image_sizes_source=source_identity.get("image_sizes"),
        on_conflict=on_conflict,
        apply=apply,
    )
    _print(report, entry.slug, source_path, applying=apply)

    if not report.ok:
        raise SystemExit("refusing to merge (see above)")
    if not apply:
        console.print(
            "dry run -- nothing written. Re-run with --apply to merge "
            f"(a snapshot of {dest_path.name} is taken first)",
            highlight=False,
        )
        return

    # Only when there is something to snapshot: merging into a recording that has never
    # been labeled creates the file, and there is no prior state to preserve.
    snapshot = _snapshot(dest_path) if dest_path.exists() else None
    save_labels(
        dest_path,
        dest,
        identity=dest_identity,
        # The source's subject id is taken only to fill a gap, never to overwrite: it groups
        # one specimen's several clips and shares an absence declaration between them, so
        # losing it costs that grouping -- but the destination's own answer wins.
        subject_id=dest.subject_id or source.subject_id,
    )
    proj.bump_iteration()
    console.print(f"[green]merged[/green] into {dest_path}")
    if snapshot is not None:
        console.print(f"pre-merge snapshot: {snapshot}", highlight=False)
    console.print(f"project iteration is now {proj.iteration}", highlight=False)
    if report.unresolved:
        queue = _write_queue(proj, entry.slug, report)
        console.print(
            f"[yellow]{len(report.unresolved)} cell(s) need a human decision[/yellow] "
            f"-- written to {queue}. They were left as the destination had them.",
            highlight=False,
        )


def _print(report, slug: str, source_path: Path, *, applying: bool) -> None:
    """Print the reconciliation, most dangerous facts first."""
    _info_line("recording:", slug)
    _info_line("source:   ", str(source_path))
    for message in report.fatal:
        console.print(f"[red]fatal:[/red] {message}", highlight=False)
    for note in report.notes:
        console.print(f"[yellow]note:[/yellow] {note}", highlight=False)
    if not report.ok:
        return

    table = Table(title="would merge" if not applying else "merged")
    table.add_column("what", style="bold")
    table.add_column("n", justify="right")
    rows = [
        ("points matched by name", len(report.points.matched)),
        ("cameras matched by name", len(report.cameras.matched)),
        ("cells taken from the source", report.taken_from_source),
        ("cells kept from the destination", report.kept_from_dest),
        ("cells already identical", report.identical),
        ("cells dropped (no destination point)", report.dropped_cells),
        ("occlusions taken", report.occluded_taken),
        ("instance seeds taken", report.seeds_taken),
        ("frames newly carrying an annotation skeleton", report.instances_added),
        ("absence declarations added", report.absent_union),
        ("frames newly marked reviewed", report.reviewed_added),
        ("conflicts needing a human", len(report.unresolved)),
    ]
    for label, count in rows:
        table.add_row(label, f"{count:,}")
    console.print(table)
    for decision in report.unresolved[:5]:
        console.print(
            f"  conflict view={decision.view} frame={decision.frame} "
            f"point={decision.point}: {decision.reason}",
            highlight=False,
        )
    if len(report.unresolved) > 5:
        console.print(f"  ... and {len(report.unresolved) - 5} more", highlight=False)


def _snapshot(path: Path) -> Path:
    """Copy ``path`` aside before it is rewritten.

    Mandatory rather than optional: the destination is irreplaceable hand work, and a merge
    that could not be undone would be a merge nobody should run.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = path.with_name(f"{path.stem}.premerge-{stamp}{path.suffix}")
    shutil.copy2(path, dest)
    return dest


def _write_queue(project, slug: str, report) -> Path:
    """Write the unresolved conflicts for review, as plain JSON."""
    out = project.root / "exports" / f"merge_conflicts_{slug}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "recording": slug,
                "summary": report.summary(),
                "conflicts": [
                    {
                        "view": c.view,
                        "frame": c.frame,
                        "point": c.point,
                        "reason": c.reason,
                        "ours_xy": c.ours_xy,
                        "theirs_xy": c.theirs_xy,
                    }
                    for c in report.unresolved
                ],
            },
            indent=2,
        )
    )
    return out
