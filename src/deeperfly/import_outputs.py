"""Pulling a stray ``deeperfly_outputs/`` into the project that already indexes it.

The gap this closes is the one :meth:`Project._warn_unseen_labels` reports and cannot fix.
A recording is one entry, so a second label set sitting beside a second copy of the same
footage reads as zero -- and that warning's two suggestions both lose something: re-point
the entry (abandoning one set) or re-adopt under an explicit ``--id`` (duplicating the
recording). Reconciling them is a merge, and :mod:`deeperfly.merge` has done that since it
landed. This connects the two.

**Identification is the part a merge cannot do for itself.** ``deeperfly labels-merge``
makes the operator name the destination recording, when content identity
(:func:`deeperfly.project.recording_id`) already knows it. Three answers, and the middle one
is the whole feature:

.. code-block:: text

    same id, and the project already READS that very file  -> nothing to do
    same id, a different file                              -> merge
    no matching id                                         -> that is `project add`

The last is a refusal, not a fallback. ``project add`` adopts by *symlink*, so a recording
the project does not know is better adopted than imported -- there is then no second copy to
reconcile, ever. Doing it here would hand the operator a new recording when they meant to
update one.

**Safety.** The source is opened read-only and never written; ``results.h5`` is never
written at all; every destination is snapshotted before the first write; the default is a dry
run; and nothing is ever copied by index -- points and cameras go through
:func:`deeperfly.merge.map_by_name`, landmarks by name against the project's own set.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .merge import MergeReport, merge_labels
from .project import OUTPUTS_DIRNAME, Project, RecordingEntry, label_stats

__all__ = [
    "OutputsSource",
    "OutputsImport",
    "ABSENT_POLICIES",
    "find_outputs",
    "identify",
    "import_outputs",
]

log = logging.getLogger("deeperfly")

#: What to do when the source declares a keypoint absent that the destination has pixels for.
#: ``union`` accepts the declaration (quarantining those pixels -- recoverable, but the
#: project's live GT total drops); ``ours`` ignores the source's declarations entirely.
ABSENT_POLICIES = ("union", "ours")


@dataclass(frozen=True)
class OutputsSource:
    """One ``deeperfly_outputs/`` on disk, and what it holds."""

    path: Path
    labels: Path | None
    results: Path | None
    rec_id: str | None = None
    id_basis: str = "none"
    stats: dict = field(default_factory=dict)
    format_version: int | None = None
    #: Every content id this recording could legitimately have, because the camera KEY is
    #: interpolated into the fingerprint and this rig names the same seven cameras three
    #: ways: ``project add``'s discovery uses file stems, a ``results.h5`` records view
    #: names, and a configured discovery uses source names. One recording, one set of
    #: footage bytes, up to three ids -- so all of them are computed and any may match.
    #: Never a *weaker* identity: every candidate still hashes basename + byte size.
    rec_id_candidates: tuple[str, ...] = ()

    #: Cells whose instance carries a seed. Counted separately from ``stats`` because a seed
    #: is not ground truth and must never be totalled with it (see the module docstring).
    seed_cells: int = 0

    @property
    def has_authored_state(self) -> bool:
        """Whether there is any human work here at all.

        Zero GT is **not** empty. Hidden marks, absence declarations and reviewed flags have
        no producer but the operator, so a set with 400 hidden marks and no pixels is still
        400 pieces of irreplaceable work -- and seeds mean an instance was *created*, which is
        also a gesture only a human makes.
        """
        s = self.stats
        return bool(
            s.get("gt_points")
            or s.get("occluded")
            or s.get("reviewed_frames")
            or s.get("absent_points")
            or self.seed_cells
        )


@dataclass
class OutputsImport:
    """What importing one source did, or would do."""

    source: OutputsSource
    outcome: str  # "merged" | "noop" | "unindexed" | "empty" | "fatal"
    reason: str = ""
    entry: RecordingEntry | None = None
    matched_by: str = ""
    merge: MergeReport | None = None
    landmarks_taken: int = 0
    landmarks_only_source: list[str] = field(default_factory=list)
    quarantined_dest_gt: int = 0
    subject_id_taken: str | None = None
    snapshot: Path | None = None

    # The loaded state, carried between the dry run and the write. Not part of the report.
    dest_identity: dict | None = field(default=None, repr=False)
    dest_labels: object | None = field(default=None, repr=False)
    dest_landmarks: object | None = field(default=None, repr=False)

    @property
    def slug(self) -> str:
        return self.entry.slug if self.entry is not None else str(self.source.path)


# -- discovery -----------------------------------------------------------------


def find_outputs(path: str | Path) -> list[OutputsSource]:
    """Every ``deeperfly_outputs/`` at or under ``path``.

    Accepts what a user already has a path to: an outputs directory, a recording directory
    holding one, a ``labels.h5``/``results.h5``, or a tree containing many. A tree is the
    case that matters -- the measured problem is 21 label files across three directory
    trees, and a command that takes one path does not solve it.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")
    if root.is_file():
        return [_describe(root.parent)]

    here = root if _is_outputs(root) else root / OUTPUTS_DIRNAME
    if _is_outputs(here):
        return [_describe(here)]

    found = sorted(
        {p for p in root.rglob(OUTPUTS_DIRNAME) if _is_outputs(p)},
        key=lambda p: str(p),
    )
    # A bare directory of loose labels.h5/results.h5 is still a source.
    if not found and (root / "labels.h5").exists():
        return [_describe(root)]
    return [_describe(p) for p in found]


def _is_outputs(path: Path) -> bool:
    return path.is_dir() and (
        (path / "labels.h5").exists() or (path / "results.h5").exists()
    )


def _describe(outputs: Path) -> OutputsSource:
    """Read one outputs directory's identity and counts, without writing anything."""
    labels = outputs / "labels.h5"
    results = outputs / "results.h5"
    stats = label_stats(labels) if labels.exists() else {}
    rec_id = basis = None
    candidates: tuple[str, ...] = ()
    if results.exists():
        rec_id, basis, candidates = _id_from_results(results)
    return OutputsSource(
        path=outputs,
        labels=labels if labels.exists() else None,
        results=results if results.exists() else None,
        rec_id=rec_id,
        id_basis=basis or "none",
        stats=stats,
        format_version=stats.get("format_version"),
        seed_cells=_seed_cells(labels) if labels.exists() else 0,
        rec_id_candidates=candidates,
    )


def _seed_cells(labels: Path) -> int:
    """How many cells carry an instance seed, read straight out of HDF5."""
    import h5py

    try:
        with h5py.File(labels, "r") as f:
            if "seeds/index" not in f:
                return 0
            import numpy as np

            return int(len(np.asarray(f["seeds/index"][()]).reshape(-1, 4)))
    except Exception:
        return 0


def _id_from_results(
    results: Path,
) -> tuple[str | None, str | None, tuple[str, ...]]:
    """``(rec_id, basis, every candidate id)`` from a ``results.h5``'s recorded footage.

    ``(None, None, ())`` when there is nothing recording-specific to fingerprint. See
    :attr:`OutputsSource.rec_id_candidates` for why there is more than one candidate.
    """
    from .project import recording_fingerprint, recording_id
    from .results import StageStore

    try:
        store = StageStore(results)
        result_footage = store.read_footage()
        pose2d = store.read_pose2d()
        n_frames = int(pose2d[0].shape[1]) if pose2d is not None else None
    except Exception as exc:
        log.warning("could not read %s: %s", results, exc)
        return None, None, ()
    if not result_footage:
        return None, None, ()

    # The footage basis first when the files still resolve, so an outputs dir sitting beside
    # its own footage lands on the same id `project add` gave it.
    footage = _resolve_recorded_footage(result_footage, results.parent)
    try:
        text, basis = recording_fingerprint(
            footage, result_footage=result_footage, n_frames=n_frames
        )
    except ValueError:
        return None, None, ()
    primary = recording_id(text)

    ids = {primary}
    if footage:
        # The same files re-keyed the other ways this rig names cameras. Each is a full
        # basename+byte-size fingerprint of the identical file set -- only the key differs.
        for rekey in (_by_stem, _by_parent_stem):
            alt = rekey(footage)
            if alt and alt != footage:
                try:
                    alt_text, _ = recording_fingerprint(alt)
                except ValueError:
                    continue
                ids.add(recording_id(alt_text))
    return primary, basis, tuple(sorted(ids))


def _by_stem(footage: dict) -> dict:
    """Re-key ``camera -> files`` by each camera's first file stem (``project add``'s way)."""
    out: dict[str, list[Path]] = {}
    for files in footage.values():
        if not files:
            return {}
        out[Path(files[0]).stem] = list(files)
    return out if len(out) == len(footage) else {}


def _by_parent_stem(footage: dict) -> dict:
    """Re-key by the containing directory's name -- the image-sequence-per-folder layout."""
    out: dict[str, list[Path]] = {}
    for files in footage.values():
        if not files:
            return {}
        out[Path(files[0]).parent.name] = list(files)
    return out if len(out) == len(footage) else {}


def _resolve_recorded_footage(result_footage: dict, anchor: Path) -> dict:
    """``camera -> [existing Path]`` from a ``read_footage`` map, or ``{}`` if any is gone.

    All-or-nothing on purpose: a partial footage basis would hash a different string than
    the one adoption produced, yielding a *third* id for one recording.
    """
    out: dict[str, list[Path]] = {}
    for camera, spec in (result_footage or {}).items():
        paths: list[Path] = []
        for flavor, base in (("abs", None), ("rel", anchor)):
            raw = (spec or {}).get(flavor) if isinstance(spec, dict) else None
            if not raw:
                continue
            candidates = [
                Path(p) if base is None else (base / str(p)).resolve() for p in raw
            ]
            if all(p.exists() for p in candidates):
                paths = candidates
                break
        if not paths:
            return {}
        out[camera] = paths
    return out


# -- identification ------------------------------------------------------------


def identify(
    project: Project, source: OutputsSource, *, recording: str | None = None
) -> tuple[RecordingEntry | None, str]:
    """``(entry, how)`` -- which indexed recording ``source`` is.

    Only two ways, and there is deliberately no third:

    1. an explicit ``recording`` slug/id -- the escape hatch;
    2. the content id (:func:`deeperfly.project.recording_id`).

    **Not** by the labels identity's footage basenames, which is the tempting fallback for an
    archived recording whose footage no longer resolves. It cannot be made safe here: on this
    rig every recording's files are named ``camera_RH.mp4`` ... ``camera_LH.mp4`` (the
    project module docstring says so outright), so basenames identify *nothing* and two
    unrelated flies compare equal. Adding ``image_sizes`` does not help -- one rig, one
    resolution -- and adding ``n_frames`` only narrows it, so two clips of equal length would
    still merge into each other. Merging one fly's ground truth into another's is
    unrecoverable in practice; requiring ``--recording`` costs one flag.
    """
    if recording:
        return project.recording(recording), "named"
    candidates = source.rec_id_candidates or ((source.rec_id,) if source.rec_id else ())
    hits = [e for e in project.recordings if e.id in candidates]
    if len(hits) == 1:
        exact = hits[0].id == source.rec_id
        return hits[0], (
            f"content id ({source.id_basis} basis)"
            if exact
            else f"content id ({source.id_basis} basis, re-keyed camera naming)"
        )
    if len(hits) > 1:
        # Two indexed entries answering to one set of footage bytes: the duplicate-adoption
        # case. Refuse rather than pick -- merging into the wrong one of two copies of the
        # same recording is the exact confusion this command exists to end.
        raise KeyError(
            f"{len(hits)} indexed recordings match this footage "
            f"({', '.join(e.slug for e in hits)}) -- they are duplicate adoptions of one "
            "recording. Name the one to merge into with --recording"
        )
    return None, ""


# -- the import ----------------------------------------------------------------


def import_outputs(
    project: Project,
    sources,
    *,
    recording: str | None = None,
    on_conflict: str = "manual",
    on_absent: str = "union",
    apply: bool = False,
) -> list[OutputsImport]:
    """Merge each source's authored state into the recording the project already indexes.

    Parameters
    ----------
    project
        The destination :class:`~deeperfly.project.Project`.
    sources
        :class:`OutputsSource` values (from :func:`find_outputs`).
    recording
        Force every source onto this recording (slug, id, or id prefix). The escape hatch
        for an archived recording whose id cannot be derived.
    on_conflict
        One of :data:`deeperfly.merge.CONFLICT_POLICIES` for cells both sides authored.
    on_absent
        One of :data:`ABSENT_POLICIES`.
    apply
        When false nothing is written -- the dry run, which is the default.

    Returns
    -------
    list of OutputsImport
        One row per source, in the order given. A per-source failure is a row, not an
        exception: one unreadable outputs directory must not abandon the rest.

    Raises
    ------
    ValueError
        If ``on_absent`` is unknown.
    """
    if on_absent not in ABSENT_POLICIES:
        raise ValueError(
            f"on_absent must be one of {list(ABSENT_POLICIES)}, got {on_absent!r}"
        )
    plans = [
        _plan(
            project,
            s,
            recording=recording,
            on_conflict=on_conflict,
            on_absent=on_absent,
        )
        for s in sources
    ]
    if not apply:
        return plans

    mergeable = [p for p in plans if p.outcome == "merged"]
    # Every snapshot before the first write, so a batch that dies half-way still leaves
    # every original recoverable rather than only the ones it had not reached.
    for plan in mergeable:
        dest = project.labels_path(plan.entry)
        if dest.exists():
            plan.snapshot = _snapshot(dest)
    for plan in mergeable:
        _write(project, plan, on_conflict=on_conflict, on_absent=on_absent)
    if mergeable:
        project.bump_iteration()
    return plans


def _plan(
    project: Project,
    source: OutputsSource,
    *,
    recording: str | None,
    on_conflict: str,
    on_absent: str,
) -> OutputsImport:
    """Decide what one source is, and dry-run its merge."""
    if source.labels is None:
        return OutputsImport(
            source,
            "empty",
            f"no labels.h5 in {source.path} -- there are no corrections here, only "
            "predictions. Importing predictions as ground truth is not something this "
            "command does",
        )
    if not source.has_authored_state:
        return OutputsImport(
            source, "empty", f"{source.labels} carries no authored state at all"
        )

    try:
        entry, how = identify(project, source, recording=recording)
    except KeyError as exc:
        return OutputsImport(source, "fatal", str(exc).strip("'"))
    if entry is None and source.rec_id is None:
        return OutputsImport(
            source,
            "unindexed",
            "this recording cannot be identified by content: its footage does not resolve "
            "and its results.h5 records none either, so there is nothing recording-specific "
            "to fingerprint (basenames repeat across every recording on this rig). Name the "
            "destination explicitly:\n    deeperfly project import-outputs "
            f"{project.root} {source.path} --recording <slug>",
        )
    if entry is None:
        return OutputsImport(
            source,
            "unindexed",
            f"no indexed recording has content id {source.rec_id}. This is an adoption, not "
            "an import -- and adopting is better, because it SYMLINKS the outputs and needs "
            f"no merge at all:\n    deeperfly project add {project.root} "
            f"{source.path.parent}\n  (if it IS an indexed recording whose id was derived "
            "under a different camera naming, pass --recording <slug>)",
        )

    dest_path = project.labels_path(entry)
    if dest_path.exists() and dest_path.resolve() == source.labels.resolve():
        return OutputsImport(
            source,
            "noop",
            "this project already reads that very file "
            f"({project.recording_dir(entry) / OUTPUTS_DIRNAME} -> {source.path}). "
            "Corrections made in the standalone editor are already here",
            entry=entry,
            matched_by=how,
        )

    plan = OutputsImport(source, "merged", "", entry=entry, matched_by=how)
    try:
        _run_merge(
            project, plan, on_conflict=on_conflict, on_absent=on_absent, apply=False
        )
    except SystemExit as exc:  # a refusal from the identity readers
        return OutputsImport(source, "fatal", str(exc), entry=entry, matched_by=how)
    except Exception as exc:
        return OutputsImport(
            source, "fatal", f"{type(exc).__name__}: {exc}", entry=entry, matched_by=how
        )
    if plan.merge is not None and not plan.merge.ok:
        plan.outcome = "fatal"
        plan.reason = "; ".join(plan.merge.fatal)
    return plan


def _run_merge(
    project: Project,
    plan: OutputsImport,
    *,
    on_conflict: str,
    on_absent: str,
    apply: bool,
) -> None:
    """Load both sides, merge them, and record the outcome on ``plan``.

    Fills ``plan.merge`` plus the landmark / subject / quarantine facts the report needs.
    Writes nothing; :func:`_write` does that.
    """
    import numpy as np

    from .cli.merge import _derived_identity, _identity
    from .gui.labels import Labels, load_labels

    entry = plan.entry
    dest_path = project.labels_path(entry)
    source_identity = _identity(plan.source.labels)
    # The source is loaded against its OWN identity, so `_check_identity` compares it with
    # itself and passes: it is reconciled by name afterwards, not by pre-agreement.
    source = load_labels(plan.source.labels, identity=source_identity)
    if source is None:
        raise ValueError(f"{plan.source.labels} holds no labels")

    if dest_path.exists():
        dest_identity = _identity(dest_path)
        dest = load_labels(dest_path, identity=dest_identity)
        if dest is None:
            raise ValueError(f"{dest_path} holds no labels")
    else:
        dest_identity = _derived_identity(project, entry) or _source_as_destination(
            plan, source_identity
        )
        dest = Labels.empty(
            len(dest_identity["camera_names"]),
            int(dest_identity["n_frames"]),
            len(dest_identity["point_names"]),
        )
    plan.dest_identity = dest_identity
    plan.dest_labels = dest

    if on_absent == "ours":
        # Drop the source's declarations before the merge sees them, rather than un-doing
        # them after: a declaration that never arrives cannot quarantine anything.
        source.absent = np.zeros_like(np.asarray(source.absent, dtype=bool))

    plan.merge = merge_labels(
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
    if not plan.merge.ok:
        return

    # A camera the source has and the destination does not takes every one of its cells with
    # it, silently, into `dropped_cells` -- which counts POINTS elsewhere. Name it.
    if plan.merge.cameras.only_source:
        plan.merge.notes.append(
            f"camera(s) {plan.merge.cameras.only_source} exist only in the source; every "
            "cell of theirs is dropped (this rig names cameras three different ways -- "
            "file stems, source names, view names -- so check which the two sides used)"
        )
    if not plan.merge.cameras.source_to_dest:
        plan.merge.fatal.append(
            f"not one camera name matches: source {list(source_identity['camera_names'])} "
            f"vs destination {list(dest_identity['camera_names'])}. Every cell would be "
            "dropped, which would report as a successful zero-cell merge"
        )
        return
    if int(source_identity.get("n_frames") or 0) > int(
        dest_identity.get("n_frames") or 0
    ):
        plan.merge.notes.append(
            f"the source has {source_identity['n_frames']} frames and the destination "
            f"{dest_identity['n_frames']}; cells beyond frame "
            f"{int(dest_identity['n_frames']) - 1} are dropped"
        )
    if plan.source.format_version and plan.source.format_version < 8:
        plan.merge.notes.append(
            f"the source is labels v{plan.source.format_version}; its per-frame instance "
            "flags are inferred from its seeds"
        )

    # What the incoming absence declarations would cost, computed before it is paid.
    absent = np.asarray(dest.absent, dtype=bool)[None, :, :]
    plan.quarantined_dest_gt = int((dest.gt_authored & absent).sum())
    if on_absent == "ours":
        plan.quarantined_dest_gt = 0

    plan.landmarks_taken, plan.landmarks_only_source = _plan_landmarks(
        project, plan, dest_identity, apply=apply
    )
    if not dest.subject_id and source.subject_id:
        plan.subject_id_taken = source.subject_id


def _source_as_destination(plan: OutputsImport, source_identity: dict) -> dict:
    """Use the source's identity for a destination that has neither labels nor a result.

    Legitimate *because the content ids matched*: the source describes the same recording, so
    its camera order and frame count are that recording's. Reported on the identity line, so
    it is never a silent assumption.
    """
    plan.matched_by += (
        " (identity taken from the source -- the destination has neither)"
    )
    return dict(source_identity)


def _plan_landmarks(
    project: Project, plan: OutputsImport, dest_identity: dict, *, apply: bool
) -> tuple[int, list[str]]:
    """Merge the source's landmark observations by name, intersected with the project's set.

    Landmarks are **project-scoped**: the names, and whether each is one fixed 3D point, come
    from the project's ``landmarks.toml``, not from either label file. So a name the project
    does not declare has nowhere to go, and ``static`` always comes from the project --
    importing the source's flags would let one recording silently redefine the geometry of
    every solve on the rig.

    Observations are additive: taken only where the destination has none. A cell both sides
    observed differently is left as the destination has it (there is no landmark conflict
    queue, and an operator's own placement in *this* project outranks an imported one).
    """
    import numpy as np

    from .gui.labels import LandmarkLabels, load_landmark_labels

    n_views = len(dest_identity["camera_names"])
    n_frames = int(dest_identity["n_frames"])
    source_lm = load_landmark_labels(
        plan.source.labels, n_views=n_views, n_frames=n_frames
    )
    if source_lm is None:
        return 0, []

    from .landmarks import LandmarkSet

    declared = LandmarkSet.load(project.root)
    if not len(declared):
        # Nothing declared: the observations have no namespace to land in. Reported rather
        # than dropped silently, because `deeperfly calibrate` is what wanted them.
        return 0, list(source_lm.names)

    dest_lm = load_landmark_labels(
        project.labels_path(plan.entry), n_views=n_views, n_frames=n_frames
    )
    if dest_lm is None or list(dest_lm.names) != declared.names:
        dest_lm = LandmarkLabels.empty(
            n_views, n_frames, declared.names, declared.static_mask
        )

    taken = 0
    only_source = [n for n in source_lm.names if n not in declared.names]
    for j, name in enumerate(source_lm.names):
        if name not in declared.names:
            continue
        k = declared.index(name)
        src = source_lm.xy[:, :, j]
        fresh = np.isfinite(src).all(axis=-1) & ~np.isfinite(dest_lm.xy[:, :, k]).all(
            axis=-1
        )
        taken += int(fresh.sum())
        if apply and fresh.any():
            dest_lm.xy[:, :, k][fresh] = src[fresh]
            dest_lm.dirty = True
    plan.dest_landmarks = dest_lm
    return taken, only_source


def _write(
    project: Project, plan: OutputsImport, *, on_conflict: str, on_absent: str
) -> None:
    """Apply one planned import. The snapshot is already taken."""
    from .gui.labels import load_landmark_labels, save_labels

    _run_merge(project, plan, on_conflict=on_conflict, on_absent=on_absent, apply=True)
    if plan.merge is None or not plan.merge.ok:
        plan.outcome = "fatal"
        plan.reason = "; ".join(plan.merge.fatal) if plan.merge else "merge failed"
        return

    dest_path = project.labels_path(plan.entry)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest = plan.dest_labels
    identity = plan.dest_identity
    landmarks = plan.dest_landmarks
    if landmarks is None:
        # Not merged (the project declares none), but still reloaded and passed back:
        # save_labels is a whole-file rewrite, so omitting them deletes the group.
        landmarks = load_landmark_labels(
            dest_path,
            n_views=len(identity["camera_names"]),
            n_frames=int(identity["n_frames"]),
        )
    save_labels(
        dest_path,
        dest,
        identity=identity,
        subject_id=dest.subject_id or plan.subject_id_taken,
        landmarks=landmarks,
    )


def _snapshot(path: Path) -> Path:
    """Copy a destination aside before it is rewritten.

    ``preimport-`` rather than ``premerge-`` so a snapshot's provenance stays legible: the
    two commands can both have touched one file.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = path.with_name(f"{path.stem}.preimport-{stamp}{path.suffix}")
    shutil.copy2(path, dest)
    return dest
