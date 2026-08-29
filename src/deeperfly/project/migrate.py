"""Skeleton edits as typed migrations -- the guard on the sharpest hazard in the project.

A skeleton's ``point_names`` is fingerprinted into every ``labels.h5``
(:func:`deeperfly.labels.store.labels_identity`) *and* written into every ``results.h5``. So a
GUI that lets an operator edit the skeleton can invalidate every label in a project with one
click, and the invalidation is not even loud: two 38-point skeletons in different orders load
each other's files happily and mean something completely different by every index.

This module makes each edit a **typed migration** with a declared effect, a dry run that
counts what it would touch, and -- for anything destructive -- a refusal to proceed silently.

.. code-block:: text

    add a point         none (a new column, all-unset)                       silent
    rename a point      remap by identity; names rewritten in labels.h5      notice
    reorder points      remap by name; on-disk COO indices rewritten         notice
    add/remove an edge  none (edges are display + the BA prior only)         silent
    change a colour     none                                                 silent
    change symmetries   none (read by flip augmentation)                     silent
    delete a point      its labels are QUARANTINED, not deleted              confirm

The one rule everything else follows from: **labels move by name, never by index.** The same
rule merging obeys (:mod:`deeperfly.labels.merge`), for the same reason, and the reason index-based
copying is not implemented anywhere in this package.

Deletion reuses the existing quarantine mechanism rather than inventing one: an absence
declaration already parks vetoed rows under ``absent/void_*`` and restores them if it is
lifted (see :mod:`deeperfly.labels.store`). A deleted point's labels go the same way, so
"delete" is recoverable by re-adding the point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "SkeletonChange",
    "MigrationPlan",
    "diff_skeletons",
    "expand_renames",
    "plan_migration",
    "apply_migration",
]

log = logging.getLogger("deeperfly")

#: Change kinds, and whether each needs the operator to confirm.
#:
#: Only deletion does. Everything else either cannot lose a label (adding, edges, colours) or
#: moves it deterministically by name (rename, reorder) -- and asking about a safe edit trains
#: people to click through the dangerous one.
DESTRUCTIVE = ("delete",)


@dataclass(frozen=True)
class SkeletonChange:
    """One difference between two skeletons."""

    kind: str  # "add"|"delete"|"rename"|"reorder"|"edges"|"colors"|"symmetries"
    detail: str
    points: tuple[str, ...] = ()

    @property
    def destructive(self) -> bool:
        return self.kind in DESTRUCTIVE


@dataclass
class MigrationPlan:
    """What changing a project's skeleton would do to its labels.

    Attributes
    ----------
    changes
        The diff, in reporting order.
    mapping
        ``old index -> new index`` for the points that survive. The *only* way labels are
        moved; a point absent here loses its position and its labels are quarantined.
    affected
        ``labels path -> {"gt": n, "occluded": n, "quarantined": n}`` -- what each sidecar
        would gain or lose. Counted before anything is written, because "this will quarantine
        1,412 labels across 6 recordings" is a decision and "done" is not.
    """

    old_names: tuple[str, ...]
    new_names: tuple[str, ...]
    changes: list[SkeletonChange] = field(default_factory=list)
    mapping: dict[int, int] = field(default_factory=dict)
    affected: dict[str, dict] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def destructive(self) -> bool:
        return any(c.destructive for c in self.changes)

    @property
    def trivial(self) -> bool:
        """Whether nothing about the *label indexing* changes.

        Bones, colours and symmetry pairs are all display or downstream-policy
        metadata: no label row moves and no sidecar is rewritten, so such an edit needs
        neither a rewrite nor a confirmation.
        """
        return all(c.kind in ("edges", "colors", "symmetries") for c in self.changes)

    @property
    def quarantined(self) -> int:
        return sum(a.get("quarantined", 0) for a in self.affected.values())

    @property
    def moved(self) -> int:
        return sum(a.get("moved", 0) for a in self.affected.values())

    def summary(self) -> dict:
        return {
            "changes": [
                {"kind": c.kind, "detail": c.detail, "points": list(c.points)}
                for c in self.changes
            ],
            "destructive": self.destructive,
            "trivial": self.trivial,
            "moved": self.moved,
            "quarantined": self.quarantined,
            "affected": self.affected,
            "errors": self.errors,
        }


def _check_renames(
    renames: dict[str, str], gone: list[str], fresh: list[str]
) -> dict[str, str]:
    """Validate a declared ``{old: new}`` against what the edit actually did.

    A declaration is only a rename if the old name really left and the new one really
    arrived. Anything else is a different edit wearing a rename's clothes -- renaming onto
    a point that already exists merges two points' labels into one column, and renaming a
    point the new skeleton still has silently duplicates it -- so every one of those is a
    refusal, not a warning.
    """
    if not renames:
        return {}
    problems = []
    for was, now in renames.items():
        if was not in gone:
            problems.append(
                f"{was!r} is not a point the new skeleton dropped, so renaming it "
                "would not move any label"
            )
        if now not in fresh:
            problems.append(
                f"{now!r} is not a point the new skeleton introduced, so renaming "
                f"{was!r} onto it would merge two points into one"
            )
    targets = list(renames.values())
    for now in sorted({n for n in targets if targets.count(n) > 1}):
        problems.append(f"{now!r} is the target of more than one rename")
    if problems:
        raise ValueError("; ".join(problems))
    return dict(renames)


def expand_renames(specs: list[str], old_names, new_names=None) -> dict[str, str]:
    """``["*_claw=*_pretarsus"]`` -> ``{"lf_claw": "lf_pretarsus", ...}``.

    The command-line form of a declared rename. A ``*`` stands for a run of characters
    captured from the old name and substituted into the new one, which is what makes the
    six-legs-at-once case one flag instead of six: the ``lf_``/``rh_`` prefixes this
    package's point names are built from are exactly what varies. At most one ``*`` per
    side, and both sides must agree on whether there is one.

    A pattern that matches no point raises rather than expanding to nothing -- a silent
    no-op here means the migration falls through to delete-plus-add and quarantines the
    labels the rename was written to save. ``new_names`` additionally checks that every
    expansion lands on a point the new skeleton has; pass ``None`` where there is no new
    skeleton to check against (renaming a checkpoint's recorded channels).
    """
    out: dict[str, str] = {}
    for spec in specs:
        was, sep, now = spec.partition("=")
        if not sep or not was or not now:
            raise ValueError(f"--rename wants OLD=NEW, got {spec!r}")
        if was.count("*") > 1 or now.count("*") > 1:
            raise ValueError(f"--rename allows at most one '*' per side, got {spec!r}")
        if ("*" in was) != ("*" in now):
            raise ValueError(
                f"--rename needs a '*' on both sides or neither, got {spec!r}"
            )
        if "*" not in was:
            matched = [was] if was in tuple(old_names) else []
            out[was] = now
        else:
            head, _, tail = was.partition("*")
            matched = [
                n
                for n in old_names
                if len(n) >= len(head) + len(tail)
                and n.startswith(head)
                and n.endswith(tail)
            ]
            for name in matched:
                stem = (
                    name[len(head) : len(name) - len(tail)]
                    if tail
                    else name[len(head) :]
                )
                out[name] = now.replace("*", stem)
        if not matched:
            raise ValueError(
                f"--rename {spec!r} matched no point in the project's skeleton"
            )
    missing = (
        []
        if new_names is None
        else [n for n in out.values() if n not in tuple(new_names)]
    )
    if missing:
        raise ValueError(
            f"--rename would produce {missing}, which the new skeleton does not have"
        )
    return out


def diff_skeletons(
    old, new, *, renames: dict[str, str] | None = None
) -> tuple[list[SkeletonChange], dict[int, int]]:
    """``(changes, old->new index mapping)`` between two :class:`~deeperfly.skeleton.Skeleton`.

    A rename is a claim that two differently-named points **mean the same thing**, and
    nothing in the two files says so: to a diff, "the claw point is now called pretarsus"
    and "the claw point is gone and a pretarsus point is new" are the same edit. So a
    rename is either declared or inferred from the one case where the inference is safe:

    ``renames``
        An explicit ``{old_name: new_name}`` the caller vouches for -- from
        ``deeperfly project skeleton --rename``, which is how a bulk rename (all six
        ``*_claw`` at once) is done. Each key must be a name the new skeleton dropped and
        each value one it introduced, or this raises: a "rename" of a point that still
        exists would silently merge two points into one.
    positional inference
        With no declaration, a name that vanished while exactly one new name appeared at
        the same index is a rename -- what an operator typing over a name did. Two at once
        is ambiguous (a whole block of points can be replaced in place, which is a
        different point set, not six renames), so each is reported as its own delete and
        add, and the delete then requires confirmation.
    """
    old_names, new_names = tuple(old.point_names), tuple(new.point_names)
    changes: list[SkeletonChange] = []

    gone = [n for n in old_names if n not in new_names]
    fresh = [n for n in new_names if n not in old_names]

    declared = _check_renames(renames or {}, gone, fresh)
    if declared:
        for was, now in declared.items():
            changes.append(SkeletonChange("rename", f"{was} -> {now}", (was, now)))
            gone.remove(was)
            fresh.remove(now)
    elif len(gone) == len(fresh) == 1:
        # One out, one in: a rename iff they sit at the same index, i.e. nothing moved.
        old_at = old_names.index(gone[0])
        new_at = new_names.index(fresh[0])
        if old_at == new_at:
            declared[gone[0]] = fresh[0]
            changes.append(
                SkeletonChange(
                    "rename",
                    f"{gone[0]} -> {fresh[0]}",
                    (gone[0], fresh[0]),
                )
            )
            gone, fresh = [], []
    renames = declared

    if fresh:
        changes.append(SkeletonChange("add", f"added {fresh}", tuple(fresh)))
    if gone:
        changes.append(SkeletonChange("delete", f"removed {gone}", tuple(gone)))

    resolved = {renames.get(n, n): i for i, n in enumerate(old_names)}
    mapping = {
        resolved[name]: j for j, name in enumerate(new_names) if name in resolved
    }
    if any(i != j for i, j in mapping.items()):
        changes.append(
            SkeletonChange("reorder", "the point order changed; labels remap by name")
        )

    if not np.array_equal(
        np.asarray(old.edges).reshape(-1, 2), np.asarray(new.edges).reshape(-1, 2)
    ):
        changes.append(
            SkeletonChange("edges", "the edges changed (display + the BA prior only)")
        )
    if tuple(old.point_colors) != tuple(new.point_colors) or tuple(
        old.edge_colors
    ) != tuple(new.edge_colors):
        changes.append(SkeletonChange("colors", "the colours changed"))
    # Symmetry is compared by NAME, not by index: a pure reorder moves both indices of
    # every pair, so an index comparison would report a symmetry change for an edit that
    # left the pairing untouched. Names are also what the emitted `[skeleton]` fragment
    # carries, so this compares what actually round-trips.
    if set(map(frozenset, old.symmetry_names)) != set(
        map(frozenset, new.symmetry_names)
    ):
        changes.append(
            SkeletonChange(
                "symmetries",
                "the left/right symmetry pairs changed (flip augmentation and the "
                "symmetrize correction read them; no label moves)",
            )
        )
    return changes, mapping


def plan_migration(
    project, new_skeleton, *, renames: dict[str, str] | None = None
) -> MigrationPlan:
    """What changing ``project``'s skeleton to ``new_skeleton`` would do.

    Counts, per ``labels.h5``, how many rows would **move** (their point index rewritten)
    and how many would be **quarantined** (their point is gone). Nothing is written.

    Parameters
    ----------
    project
        The :class:`~deeperfly.project.Project`.
    new_skeleton
        The proposed :class:`~deeperfly.skeleton.Skeleton`.
    renames
        Points the caller declares are the same point under a new name, ``{old: new}``.
        See :func:`diff_skeletons`: without it a bulk rename reads as delete-plus-add and
        quarantines every label on those points.

    Returns
    -------
    MigrationPlan
        The plan, with ``errors`` naming any sidecar that could not be read **or that is not
        on the axis the project declares** -- either of which blocks the migration rather
        than being skipped, because a half-migrated project is worse than an unmigrated one.

    Notes
    -----
    The axis check is the load-bearing one. ``mapping`` is derived from ``skeleton.toml``
    alone, and :func:`_rewrite` reads each sidecar's identity *out of the sidecar*, so
    :func:`~deeperfly.labels.store._check_identity` compares that file with itself and can
    never fire. Nothing else asks a ``labels.h5`` which point order its rows are actually
    on. Without this comparison, a project whose ``skeleton.toml`` has drifted from its
    sidecars -- by a hand edit, or just by adopting a recording produced under a different
    config, which :meth:`~deeperfly.project.Project.add_recording` does not check -- has
    every label silently transposed and then restamped as consistent.
    """
    old = project.skeleton()
    changes, mapping = diff_skeletons(old, new_skeleton, renames=renames)
    plan = MigrationPlan(
        old_names=tuple(old.point_names),
        new_names=tuple(new_skeleton.point_names),
        changes=changes,
        mapping=mapping,
    )
    survivors = set(mapping)
    for entry in project.recordings:
        path = project.labels_path(entry)
        if not path.exists():
            continue
        try:
            stored = _sidecar_point_names(path)
            counts = _count(path, survivors, mapping)
        except Exception as exc:
            plan.errors.append(f"{path}: {exc}")
            continue
        if stored is not None and tuple(stored) != plan.old_names:
            plan.errors.append(_axis_error(path, stored, plan.old_names))
            continue
        plan.affected[str(path)] = counts
    return plan


def _sidecar_point_names(path: Path) -> list[str] | None:
    """The point order a ``labels.h5``'s rows are actually on, or ``None`` if unstamped."""
    import json

    import h5py

    with h5py.File(path, "r") as f:
        meta = json.loads(f.attrs.get("meta", "{}"))
    names = (meta.get("identity") or {}).get("point_names")
    return None if names is None else [str(n) for n in names]


def _axis_error(path: Path, stored: list[str], declared: tuple[str, ...]) -> str:
    """Why this sidecar cannot be migrated, and what to do about it.

    Two distinguishable cases, because they call for different repairs: a pure reordering is
    a bookkeeping mismatch the project can be re-pointed at, while a different *set* of names
    means the sidecar was authored against another skeleton entirely.
    """
    if sorted(stored) == sorted(declared):
        return (
            f"{path} holds labels on a DIFFERENT POINT ORDER than the project declares "
            f"(the same {len(stored)} names, reordered). Migrating with the project's "
            "mapping would silently transpose every label in it. Point the project's "
            "skeleton.toml at the order these labels were authored on, then migrate"
        )
    only_sidecar = [n for n in stored if n not in declared]
    only_project = [n for n in declared if n not in stored]
    return (
        f"{path} holds labels on a different skeleton than the project declares "
        f"({len(stored)} points vs {len(declared)}"
        + (f"; only in the labels: {only_sidecar[:6]}" if only_sidecar else "")
        + (f"; only in the project: {only_project[:6]}" if only_project else "")
        + "). It was authored against another skeleton, so the project's mapping does not "
        "describe it -- reconcile it with 'deeperfly labels-merge', which maps by name"
    )


def _count(path: Path, survivors: set[int], mapping: dict[int, int]) -> dict:
    """How many rows in one sidecar would move / be quarantined."""
    import h5py

    from .core import _coo_rows

    moved = quarantined = occluded = 0
    with h5py.File(path, "r") as f:
        for key, is_gt in (("gt/index", True), ("occluded/index", False)):
            if key not in f:
                continue
            rows = _coo_rows(f[key][()])
            if not len(rows):
                continue
            points = rows[:, -1]
            keep = (
                np.isin(points, list(survivors))
                if survivors
                else np.zeros(len(points), dtype=bool)
            )
            changed = sum(
                1 for p in points[keep] if mapping.get(int(p), int(p)) != int(p)
            )
            if is_gt:
                moved += changed
                quarantined += int((~keep).sum())
            else:
                occluded += changed
                quarantined += int((~keep).sum())
    return {"moved": moved, "occluded_moved": occluded, "quarantined": quarantined}


def apply_migration(
    project, new_skeleton, plan: MigrationPlan, *, snapshot=True
) -> dict:
    """Rewrite every label sidecar onto ``new_skeleton``, then adopt it.

    Order matters and is deliberate: a **package snapshot first**, then every sidecar, then
    the skeleton file last. If a sidecar fails mid-way the project's declared skeleton still
    matches the un-migrated ones, so the state is consistent and the snapshot is the way
    back.

    Parameters
    ----------
    project, new_skeleton, plan
        As from :func:`plan_migration`.
    snapshot
        Write a pre-migration ``.dfpkg`` first. On by default and only disabled by tests:
        this is the one operation that can lose labels.

    Returns
    -------
    dict
        ``{"snapshot": path|None, "migrated": [paths], "moved": n, "quarantined": n}``.

    Raises
    ------
    ValueError
        If ``plan`` carries errors -- a half-migrated project is worse than an unmigrated
        one, so an unreadable sidecar blocks the whole thing.
    """
    if plan.errors:
        raise ValueError(
            "refusing to migrate: "
            + "; ".join(plan.errors)
            + ". Fix or remove those sidecars first -- a half-migrated project is worse "
            "than an unmigrated one"
        )
    result: dict = {"snapshot": None, "migrated": [], "moved": 0, "quarantined": 0}
    if snapshot:
        from .package import export_package

        stamp = _stamp()
        out = project.root / "exports" / f"pre-skeleton-{stamp}.dfpkg"
        export_package(project, out, embed="none")
        result["snapshot"] = str(out)

    for entry in project.recordings:
        path = project.labels_path(entry)
        if not path.exists():
            continue
        moved, quarantined = _rewrite(path, plan, new_skeleton)
        result["migrated"].append(str(path))
        result["moved"] += moved
        result["quarantined"] += quarantined

    # The skeleton file last: until it changes, the project still describes what the
    # sidecars contained.
    project.skeleton_path().write_text(_skeleton_toml(new_skeleton))
    project.bump_iteration()
    log.info(
        "migrated %d sidecar(s): %d label(s) moved, %d quarantined",
        len(result["migrated"]),
        result["moved"],
        result["quarantined"],
    )
    return result


def _rewrite(path: Path, plan: MigrationPlan, new_skeleton) -> tuple[int, int]:
    """Rewrite one sidecar onto the new point axis. Returns ``(moved, quarantined)``."""
    import json

    import h5py

    from ..labels.merge import SkeletonMapping, remap_labels
    from ..labels.store import load_labels, save_labels

    with h5py.File(path, "r") as f:
        identity = json.loads(f.attrs["meta"])["identity"]
    labels = load_labels(path, identity=identity)
    if labels is None:
        return 0, 0
    n_views = len(identity["camera_names"])
    n_frames = int(identity["n_frames"])

    before = int(labels.gt_authored.sum())
    mapping = SkeletonMapping(
        source_to_dest=dict(plan.mapping),
        matched=[plan.new_names[j] for j in plan.mapping.values()],
        reordered=any(i != j for i, j in plan.mapping.items()),
    )
    identity_cams = SkeletonMapping(
        source_to_dest={i: i for i in range(n_views)},
        matched=list(identity["camera_names"]),
    )
    migrated = remap_labels(
        labels,
        points=mapping,
        cameras=identity_cams,
        n_views=n_views,
        n_frames=n_frames,
        n_points=len(plan.new_names),
    )
    after = int(migrated.gt_authored.sum())
    moved = sum(
        1
        for old, new in plan.mapping.items()
        if old != new and labels.gt_authored[:, :, old].any()
    )

    new_identity = dict(identity)
    new_identity["point_names"] = list(plan.new_names)
    save_labels(
        path,
        migrated,
        identity=new_identity,
        subject_id=labels.subject_id,
    )
    return moved, max(0, before - after)


def _skeleton_toml(skeleton) -> str:
    """A skeleton as a ``[skeleton]`` file: points, edges, symmetries, colours.

    Written from the object rather than lifted from a file, because a migrated skeleton has
    no source file yet. Every one of the four is emitted by NAME, so the fragment survives
    a later reorder -- and all four are emitted, because a migration rewrites the whole
    table: dropping the symmetries here would silently disable flip augmentation on the
    first skeleton edit a project ever makes, and dropping the colours would drop the
    operator's palette back to a colormap.
    """
    from .. import _toml

    lines = [
        "# The project's skeleton. Rewritten by a skeleton migration; the prose from the",
        "# original file is not preserved, but the structure is exact.",
        "[skeleton]",
        f"name = {_toml.value(skeleton.name)}",
        f"points = {_toml.value(list(skeleton.point_names))}",
    ]
    names = tuple(skeleton.point_names)
    edges = [
        [names[int(a)], names[int(b)]]
        for a, b in np.asarray(skeleton.edges, dtype=int).reshape(-1, 2)
    ]
    if edges:
        lines += [
            "",
            "# The edges, as point pairs. Direction is kept but carries no colour",
            "# meaning; an unnamed edge averages its two endpoints.",
            f"edges = {_toml.value(edges)}",
        ]
    if skeleton.n_symmetries:
        lines += [
            "",
            "# Left/right mirror pairs (unordered; each point in at most one pair). The",
            "# loader checks they are an automorphism of the edges above.",
            "point_symmetries = "
            f"{_toml.value([list(p) for p in skeleton.symmetry_names])}",
        ]
    # Per point and spelled out rather than compacted into `*` patterns: a generated
    # pattern would be a guess about which points are meant to share a colour, and the
    # only fact in hand is that these ones do.
    lines += ["", "[skeleton.point_colors]"]
    for name, color in zip(skeleton.point_names, skeleton.point_colors):
        lines.append(f"{_toml.key(name)} = {_toml.value(color)}")
    # Only the edges whose colour the endpoint average would NOT reproduce. Emitting all
    # of them would round-trip just as exactly and bury the handful that were chosen
    # deliberately under one line per leg segment.
    explicit = [
        (f"{names[int(a)]}--{names[int(b)]}", color)
        for (a, b), color, derived in zip(
            np.asarray(skeleton.edges, dtype=int).reshape(-1, 2),
            skeleton.edge_colors,
            _derived_edge_colors(skeleton),
        )
        if color != derived
    ]
    if explicit:
        lines += [
            "",
            "# Edges whose colour is not their endpoints' average.",
            "[skeleton.edge_colors]",
        ]
        for key, color in explicit:
            lines.append(f"{_toml.key(key)} = {_toml.value(color)}")
    return "\n".join(lines) + "\n"


def _derived_edge_colors(skeleton) -> tuple[str, ...]:
    """What ``edge_colors`` would default to for ``skeleton`` -- its endpoint averages."""
    from ..skeleton import Skeleton

    return Skeleton(
        name=skeleton.name,
        point_names=skeleton.point_names,
        edges=skeleton.edges,
        point_colors=skeleton.point_colors,
    ).edge_colors


def _stamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
