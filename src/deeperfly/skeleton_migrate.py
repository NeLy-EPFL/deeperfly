"""Skeleton edits as typed migrations -- the guard on the sharpest hazard in the project.

A skeleton's ``point_names`` is fingerprinted into every ``labels.h5``
(:func:`deeperfly.gui.labels.labels_identity`) *and* written into every ``results.h5``. So a
GUI that lets an operator edit the skeleton can invalidate every label in a project with one
click, and the invalidation is not even loud: two 38-point skeletons in different orders load
each other's files happily and mean something completely different by every index.

This module makes each edit a **typed migration** with a declared effect, a dry run that
counts what it would touch, and -- for anything destructive -- a refusal to proceed silently.

.. code-block:: text

    add a point         none (a new column, all-unset)                       silent
    rename a point      remap by identity; names rewritten in labels.h5      notice
    reorder points      remap by name; on-disk COO indices rewritten         notice
    add/remove a bone   none (bones are display + the BA prior only)         silent
    change limb/palette none                                                 silent
    change symmetries   none (read by flip aug / mirror check / chirality)   silent
    delete a point      its labels are QUARANTINED, not deleted              confirm

The one rule everything else follows from: **labels move by name, never by index.** The same
rule merging obeys (:mod:`deeperfly.merge`), for the same reason, and the reason index-based
copying is not implemented anywhere in this package.

Deletion reuses the existing quarantine mechanism rather than inventing one: an absence
declaration already parks vetoed rows under ``absent/void_*`` and restores them if it is
lifted (see :mod:`deeperfly.gui.labels`). A deleted point's labels go the same way, so
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
    "plan_migration",
    "apply_migration",
]

log = logging.getLogger("deeperfly")

#: Change kinds, and whether each needs the operator to confirm.
#:
#: Only deletion does. Everything else either cannot lose a label (adding, bones, palette) or
#: moves it deterministically by name (rename, reorder) -- and asking about a safe edit trains
#: people to click through the dangerous one.
DESTRUCTIVE = ("delete",)


@dataclass(frozen=True)
class SkeletonChange:
    """One difference between two skeletons."""

    kind: str  # "add"|"delete"|"rename"|"reorder"|"bones"|"limbs"|"symmetries"
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

        Bones, limbs/palette and symmetry pairs are all display or downstream-policy
        metadata: no label row moves and no sidecar is rewritten, so such an edit needs
        neither a rewrite nor a confirmation.
        """
        return all(c.kind in ("bones", "limbs", "symmetries") for c in self.changes)

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


def diff_skeletons(old, new) -> tuple[list[SkeletonChange], dict[int, int]]:
    """``(changes, old->new index mapping)`` between two :class:`~deeperfly.skeleton.Skeleton`.

    Renames are detected **positionally**, and only when the rest of the ordering is
    otherwise unchanged: a name that vanished while a new one appeared at the same index is
    reported as a rename rather than a delete-plus-add, because that is what an operator
    typing over a name did. When more than one name changed at once the inference is unsafe,
    so each is reported as its own delete and add -- and the delete then requires
    confirmation, which is the correct outcome for an ambiguous edit.
    """
    old_names, new_names = tuple(old.point_names), tuple(new.point_names)
    changes: list[SkeletonChange] = []

    gone = [n for n in old_names if n not in new_names]
    fresh = [n for n in new_names if n not in old_names]

    renames: dict[str, str] = {}
    if len(gone) == len(fresh) == 1:
        # One out, one in: a rename iff they sit at the same index, i.e. nothing moved.
        old_at = old_names.index(gone[0])
        new_at = new_names.index(fresh[0])
        if old_at == new_at:
            renames[gone[0]] = fresh[0]
            changes.append(
                SkeletonChange(
                    "rename",
                    f"{gone[0]} -> {fresh[0]}",
                    (gone[0], fresh[0]),
                )
            )
            gone, fresh = [], []

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
        np.asarray(old.bones).reshape(-1, 2), np.asarray(new.bones).reshape(-1, 2)
    ):
        changes.append(
            SkeletonChange("bones", "the bones changed (display + the BA prior only)")
        )
    if tuple(old.limb_names) != tuple(new.limb_names) or old.palette != new.palette:
        changes.append(SkeletonChange("limbs", "the limbs or palette changed"))
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
                "the left/right symmetry pairs changed (flip augmentation, the "
                "[pose2d.output_points] mirror check and the chirality QC read them; "
                "no label moves)",
            )
        )
    return changes, mapping


def plan_migration(project, new_skeleton) -> MigrationPlan:
    """What changing ``project``'s skeleton to ``new_skeleton`` would do.

    Counts, per ``labels.h5``, how many rows would **move** (their point index rewritten)
    and how many would be **quarantined** (their point is gone). Nothing is written.

    Parameters
    ----------
    project
        The :class:`~deeperfly.project.Project`.
    new_skeleton
        The proposed :class:`~deeperfly.skeleton.Skeleton`.

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
    :func:`~deeperfly.gui.labels._check_identity` compares that file with itself and can
    never fire. Nothing else asks a ``labels.h5`` which point order its rows are actually
    on. Without this comparison, a project whose ``skeleton.toml`` has drifted from its
    sidecars -- by a hand edit, or just by adopting a recording produced under a different
    config, which :meth:`~deeperfly.project.Project.add_recording` does not check -- has
    every label silently transposed and then restamped as consistent.
    """
    old = project.skeleton()
    changes, mapping = diff_skeletons(old, new_skeleton)
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

    from .project import _coo_rows

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

    from .gui.labels import load_labels, load_landmark_labels, save_labels
    from .merge import SkeletonMapping, remap_labels

    with h5py.File(path, "r") as f:
        identity = json.loads(f.attrs["meta"])["identity"]
    labels = load_labels(path, identity=identity)
    if labels is None:
        return 0, 0
    n_views = len(identity["camera_names"])
    n_frames = int(identity["n_frames"])
    landmarks = load_landmark_labels(path, n_views=n_views, n_frames=n_frames)

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
        landmarks=landmarks,
    )
    return moved, max(0, before - after)


def _skeleton_toml(skeleton) -> str:
    """A skeleton as a ``[skeleton]`` config fragment.

    Written from the object rather than lifted from a file, because a migrated skeleton has
    no source file yet. ``limb_points`` is the source of truth the parser reads -- ``bones``
    are derived from it -- so the chains are reconstructed from ``limb_id`` and the bone
    list rather than emitted directly.
    """
    from . import _toml

    lines = [
        "# The project's skeleton. Rewritten by a skeleton migration; the prose from the",
        "# original file is not preserved, but the structure is exact.",
        "[skeleton]",
        f"name = {_toml.value(skeleton.name)}",
        f"point_names = {_toml.value(list(skeleton.point_names))}",
    ]
    # By NAME, so the emitted fragment survives a later reorder -- and emitted at all,
    # because a migration rewrites the whole [skeleton] table: dropping the pairs here
    # would silently disable the mirror check, flip augmentation and the chirality QC on
    # the first skeleton edit a project ever makes.
    if skeleton.n_symmetries:
        lines += [
            "",
            "# Left/right mirror pairs (unordered; each point in at most one pair).",
            f"symmetries = {_toml.value([list(p) for p in skeleton.symmetry_names])}",
        ]
    lines += [
        "",
        "# Each limb's points in kinematic-chain order (the bones are the consecutive pairs).",
        "[skeleton.limb_points]",
    ]
    chains = _limb_chains(skeleton)
    for limb, points in chains.items():
        lines.append(f"{_toml.key(limb)} = {_toml.value(points)}")
    if skeleton.palette:
        lines += ["", "[skeleton.limb_palette]"]
        for limb, color in skeleton.palette.items():
            lines.append(f"{_toml.key(limb)} = {_toml.value(color)}")
    return "\n".join(lines) + "\n"


def _limb_chains(skeleton) -> dict[str, list[str]]:
    """``limb name -> its points in chain order``, recovered from ``limb_id`` + ``bones``.

    A limb's points are ordered by walking its bones from the endpoint that is never a
    bone's *target*; a limb with no bones (a single antenna point) is its member list.
    """
    limb_id = np.asarray(skeleton.limb_id)
    bones = np.asarray(skeleton.bones, dtype=int).reshape(-1, 2)
    out: dict[str, list[str]] = {}
    for lid, limb in enumerate(skeleton.limb_names):
        members = [int(i) for i in np.nonzero(limb_id == lid)[0]]
        if not members:
            continue
        inside = [b for b in bones if b[0] in members and b[1] in members]
        if not inside:
            out[limb] = [skeleton.point_names[i] for i in members]
            continue
        targets = {int(b[1]) for b in inside}
        start = next((m for m in members if m not in targets), members[0])
        nxt = {int(a): int(b) for a, b in inside}
        chain, seen = [start], {start}
        while chain[-1] in nxt and nxt[chain[-1]] not in seen:
            chain.append(nxt[chain[-1]])
            seen.add(chain[-1])
        chain += [m for m in members if m not in seen]
        out[limb] = [skeleton.point_names[i] for i in chain]
    return out


def _stamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
