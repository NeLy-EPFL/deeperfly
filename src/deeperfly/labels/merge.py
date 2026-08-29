"""Merging label sets -- reconciling ground truth that arrived in several places.

The measured problem this exists for: ``~/fly-pose-data`` holds 21 non-backup ``labels.h5``
files across three parallel directory trees, carrying roughly 29,500 *distinct*
human-placed points, of which a hand-written manifest tracks 9,175. The other ~20,000 are
real hand labels that no training set has ever seen. The blocker is not labeling effort; it
is reconciliation.

Four things have to be reconciled, and one of them is a trap:

**Skeleton.** Matched **by name, never by index**. Two 38-point skeletons in different
orders are the nightmare case: index-based copying between them is a silent, total
corruption, and every cell would still look plausible afterwards. Today's exact-match
identity check (:func:`deeperfly.labels.store._check_identity`) is what prevents it by
refusing outright; merging replaces that refusal with a name-based remap, and index copying
is *not implemented at all* so it cannot be reached by accident.

**Cameras.** Same treatment. A same-named camera whose ``image_sizes`` differ is **fatal**:
ground truth is stored in footage pixels, so the stored coordinates would mean something
else entirely.

**Recordings.** Deduplicated by content id (:func:`deeperfly.project.recording_id`), so a
backup copy is recognized rather than double-counted.

**Cells.** Per ``(view, frame, point)``. A cell only one side authored is taken; a cell
both authored identically is a no-op. A genuine disagreement is two operators disagreeing
about where a keypoint is, and nothing in the data ranks one above the other, so it goes to
a review queue rather than being resolved by a coin flip.

Every piece of authored state in a ``labels.h5`` travels, not just the pixels: ``gt``,
``occluded`` (the **hidden** flag -- "hold this cell out of the training loss"), ``seeds``,
``instance``, ``reviewed`` and ``absent``. Two of those have rules worth stating here,
because getting them wrong is silent:

- **Seeds are additive, never overwritten.** A seed is where an instance's keypoint
  *started* -- evidence, not a claim -- so two sides cannot conflict over one. But
  overwriting a seed is reseeding, and reseeding is an explicit operator gesture; a merge
  must not do it as a side effect.
- **The instance flag is unioned AND implied.** A frame this merge put a GT or seed row into
  gains the flag even if the source never carried one. Without that, imported labels land in
  a frame the editor treats as having no annotation skeleton and the whole frame drops back
  to the pre-v8 display layer -- which looks like the labels went missing.

Every merge is **dry-run by default** and writes a pre-merge snapshot before applying.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .store import Labels

__all__ = [
    "SkeletonMapping",
    "CellDecision",
    "MergeReport",
    "map_by_name",
    "remap_labels",
    "merge_labels",
    "CONFLICT_POLICIES",
]

log = logging.getLogger("deeperfly")

#: How to resolve a cell both sides authored differently. ``"manual"`` defers to a review
#: queue -- the honest answer when neither side is preferable.
CONFLICT_POLICIES = ("manual", "ours", "theirs", "newest")


@dataclass
class SkeletonMapping:
    """How one label set's point/camera axes map onto another's, by name.

    Attributes
    ----------
    source_to_dest
        ``source index -> destination index`` for the names present in both.
    matched, only_source, only_dest
        The names in each category, for the report.
    reordered
        Whether any matched name changed index. When true, a copy that ignored names
        would have silently transposed data -- which is the whole reason this exists.
    """

    source_to_dest: dict[int, int] = field(default_factory=dict)
    matched: list[str] = field(default_factory=list)
    only_source: list[str] = field(default_factory=list)
    only_dest: list[str] = field(default_factory=list)
    reordered: bool = False

    @property
    def complete(self) -> bool:
        """Whether every source name has a destination (nothing would be dropped)."""
        return not self.only_source


def map_by_name(source: list[str], dest: list[str]) -> SkeletonMapping:
    """Map ``source`` names onto ``dest`` names positionally-safely.

    Parameters
    ----------
    source, dest
        Ordered name lists (skeleton points, or camera names).

    Returns
    -------
    SkeletonMapping
        The mapping plus what it could not match.
    """
    dest_index = {name: i for i, name in enumerate(dest)}
    mapping = SkeletonMapping()
    for i, name in enumerate(source):
        j = dest_index.get(name)
        if j is None:
            mapping.only_source.append(name)
            continue
        mapping.source_to_dest[i] = j
        mapping.matched.append(name)
        if i != j:
            mapping.reordered = True
    source_set = set(source)
    mapping.only_dest = [n for n in dest if n not in source_set]
    return mapping


def remap_labels(
    source: Labels,
    *,
    points: SkeletonMapping,
    cameras: SkeletonMapping,
    n_views: int,
    n_frames: int,
    n_points: int,
) -> Labels:
    """Re-express ``source`` on the destination's axes.

    Cells whose point or camera has no destination are **dropped** (they have nowhere to
    go); frames beyond ``n_frames`` are dropped too. Everything that does map is carried
    across verbatim; the merge decides *which* value wins later.

    Returns
    -------
    Labels
        A destination-shaped overlay holding the mapped subset of ``source``.
    """
    out = Labels.empty(n_views, n_frames, n_points)
    gt_mask = source.gt_authored
    seed_mask = np.isfinite(source.seeds).all(axis=-1)
    for v_src, v_dst in cameras.source_to_dest.items():
        if v_src >= gt_mask.shape[0] or v_dst >= n_views:
            continue
        for p_src, p_dst in points.source_to_dest.items():
            if p_src >= gt_mask.shape[2] or p_dst >= n_points:
                continue
            frames = min(source.gt.shape[1], n_frames)
            col = gt_mask[v_src, :frames, p_src]
            if col.any():
                rows = np.nonzero(col)[0]
                out.gt[v_dst, rows, p_dst] = source.gt[v_src, rows, p_src]
            occ = source.occluded[v_src, :frames, p_src]
            if occ.any():
                out.occluded[v_dst, np.nonzero(occ)[0], p_dst] = True
            # Seeds move by name too. They are the instance's start positions -- authored
            # state, persisted precisely so a re-run cannot silently re-solve every non-GT
            # point -- so dropping them here would discard what the file exists to keep.
            sd = seed_mask[v_src, :frames, p_src]
            if sd.any():
                rows = np.nonzero(sd)[0]
                out.seeds[v_dst, rows, p_dst] = source.seeds[v_src, rows, p_src]

    frames = min(source.reviewed.shape[0], n_frames)
    out.reviewed[:frames] |= source.reviewed[:frames]
    # The per-frame instance flag is view- and point-independent, so it needs no name
    # reconciliation -- only truncation to the destination's frame count.
    inst_frames = min(source.instance.shape[0], n_frames)
    out.instance[:inst_frames] |= source.instance[:inst_frames]
    absent = np.asarray(source.absent, dtype=bool)
    for p_src, p_dst in points.source_to_dest.items():
        if p_src < absent.shape[1] and p_dst < n_points:
            rows = min(absent.shape[0], n_frames)
            out.absent[:rows, p_dst] |= absent[:rows, p_src]
    out.dirty = False
    return out


@dataclass(frozen=True)
class CellDecision:
    """One cell both sides authored, and what was done about it."""

    view: int
    frame: int
    point: int
    outcome: str  # "ours" | "theirs" | "manual" | "identical"
    reason: str
    ours_xy: tuple[float, float] | None = None
    theirs_xy: tuple[float, float] | None = None


@dataclass
class MergeReport:
    """What a merge did, or would do. Printed by the CLI and stored beside the project."""

    points: SkeletonMapping
    cameras: SkeletonMapping
    taken_from_source: int = 0
    kept_from_dest: int = 0
    identical: int = 0
    conflicts: list[CellDecision] = field(default_factory=list)
    dropped_cells: int = 0
    occluded_taken: int = 0
    absent_union: int = 0
    reviewed_added: int = 0
    #: Instance seeds taken where the destination had none (never overwritten).
    seeds_taken: int = 0
    #: Frames that gained an annotation skeleton -- unioned from the source, plus every
    #: frame this merge put a GT or seed row into. Without the second half, imported labels
    #: land in a frame the editor treats as un-annotated.
    instances_added: int = 0
    fatal: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.fatal

    @property
    def unresolved(self) -> list[CellDecision]:
        return [c for c in self.conflicts if c.outcome == "manual"]

    def summary(self) -> dict:
        return {
            "points_matched": len(self.points.matched),
            "points_only_source": self.points.only_source,
            "points_only_dest": self.points.only_dest,
            "points_reordered": self.points.reordered,
            "cameras_matched": len(self.cameras.matched),
            "cameras_only_source": self.cameras.only_source,
            "taken_from_source": self.taken_from_source,
            "kept_from_dest": self.kept_from_dest,
            "identical": self.identical,
            "conflicts": len(self.conflicts),
            "unresolved": len(self.unresolved),
            "dropped_cells": self.dropped_cells,
            "occluded_taken": self.occluded_taken,
            "absent_union": self.absent_union,
            "reviewed_added": self.reviewed_added,
            "seeds_taken": self.seeds_taken,
            "instances_added": self.instances_added,
            "fatal": self.fatal,
            "notes": self.notes,
        }


def merge_labels(
    dest: Labels,
    source: Labels,
    *,
    point_names_dest: list[str],
    point_names_source: list[str],
    camera_names_dest: list[str],
    camera_names_source: list[str],
    image_sizes_dest: dict | None = None,
    image_sizes_source: dict | None = None,
    on_conflict: str = "manual",
    source_is_newer: bool = True,
    apply: bool = True,
) -> MergeReport:
    """Merge ``source`` labels into ``dest``, in place when ``apply``.

    Parameters
    ----------
    dest, source
        The two overlays. ``dest`` is mutated only when ``apply``.
    point_names_dest, point_names_source
        The two skeletons' point orders. Matched by **name**.
    camera_names_dest, camera_names_source
        The two camera name lists. Matched by name.
    image_sizes_dest, image_sizes_source
        ``camera -> (h, w)``. A same-named camera whose sizes differ is **fatal**: the
        stored pixels would mean something else.
    on_conflict
        One of :data:`CONFLICT_POLICIES` for cells both sides authored differently.
    source_is_newer
        Which side ``"newest"`` prefers.
    apply
        When false nothing is written -- the dry run.

    Returns
    -------
    MergeReport
        Everything that happened, or would.

    Raises
    ------
    ValueError
        If ``on_conflict`` is unknown.
    """
    if on_conflict not in CONFLICT_POLICIES:
        raise ValueError(
            f"on_conflict must be one of {list(CONFLICT_POLICIES)}, got {on_conflict!r}"
        )
    points = map_by_name(point_names_source, point_names_dest)
    cameras = map_by_name(camera_names_source, camera_names_dest)
    report = MergeReport(points=points, cameras=cameras)

    for name in cameras.matched:
        a = (image_sizes_dest or {}).get(name)
        b = (image_sizes_source or {}).get(name)
        if a and b and tuple(a) != tuple(b):
            report.fatal.append(
                f"camera {name!r} is {tuple(a)} in the destination and {tuple(b)} in the "
                "source -- ground truth is stored in footage pixels, so merging would "
                "reinterpret every coordinate. Re-label against matching footage"
            )
    if points.only_source:
        report.notes.append(
            f"point(s) {points.only_source} exist only in the source and their labels "
            "will be dropped -- add them to the destination skeleton first to keep them"
        )
    if points.reordered:
        report.notes.append(
            "the skeletons agree on names but not on ORDER; labels are remapped by name "
            "(an index-based copy here would have silently transposed every point)"
        )
    if not report.ok:
        return report

    n_views, n_frames, n_points = (
        dest.gt.shape[0],
        dest.gt.shape[1],
        dest.gt.shape[2],
    )
    mapped = remap_labels(
        source,
        points=points,
        cameras=cameras,
        n_views=n_views,
        n_frames=n_frames,
        n_points=n_points,
    )
    report.dropped_cells = int(source.gt_authored.sum()) - int(mapped.gt_authored.sum())

    theirs = mapped.gt_authored
    ours = dest.gt_authored
    for v, t, p in zip(*np.nonzero(theirs)):
        their_xy = mapped.gt[v, t, p]
        if not ours[v, t, p]:
            if apply:
                dest.set_gt(v, t, p, their_xy)
            report.taken_from_source += 1
            continue
        our_xy = dest.gt[v, t, p]
        if np.allclose(our_xy, their_xy, atol=1e-9):
            report.identical += 1
            continue
        decision = _resolve(v, t, p, our_xy, their_xy, on_conflict, source_is_newer)
        report.conflicts.append(decision)
        if decision.outcome == "theirs":
            if apply:
                dest.set_gt(v, t, p, their_xy)
            report.taken_from_source += 1
        elif decision.outcome == "ours":
            report.kept_from_dest += 1

    # Hidden marks: taken where the destination has none of its own.
    #
    # Deliberately NOT gated on the destination having a GT pixel. The flag is its own axis
    # ("hold this cell out of the training loss"), so it composes with any pixel rather than
    # competing with one (see the labels module docstring). Under the old rule a source mark
    # was silently discarded on every cell the destination had a pixel for -- which is exactly
    # the pairing worth keeping, and one nothing else in the pipeline can reproduce.
    for v, t, p in zip(*np.nonzero(mapped.occluded)):
        if dest.occluded[v, t, p]:
            continue
        if apply:
            dest.set_occluded(v, t, p, True)
        report.occluded_taken += 1

    # Seeds: additive only, never overwriting. A seed is *evidence* (where the instance's
    # keypoint started), not a claim, so there is nothing for two sides to conflict over --
    # but overwriting one is reseeding, which is an explicit operator gesture in the editor
    # and must not happen as a side effect of a merge.
    theirs_seeded = np.isfinite(mapped.seeds).all(axis=-1)
    ours_seeded = np.isfinite(dest.seeds).all(axis=-1)
    fresh_seeds = theirs_seeded & ~ours_seeded
    report.seeds_taken = int(fresh_seeds.sum())
    if apply and report.seeds_taken:
        dest.seeds[fresh_seeds] = mapped.seeds[fresh_seeds]
        dest.dirty = True

    # The instance flag: unioned, and then *implied* for every frame this merge put a GT or
    # seed row into. The second half is not cosmetic -- it is what stops imported labels
    # landing in a frame the editor treats as having no annotation skeleton, which drops the
    # whole frame back to the pre-v8 display layer.
    implied = np.zeros(n_frames, dtype=bool)
    if report.taken_from_source or report.seeds_taken:
        implied |= (theirs & ~ours).any(axis=(0, 2))
        implied |= fresh_seeds.any(axis=(0, 2))
    wanted = (np.asarray(mapped.instance, dtype=bool) | implied) & ~np.asarray(
        dest.instance, dtype=bool
    )
    report.instances_added = int(wanted.sum())
    if apply and report.instances_added:
        dest.instance |= wanted
        dest.dirty = True

    # Absence is unioned rather than resolved: both sides are claims about the animal, and
    # a declaration is non-destructive by construction (the labels underneath are
    # quarantined, not deleted), so keeping both loses nothing.
    new_absent = np.asarray(mapped.absent, dtype=bool) & ~np.asarray(
        dest.absent, dtype=bool
    )
    report.absent_union = int(new_absent.sum())
    if apply and report.absent_union:
        dest.absent = np.asarray(dest.absent, dtype=bool) | np.asarray(
            mapped.absent, dtype=bool
        )
        dest.dirty = True

    added_reviewed = mapped.reviewed & ~dest.reviewed
    report.reviewed_added = int(added_reviewed.sum())
    if apply and report.reviewed_added:
        dest.reviewed |= mapped.reviewed
        dest.dirty = True

    return report


def _resolve(
    view, frame, point, our_xy, their_xy, policy, source_is_newer
) -> CellDecision:
    """Decide one conflicting cell.

    There is one kind of ground truth -- a pixel an operator created -- so two sides that
    disagree are two humans disagreeing, and nothing in the data ranks one above the other.
    The policy decides, and its honest default is to ask.
    """
    kwargs = dict(
        view=int(view),
        frame=int(frame),
        point=int(point),
        ours_xy=(float(our_xy[0]), float(our_xy[1])),
        theirs_xy=(float(their_xy[0]), float(their_xy[1])),
    )
    if policy == "ours":
        return CellDecision(outcome="ours", reason="policy=ours", **kwargs)
    if policy == "theirs":
        return CellDecision(outcome="theirs", reason="policy=theirs", **kwargs)
    if policy == "newest":
        return CellDecision(
            outcome="theirs" if source_is_newer else "ours",
            reason="policy=newest",
            **kwargs,
        )
    distance = float(np.linalg.norm(np.asarray(our_xy) - np.asarray(their_xy)))
    return CellDecision(
        outcome="manual",
        reason=f"both authored, {distance:.1f} px apart",
        **kwargs,
    )
