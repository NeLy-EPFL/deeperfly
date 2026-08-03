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
identity check (:func:`deeperfly.gui.labels._check_identity`) is what prevents it by
refusing outright; merging replaces that refusal with a name-based remap, and index copying
is *not implemented at all* so it cannot be reached by accident.

**Cameras.** Same treatment. A same-named camera whose ``image_sizes`` differ is **fatal**:
ground truth is stored in footage pixels, so the stored coordinates would mean something
else entirely.

**Recordings.** Deduplicated by content id (:func:`deeperfly.project.recording_id`), so a
backup copy is recognized rather than double-counted.

**Cells.** Per ``(view, frame, point)``, with a **provenance-aware** default: a human's
drag beats a bulk-confirmed reprojection, because the second is the model's own guess
promoted to ground truth. Anything genuinely ambiguous goes to a review queue rather than
being resolved by a coin flip.

Every merge is **dry-run by default** and writes a pre-merge snapshot before applying.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .gui.labels import Labels, Provenance

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

#: Provenance ranking for the automatic tie-break. Higher wins. A human drag beats a
#: confirmed prediction beats a bulk-confirmed *projection* (which is the model's own
#: reprojected guess), and a placeholder seed -- a drag handle the editor invented at the
#: image edge -- loses to everything, because it was never a claim about the animal.
_PROVENANCE_RANK = {
    Provenance.DRAGGED: 3,
    Provenance.CONFIRMED_PREDICTION: 2,
    Provenance.CONFIRMED_PROJECTION: 1,
    Provenance.PLACEHOLDER_SEED: 0,
    Provenance.NONE: -1,
}


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
    across verbatim, including provenance -- the merge decides *which* value wins later,
    and it needs the provenance to do so.

    Returns
    -------
    Labels
        A destination-shaped overlay holding the mapped subset of ``source``.
    """
    out = Labels.empty(n_views, n_frames, n_points)
    gt_mask = source.gt_authored
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
                out.gt_provenance[v_dst, rows, p_dst] = source.gt_provenance[
                    v_src, rows, p_src
                ]
            occ = source.occluded[v_src, :frames, p_src]
            if occ.any():
                out.occluded[v_dst, np.nonzero(occ)[0], p_dst] = True

    frames = min(source.reviewed.shape[0], n_frames)
    out.reviewed[:frames] |= source.reviewed[:frames]
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
    ours_provenance: int = 0
    theirs_provenance: int = 0


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
        One of :data:`CONFLICT_POLICIES` for cells the provenance rule cannot settle.
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
        their_prov = int(mapped.gt_provenance[v, t, p])
        if not ours[v, t, p]:
            if apply:
                dest.set_gt(v, t, p, their_xy, provenance=their_prov)
            report.taken_from_source += 1
            continue
        our_xy = dest.gt[v, t, p]
        our_prov = int(dest.gt_provenance[v, t, p])
        if np.allclose(our_xy, their_xy, atol=1e-9):
            report.identical += 1
            continue
        decision = _resolve(
            v,
            t,
            p,
            our_xy,
            their_xy,
            our_prov,
            their_prov,
            on_conflict,
            source_is_newer,
        )
        report.conflicts.append(decision)
        if decision.outcome == "theirs":
            if apply:
                dest.set_gt(v, t, p, their_xy, provenance=their_prov)
            report.taken_from_source += 1
        elif decision.outcome == "ours":
            report.kept_from_dest += 1

    # Occlusions: taken where the destination has neither a label nor an occlusion. An
    # occlusion is a weaker statement than a pixel, so it never displaces one.
    for v, t, p in zip(*np.nonzero(mapped.occluded)):
        if dest.gt_authored[v, t, p] or dest.occluded[v, t, p]:
            continue
        if apply:
            dest.set_occluded(v, t, p, True)
        report.occluded_taken += 1

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
    view, frame, point, our_xy, their_xy, our_prov, their_prov, policy, source_is_newer
) -> CellDecision:
    """Decide one conflicting cell.

    Provenance decides first, whatever the policy: a human's drag beating a bulk-confirmed
    reprojection is not a preference, it is the difference between evidence and the model's
    own output. Only when both sides are the same *kind* of label does the policy apply --
    and its honest default is to ask.
    """
    kwargs = dict(
        view=int(view),
        frame=int(frame),
        point=int(point),
        ours_xy=(float(our_xy[0]), float(our_xy[1])),
        theirs_xy=(float(their_xy[0]), float(their_xy[1])),
        ours_provenance=our_prov,
        theirs_provenance=their_prov,
    )
    our_rank = _PROVENANCE_RANK.get(our_prov, -1)
    their_rank = _PROVENANCE_RANK.get(their_prov, -1)
    if our_rank != their_rank:
        better = "theirs" if their_rank > our_rank else "ours"
        return CellDecision(
            outcome=better,
            reason=f"provenance: {_name(their_prov)} vs {_name(our_prov)}",
            **kwargs,
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
        reason=f"both {_name(our_prov)}, {distance:.1f} px apart",
        **kwargs,
    )


def _name(code: int) -> str:
    return {
        Provenance.DRAGGED: "dragged",
        Provenance.CONFIRMED_PREDICTION: "confirmed_prediction",
        Provenance.CONFIRMED_PROJECTION: "confirmed_projection",
        Provenance.PLACEHOLDER_SEED: "placeholder_seed",
        Provenance.NONE: "none",
    }.get(int(code), f"code{code}")
