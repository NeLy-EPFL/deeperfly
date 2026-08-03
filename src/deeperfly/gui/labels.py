"""The ground-truth annotation sidecar (``labels.h5``).

The keypoint editor is a *ground-truth annotation* tool, not a prediction editor:
the operator's 2D labels are the source of truth and the 3D pose is a pure derived
function of them (:mod:`deeperfly.gui.solve`). So the only state worth persisting
is what the operator actually authored -- and that is, per ``(view, frame, point)``,
a tri-state:

.. code-block:: text

    unset      -> fall back to the detector's prediction, else the 3D reprojection
    gt(x, y)   -> an affirmed 2D pixel (with provenance: how it was authored)
    occluded   -> "a human cannot place this point from this view" (dropped from 3D)

``gt`` and ``occluded`` are mutually exclusive. Alongside that per-``(view, frame,
point)`` tri-state, each *frame* carries one authored ``reviewed`` flag -- the
operator ticking "I have finished checking this frame". Everything else --
predictions, projections, the 3D point, reprojection error -- is derived, so it is
never stored: no dense NaN arrays, no duplicate of the detector's output.

**Absence (schema v3, per-frame in v4).** One more thing the operator can author is not a
property of a *view* at all: a keypoint may not *exist on this animal* -- an amputated leg,
an ablated antenna. That is categorically different from ``occluded`` ("it exists but I
cannot place it from *this view*") and from ``unset`` ("nobody has looked yet"), and it is
view-independent by construction. It *can* vary over time, though -- a leg lost to autotomy
part-way through a recording -- so it is stored per ``(frame, point)``:

.. code-block:: text

    absent      (T, P) bool  -> "this keypoint is not on this animal, in this frame"

Most declarations cover the whole recording (an animal that arrives with a leg missing
keeps it missing), which is why the editor offers a one-gesture "apply to the entire
recording" alongside the per-frame toggle, and why the on-disk form is **run-length spans**:
a whole-recording declaration is a single row no matter how long the recording.

A caution the schema deliberately does *not* enforce: absence and occlusion are opposites
at export time -- an occluded cell is a positive "unplaceable from this view" training
label, an absent one is excluded from supervision entirely. So marking a joint absent in
the handful of frames where it is merely hidden discards real training signal instead of
contributing it. Absence is a claim about the animal; occlusion is a claim about the view.

Absence acts as a **read-time veto** rather than a fourth exclusive cell value: the
authored pixels and occlusions stay in memory untouched, and the derived masks every
consumer reads (:attr:`Labels.has_gt`, :attr:`Labels.occluded_effective`) simply mask
the absent points out. Declaring -- and un-declaring -- absence therefore destroys
nothing, and the two masks are the whole contract:

.. code-block:: text

    gt_authored         raw "a pixel is stored here"   -- persistence and undo only
    has_gt              gt_authored & ~absent          -- everything else
    occluded_effective  occluded    & ~absent          -- everything else

On disk (schema v2) the deltas are stored **sparsely** (COO), which is tiny next to
``results.h5`` and, unlike the old dense ``corrections.h5``, carries no copy of the
prediction NaN pattern:

.. code-block:: text

    attrs["meta"]   json {deeperfly_labels_format_version, created_utc, identity,
                          subject_id}   (subject_id added in v3; optional, may be null)
    gt/
        index       (N, 3) int32   [view, frame, point]
        xy          (N, 2) float64  affirmed 2D pixel (footage space)
        provenance  (N,)   uint8    1=dragged, 2=confirmed_prediction, 3=confirmed_projection,
                                    4=placeholder_seed (v5; an editor-invented drag handle,
                                    never exported -- see :class:`Provenance`)
    occluded/
        index       (M, 3) int32   [view, frame, point]
    reviewed/                       (added in v2; absent in a v1 file -> no frames reviewed)
        index       (K,)   int32   frame indices the operator marked reviewed
    absent/                         (added in v3; absent in a v1/v2 file -> nothing absent)
        index       (Q,)   int32   points absent in EVERY frame. This was v3's whole
                                   representation and is still written, so a v3 reader
                                   sees the whole-recording declarations (and simply
                                   misses any partial ones).
        spans       (S, 3) int32   [point, t0, t1) run-length runs of absence (v4). The
                                   authoritative form; ``index`` is derived from it.
        void_gt/                    the gt rows the declaration vetoes (quarantine)
            index   (N', 3) int32
            xy      (N', 2) float64
            provenance (N',) uint8
        void_occluded/
            index   (M', 3) int32   the occlusion rows the declaration vetoes

``gt/`` and ``occluded/`` hold **live rows only** -- the vetoed ones move to
``absent/void_*``. That is what makes the *file* self-consistent for a consumer that
reads ``gt/index`` straight out of HDF5 without going through :func:`export_gt`, while
still losing nothing: un-declaring a point restores its rows from quarantine on the
next load.

``identity`` fingerprints the *recording* these labels annotate (skeleton points,
camera names, frame count, image sizes, footage basenames) so a stale sidecar is
refused. It deliberately excludes the predictions and ``created_utc`` themselves, so
re-running detection/triangulation on the *same* recording keeps the labels valid --
ground truth is absolute, not relative to what the network happened to predict. It also
deliberately excludes ``absent``, which is point-indexed and so already domain-checked
by the exact ``point_names`` match -- keeping it out means every existing sidecar stays
loadable and declaring an amputation does not orphan the labels authored beside it.

In memory the overlay is kept **dense** per ``(V, T, P)`` (mirroring the result), so
resolving a frame's effective points on the interactive hot path stays ``O(V*P)``
and independent of how much has been labeled -- the sparse form is disk-only.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from jaxtyping import Bool, Float

__all__ = [
    "Labels",
    "absent_to_spans",
    "spans_to_absent",
    "Provenance",
    "save_labels",
    "load_labels",
    "labels_identity",
    "migrate_from_corrections",
    "export_gt",
    "export_absent",
    "resolve_point_names",
    "LABELS_FORMAT_VERSION",
]

log = logging.getLogger("deeperfly")

LABELS_FORMAT_VERSION = 5


class Provenance:
    """How a GT pixel was authored (stored per ``gt`` row so it can be filtered).

    ``dragged`` and ``confirmed_prediction`` are the operator affirming a pixel they
    looked at; ``confirmed_projection`` is a bulk-accepted triangulation guess (the
    network never fired there), so an export/solve can down-weight or drop it.

    ``placeholder_seed`` is not a claim about the fly at all. When a bulk confirm has
    no pixel to offer -- the reprojection landed off the image, or there is no 3D at
    all -- the editor still stores *something* so the operator has a dot to grab and
    drag (:meth:`EditorState._grabbable`, which clamps it into frame). That coordinate
    is invented, and it is invented *at the image edge*, which is the worst possible
    place to teach a heatmap. It is therefore dropped by :func:`export_gt`
    unconditionally, including under ``include_projection=True``: a drag handle is
    machinery, not evidence, and no training flag should be able to turn it into a
    label. The moment the operator drags it, :meth:`EditorState.drag` restamps it
    ``dragged`` and it becomes real.

    The distinction is why v5 exists. Before it, a clamped seed and a genuine
    reprojected pixel shared code 3, so "include projections in training" could not be
    said without also saying "train on fabricated edge coordinates".
    """

    NONE = 0
    DRAGGED = 1
    CONFIRMED_PREDICTION = 2
    CONFIRMED_PROJECTION = 3
    PLACEHOLDER_SEED = 4


@dataclass
class Labels:
    """In-memory ground-truth overlay on a :class:`~deeperfly.results.PoseResult`.

    The per-``(view, frame, point)`` arrays are dense ``(V, T, P)``-shaped (2D pixels
    carry a trailing 2). A GT pixel is *stored* iff ``gt`` is finite there, which is
    exactly ``provenance != 0``; ``occluded`` marks views the operator flagged
    unusable. ``reviewed`` is a separate per-*frame* ``(T,)`` flag (the operator's "I
    have checked this frame"), independent of the point labels. ``absent`` is a
    per-``(frame, point)`` ``(T, P)`` flag -- "this keypoint is not on this animal" -- which
    is view-independent but *may* vary over time (a leg lost to autotomy part-way through).

    Two masks, and the difference matters at every call site:

    - :attr:`gt_authored` -- the raw "a pixel is stored here". Used **only** by
      persistence and undo, which must not lose what the operator typed.
    - :attr:`has_gt` -- ``gt_authored`` with the absent points vetoed out. This is what
      every *consumer* (the solve, the display, the export, progress accounting) reads.
      :attr:`occluded_effective` is the same veto over ``occluded``.

    Because the veto is applied on read, declaring a point absent destroys nothing and
    un-declaring it restores every pixel and occlusion underneath byte for byte.

    The invariants (``gt`` and ``occluded`` disjoint; ``provenance`` set iff ``gt``
    finite) are maintained by the mutators and re-checked on load. ``dirty`` tracks
    unsaved changes.
    """

    gt: Float[np.ndarray, "V T P 2"]
    gt_provenance: np.ndarray  # (V, T, P) uint8, a Provenance value
    occluded: Bool[np.ndarray, "V T P"]
    reviewed: Bool[np.ndarray, "T"]  # per-frame "operator has checked this frame"
    #: Per-``(frame, point)`` ``(T, P)`` "this keypoint is not on this animal" (v4; v3 was
    #: per-point). Vetoes the derived masks. View-independent by construction -- an
    #: amputated joint is missing from every camera at once, which is exactly what
    #: distinguishes it from ``occluded``.
    absent: Bool[np.ndarray, "T P"] | None = None
    #: Optional free-text animal identifier, so one animal's several recordings can be
    #: grouped (and an absence declaration copied between them). Persisted in ``meta``.
    subject_id: str | None = None
    dirty: bool = field(default=False)

    def __post_init__(self) -> None:
        # ``absent`` is optional in the constructor so every existing call site keeps
        # working; normalize it to a real (T, P) array. A (P,) value is accepted and
        # broadcast, which is how a v3 whole-recording declaration loads.
        n_frames, n_points = self.gt.shape[1], self.gt.shape[2]
        if self.absent is None:
            self.absent = np.zeros((n_frames, n_points), dtype=bool)
            return
        a = np.asarray(self.absent, dtype=bool)
        if a.ndim == 1:
            a = np.broadcast_to(a.reshape(1, -1), (n_frames, n_points))
        self.absent = np.array(a, dtype=bool, copy=True).reshape(n_frames, n_points)

    @classmethod
    def empty(cls, n_views: int, n_frames: int, n_points: int) -> Labels:
        """An overlay with no GT, nothing occluded, no frame reviewed, nothing absent."""
        return cls(
            gt=np.full((n_views, n_frames, n_points, 2), np.nan),
            gt_provenance=np.zeros((n_views, n_frames, n_points), dtype=np.uint8),
            occluded=np.zeros((n_views, n_frames, n_points), dtype=bool),
            reviewed=np.zeros(n_frames, dtype=bool),
            absent=np.zeros(n_points, dtype=bool),
        )

    # -- derived masks --------------------------------------------------------

    @property
    def gt_authored(self) -> Bool[np.ndarray, "V T P"]:
        """Raw per-``(view, frame, point)`` "a GT pixel is stored here".

        **Not** the consumer mask -- it ignores :attr:`absent`. Use it only where losing
        an authored pixel would be wrong (persistence, undo, the load-time tie break).
        """
        return np.isfinite(self.gt).all(axis=-1)

    @property
    def has_gt(self) -> Bool[np.ndarray, "V T P"]:
        """Per-``(view, frame, point)`` whether a *usable* GT pixel is present.

        :attr:`gt_authored` with the absent points vetoed out -- the mask every consumer
        should read.
        """
        return self.gt_authored & ~self._absent_bcast

    @property
    def occluded_effective(self) -> Bool[np.ndarray, "V T P"]:
        """:attr:`occluded` with the absent points vetoed out.

        An absent point is not "occluded in every view" -- it is not there at all, so it
        must not be exported as a positive "unplaceable from this view" label.
        """
        return self.occluded & ~self._absent_bcast

    @property
    def _absent_bcast(self) -> Bool[np.ndarray, "V T P"]:
        """:attr:`absent` broadcast to the ``(V, T, P)`` cell shape (absence spans views)."""
        return np.asarray(self.absent, dtype=bool)[None, :, :]

    @property
    def any_labels(self) -> bool:
        """Whether any GT pixel, occlusion or absence declaration has been authored."""
        return bool(self.gt_authored.any() or self.occluded.any() or self.absent.any())

    # -- absence accessors ----------------------------------------------------
    #
    # Every consumer goes through these rather than touching ``absent`` directly, so
    # that a future per-frame-range form (autotomy: a leg lost mid-recording, stored as
    # an ``absent/spans`` sibling) can re-back them without touching a single caller.

    def absent_at(self, frame: int) -> Bool[np.ndarray, "P"]:
        """``(P,)`` which points are not on this animal at ``frame``."""
        return np.asarray(self.absent, dtype=bool)[int(frame)]

    def absent_all_frames(self) -> Bool[np.ndarray, "P"]:
        """``(P,)`` which points are absent in *every* frame of the recording.

        The stricter question, and the one **structural** consumers must ask: truncating
        the IK body plan or dropping a point from bundle adjustment is a per-recording
        decision that no per-frame flag can express, so it may only act on a point that
        never exists. A leg lost at frame 900 stays in the plan -- it was there, its bone
        lengths are measurable from the earlier frames, and its angles up to the loss are
        exactly what a leg-loss study wants.
        """
        a = np.asarray(self.absent, dtype=bool)
        return a.all(axis=0) if a.size else np.zeros(a.shape[1], dtype=bool)

    def absent_any_frame(self) -> Bool[np.ndarray, "P"]:
        """``(P,)`` which points are absent in *at least one* frame (for reporting)."""
        a = np.asarray(self.absent, dtype=bool)
        return a.any(axis=0) if a.size else np.zeros(a.shape[1], dtype=bool)

    # -- mutators (maintain the invariants) -----------------------------------

    def set_gt(
        self,
        view: int,
        frame: int,
        point: int,
        xy,
        *,
        provenance: int = Provenance.DRAGGED,
    ) -> None:
        """Author a GT pixel for ``point`` in ``view`` at ``frame`` (clears occluded)."""
        self.gt[view, frame, point] = np.asarray(xy, dtype=float)
        self.gt_provenance[view, frame, point] = np.uint8(provenance)
        self.occluded[view, frame, point] = False
        self.dirty = True

    def clear_gt(self, view: int, frame: int, point: int) -> None:
        """Drop just the GT pixel for ``point`` in ``view`` (occlusion untouched)."""
        self.gt[view, frame, point] = np.nan
        self.gt_provenance[view, frame, point] = Provenance.NONE
        self.dirty = True

    def set_occluded(self, view: int, frame: int, point: int, value: bool) -> None:
        """Flag ``point`` in ``view`` occluded (or clear it); setting drops any GT."""
        self.occluded[view, frame, point] = bool(value)
        if value:
            self.gt[view, frame, point] = np.nan
            self.gt_provenance[view, frame, point] = Provenance.NONE
        self.dirty = True

    def clear_view(self, view: int, frame: int, point: int) -> None:
        """Reset one ``(view, frame, point)`` to ``unset`` (drop GT and occlusion)."""
        self.gt[view, frame, point] = np.nan
        self.gt_provenance[view, frame, point] = Provenance.NONE
        self.occluded[view, frame, point] = False
        self.dirty = True

    def clear_point(self, frame: int, point: int) -> None:
        """Reset every view of ``point`` at ``frame`` to ``unset``."""
        self.gt[:, frame, point] = np.nan
        self.gt_provenance[:, frame, point] = Provenance.NONE
        self.occluded[:, frame, point] = False
        self.dirty = True

    def clear_frame(self, frame: int) -> None:
        """Reset every label in ``frame`` to ``unset``.

        The per-frame ``reviewed`` flag and the per-point ``absent`` declaration are not
        point labels, so both are left untouched -- otherwise the everyday "reset this
        frame" would silently un-declare an amputation for the whole recording.
        """
        self.gt[:, frame] = np.nan
        self.gt_provenance[:, frame] = Provenance.NONE
        self.occluded[:, frame] = False
        self.dirty = True

    def set_reviewed(self, frame: int, value: bool) -> None:
        """Mark ``frame`` reviewed (or clear it) -- the operator's per-frame check flag."""
        self.reviewed[frame] = bool(value)
        self.dirty = True

    def set_absent(self, points, value: bool, frames=None) -> None:
        """Declare (or un-declare) ``points`` as not being on this animal.

        ``frames`` is a frame index, an iterable of them, or ``None`` for **the whole
        recording** -- the convenience for the common case, an animal that arrives with a
        leg already missing. View-independent either way: an amputated joint is missing
        from every camera at once, which is what separates this from ``occluded``.

        Deliberately **non-destructive**: the GT pixels and occlusions already authored
        there stay in memory (vetoed out of :attr:`has_gt` / :attr:`occluded_effective`,
        quarantined on save), so un-declaring restores them exactly.
        """
        idx = np.atleast_1d(np.asarray(points, dtype=int)).reshape(-1)
        if frames is None:
            self.absent[:, idx] = bool(value)
        else:
            rows = np.atleast_1d(np.asarray(frames, dtype=int)).reshape(-1)
            self.absent[np.ix_(rows, idx)] = bool(value)
        self.dirty = True


# -- identity fingerprint -----------------------------------------------------


def labels_identity(
    *,
    point_names: list[str],
    camera_names: list[str],
    n_frames: int,
    image_sizes: dict[str, tuple[int, int]] | None = None,
    footage: dict | None = None,
) -> dict:
    """Fingerprint the *recording* a set of labels annotates.

    Two parts: the **index domain** (``point_names`` / ``camera_names`` /
    ``n_frames``), which the ``(view, frame, point)`` keys index into, and a
    **recording fingerprint** (``image_sizes`` -- the pixel space GT lives in -- and
    the footage file basenames). Predictions and ``created_utc`` are intentionally
    excluded, so re-running the pipeline on the same recording does not invalidate
    labels. ``footage`` is the mapping :meth:`StageStore.read_footage` returns
    (``camera -> {"abs"|"rel": [paths]}``); only the basenames are kept.
    """
    return {
        "point_names": list(point_names),
        "camera_names": list(camera_names),
        "n_frames": int(n_frames),
        "image_sizes": {
            str(k): [int(v[0]), int(v[1])] for k, v in (image_sizes or {}).items()
        },
        "footage": _footage_basenames(footage),
    }


def _footage_basenames(footage: dict | None) -> dict[str, list[str]]:
    """``camera -> sorted footage file basenames`` from a StageStore footage map."""
    out: dict[str, list[str]] = {}
    for cam, spec in (footage or {}).items():
        paths: list[str] = []
        if isinstance(spec, dict):
            for key in ("rel", "abs"):
                paths = list(spec.get(key) or [])
                if paths:
                    break
        elif isinstance(spec, (list, tuple)):
            paths = list(spec)
        out[str(cam)] = sorted(os.path.basename(str(p)) for p in paths)
    return out


def _check_identity(stored: dict, current: dict, path: Path) -> None:
    """Raise ``ValueError`` if ``stored`` labels do not belong to ``current``.

    The index domain must match exactly (name-based remap of reordered points/cameras
    is a future enhancement); the recording fingerprint must match wherever both
    sides carry it (a bare ``results.h5`` with no footage/sizes cannot be checked on
    those fields and falls back to the index domain).
    """
    for key in ("point_names", "camera_names", "n_frames"):
        if stored.get(key) != current.get(key):
            raise ValueError(
                f"{path} labels do not match this result ({key} differs); "
                "they belong to a different result"
            )
    for key in ("image_sizes", "footage"):
        s, c = stored.get(key), current.get(key)
        if s and c and s != c:
            raise ValueError(
                f"{path} labels belong to a different recording ({key} differs -- e.g. "
                "a changed resolution/crop or different footage); the stored GT pixels "
                "would be misinterpreted, so they are refused"
            )


# -- persistence --------------------------------------------------------------


def _coo_gt(
    labels: Labels, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(index (N,3), xy (N,2), provenance (N,))`` for the GT cells ``mask`` selects."""
    v, t, p = np.nonzero(mask)  # nonzero preserves axis order: view, frame, point
    return (
        np.stack([v, t, p], axis=1).astype(np.int32),
        labels.gt[mask].astype(np.float64),
        labels.gt_provenance[mask].astype(np.uint8),
    )


def absent_to_spans(absent: np.ndarray) -> np.ndarray:
    """``(T, P)`` absence -> ``(S, 3)`` ``[point, t0, t1)`` run-length spans.

    Run-length rather than one row per absent cell: the overwhelmingly common declaration
    covers the whole recording, and that must not cost 3010 rows per keypoint just because
    the representation *can* express a per-frame pattern.
    """
    a = np.asarray(absent, dtype=bool)
    if a.ndim != 2 or not a.any():
        return np.zeros((0, 3), dtype=np.int32)
    spans: list[tuple[int, int, int]] = []
    for p_i in np.nonzero(a.any(axis=0))[0]:
        col = a[:, p_i]
        # Edge-detect on the padded column: +1 opens a run, -1 closes it.
        edges = np.diff(np.concatenate(([0], col.view(np.int8), [0])))
        starts = np.nonzero(edges == 1)[0]
        ends = np.nonzero(edges == -1)[0]
        spans.extend((int(p_i), int(t0), int(t1)) for t0, t1 in zip(starts, ends))
    return np.asarray(spans, dtype=np.int32).reshape(-1, 3)


def spans_to_absent(spans: np.ndarray, n_frames: int, n_points: int) -> np.ndarray:
    """``(S, 3)`` ``[point, t0, t1)`` spans -> a dense ``(T, P)`` absence mask.

    Out-of-range rows are clipped rather than dropped, and overlapping runs simply
    union, so a hand-edited or older file cannot raise here.
    """
    out = np.zeros((n_frames, n_points), dtype=bool)
    for row in np.asarray(spans, dtype=np.int64).reshape(-1, 3):
        p_i, t0, t1 = int(row[0]), int(row[1]), int(row[2])
        if not (0 <= p_i < n_points):
            continue
        t0, t1 = max(0, t0), min(n_frames, t1)
        if t1 > t0:
            out[t0:t1, p_i] = True
    return out


def _coo_cells(mask: np.ndarray) -> np.ndarray:
    """``(M, 3)`` ``[view, frame, point]`` for the cells ``mask`` selects."""
    v, t, p = np.nonzero(mask)
    return np.stack([v, t, p], axis=1).astype(np.int32)


def save_labels(
    path: str | Path, labels: Labels, *, identity: dict, subject_id: str | None = None
) -> None:
    """Write ``labels`` to a sparse ``labels.h5`` sidecar (overwriting ``path``).

    Only authored entries are written (COO), stamped with ``identity``. Clears
    ``labels.dirty`` on success. ``results.h5`` is never touched.

    Rows an absence declaration vetoes are **quarantined**, not dropped: ``gt/`` and
    ``occluded/`` carry live rows only, so a consumer reading ``gt/index`` straight out
    of HDF5 (rather than through :func:`export_gt`) sees a self-consistent file, while
    ``absent/void_gt`` and ``absent/void_occluded`` keep the authored rows so
    un-declaring the point restores them on the next load.

    ``subject_id`` overrides ``labels.subject_id`` when given.
    """
    vetoed = labels._absent_bcast  # (1, 1, P) -> broadcasts over the cell arrays
    raw_gt = labels.gt_authored  # (V, T, P)
    live_gt_mask = raw_gt & ~vetoed
    void_gt_mask = raw_gt & vetoed
    live_occ_mask = labels.occluded & ~vetoed
    void_occ_mask = labels.occluded & vetoed

    gt_index, gt_xy, gt_prov = _coo_gt(labels, live_gt_mask)
    vgt_index, vgt_xy, vgt_prov = _coo_gt(labels, void_gt_mask)
    rev_index = np.nonzero(labels.reviewed)[0].astype(np.int32)  # (K,) frame indices
    absent_spans = absent_to_spans(labels.absent)
    # ``index`` stays the whole-recording subset: it is what a v3 reader understands, and
    # writing it means such a reader degrades to "misses the partial declarations" rather
    # than "sees none of them".
    absent_index = np.nonzero(labels.absent_all_frames())[0].astype(np.int32)

    meta = {
        "deeperfly_labels_format_version": LABELS_FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "subject_id": subject_id if subject_id is not None else labels.subject_id,
    }
    with h5py.File(path, "w") as f:
        f.attrs["meta"] = json.dumps(meta)
        g = f.create_group("gt")
        g.create_dataset("index", data=gt_index, dtype="int32")
        g.create_dataset("xy", data=gt_xy, dtype="float64")
        g.create_dataset("provenance", data=gt_prov, dtype="uint8")
        o = f.create_group("occluded")
        o.create_dataset("index", data=_coo_cells(live_occ_mask), dtype="int32")
        r = f.create_group("reviewed")
        r.create_dataset("index", data=rev_index, dtype="int32")
        a = f.create_group("absent")
        a.create_dataset("index", data=absent_index, dtype="int32")
        a.create_dataset("spans", data=absent_spans, dtype="int32")
        vg = a.create_group("void_gt")
        vg.create_dataset("index", data=vgt_index, dtype="int32")
        vg.create_dataset("xy", data=vgt_xy, dtype="float64")
        vg.create_dataset("provenance", data=vgt_prov, dtype="uint8")
        vo = a.create_group("void_occluded")
        vo.create_dataset("index", data=_coo_cells(void_occ_mask), dtype="int32")
    labels.dirty = False


def load_labels(path: str | Path, *, identity: dict) -> Labels | None:
    """Read a ``labels.h5`` sidecar into a dense :class:`Labels`, or ``None`` if absent.

    Validates the stored identity against ``identity`` (raising ``ValueError`` on a
    different recording/result) and normalizes the sparse lists: out-of-range or
    non-finite GT rows are dropped, duplicate ``(view, frame, point)`` keys resolve
    last-write-wins, and any ``(view, frame, point)`` present in both ``gt`` and
    ``occluded`` keeps the GT (which carries an authored pixel) and drops the
    occlusion -- so the on-disk invariants cannot desync the in-memory overlay. That tie
    is resolved on the **raw** stored mask, deliberately ignoring the absence veto, so a
    round trip through a declaration is idempotent.

    ``reviewed`` is optional (a v1 file has none -> no frames reviewed) and so is
    ``absent`` (a v1/v2 file has none -> nothing absent). The quarantined
    ``absent/void_*`` rows are merged back into the overlay, so the pixels an absence
    declaration vetoed survive the round trip and reappear if it is lifted.

    A file written by a *newer* deeperfly is refused rather than silently misread.
    """
    p = Path(path)
    if not p.exists():
        return None
    with h5py.File(p, "r") as f:
        meta = json.loads(f.attrs["meta"])  # type: ignore[arg-type]
        stored_identity = meta.get("identity", {})
        stored_version = int(meta.get("deeperfly_labels_format_version", 1))
        gt_index = np.asarray(f["gt/index"][()], dtype=np.int64).reshape(-1, 3)  # type: ignore[index]
        gt_xy = np.asarray(f["gt/xy"][()], dtype=float).reshape(-1, 2)  # type: ignore[index]
        gt_prov = np.asarray(f["gt/provenance"][()], dtype=np.uint8).reshape(-1)  # type: ignore[index]
        occ_index = np.asarray(f["occluded/index"][()], dtype=np.int64).reshape(-1, 3)  # type: ignore[index]
        rev_index = (  # optional group: pre-v2 files carry no review progress
            np.asarray(f["reviewed/index"][()], dtype=np.int64).reshape(-1)  # type: ignore[index]
            if "reviewed" in f
            else np.empty(0, dtype=np.int64)
        )
        subject_id = meta.get("subject_id")
        # Optional v3 absence group, with the rows the declaration quarantined.
        absent_index = np.empty(0, dtype=np.int64)
        absent_spans = None  # None -> no v4 spans; fall back to the v3 index
        vgt_index = np.empty((0, 3), dtype=np.int64)
        vgt_xy = np.empty((0, 2), dtype=float)
        vgt_prov = np.empty(0, dtype=np.uint8)
        vocc_index = np.empty((0, 3), dtype=np.int64)
        if "absent" in f:
            absent_index = np.asarray(f["absent/index"][()], dtype=np.int64).reshape(-1)  # type: ignore[index]
            if "absent/spans" in f:
                absent_spans = np.asarray(  # type: ignore[index]
                    f["absent/spans"][()], dtype=np.int64
                ).reshape(-1, 3)
            if "absent/void_gt" in f:
                vgt_index = np.asarray(  # type: ignore[index]
                    f["absent/void_gt/index"][()], dtype=np.int64
                ).reshape(-1, 3)
                vgt_xy = np.asarray(  # type: ignore[index]
                    f["absent/void_gt/xy"][()], dtype=float
                ).reshape(-1, 2)
                vgt_prov = np.asarray(  # type: ignore[index]
                    f["absent/void_gt/provenance"][()], dtype=np.uint8
                ).reshape(-1)
            if "absent/void_occluded" in f:
                vocc_index = np.asarray(  # type: ignore[index]
                    f["absent/void_occluded/index"][()], dtype=np.int64
                ).reshape(-1, 3)

    if stored_version > LABELS_FORMAT_VERSION:
        raise ValueError(
            f"{p} was written by a newer deeperfly (labels format v{stored_version}, "
            f"this build understands v{LABELS_FORMAT_VERSION}); refusing to read it "
            "rather than silently dropping state it carries"
        )
    _check_identity(stored_identity, identity, p)

    # Quarantined rows rejoin the live ones -- they are authored state, just vetoed.
    if vgt_index.size:
        gt_index = np.concatenate([gt_index, vgt_index])
        gt_xy = np.concatenate([gt_xy, vgt_xy])
        gt_prov = np.concatenate([gt_prov, vgt_prov])
    if vocc_index.size:
        occ_index = np.concatenate([occ_index, vocc_index])

    n_views = len(identity["camera_names"])
    n_frames = int(identity["n_frames"])
    n_points = len(identity["point_names"])
    labels = Labels.empty(n_views, n_frames, n_points)

    def _in_range(idx: np.ndarray) -> np.ndarray:
        v, t, pt = idx[:, 0], idx[:, 1], idx[:, 2]
        return (
            (v >= 0)
            & (v < n_views)
            & (t >= 0)
            & (t < n_frames)
            & (pt >= 0)
            & (pt < n_points)
        )

    # GT rows: keep in-range, finite, last-write-wins on duplicate keys.
    if gt_index.size:
        keep = _in_range(gt_index) & np.isfinite(gt_xy).all(axis=1)
        n_dropped = int((~keep).sum())
        if n_dropped:
            log.warning("%s: dropped %d out-of-range/NaN GT row(s)", p, n_dropped)
        for (v, t, pt), xy, prov in zip(gt_index[keep], gt_xy[keep], gt_prov[keep]):
            labels.gt[v, t, pt] = xy
            labels.gt_provenance[v, t, pt] = prov
    # Occluded rows: keep in-range, but GT wins the disjointness tie. The tie is decided
    # on the RAW authored mask, not ``has_gt``: under the absence veto ``has_gt`` reads
    # False for a declared point, which would start *retaining* an occlusion that a v2
    # load drops -- and load/save would stop being idempotent across a declaration.
    raw_gt = labels.gt_authored
    if occ_index.size:
        keep = _in_range(occ_index)
        for v, t, pt in occ_index[keep]:
            if raw_gt[v, t, pt]:
                log.warning(
                    "%s: (view=%d, frame=%d, point=%d) is both GT and occluded; "
                    "keeping the GT",
                    p,
                    v,
                    t,
                    pt,
                )
                continue
            labels.occluded[v, t, pt] = True
    # Reviewed frames: keep in-range indices (a stale/oversized index is dropped).
    if rev_index.size:
        keep = (rev_index >= 0) & (rev_index < n_frames)
        labels.reviewed[rev_index[keep]] = True
    # Absence. ``spans`` is authoritative when present (v4); a v3 file has only ``index``,
    # which meant "absent in every frame", so it materializes as a full column.
    if absent_spans is not None and absent_spans.size:
        labels.absent |= spans_to_absent(absent_spans, n_frames, n_points)
    elif absent_index.size:
        keep = (absent_index >= 0) & (absent_index < n_points)
        if int((~keep).sum()):
            log.warning(
                "%s: dropped %d out-of-range absent point index/indices",
                p,
                int((~keep).sum()),
            )
        labels.absent[:, absent_index[keep]] = True
    n_void = int((raw_gt & labels._absent_bcast).sum())
    n_void_occ = int((labels.occluded & labels._absent_bcast).sum())
    if n_void or n_void_occ:
        log.info(
            "%s: %d GT row(s) and %d occlusion(s) quarantined by the absent declaration",
            p,
            n_void,
            n_void_occ,
        )
    labels.subject_id = subject_id
    labels.dirty = False
    return labels


# -- migration from the legacy corrections.h5 ---------------------------------


def migrate_from_corrections(
    corrections,
    result_pts2d: np.ndarray,
    *,
    keep_ambiguous_occluded: bool = False,
) -> tuple[Labels, dict]:
    """Convert a legacy dense :class:`~deeperfly.gui.corrections.Corrections` to labels.

    This preserves the operator's authored 2D *pixels* as GT but not the old solve
    semantics -- the 3D re-derives under the new policy. The mapping (and what it
    drops) is:

    - ``pts2d_edited`` (incl. the old ``fixed`` finalized pixels) with a finite pixel
      -> **GT** (provenance ``dragged``). ``fixed`` collapses into GT.
    - ``pts2d_invisible`` on a view the detector *did* see (``isfinite`` prediction)
      -> **occluded** (a confident human "delete this view").
    - ``pts2d_invisible`` where the detector *also* missed is ambiguous: the fresh
      overlay seeded ``invisible`` from NaN predictions, so it cannot be told from a
      real human occlusion. Dropped by default (re-derives as *unset*); set
      ``keep_ambiguous_occluded`` to keep them and un-occlude the false positives by
      hand.
    - ``pts3d_edited`` with no fixed 2D cannot be expressed as a 2D label and is
      dropped.

    Returns ``(labels, report)`` where ``report`` counts each bucket.
    """
    pts2d = np.asarray(result_pts2d, dtype=float)
    pred_finite = np.isfinite(pts2d).all(axis=-1)  # (V, T, P)
    n_views, n_frames, n_points = pred_finite.shape
    labels = Labels.empty(n_views, n_frames, n_points)

    gt_mask = corrections.pts2d_edited & np.isfinite(corrections.pts2d).all(axis=-1)
    for v, t, pt in zip(*np.nonzero(gt_mask)):
        labels.gt[v, t, pt] = corrections.pts2d[v, t, pt]
        labels.gt_provenance[v, t, pt] = Provenance.DRAGGED

    occ_confident = corrections.pts2d_invisible & pred_finite & ~gt_mask
    occ_ambiguous = corrections.pts2d_invisible & ~pred_finite & ~gt_mask
    occ_mask = occ_confident | (occ_ambiguous if keep_ambiguous_occluded else False)
    labels.occluded = np.asarray(occ_mask, dtype=bool)

    # A pure-3D edit with no fixed 2D pixel cannot become a 2D label.
    fixed = getattr(corrections, "pts2d_fixed", np.zeros_like(gt_mask))
    pts3d_only = corrections.pts3d_edited & ~fixed.any(axis=0)  # (T, P)

    labels.dirty = False
    report = {
        "gt": int(gt_mask.sum()),
        "occluded": int(labels.occluded.sum()),
        "dropped_ambiguous_occluded": 0
        if keep_ambiguous_occluded
        else int(occ_ambiguous.sum()),
        "dropped_pts3d_only": int(pts3d_only.sum()),
    }
    return labels, report


# -- export (the training/eval consumer seam) ---------------------------------


def export_gt(
    labels: Labels, *, include_projection: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense ground-truth arrays for training/eval, filtered by provenance.

    Returns ``(gt_xy (V,T,P,2), gt_mask (V,T,P) bool, occluded (V,T,P) bool)``. The GT
    is in **footage pixel space** -- the coordinate the operator clicked. By default
    ``confirmed_projection`` GT (the model's own reprojected guess) is excluded so the
    export is only human-placed / prediction-confirmed pixels; ``include_projection``
    keeps it. ``placeholder_seed`` GT is dropped either way -- see :class:`Provenance`.

    To train the 2D detector, transform ``gt_xy`` from footage space into each
    pathway's model-input space by inverting the pathway ``FrameTransform``
    (mirror/crop/resize). That plan lives in the run config, not ``results.h5``, so the
    transform is applied by the training pipeline -- this function is the neutral,
    footage-space contract it consumes.

    Points declared absent are excluded from **both** returned masks: an amputated
    keypoint is not ground truth (so it must not be supervised) and it is not occluded
    either (so it must not be exported as a positive "unplaceable from this view" label,
    which is what a naive all-views-occluded workaround would teach the detector). Ask
    :func:`export_absent` for the declaration itself.
    """
    mask = labels.has_gt.copy()
    if not include_projection:
        mask &= labels.gt_provenance != Provenance.CONFIRMED_PROJECTION
    # Unconditional, and deliberately not behind ``include_projection``: a placeholder
    # seed is a coordinate the editor invented so the operator would have something to
    # grab, clamped to the image edge when the reprojection fell outside. It is the one
    # provenance that is never a claim about where the keypoint is.
    mask &= labels.gt_provenance != Provenance.PLACEHOLDER_SEED
    gt_xy = np.where(mask[..., None], labels.gt, np.nan)
    return gt_xy, mask, labels.occluded_effective


def export_absent(labels: Labels) -> Bool[np.ndarray, "T P"]:
    """``(T, P)`` which keypoints are not on this animal, per frame.

    Per frame rather than per point because absence *can* change over a recording (a leg
    lost to autotomy). ``labels.absent_all_frames()`` is the whole-recording subset, which
    is what structural consumers -- the IK body plan, dropping a point from bundle
    adjustment -- must use instead, since those are decided once per recording.

    A strictly additive companion to :func:`export_gt` (whose 3-tuple shape several
    consumers unpack and one subscripts, so it is deliberately left alone). A consumer
    that predates this call should treat a missing declaration as all-False.

    The intended training semantics is **masking**, not negative supervision: an absent
    keypoint contributes no gradient at all. Supervising it with an all-zero target
    heatmap would be the stronger claim ("learn that nothing is here"), and it needs both
    a decode contract that can abstain and enough amputee animals to calibrate one --
    neither of which exists yet.
    """
    return np.asarray(labels.absent, dtype=bool).copy()


def resolve_point_names(names, point_names: list[str]) -> list[int]:
    """Resolve keypoint ``names`` to indices into ``point_names``.

    Accepts exact names and ``fnmatch`` globs (``lf_*``), and **raises** ``ValueError``
    on a name that matches nothing -- a typo must not silently declare nothing absent.
    Returns sorted unique indices.
    """
    import fnmatch

    wanted = [names] if isinstance(names, str) else list(names)
    out: set[int] = set()
    for name in wanted:
        hits = [i for i, n in enumerate(point_names) if fnmatch.fnmatchcase(n, name)]
        if not hits:
            raise ValueError(
                f"no skeleton point matches {name!r}; known points: "
                + ", ".join(point_names)
            )
        out.update(hits)
    return sorted(out)
