"""The ground-truth annotation sidecar (``labels.h5``).

The keypoint editor is a *ground-truth annotation* tool, not a prediction editor:
the operator's 2D labels are the source of truth and the 3D pose is a pure derived
function of them (:mod:`deeperfly.gui.solve`). So the only state worth persisting
is what the operator actually authored.

**Instances (schema v8).** The unit of annotation is an *instance*: one annotation
skeleton, created in a frame by double-clicking the detected skeleton, which seeds a
position for every ``(view, point)`` and from then on is its own object. Before v8 there
was no such thing -- the overlay was a per-cell tri-state over the detections, where
``unset`` meant "defer to whatever the detector said". That third value existed only
because the cell had no position of its own; once an instance owns its positions there is
nothing for it to mean, and the model collapses to two facts per cell plus one flag:

.. code-block:: text

    seeds(x, y)  -> where the instance's keypoint started (every cell, once created)
    gt(x, y)     -> the operator dragged/placed it here; overrides the seed
    hidden       -> "do not include this cell in the training loss"

**The hidden flag is its own binary axis, fully orthogonal to ``gt``.** It carries exactly
one meaning -- *whether this cell is supervised* -- and it is the only per-cell thing in
the schema that is neither a position nor derived from one. Orthogonal is meant literally:
setting or clearing it never reads or writes ``gt``, ``seeds`` or the 3D, and nothing reads
it to decide where a keypoint is drawn, whether it can be labeled, or what the solve sees.
All four combinations are legal and each says something different:

.. code-block:: text

    gt, not hidden   the ordinary label -- a pixel, supervised
    gt, hidden       a pixel the operator stands behind but withholds from the loss
    no gt, hidden    nothing to supervise anyway; the flag records the intent
    no gt, not hidden  unlabeled

Its storage keeps the older name ``occluded``: through v8 the flag meant "a human cannot
see this keypoint in this view", and it was read as an instruction -- first to drop the
view from the 3D solve, later to suppress its detection on screen. Both jobs are gone (the
robust solve down-weights a bad observation on its merits, and hiding a cell must not take
away the position the operator needs in order to place it), leaving the loss switch, which
is the one thing no geometry can supply. Only the meaning narrowed, so no stored bit
changes and no file needs migrating; the group and the attribute stay ``occluded`` and the
editor calls the flag what it now is -- **Hidden**.

A non-GT cell contributes its **seed** to its point's 3D solve, precisely where the
detections used to contribute -- so the solve, its Huber depth estimate for the one-GT
case, and the undo history all carry over untouched. Each *frame* additionally carries one
``reviewed`` flag. Everything else -- the 3D point, the reprojections, reprojection error
-- stays derived and unstored.

The instance axis is single-valued in this build (one animal), but it is the axis the COO
index has reserved since v6, so multiple instances per frame land in a slot already on
disk rather than needing a second migration of the one dataset that cannot be regenerated.

**Absence (schema v3, per-frame in v4).** One more thing the operator can author is not a
property of a *view* at all: a keypoint may not *exist on this animal* -- an amputated leg,
an ablated antenna. That is categorically different from ``hidden`` ("it is there, and I am
holding this one cell out of the loss") and from ``unset`` ("nobody has looked yet"), and it
is view-independent by construction. It *can* vary over time, though -- a leg lost to autotomy
part-way through a recording -- so it is stored per ``(frame, point)``:

.. code-block:: text

    absent      (T, P) bool  -> "this keypoint is not on this animal, in this frame"

Most declarations cover the whole recording (an animal that arrives with a leg missing
keeps it missing), which is why the editor offers a one-gesture "apply to the entire
recording" alongside the per-frame toggle, and why the on-disk form is **run-length spans**:
a whole-recording declaration is a single row no matter how long the recording.

A caution the schema deliberately does *not* enforce: absence is not a bulk ``hidden``.
Both keep a cell out of the loss, but absence also destroys the keypoint's *position* --
in every view, and for the structural consumers (the IK body plan, bundle adjustment) for
the whole recording -- while ``hidden`` leaves the joint exactly where it is and withholds
only the supervision. So declaring a joint absent in the handful of frames where it is
merely awkward to see costs the position too, and hiding an amputated joint view by view
leaves a phantom limb standing in every canvas. Absence is a claim about the animal;
``hidden`` is a claim about what to train on.

Absence acts as a **read-time veto** rather than a fourth exclusive cell value: the
authored pixels and occlusions stay in memory untouched, and the derived masks every
consumer reads (:attr:`Labels.has_gt`, :attr:`Labels.occluded_effective`) simply mask
the absent points out. Declaring -- and un-declaring -- absence therefore destroys
nothing, and the two masks are the whole contract:

.. code-block:: text

    gt_authored         raw "a pixel is stored here"   -- persistence and undo only
    has_gt              gt_authored & ~absent          -- everything else
    occluded_effective  occluded    & ~absent          -- everything else

**One kind of ground truth (schema v7).** v5/v6 tagged each ``gt`` row with a
``provenance`` code recording which layer its pixel had been copied out of -- dragged,
confirmed prediction, confirmed projection, or an editor-invented placeholder. v7 drops
it. GT is created by *dragging a point from a proposed initial location* (a detection or a
reprojection), and which proposal it started from is not a property of the label that
results; the proposal layers are still there to be read separately whenever a consumer
wants the precedence GT -> detection -> projection. Migrating a v5/v6 file keeps every row
except the invented placeholder seeds, which were never exportable.

On disk (schema v2) the deltas are stored **sparsely** (COO), which is tiny next to
``results.h5`` and, unlike the old dense ``corrections.h5``, carries no copy of the
prediction NaN pattern:

.. code-block:: text

    attrs["meta"]   json {deeperfly_labels_format_version, created_utc, identity,
                          subject_id}   (subject_id added in v3; optional, may be null)
    gt/
        index       (N, 3) int32   [view, frame, point]
        xy          (N, 2) float64  the 2D pixel the operator created (footage space)
    seeds/                          (v8) the instance's evidence-backed start positions
        index       (S, 3) int32   [view, frame, point]
        xy          (S, 2) float64
    instance/                       (v8) frames carrying an annotation skeleton
        index       (J,)   int32
    occluded/                       the **hidden** flag -- the cells held out of the loss
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from jaxtyping import Bool, Float

__all__ = [
    "Labels",
    "LandmarkLabels",
    "absent_to_spans",
    "spans_to_absent",
    "save_labels",
    "load_labels",
    "load_landmark_labels",
    "labels_identity",
    "migrate_from_corrections",
    "export_gt",
    "export_absent",
    "resolve_point_names",
    "LABELS_FORMAT_VERSION",
]

log = logging.getLogger("deeperfly")

#: utf-8 variable-length strings (the landmark names are the first strings this file
#: stores; the rest of the schema is numeric).
_STR = h5py.string_dtype("utf-8")

LABELS_FORMAT_VERSION = 8

#: The COO index width from v6 on: ``[view, frame, instance, point]``. Earlier versions
#: wrote ``[view, frame, point]``.
#:
#: The instance column is reserved *now*, while there is exactly one animal, because this
#: is the one dataset in the project that cannot be regenerated -- and widening a stored
#: index later would mean a second migration of it. Writing it costs one column; deferring
#: it costs the migration twice. See ``design/project-system-plan.md`` §10.1.
#:
#: This build is single-animal: every row is written with ``instance = 0``, and a row
#: carrying a different instance is **refused on load** rather than collapsed into animal
#: zero. A silent collapse would merge two animals' keypoints into one skeleton, which
#: looks like a labeling mistake rather than a format mismatch.
_INDEX_WIDTH_V6 = 4


#: v5/v6 stored a per-row ``provenance`` code saying which layer a GT pixel had been
#: *copied out of* (dragged / confirmed_prediction / confirmed_projection /
#: placeholder_seed). v7 drops it: there is one kind of ground truth -- a pixel the
#: operator created -- and the layer a proposal came from is not a property of the
#: resulting label. The one code that was never a claim about the animal was
#: ``placeholder_seed = 4``, a coordinate the editor invented (clamped to the image edge)
#: purely to give the operator something to grab; ``export_gt`` dropped it
#: unconditionally. Migration must drop those rows rather than carry them over, or
#: fabricated edge pixels would become indistinguishable from real labels.
_LEGACY_PLACEHOLDER_SEED = 4


@dataclass
class Labels:
    """In-memory ground-truth overlay on a :class:`~deeperfly.results.PoseResult`.

    The per-``(view, frame, point)`` arrays are dense ``(V, T, P)``-shaped (2D pixels
    carry a trailing 2). A GT pixel is *stored* iff ``gt`` is finite there, which is
    exactly ``isfinite(gt)``; ``occluded`` is the **hidden** flag -- the cells to hold out
    of the training loss -- and it is an independent axis, not a third value of ``gt`` (see
    the module docstring). ``reviewed`` is a separate per-*frame* ``(T,)`` flag (the
    operator's "I have checked this frame"), independent of the point labels. ``absent`` is a
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

    There is no cross-invariant to maintain between ``gt`` and ``occluded``: all four
    combinations are legal, so neither the mutators nor the loader may resolve one against
    the other. ``dirty`` tracks unsaved changes.
    """

    gt: Float[np.ndarray, "V T P 2"]
    #: The **hidden** flag ``(V, T, P)``: "hold this cell out of the training loss". Its own
    #: axis -- it stores no position, gates no position, and is read by nothing but the
    #: export and the marker that reports it. Named ``occluded`` for its storage group; see
    #: the module docstring for what narrowed and why nothing had to migrate.
    occluded: Bool[np.ndarray, "V T P"]
    reviewed: Bool[np.ndarray, "T"]  # per-frame "operator has checked this frame"
    #: Per-frame "an annotation skeleton has been created here". Explicit rather than
    #: derived from ``isfinite(seeds).any()``, because a seed is only written where there is
    #: real evidence for it -- a detection or a reprojection. A frame whose keypoints the
    #: network predicts nowhere (an ipsilateral-only model, say) gets an instance with *no*
    #: finite seeds at all, and it still has to count as created.
    instance: Bool[np.ndarray, "T"] | None = None
    #: The **instance seeds** ``(V, T, P, 2)``: where each keypoint started when the
    #: operator created an annotation skeleton in that frame, NaN in frames with no
    #: instance yet. This is the array that used to be the detector's job (see
    #: :attr:`EditorState.detections`): a non-GT cell's contribution to its point's 3D
    #: solve. Persisted, and deliberately so -- deriving it at load time from whatever the
    #: detections happen to be *then* would silently re-solve every non-GT point after a
    #: model re-run, which is the same trap an unpersisted ray-slid depth was.
    seeds: Float[np.ndarray, "V T P 2"] | None = None
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
        n_views, n_frames, n_points = (
            self.gt.shape[0],
            self.gt.shape[1],
            self.gt.shape[2],
        )
        if self.seeds is None:
            self.seeds = np.full((n_views, n_frames, n_points, 2), np.nan)
        if self.instance is None:
            self.instance = np.zeros(n_frames, dtype=bool)
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
            occluded=np.zeros((n_views, n_frames, n_points), dtype=bool),
            reviewed=np.zeros(n_frames, dtype=bool),
            absent=np.zeros(n_points, dtype=bool),
            seeds=np.full((n_views, n_frames, n_points, 2), np.nan),
            instance=np.zeros(n_frames, dtype=bool),
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
        """The **hidden** flag with the absent points vetoed out.

        Absence already excludes the point from supervision everywhere, so a "hold this out
        of the loss" mark under it has nothing left to hold out; reporting it anyway would
        make an amputated joint look like 7 deliberate hold-outs in every export and every
        count. The raw bit stays in :attr:`occluded` (un-declaring restores it).
        """
        return self.occluded & ~self._absent_bcast

    @property
    def _absent_bcast(self) -> Bool[np.ndarray, "V T P"]:
        """:attr:`absent` broadcast to the ``(V, T, P)`` cell shape (absence spans views)."""
        return np.asarray(self.absent, dtype=bool)[None, :, :]

    @property
    def any_labels(self) -> bool:
        """Whether any GT pixel, hidden mark or absence declaration has been authored."""
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

    def set_gt(self, view: int, frame: int, point: int, xy) -> None:
        """Create a GT pixel for ``point`` in ``view`` at ``frame``.

        Writes the position and nothing else. In particular it does not read or clear the
        **hidden** flag: whether a cell is supervised is a separate decision from where its
        keypoint is, and a cell can legitimately carry a pixel the operator stands behind
        and still be held out of the loss (see :meth:`set_occluded`).
        """
        self.gt[view, frame, point] = np.asarray(xy, dtype=float)
        self.dirty = True

    def clear_gt(self, view: int, frame: int, point: int) -> None:
        """Drop just the GT pixel for ``point`` in ``view`` (the hidden flag untouched)."""
        self.gt[view, frame, point] = np.nan
        self.dirty = True

    def set_occluded(self, view: int, frame: int, point: int, value: bool) -> None:
        """Set (or clear) the **hidden** flag on one cell: hold it out of the training loss.

        The whole of what this writes is one bit. It does not clear a pixel, does not move a
        keypoint, does not touch triangulation, and no consumer of a *position* reads it --
        so it composes freely with :meth:`set_gt` in either order and all four combinations
        mean what they say. Its independence is the point: the training loss is the one
        thing about a cell that geometry cannot decide, so it needs a switch of its own.
        """
        self.occluded[view, frame, point] = bool(value)
        self.dirty = True

    def clear_view(self, view: int, frame: int, point: int) -> None:
        """Reset one ``(view, frame, point)``: drop the GT pixel *and* the hidden flag.

        The one verb that deliberately spans both axes, because "start over on this cell" is
        a gesture the operator asks for by name (the editor's Reset). Every other mutator
        touches exactly one of them.
        """
        self.gt[view, frame, point] = np.nan
        self.occluded[view, frame, point] = False
        self.dirty = True

    def clear_point(self, frame: int, point: int) -> None:
        """Reset every view of ``point`` at ``frame`` to ``unset``."""
        self.gt[:, frame, point] = np.nan
        self.occluded[:, frame, point] = False
        self.dirty = True

    def clear_frame(self, frame: int) -> None:
        """Reset every label in ``frame`` to ``unset``.

        The per-frame ``reviewed`` flag and the per-point ``absent`` declaration are not
        point labels, so both are left untouched -- otherwise the everyday "reset this
        frame" would silently un-declare an amputation for the whole recording.
        """
        self.gt[:, frame] = np.nan
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


@dataclass
class LandmarkLabels:
    """Observed calibration landmarks for one recording (a ``landmarks/`` group).

    Deliberately a *separate* overlay from :class:`Labels`, sharing only the file. A
    landmark is not a skeleton point: it must never reach the detector's output mapping,
    the IK body plan, the bone-length priors, the training export or the rendered videos,
    and it must not perturb the fingerprinted ``point_names`` that every existing sidecar
    is validated against. Two namespaces enforce both by construction; one namespace with
    a flag would need a filter at every call site and would invalidate every label already
    authored. See :mod:`deeperfly.landmarks` for why they exist at all.

    Attributes
    ----------
    names
        The landmark names in ``L``-axis order (from the project's ``landmarks.toml``).
    static
        ``(L,)`` whether each is one fixed 3D point over time. Stored alongside the
        observations so a solve reading the file alone knows how to treat each column.
    xy
        ``(V, T, L, 2)`` observed pixel, NaN where unobserved. A landmark is always
        operator-placed, so ``isfinite(xy)`` is the whole of "this one is observed".
    """

    names: tuple[str, ...]
    static: Bool[np.ndarray, "L"]
    xy: Float[np.ndarray, "V T L 2"]
    dirty: bool = field(default=False)

    @classmethod
    def empty(cls, n_views: int, n_frames: int, names, static=None) -> LandmarkLabels:
        names = tuple(str(n) for n in names)
        n = len(names)
        return cls(
            names=names,
            static=np.ones(n, dtype=bool)
            if static is None
            else np.asarray(static, dtype=bool).reshape(n),
            xy=np.full((n_views, n_frames, n, 2), np.nan),
        )

    @property
    def observed(self) -> Bool[np.ndarray, "V T L"]:
        """``(V, T, L)`` where a pixel is stored."""
        return np.isfinite(self.xy).all(axis=-1)

    @property
    def any_labels(self) -> bool:
        return bool(self.observed.any())

    def index(self, name: str) -> int:
        """The ``L``-axis index of ``name``.

        Raises
        ------
        KeyError
            If this recording's landmark set has no such name.
        """
        try:
            return self.names.index(str(name))
        except ValueError:
            raise KeyError(
                f"no landmark named {name!r} (have {list(self.names)})"
            ) from None

    def set(self, view: int, frame: int, landmark: int, xy) -> None:
        """Place (or move) a landmark observation."""
        self.xy[view, frame, landmark] = np.asarray(xy, dtype=float)
        self.dirty = True

    def clear(self, view: int, frame: int, landmark: int) -> None:
        """Drop one landmark observation."""
        self.xy[view, frame, landmark] = np.nan
        self.dirty = True

    def counts(self) -> dict[str, int]:
        """``name -> how many (view, frame) cells observe it`` (for the readiness report)."""
        obs = self.observed
        return {name: int(obs[:, :, i].sum()) for i, name in enumerate(self.names)}


def _write_landmarks(f, landmarks: LandmarkLabels | None) -> None:
    """Write the ``landmarks/`` group, or nothing when there is nothing to write.

    Skipped entirely for an empty set, so a recording that never used landmarks produces a
    file byte-identical to one from before they existed.
    """
    if landmarks is None or not landmarks.names:
        return
    obs = landmarks.observed
    v, t, lm = np.nonzero(obs)
    g = f.create_group("landmarks")
    g.create_dataset(
        "names", data=np.array(list(landmarks.names), dtype=object), dtype=_STR
    )
    g.create_dataset("static", data=np.asarray(landmarks.static, dtype=bool))
    g.create_dataset(
        "index", data=np.stack([v, t, lm], axis=1).astype(np.int32), dtype="int32"
    )
    g.create_dataset("xy", data=landmarks.xy[obs].astype(np.float64), dtype="float64")


def load_landmark_labels(
    path: str | Path, *, n_views: int, n_frames: int
) -> LandmarkLabels | None:
    """Read the ``landmarks/`` group of a ``labels.h5``, or ``None`` if it has none.

    Read separately from :func:`load_labels` rather than folded into it: the landmark set
    is defined by the *project*, not by the recording's skeleton, so a consumer that does
    not care about calibration should not have to know they exist. Out-of-range rows are
    dropped (a hand-edited or stale file must not raise here).

    Parameters
    ----------
    path
        The ``labels.h5``.
    n_views, n_frames
        The recording's dimensions, to size the dense arrays and range-check the rows.

    Returns
    -------
    LandmarkLabels or None
        The overlay, or ``None`` when the file is absent or carries no landmarks.
    """
    p = Path(path)
    if not p.exists():
        return None
    with h5py.File(p, "r") as f:
        if "landmarks" not in f:
            return None
        g = f["landmarks"]
        names = tuple(
            n.decode() if isinstance(n, bytes) else str(n) for n in g["names"][()]
        )
        static = np.asarray(g["static"][()], dtype=bool).reshape(len(names))
        index = np.asarray(g["index"][()], dtype=np.int64).reshape(-1, 3)
        xy = np.asarray(g["xy"][()], dtype=float).reshape(-1, 2)

    out = LandmarkLabels.empty(n_views, n_frames, names, static)
    if index.size:
        v, t, lm = index[:, 0], index[:, 1], index[:, 2]
        keep = (
            (v >= 0)
            & (v < n_views)
            & (t >= 0)
            & (t < n_frames)
            & (lm >= 0)
            & (lm < len(names))
            & np.isfinite(xy).all(axis=1)
        )
        if int((~keep).sum()):
            log.warning(
                "%s: dropped %d out-of-range/NaN landmark row(s)", p, int((~keep).sum())
            )
        for (vi, ti, li), pt in zip(index[keep], xy[keep]):
            out.xy[vi, ti, li] = pt
    out.dirty = False
    return out


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
    """``camera -> sorted footage file basenames`` from any footage pointer map.

    One line, because :func:`deeperfly.footage.basenames` is the shared projection -- and it
    is deliberately byte-identical to what this function used to compute, so adopting it
    cannot invalidate a single stored label.
    """
    from ..footage import basenames

    return {str(cam): basenames(spec) for cam, spec in (footage or {}).items()}


def _check_identity(stored: dict, current: dict, path: Path) -> dict[int, int] | None:
    """Validate that ``stored`` labels belong to ``current``, and say how to read them.

    Returns
    -------
    dict of int to int, or None
        ``stored view index -> current view index`` when the two disagree about the camera
        axis but describe the same cameras (see :func:`_camera_correspondence`); ``None``
        when no remap is needed. The caller applies it to the stored COO rows.

    Raises
    ------
    ValueError
        If the two do not describe the same recording at all.

    Notes
    -----
    ``point_names`` and ``n_frames`` must still match **exactly**. Points are the axis whose
    reordering is a project-wide migration (``deeperfly project skeleton``), and remapping
    them silently here would bypass its dry run and its confirmation.

    Cameras are different, and they had to become so: this rig names the same seven cameras
    three ways -- file stems from a bare directory, source names from a config, view names
    from the detection plan -- so labels authored *before* a recording was ever run were
    stamped with one naming and then refused after the first run, which is the whole
    label-first-then-calibrate workflow walking into a wall.
    """
    for key in ("point_names", "n_frames"):
        if stored.get(key) != current.get(key):
            raise ValueError(
                f"{path} labels do not match this result ({key} differs); "
                "they belong to a different result"
            )
    remap = _camera_correspondence(stored, current, path)
    # Sizes and footage are compared per camera *through* the correspondence, so a rename
    # does not read as a different recording.
    _check_per_camera(stored, current, remap, path)
    return remap


def _camera_correspondence(
    stored: dict, current: dict, path: Path
) -> dict[int, int] | None:
    """``stored view -> current view``, or ``None`` when the axes already agree.

    Three attempts, in decreasing confidence, and never an index-based fallback:

    1. **Identical name lists** -- nothing to do.
    2. **By name.** Handles a pure reordering: a from-scratch session names its views in the
       alphabetical order of ``recording.toml``'s footage table, while a run names them in
       config order.
    3. **By footage.** Handles a *rename*, which name matching cannot: ``camera_F`` and ``f``
       share no string, but if both are recorded against the same footage file they are the
       same camera. This matches on what a camera *is* rather than on what it is called.

    Raises
    ------
    ValueError
        When neither correspondence is complete. A partial one is refused rather than applied
        to the cameras it does cover: the uncovered views' rows would silently vanish.
    """
    s_names = list(stored.get("camera_names") or [])
    c_names = list(current.get("camera_names") or [])
    if s_names == c_names:
        return None
    if not s_names or not c_names:
        raise ValueError(
            f"{path} labels do not match this result (camera_names differs); "
            "they belong to a different result"
        )

    by_name = {i: c_names.index(n) for i, n in enumerate(s_names) if n in c_names}
    if len(by_name) == len(s_names):
        log.info(
            "%s: the camera order changed (%s -> %s); its rows are remapped by NAME",
            path.name,
            s_names,
            c_names,
        )
        return by_name

    by_footage = _match_by_footage(stored, current, s_names, c_names)
    if by_footage is not None:
        log.info(
            "%s: the cameras were renamed (%s -> %s); its rows are remapped by matching "
            "each camera's FOOTAGE, which is what identifies it",
            path.name,
            s_names,
            c_names,
        )
        return by_footage

    raise ValueError(
        f"{path} labels do not match this result: its cameras are {s_names} and this "
        f"result's are {c_names}, and they cannot be put in correspondence by name or by "
        "footage. The stored GT pixels would be attributed to the wrong view, so they are "
        "refused"
    )


def _match_by_footage(stored, current, s_names, c_names) -> dict[int, int] | None:
    """Camera correspondence from each camera's footage basenames, or ``None``."""
    s_foot, c_foot = stored.get("footage") or {}, current.get("footage") or {}
    if not s_foot or not c_foot:
        return None

    # basename tuple -> index, on each side. A repeated tuple makes the match ambiguous, and
    # an ambiguous match is no match.
    def _index(names, footage):
        out: dict[tuple, int] = {}
        for i, name in enumerate(names):
            key = tuple(_footage_basenames({name: footage.get(name)}).get(name) or ())
            if not key or key in out:
                return None
            out[key] = i
        return out

    s_index, c_index = _index(s_names, s_foot), _index(c_names, c_foot)
    if s_index is None or c_index is None:
        return None
    mapping = {i: c_index[key] for key, i in s_index.items() if key in c_index}
    return mapping if len(mapping) == len(s_names) else None


def _remap_views(index: np.ndarray, remap: dict[int, int]) -> np.ndarray:
    """Rewrite the view column of COO rows. A view with no home is dropped, loudly.

    Dropping cannot happen through :func:`_camera_correspondence`, which refuses a partial
    correspondence -- this is the belt-and-braces path, so a future caller that hands over an
    incomplete map loses rows visibly rather than mis-attributing them.
    """
    if not index.size:
        return index
    out = np.array(index, copy=True)
    keep = np.zeros(len(out), dtype=bool)
    for i, row in enumerate(out):
        j = remap.get(int(row[0]))
        if j is None:
            continue
        out[i, 0] = j
        keep[i] = True
    if not keep.all():
        log.warning(
            "dropped %d label row(s) whose camera has no place in this result",
            int((~keep).sum()),
        )
    return out[keep]


def _check_per_camera(
    stored: dict, current: dict, remap: dict[int, int] | None, path: Path
) -> None:
    """Compare ``image_sizes`` and ``footage`` per camera, through the correspondence.

    Compared *through* the remap rather than as whole dicts, because a rename changes every
    key of both maps while changing nothing about the recording. A same-camera size mismatch
    is still fatal: ground truth is stored in footage pixels.
    """
    s_names = list(stored.get("camera_names") or [])
    c_names = list(current.get("camera_names") or [])
    pairs = (
        [(n, n) for n in s_names]
        if remap is None
        else [(s_names[i], c_names[j]) for i, j in remap.items()]
    )
    for key in ("image_sizes", "footage"):
        s, c = stored.get(key) or {}, current.get(key) or {}
        if not s or not c:
            continue
        for s_name, c_name in pairs:
            a, b = s.get(s_name), c.get(c_name)
            if a is None or b is None:
                continue
            if key == "footage":
                a = _footage_basenames({s_name: a}).get(s_name)
                b = _footage_basenames({c_name: b}).get(c_name)
            if a != b:
                raise ValueError(
                    f"{path} labels belong to a different recording ({key} differs for "
                    f"camera {s_name!r}/{c_name!r} -- e.g. a changed resolution/crop or "
                    "different footage); the stored GT pixels would be misinterpreted, so "
                    "they are refused"
                )


# -- persistence --------------------------------------------------------------


def _coo_gt(labels: Labels, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(index (N,4), xy (N,2))`` for the GT cells ``mask`` selects.

    The index is ``[view, frame, instance, point]``; ``instance`` is 0 throughout in this
    single-animal build (see :data:`_INDEX_WIDTH_V6`).
    """
    v, t, p = np.nonzero(mask)  # nonzero preserves axis order: view, frame, point
    return (
        np.stack([v, t, np.zeros_like(v), p], axis=1).astype(np.int32),
        labels.gt[mask].astype(np.float64),
    )


def _coo_seeds(labels: Labels) -> tuple[np.ndarray, np.ndarray]:
    """``(index (S,4), xy (S,2))`` for every cell whose instance has a seed."""
    mask = np.isfinite(labels.seeds).all(axis=-1)
    v, t, p = np.nonzero(mask)
    return (
        np.stack([v, t, np.zeros_like(v), p], axis=1).astype(np.int32),
        labels.seeds[mask].astype(np.float64),
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
    """``(M, 4)`` ``[view, frame, instance, point]`` for the cells ``mask`` selects."""
    v, t, p = np.nonzero(mask)
    return np.stack([v, t, np.zeros_like(v), p], axis=1).astype(np.int32)


def _read_cells(raw, path, what: str) -> np.ndarray:
    """A stored COO cell index, normalized to ``(N, 3)`` ``[view, frame, point]``.

    Accepts both widths: v6+ stores ``[view, frame, instance, point]``, v5 and earlier
    stored ``[view, frame, point]``. Rows for a non-zero instance are **dropped with a
    warning** -- this build has one animal per recording, and quietly folding a second
    animal's keypoints into the first would read as a labeling error rather than a version
    mismatch.
    """
    arr = np.asarray(raw, dtype=np.int64)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[1] == 3:
        return arr
    if arr.shape[1] != _INDEX_WIDTH_V6:
        raise ValueError(
            f"{path}: {what} index has {arr.shape[1]} columns; expected 3 "
            f"([view, frame, point]) or {_INDEX_WIDTH_V6} "
            "([view, frame, instance, point])"
        )
    keep = arr[:, 2] == 0
    n_dropped = int((~keep).sum())
    if n_dropped:
        log.warning(
            "%s: dropped %d %s row(s) belonging to instance != 0 -- this build tracks "
            "one animal per recording, and merging a second animal's points into the "
            "first would be worse than losing them",
            path,
            n_dropped,
            what,
        )
    return arr[keep][:, [0, 1, 3]]


def save_labels(
    path: str | Path,
    labels: Labels,
    *,
    identity: dict,
    subject_id: str | None = None,
    landmarks: "LandmarkLabels | None" = None,
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

    ``landmarks``, when given, writes the ``landmarks/`` group alongside -- the calibration
    observations, which share the file but not the point namespace (see
    :class:`LandmarkLabels`). Omitting it leaves any existing group *out* of the rewritten
    file, so a caller that loaded landmarks must pass them back; this is a whole-file
    rewrite, as the module docstring notes.
    """
    vetoed = labels._absent_bcast  # (1, 1, P) -> broadcasts over the cell arrays
    raw_gt = labels.gt_authored  # (V, T, P)
    live_gt_mask = raw_gt & ~vetoed
    void_gt_mask = raw_gt & vetoed
    live_occ_mask = labels.occluded & ~vetoed
    void_occ_mask = labels.occluded & vetoed

    gt_index, gt_xy = _coo_gt(labels, live_gt_mask)
    vgt_index, vgt_xy = _coo_gt(labels, void_gt_mask)
    # Seeds are not vetoed by absence: they are the instance's own geometry, not a claim
    # about the animal, and quarantining them would make un-declaring an amputation
    # silently re-seed the joint somewhere else.
    seed_index, seed_xy = _coo_seeds(labels)
    rev_index = np.nonzero(labels.reviewed)[0].astype(np.int32)  # (K,) frame indices
    inst_index = np.nonzero(labels.instance)[0].astype(np.int32)
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
        sd = f.create_group("seeds")
        sd.create_dataset("index", data=seed_index, dtype="int32")
        sd.create_dataset("xy", data=seed_xy, dtype="float64")
        o = f.create_group("occluded")
        o.create_dataset("index", data=_coo_cells(live_occ_mask), dtype="int32")
        r = f.create_group("reviewed")
        r.create_dataset("index", data=rev_index, dtype="int32")
        it = f.create_group("instance")
        it.create_dataset("index", data=inst_index, dtype="int32")
        a = f.create_group("absent")
        a.create_dataset("index", data=absent_index, dtype="int32")
        a.create_dataset("spans", data=absent_spans, dtype="int32")
        vg = a.create_group("void_gt")
        vg.create_dataset("index", data=vgt_index, dtype="int32")
        vg.create_dataset("xy", data=vgt_xy, dtype="float64")
        vo = a.create_group("void_occluded")
        vo.create_dataset("index", data=_coo_cells(void_occ_mask), dtype="int32")
        _write_landmarks(f, landmarks)
    labels.dirty = False
    if landmarks is not None:
        landmarks.dirty = False


def load_labels(path: str | Path, *, identity: dict) -> Labels | None:
    """Read a ``labels.h5`` sidecar into a dense :class:`Labels`, or ``None`` if absent.

    Validates the stored identity against ``identity`` (raising ``ValueError`` on a
    different recording/result) and normalizes the sparse lists: out-of-range or
    non-finite GT rows are dropped and duplicate ``(view, frame, point)`` keys resolve
    last-write-wins. A cell listed in **both** ``gt`` and ``occluded`` keeps both -- the
    two are independent axes, so there is no tie to break, and a reader that dropped one
    would silently re-supervise a pixel the operator withheld.

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
        gt_index = _read_cells(f["gt/index"][()], p, "gt")  # type: ignore[index]
        gt_xy = np.asarray(f["gt/xy"][()], dtype=float).reshape(-1, 2)  # type: ignore[index]
        # v5/v6 only: the legacy per-row provenance, read solely so the migration can
        # drop the invented placeholder-seed rows (see :data:`_LEGACY_PLACEHOLDER_SEED`).
        gt_prov = (
            np.asarray(f["gt/provenance"][()], dtype=np.uint8).reshape(-1)  # type: ignore[index]
            if "gt/provenance" in f
            else np.zeros(len(gt_xy), dtype=np.uint8)
        )
        occ_index = _read_cells(f["occluded/index"][()], p, "occluded")  # type: ignore[index]
        # v8: the instance seeds. A pre-v8 file has none -- its frames carry labels but no
        # instance, so the editor falls back to the detections as the solve's evidence
        # exactly as it did before, and the first edit in a frame seeds it.
        seed_index = np.empty((0, 3), dtype=np.int64)
        seed_xy = np.empty((0, 2), dtype=float)
        if "seeds" in f:
            seed_index = _read_cells(f["seeds/index"][()], p, "seeds")  # type: ignore[index]
            seed_xy = np.asarray(f["seeds/xy"][()], dtype=float).reshape(-1, 2)  # type: ignore[index]
        inst_index = (
            np.asarray(f["instance/index"][()], dtype=np.int64).reshape(-1)  # type: ignore[index]
            if "instance" in f
            else np.empty(0, dtype=np.int64)
        )
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
                vgt_index = _read_cells(  # type: ignore[index]
                    f["absent/void_gt/index"][()], p, "quarantined gt"
                )
                vgt_xy = np.asarray(  # type: ignore[index]
                    f["absent/void_gt/xy"][()], dtype=float
                ).reshape(-1, 2)
                vgt_prov = (
                    np.asarray(  # type: ignore[index]
                        f["absent/void_gt/provenance"][()], dtype=np.uint8
                    ).reshape(-1)
                    if "absent/void_gt/provenance" in f
                    else np.zeros(len(vgt_xy), dtype=np.uint8)
                )
            if "absent/void_occluded" in f:
                vocc_index = _read_cells(  # type: ignore[index]
                    f["absent/void_occluded/index"][()], p, "quarantined occlusion"
                )

    if stored_version > LABELS_FORMAT_VERSION:
        raise ValueError(
            f"{p} was written by a newer deeperfly (labels format v{stored_version}, "
            f"this build understands v{LABELS_FORMAT_VERSION}); refusing to read it "
            "rather than silently dropping state it carries"
        )
    view_remap = _check_identity(stored_identity, identity, p)

    # Quarantined rows rejoin the live ones -- they are authored state, just vetoed.
    if vgt_index.size:
        gt_index = np.concatenate([gt_index, vgt_index])
        gt_xy = np.concatenate([gt_xy, vgt_xy])
        gt_prov = np.concatenate([gt_prov, vgt_prov])
    if vocc_index.size:
        occ_index = np.concatenate([occ_index, vocc_index])

    # The camera axis moved (a reorder, or a rename bridged by footage). Rewrite the view
    # column of every stored row before it is materialized -- once, here, so no consumer
    # downstream has to know the axis ever differed.
    if view_remap is not None:
        gt_index = _remap_views(gt_index, view_remap)
        occ_index = _remap_views(occ_index, view_remap)
        seed_index = _remap_views(seed_index, view_remap)

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

    # v5/v6 -> v7: an invented placeholder-seed row was never a claim about the animal
    # (``export_gt`` dropped it unconditionally), and v7 has no provenance column to keep
    # telling it apart from a real label. Drop it at the boundary rather than promote it.
    if gt_index.size and (gt_prov == _LEGACY_PLACEHOLDER_SEED).any():
        legacy = gt_prov == _LEGACY_PLACEHOLDER_SEED
        log.warning(
            "%s: dropped %d editor-invented placeholder GT row(s) while migrating to "
            "labels format v%d (they were never exportable)",
            p,
            int(legacy.sum()),
            LABELS_FORMAT_VERSION,
        )
        gt_index, gt_xy = gt_index[~legacy], gt_xy[~legacy]

    # GT rows: keep in-range, finite, last-write-wins on duplicate keys.
    if gt_index.size:
        keep = _in_range(gt_index) & np.isfinite(gt_xy).all(axis=1)
        n_dropped = int((~keep).sum())
        if n_dropped:
            log.warning("%s: dropped %d out-of-range/NaN GT row(s)", p, n_dropped)
        for (v, t, pt), xy in zip(gt_index[keep], gt_xy[keep]):
            labels.gt[v, t, pt] = xy
    # Seed rows: the instance's starting geometry, kept wherever it is in range and finite.
    if seed_index.size:
        keep = _in_range(seed_index) & np.isfinite(seed_xy).all(axis=1)
        if int((~keep).sum()):
            log.warning(
                "%s: dropped %d out-of-range/NaN seed row(s)", p, int((~keep).sum())
            )
        for (v, t, pt), xy in zip(seed_index[keep], seed_xy[keep]):
            labels.seeds[v, t, pt] = xy
    # Hidden rows: keep every in-range one. There is no disjointness tie to break -- the
    # flag is its own axis (see Labels.set_occluded), so a cell that is both hand-placed and
    # held out of the loss is a legitimate, useful state rather than a corrupt one. Pre-v8
    # writers could not produce it; readers must not discard it.
    if occ_index.size:
        keep = _in_range(occ_index)
        for v, t, pt in occ_index[keep]:
            labels.occluded[v, t, pt] = True
    # Reviewed frames: keep in-range indices (a stale/oversized index is dropped).
    if rev_index.size:
        keep = (rev_index >= 0) & (rev_index < n_frames)
        labels.reviewed[rev_index[keep]] = True
    if inst_index.size:
        keep = (inst_index >= 0) & (inst_index < n_frames)
        labels.instance[inst_index[keep]] = True
    else:
        # Pre-v8, or a v8 file written before the flag: a frame with any seed had an
        # instance, which is exactly what the old derived test said.
        labels.instance |= np.isfinite(labels.seeds).all(axis=-1).any(axis=(0, 2))
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
    n_void = int((labels.gt_authored & labels._absent_bcast).sum())
    n_void_occ = int((labels.occluded & labels._absent_bcast).sum())
    if n_void or n_void_occ:
        log.info(
            "%s: %d GT row(s) and %d hidden mark(s) quarantined by the absent declaration",
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
      -> **GT**. ``fixed`` collapses into GT.
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
    labels: Labels,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense ground-truth arrays for training/eval.

    Returns ``(gt_xy (V,T,P,2), gt_mask (V,T,P) bool, hidden (V,T,P) bool)``. The GT
    is in **footage pixel space** -- the coordinate the operator clicked.

    The third mask is the **hidden** flag, and it is returned *beside* ``gt_mask`` rather
    than folded into it because the two answer different questions -- "is there a label
    here?" and "should it be trained on?". A trainer's supervision mask is their conjunction:

    .. code-block:: python

        gt_xy, gt_mask, hidden = export_gt(labels)
        supervise = gt_mask & ~hidden        # what the loss sees

    Pre-multiplying would lose the distinction the flag exists to record: a cell whose pixel
    is deliberately withheld would become indistinguishable from one nobody ever labeled,
    and re-including it later (the flag is a decision, and decisions get revised) would need
    the file re-read rather than one mask dropped.

    There is nothing to filter: a GT pixel is a pixel the operator created, and that is
    the only kind there is. A cell with *no* GT is not this function's business -- the
    consumer decides what to fall back to, in the editor's own precedence order (GT,
    else the detection, else the reprojection of the derived 3D), which needs the
    detections and the rig this function deliberately does not take.

    To train the 2D detector, transform ``gt_xy`` from footage space into each
    pathway's model-input space by inverting the pathway ``FrameTransform``
    (mirror/crop/resize). That plan lives in the run config, not ``results.h5``, so the
    transform is applied by the training pipeline -- this function is the neutral,
    footage-space contract it consumes.

    Points declared absent are excluded from **both** returned masks: an amputated
    keypoint is not ground truth (so it must not be supervised), and a hold-out mark on a
    cell that is already unsupervised is not a decision anyone made -- reporting it would
    read as seven deliberate hold-outs per amputated joint. Ask :func:`export_absent` for
    the declaration itself.
    """
    mask = labels.has_gt.copy()
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
