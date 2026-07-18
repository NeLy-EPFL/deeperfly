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

``gt`` and ``occluded`` are mutually exclusive. Everything else -- predictions,
projections, the 3D point, reprojection error, review progress -- is derived, so it
is never stored: no dense NaN arrays, no duplicate of the detector's output.

On disk (schema v1) the two deltas are stored **sparsely** (COO), which is tiny
next to ``results.h5`` and, unlike the old dense ``corrections.h5``, carries no copy
of the prediction NaN pattern:

.. code-block:: text

    attrs["meta"]   json {deeperfly_labels_format_version, created_utc, identity}
    gt/
        index       (N, 3) int32   [view, frame, point]
        xy          (N, 2) float64  affirmed 2D pixel (footage space)
        provenance  (N,)   uint8    1=dragged, 2=confirmed_prediction, 3=confirmed_projection
    occluded/
        index       (M, 3) int32   [view, frame, point]

In memory the overlay is kept **dense** per ``(V, T, P)`` (mirroring the result), so
resolving a frame's effective points on the interactive hot path stays ``O(V*P)``
and independent of how much has been labelled -- the sparse form is disk-only.

``identity`` fingerprints the *recording* these labels annotate (skeleton points,
camera names, frame count, image sizes, footage basenames) so a stale sidecar is
refused. It deliberately excludes the predictions and ``created_utc`` themselves, so
re-running detection/triangulation on the *same* recording keeps the labels valid --
ground truth is absolute, not relative to what the network happened to predict.
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
    "Provenance",
    "save_labels",
    "load_labels",
    "labels_identity",
    "migrate_from_corrections",
    "export_gt",
    "LABELS_FORMAT_VERSION",
]

log = logging.getLogger("deeperfly")

LABELS_FORMAT_VERSION = 1


class Provenance:
    """How a GT pixel was authored (stored per ``gt`` row so it can be filtered).

    ``dragged`` and ``confirmed_prediction`` are the operator affirming a pixel they
    looked at; ``confirmed_projection`` is a bulk-accepted triangulation guess (the
    network never fired there), so an export/solve can down-weight or drop it.
    """

    NONE = 0
    DRAGGED = 1
    CONFIRMED_PREDICTION = 2
    CONFIRMED_PROJECTION = 3


@dataclass
class Labels:
    """In-memory ground-truth overlay on a :class:`~deeperfly.results.PoseResult`.

    All arrays are dense ``(V, T, P)``-shaped (2D pixels carry a trailing 2). A GT
    pixel is present iff ``gt`` is finite there, which is exactly ``provenance != 0``;
    ``occluded`` marks views the operator flagged unusable. The invariants (``gt``
    and ``occluded`` disjoint; ``provenance`` set iff ``gt`` finite) are maintained by
    the mutators and re-checked on load. ``dirty`` tracks unsaved changes.
    """

    gt: Float[np.ndarray, "V T P 2"]
    gt_provenance: np.ndarray  # (V, T, P) uint8, a Provenance value
    occluded: Bool[np.ndarray, "V T P"]
    dirty: bool = field(default=False)

    @classmethod
    def empty(cls, n_views: int, n_frames: int, n_points: int) -> Labels:
        """An overlay with no GT and nothing occluded for a ``(V, T, P)`` result."""
        return cls(
            gt=np.full((n_views, n_frames, n_points, 2), np.nan),
            gt_provenance=np.zeros((n_views, n_frames, n_points), dtype=np.uint8),
            occluded=np.zeros((n_views, n_frames, n_points), dtype=bool),
        )

    # -- derived masks --------------------------------------------------------

    @property
    def has_gt(self) -> Bool[np.ndarray, "V T P"]:
        """Per-``(view, frame, point)`` whether a GT pixel is present."""
        return np.isfinite(self.gt).all(axis=-1)

    @property
    def any_labels(self) -> bool:
        """Whether any GT pixel or occlusion has been authored."""
        return bool(self.has_gt.any() or self.occluded.any())

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
        """Reset every label in ``frame`` to ``unset``."""
        self.gt[:, frame] = np.nan
        self.gt_provenance[:, frame] = Provenance.NONE
        self.occluded[:, frame] = False
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


def save_labels(path: str | Path, labels: Labels, *, identity: dict) -> None:
    """Write ``labels`` to a sparse ``labels.h5`` sidecar (overwriting ``path``).

    Only authored GT/occluded entries are written (COO), stamped with ``identity``.
    Clears ``labels.dirty`` on success. ``results.h5`` is never touched.
    """
    has_gt = labels.has_gt  # (V, T, P)
    gv, gf, gp = np.nonzero(has_gt)  # nonzero preserves axis order: view, frame, point
    gt_index = np.stack([gv, gf, gp], axis=1).astype(
        np.int32
    )  # (N, 3) [view, frame, point]
    gt_xy = labels.gt[has_gt].astype(np.float64)  # (N, 2)
    gt_prov = labels.gt_provenance[has_gt].astype(np.uint8)  # (N,)
    ov, of, op = np.nonzero(labels.occluded)
    occ_index = np.stack([ov, of, op], axis=1).astype(np.int32)  # (M, 3)

    meta = {
        "deeperfly_labels_format_version": LABELS_FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
    }
    with h5py.File(path, "w") as f:
        f.attrs["meta"] = json.dumps(meta)
        g = f.create_group("gt")
        g.create_dataset("index", data=gt_index, dtype="int32")
        g.create_dataset("xy", data=gt_xy, dtype="float64")
        g.create_dataset("provenance", data=gt_prov, dtype="uint8")
        o = f.create_group("occluded")
        o.create_dataset("index", data=occ_index, dtype="int32")
    labels.dirty = False


def load_labels(path: str | Path, *, identity: dict) -> Labels | None:
    """Read a ``labels.h5`` sidecar into a dense :class:`Labels`, or ``None`` if absent.

    Validates the stored identity against ``identity`` (raising ``ValueError`` on a
    different recording/result) and normalises the sparse lists: out-of-range or
    non-finite GT rows are dropped, duplicate ``(view, frame, point)`` keys resolve
    last-write-wins, and any ``(view, frame, point)`` present in both ``gt`` and
    ``occluded`` keeps the GT (which carries an authored pixel) and drops the
    occlusion -- so the on-disk invariants cannot desync the in-memory overlay.
    """
    p = Path(path)
    if not p.exists():
        return None
    with h5py.File(p, "r") as f:
        meta = json.loads(f.attrs["meta"])  # type: ignore[arg-type]
        stored_identity = meta.get("identity", {})
        gt_index = np.asarray(f["gt/index"][()], dtype=np.int64).reshape(-1, 3)  # type: ignore[index]
        gt_xy = np.asarray(f["gt/xy"][()], dtype=float).reshape(-1, 2)  # type: ignore[index]
        gt_prov = np.asarray(f["gt/provenance"][()], dtype=np.uint8).reshape(-1)  # type: ignore[index]
        occ_index = np.asarray(f["occluded/index"][()], dtype=np.int64).reshape(-1, 3)  # type: ignore[index]

    _check_identity(stored_identity, identity, p)

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
    # Occluded rows: keep in-range, but GT wins the disjointness tie.
    if occ_index.size:
        keep = _in_range(occ_index)
        for v, t, pt in occ_index[keep]:
            if labels.has_gt[v, t, pt]:
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
      real human occlusion. Dropped by default (re-derives as ``absent``); set
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
    keeps it.

    To train the 2D detector, transform ``gt_xy`` from footage space into each
    pathway's model-input space by inverting the pathway ``FrameTransform``
    (mirror/crop/resize). That plan lives in the run config, not ``results.h5``, so the
    transform is applied by the training pipeline -- this function is the neutral,
    footage-space contract it consumes.
    """
    mask = labels.has_gt.copy()
    if not include_projection:
        mask &= labels.gt_provenance != Provenance.CONFIRMED_PROJECTION
    gt_xy = np.where(mask[..., None], labels.gt, np.nan)
    return gt_xy, mask, labels.occluded.copy()
