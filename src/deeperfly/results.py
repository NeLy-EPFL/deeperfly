"""Self-contained HDF5 result container for the pose pipeline.

``results.h5`` (schema v3) stores each pipeline stage's output in its own group,
so a stage never overwrites another stage's data and any downstream stage can
be re-run later from pristine upstream outputs:

.. code-block:: text

    attrs["meta"]            json: {deeperfly_format_version: 3, created_utc, ...}
    skeleton/                the skeleton (point names, bones, symmetries, colors)
    pose2d/
        points               (V, T, P, 2) arg-max 2D detections (visibility-masked)
        conf                 (V, T, P) detection confidences
        cameras/             the config rig as built at detect time
        attrs["image_sizes"] json {camera_name: [h, w]} of the raw footage frames
        candidates/          top-K peaks (xy, score) -- present iff the
                             pictorial_structures stage was enabled at detect time
    bundle_adjustment/
        cameras/             the BA-refined rig
    pictorial_structures/
        points               (V, T, P, 2) PS-corrected 2D
        points3d             (T, P, 3) initial 3D estimate
        reproj_error         (V, T, P)
    triangulation/
        points               (V, T, P, 2) cleaned 2D (outlier-rejecting methods)
        points3d             (T, P, 3)
        reproj_error         (V, T, P)
    eks/
        points3d             (T, P, 3) the smoothed 3D
        posterior_var        (T, P, 3) the smoother's per-axis posterior variance
        smooth_param         (P,) the fitted per-keypoint process-noise scale
    postprocess/
        points3d             (T, P, 3) 3D after the correction chain
        points2d_override    (V, R, 2) the 2D the ops froze in pixel space
        points2d_override_cols  (R,) which skeleton columns those are
    inverse_kinematics/
        angles               (T, D) fitted joint angles (radians)
        angle_names          (D,) the angle names, in column order
        points3d             (T, P, 3) fitted model joints (world; skeleton order)
        attrs["template"]    the template name; attrs["alignment"] the body frame

:class:`StageStore` is the per-stage read/write access used by the staged run;
:class:`PoseResult` is the assembled in-memory view (the *best* points present:
postprocess over eks over triangulation over pictorial over pose2d, BA
cameras over the config rig). The
HDF5 file fully reconstructs the cameras and skeleton, so results are portable
without the original config files.

Arrays use the view-leading layout: ``pts2d`` is ``(V, T, P, 2)``, ``conf`` is
``(V, T, P)``, ``pts3d`` is ``(T, P, 3)``. NaN encodes missing observations /
un-triangulated points and is preserved by the float32 datasets.

**What a stage stores, and what it reconstructs (v3).** Every stage used to keep a
full ``(V, T, P, 2)`` 2D array and a full ``(V, T, P)`` error, which on an eight-view
recording is 14.6 MB per stage before any of them says anything new. A stage now stores
only what cannot be rebuilt from what its neighbors already store, decided per write by
:func:`_reduce_pts2d` rather than hardcoded per stage -- so a new stage, or an op that
starts moving pixels per frame, gets the right answer without editing this module:

* A 2D that *is* ``cameras.project(points3d)`` is not stored at all (the smoother's
  is, exactly). ``attrs["points2d_storage"] = "derived"``.
* A 2D that is that projection except on a few columns held constant over time is
  stored as just those constants -- the correction chain's frozen thorax-coxae come
  to 112 numbers instead of 9.8 MB. ``attrs["points2d_storage"] = "override"``.
* A 2D that is an independent pixel measurement is stored whole: the detections, the
  pictorial candidate selection, and triangulation's outlier-cleaned observations.
  ``attrs["points2d_storage"] = "full"``.

``reproj_error`` is dropped only when a recomputation reproduces it *and* the stage's
2D was not stored whole. The second half is not an optimization but a safeguard: the
stored error is the only witness that an outside tool overwrote a stage's 2D with
something other than the detections (see :func:`deeperfly.acquisition.stored_vs_pose2d`),
and a recomputed error agrees with the stored 3D by construction, so it can never
disagree with itself. Deriving it would silently retire that check.

Reading is version-agnostic: every reader prefers a stored array and falls back to
reconstruction, so a v2 file -- which stores everything -- needs no special case.
:data:`READABLE_VERSIONS` is what :meth:`PoseResult.load` accepts; ``deeperfly repack``
rewrites an older file in place (see :func:`repack`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
from jaxtyping import Bool, Float

from .cameras import CameraGroup
from .config import STAGES
from .skeleton import Skeleton

if TYPE_CHECKING:
    from .pictorial import Candidates

__all__ = ["PoseResult", "StageStore", "repack"]

FORMAT_VERSION = 3

#: Schema versions this build can *read*. v3 dropped only datasets that are exactly
#: reconstructible from the ones it kept, so the readers here -- which prefer a stored
#: array and reconstruct only what is missing -- serve both without a version branch.
#: Writing is always the current version; :func:`repack` converts.
READABLE_VERSIONS = (2, FORMAT_VERSION)

_STR = h5py.string_dtype("utf-8")

#: Point arrays are stored as float32 and deflated. They hold pixel coordinates decoded
#: from a heat-map arg-max and the millimeter 3D fitted from them; float32 resolves those
#: to ~6e-05 px against the ~1 px the detector can actually localize, so the eight extra
#: digits float64 carries are noise. Incompressible noise, at that -- which is why
#: narrowing the dtype saves twice what deflating alone does (2.00x vs 1.32x measured),
#: and why the two together beat either.
_STORE_DTYPE = np.float32

#: The deflate filter, and the one knob here with a real cost. Measured on an 8-view
#: 2007-frame recording (66 MB as v2), whole-file size against the wall clock of reading a
#: whole array back:
#:
#: ===========  =======  ==============  ==================
#: filter       size     read_pose2d     PoseResult.load
#: ===========  =======  ==============  ==================
#: v2, none     65.9 MB     2.4 ms           7.2 ms
#: none         19.4 MB     5.1 ms          28.2 ms
#: lzf          18.4 MB    14.2 ms          42.2 ms
#: gzip-1       15.7 MB    33.6 ms          65.0 ms
#: gzip-4       15.6 MB    32.7 ms          64.1 ms
#: ===========  =======  ==============  ==================
#:
#: Three things that table settles. Level 4 is free relative to level 1 (same size, same
#: time), so there is no reason to run the cheap setting. Most of the win -- 3.4x of 4.2x
#: -- is the reduction and the dtype, which cost almost nothing to read; deflate buys the
#: last 24% for a 6x read slowdown. And ``load``'s 28 ms floor with no filter at all is
#: the reprojection that rebuilds the dropped 2D, not decompression.
#:
#: Deflate stays on because 64 ms to open a recording is nothing next to the 157-252 ms a
#: single video frame costs to decode, and nothing reads these arrays in a hot loop. If
#: something ever does, ``None`` here is the setting to reach for -- it keeps 3.4x.
_COMPRESSION = "gzip"
_COMPRESSION_OPTS = 4

#: Below this many elements a filter costs more in chunk bookkeeping than it saves, and
#: the threshold is also what keeps the policy away from the small float64 arrays whose
#: precision is load-bearing: a rig's rvecs/tvecs/intrs and the skeleton.
_FILTER_MIN_SIZE = 4096

#: How far a rebuilt array may sit from the stored one and still count as the same array.
#: Both sides are computed by the same projection code from the same float64 inputs, so
#: the honest gap is zero; this is slack against a future rig whose projection is not
#: bit-reproducible, and it is still ~4 orders below the float32 storage step.
_DERIVE_ATOL = 1e-9


def _put(g: h5py.Group, name: str, arr, *, dtype=_STORE_DTYPE) -> h5py.Dataset:
    """Create ``g[name]`` under the storage policy: float32 and deflated when it is big.

    Small arrays are written verbatim, which is what keeps camera and skeleton data in
    float64 without this function needing to know what a camera is.
    """
    a = np.asarray(arr)
    if dtype is not None and a.dtype == np.float64 and a.size >= _FILTER_MIN_SIZE:
        a = a.astype(dtype)
    kw = (
        dict(chunks=True, compression=_COMPRESSION, compression_opts=_COMPRESSION_OPTS)
        if a.size >= _FILTER_MIN_SIZE
        else {}
    )
    return g.create_dataset(name, data=a, **kw)


def _project(cameras: CameraGroup, pts3d) -> np.ndarray:
    """``cameras.project(pts3d)`` as a writable float64 array.

    ``np.array`` and not ``np.asarray``: the projection comes back from jax read-only,
    and the override path assigns into it.
    """
    return np.array(cameras.project(np.asarray(pts3d, dtype=float)), dtype=float)


def _same(a, b, *, atol: float = _DERIVE_ATOL) -> bool:
    """Whether two arrays match to ``atol``, counting NaN as equal to NaN only.

    ``np.allclose`` cannot express that: its ``equal_nan`` makes NaN equal to NaN but
    it has no way to *require* the NaN patterns to agree, and here they carry meaning --
    a NaN is "this view did not see it", which is not interchangeable with a number.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False
    na, nb = np.isnan(a), np.isnan(b)
    if not np.array_equal(na, nb):
        return False
    m = ~na
    return bool(m.sum() == 0 or np.allclose(a[m], b[m], rtol=0, atol=atol))


def _constant_over_time(x: np.ndarray) -> bool:
    """Whether ``x`` ``(V, T, ...)`` is the same at every frame (NaN counting as equal)."""
    return x.shape[1] == 0 or _same(x, np.broadcast_to(x[:, :1], x.shape), atol=0.0)


def _reduce_pts2d(pts2d, pts3d, cameras: CameraGroup | None):
    """How much of ``pts2d`` must be stored, given that ``pts3d`` and the rig are.

    Returns ``(kind, payload)``:

    ``("derived", None)``
        ``pts2d`` *is* ``project(pts3d)``; nothing needs storing.
    ``("override", (cols, values))``
        it is that projection except on ``cols``, where it is constant over time --
        a frozen pixel measurement. Only the ``(V, len(cols), 2)`` constants are stored.
    ``("full", pts2d)``
        it is an independent estimate of its own; all of it is stored.

    Measured rather than declared per stage, so the classification cannot drift from
    what the stages actually produce. A column that differs in only *some* frames is
    not an override -- it is per-frame information, and falls through to ``full``.
    """
    if pts2d is None:
        return "absent", None
    pts2d = np.asarray(pts2d, dtype=float)
    if pts3d is None or cameras is None:
        return "full", pts2d
    proj = _project(cameras, pts3d)
    if proj.shape != pts2d.shape:
        return "full", pts2d
    if _same(proj, pts2d):
        return "derived", None
    finite = ~(np.isnan(proj) | np.isnan(pts2d))
    differs = ~(np.isclose(proj, pts2d, rtol=0, atol=_DERIVE_ATOL) | ~finite).all(-1)
    cols = np.flatnonzero(differs.any(axis=(0, 1)))
    if cols.size and cols.size < pts2d.shape[2]:
        rest = np.ones(pts2d.shape[2], dtype=bool)
        rest[cols] = False
        block = pts2d[:, :, cols, :]
        if _same(proj[:, :, rest], pts2d[:, :, rest]) and _constant_over_time(block):
            return "override", (cols.astype(np.int32), block[:, 0])
    return "full", pts2d


def _rebuild_pts2d(g: h5py.Group, cameras: CameraGroup | None) -> np.ndarray | None:
    """A points group's 2D: the stored array, or the one v3 left to be reconstructed.

    ``None`` when neither is available -- an unfinished group, or a derived 2D in a file
    whose rig is missing, which is the one case where the reconstruction cannot be done
    and a caller must be told rather than handed a guess.
    """
    if "points" in g:
        return np.asarray(g["points"][()], dtype=float)
    if "points3d" not in g or cameras is None:
        return None
    out = _project(cameras, g["points3d"][()])
    if "points2d_override" in g:
        cols = np.asarray(g["points2d_override_cols"][()], dtype=int)
        vals = np.asarray(g["points2d_override"][()], dtype=float)
        out[:, :, cols, :] = vals[:, None]
    return out


def _rebuild_reproj_error(
    g: h5py.Group, obs2d, cameras: CameraGroup | None
) -> np.ndarray | None:
    """A points group's reprojection error: stored, else recomputed against ``obs2d``.

    ``obs2d`` is the ``pose2d`` detections, which is what every stage that lets its error
    be recomputed measured against (the stages that measured against their own cleaned 2D
    store it, so they never reach this path).
    """
    if "reproj_error" in g:
        return np.asarray(g["reproj_error"][()], dtype=float)
    if "points3d" not in g or cameras is None or obs2d is None:
        return None
    proj = _project(cameras, g["points3d"][()])
    obs = np.asarray(obs2d, dtype=float)
    if proj.shape != obs.shape:
        return None
    return np.linalg.norm(proj - obs, axis=-1)


def _cameras_from(f: h5py.File) -> CameraGroup | None:
    """The rig a reconstruction should use: BA-refined when present, else the config rig.

    The same preference :meth:`PoseResult.load` applies, and it has to be, or a rebuilt
    2D would come off a different rig than the stage that produced it used.
    """
    for group in ("bundle_adjustment/cameras", "pose2d/cameras"):
        if group in f:
            return _read_cameras(f[group])  # type: ignore[arg-type]
    return None


def _keep_reproj_error(reproj_error, pts3d, obs2d, cameras, *, kind: str) -> bool:
    """Whether a stage's reprojection error has to be stored rather than recomputed.

    Two reasons to keep it, and only the first is about bytes:

    1. A recomputation would not reproduce it (no rig, no observations, or a stage that
       measured against something other than the detections). Nothing is derivable here.
    2. ``kind == "full"`` -- the stage's 2D is stored whole, which is exactly the 2D an
       outside tool can overwrite. The stored error is then the only witness to that
       substitution, because a recomputed one agrees with the stored 3D by construction.
       See :func:`deeperfly.acquisition.stored_vs_pose2d`, which reads it for that.
    """
    if reproj_error is None:
        return False
    if kind == "full":
        return True
    if pts3d is None or cameras is None or obs2d is None:
        return True
    proj = _project(cameras, pts3d)
    obs = np.asarray(obs2d, dtype=float)
    if proj.shape != obs.shape:
        return True
    return not _same(reproj_error, np.linalg.norm(proj - obs, axis=-1))


def _write_points_group(
    g: h5py.Group,
    *,
    pts2d,
    pts3d,
    reproj_error,
    extra: dict | None = None,
    meta: dict | None = None,
    cameras: CameraGroup | None = None,
    obs2d=None,
) -> str:
    """Fill a freshly created points group, storing only what cannot be rebuilt.

    Shared by :meth:`StageStore.write_points` and :func:`repack` so the two cannot
    disagree about what a v3 group contains. Returns the ``points2d_storage`` kind.
    """
    kind, payload = _reduce_pts2d(pts2d, pts3d, cameras)
    if kind == "full":
        _put(g, "points", payload)
    elif kind == "override":
        cols, values = payload
        _put(g, "points2d_override", values)
        g.create_dataset("points2d_override_cols", data=cols)
    if pts3d is not None:
        _put(g, "points3d", pts3d)
    if _keep_reproj_error(reproj_error, pts3d, obs2d, cameras, kind=kind):
        _put(g, "reproj_error", reproj_error)
    for name, arr in sorted((extra or {}).items()):
        if arr is not None:
            _put(g, name, arr)
    g.attrs["points2d_storage"] = kind
    if meta:
        g.attrs["meta"] = json.dumps(meta, default=str)
    return kind


def _is_newer(version) -> bool:
    """Whether a stored ``deeperfly_format_version`` is from a *later* build than this one.

    Tolerant of junk: an unparseable version is not "newer", it is unrecognized, and the
    caller's older-file path (regenerate) is the right answer for it.
    """
    try:
        return int(version) > FORMAT_VERSION
    except (TypeError, ValueError):
        return False


def _version_in(f) -> int | None:
    """An open result file's schema version, or ``None`` when absent/unparseable."""
    try:
        return int(
            json.loads(f.attrs.get("meta", "{}")).get("deeperfly_format_version")
        )
    except (TypeError, ValueError):
        return None


def stored_version(path: str | Path) -> int | None:
    """The schema version recorded in a result file, or ``None``.

    ``None`` for a missing, unreadable, or non-deeperfly file -- every caller here treats
    those the same way it treats an old one, so they do not need telling apart.
    """
    if not Path(path).exists():
        return None
    try:
        with h5py.File(path, "r") as f:
            return _version_in(f)
    except OSError:
        return None


def _newer_message(path, version) -> str:
    """The refusal every deeperfly artifact shares, so it reads the same wherever it lands."""
    return (
        f"{path} was written by a newer deeperfly (result format v{version}, this build "
        f"understands v{FORMAT_VERSION}); refusing to read it rather than silently "
        "dropping state it carries"
    )


#: The dataset whose presence means a stage's output is complete, per stage. Every 3D
#: stage is marked by its ``points3d`` rather than its ``points``, because whether the
#: 2D is stored is now a property of the *data* (v3 drops a 2D its 3D reprojects to) --
#: marking a stage by an array the writer is entitled to omit would make a completed
#: stage look unfinished and recompute forever.
_STAGE_MARKER = {
    "pose2d": "pose2d/points",
    "bundle_adjustment": "bundle_adjustment/cameras",
    "pictorial_structures": "pictorial_structures/points3d",
    "triangulation": "triangulation/points3d",
    "eks": "eks/points3d",
    "postprocess": "postprocess/points3d",
    "inverse_kinematics": "inverse_kinematics/angles",
}


def _write_animal(f, *, absent, subject_id) -> None:
    """Write the ``animal/`` group (which keypoints are not on this specimen).

    Skipped entirely when there is nothing to record, so an ordinary run's file is
    byte-identical to before this existed.
    """
    if absent is None and subject_id is None:
        return
    a = np.asarray(absent, dtype=bool) if absent is not None else None
    if (a is None or not a.any()) and subject_id is None:
        return
    g = f.create_group("animal")
    if a is not None:
        g.create_dataset("absent", data=a)
    if subject_id is not None:
        g.attrs["subject_id"] = str(subject_id)


def _read_animal(f) -> "tuple[np.ndarray | None, str | None]":
    """``(absent (P,) | None, subject_id | None)`` from the optional ``animal/`` group."""
    if "animal" not in f:
        return None, None
    g = f["animal"]
    absent = np.asarray(g["absent"][()], dtype=bool) if "absent" in g else None
    subject = g.attrs.get("subject_id")
    return absent, (str(subject) if subject is not None else None)


@dataclass
class PoseResult:
    """A complete multi-view pose-estimation result for one recording.

    ``cameras`` is ``None`` only for an **uncalibrated** result -- a recording whose rig
    has never been solved, held open in the editor so its 2D can be labeled (see
    :meth:`uncalibrated`). Nothing the pipeline writes or :meth:`load` reads is ever
    camera-less: a stored file always carries a rig. Every geometric consumer is already
    gated on ``pts3d is not None``, which an uncalibrated result also leaves ``None``, so
    the two states travel together -- but check :attr:`has_cameras` rather than assuming.
    """

    cameras: CameraGroup | None
    skeleton: Skeleton
    pts2d: Float[np.ndarray, "V T P 2"]
    conf: Float[np.ndarray, "V T P"] | None = None
    pts3d: Float[np.ndarray, "T P 3"] | None = None
    reproj_error: Float[np.ndarray, "V T P"] | None = None
    nmf_pts3d: Float[np.ndarray, "T P 3"] | None = None
    nmf_angles: Float[np.ndarray, "T D"] | None = None
    nmf_angle_names: list[str] | None = None
    nmf_chain_scales: dict[str, float] = field(default_factory=dict)
    #: ``chain name -> (3,)`` model-unit shift putting a chain's base where the
    #: recording's own base landmark was measured (the head's ``neck``). The mesh
    #: overlay needs it for the same reason it needs ``nmf_chain_scales``: the solved
    #: plan baked the shift into the angles, so drawing without it puts the head on a
    #: different pivot than the one it was fitted about. Empty for a file written before
    #: chains had base landmarks, which is correctly no shift.
    nmf_chain_offsets: dict[str, np.ndarray] = field(default_factory=dict)
    nmf_body_scale: float = 1.0
    #: The body plan the fit was solved on, as JSON, when the file recorded one. Lets
    #: the editor's live re-fit run on exactly the pipeline's geometry instead of
    #: re-deriving it. ``None`` for a file written before plans were stored -- every
    #: consumer must cope, since the stored fit and its overlays do not need it.
    nmf_body_plan: str | None = None
    #: ``(T, P)`` which skeleton keypoints are **not on this animal** -- an amputated leg,
    #: an ablated antenna -- per frame, since a limb can be lost part-way through a
    #: recording. Columns align to ``skeleton.point_names``. ``None`` (the common case)
    #: means nothing is declared absent; consumers must treat that as all-False. Persisted
    #: in a top-level ``animal/`` group, which is deliberately *not* a pipeline stage: it
    #: is an operator-authored fact about the specimen, so no stage recompute discards it.
    absent: Bool[np.ndarray, "T P"] | None = None
    #: Optional animal identifier, so one specimen's several recordings can be grouped.
    subject_id: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def nmf_head_scale(self) -> float:
        """Data-estimated overlay head size (1.0 if unknown); see ``nmf_chain_scales``."""
        return float(self.nmf_chain_scales.get("head", 1.0))

    @property
    def nmf_abdomen_scale(self) -> float:
        """Data-estimated overlay abdomen size (1.0 if unknown)."""
        return float(self.nmf_chain_scales.get("abdomen", 1.0))

    def __post_init__(self) -> None:
        self.pts2d = np.asarray(self.pts2d, dtype=float)
        for name in ("conf", "pts3d", "reproj_error", "nmf_pts3d"):
            arr = getattr(self, name)
            if arr is not None:
                setattr(self, name, np.asarray(arr, dtype=float))

    @property
    def n_views(self) -> int:
        return self.pts2d.shape[0]

    @property
    def n_frames(self) -> int:
        return self.pts2d.shape[1]

    @property
    def has_cameras(self) -> bool:
        """Whether this result carries a camera rig (false = uncalibrated)."""
        return self.cameras is not None

    @classmethod
    def uncalibrated(
        cls,
        skeleton: Skeleton,
        *,
        n_views: int,
        n_frames: int,
        view_names: list[str] | None = None,
    ) -> PoseResult:
        """An empty result for a recording with no rig and no detections.

        This is the from-scratch starting point: footage exists, nothing has been
        detected, no camera has been calibrated, and the operator is about to label 2D by
        hand. Every observation is ``NaN`` and there is no 3D, so the editor's derived-3D
        machinery stays inert and each view is an independent 2D canvas.

        Deliberately **not** persisted: it is a scaffold for an editing session, and
        writing it would create a ``results.h5`` claiming a pipeline ran. The labels the
        operator authors go to ``labels.h5``, which is the durable artifact.

        Parameters
        ----------
        skeleton
            The project's skeleton (fixes the point axis).
        n_views, n_frames
            The rig's view count and the footage's frame count.
        view_names
            Camera names, for display. Defaults to ``view0 ... viewN``.

        Returns
        -------
        PoseResult
            An all-NaN, camera-less result.
        """
        n_points = len(skeleton.point_names)
        return cls(
            cameras=None,
            skeleton=skeleton,
            pts2d=np.full((int(n_views), int(n_frames), n_points, 2), np.nan),
            meta={
                "uncalibrated": True,
                "view_names": list(view_names)
                if view_names
                else [f"view{i}" for i in range(int(n_views))],
            },
        )

    # -- serialization -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the result to an HDF5 file (overwriting ``path``).

        The library one-shot: ``pts2d``/``conf`` go to ``pose2d/`` and, when a 3D
        pose is present, the (possibly cleaned) 2D, 3D and reprojection error go
        to ``triangulation/`` -- so :meth:`load` round-trips the assembled view.
        ``pts2d`` is duplicated into both groups in that case (it is small next
        to the footage).

        ``absent`` (keypoints not on this animal) is written to a top-level ``animal/``
        group. This is a whole-file rewrite, so the group must be written here or a plain
        load -> mutate -> save round trip would silently drop the declaration.

        Parameters
        ----------
        path
            Destination ``.h5`` path; an existing file is overwritten.
        """
        meta = {
            "deeperfly_format_version": FORMAT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            **self.meta,
        }
        if self.cameras is None:
            raise ValueError(
                "this is an uncalibrated result (no camera rig) and cannot be saved: it "
                "is an editing scaffold, and a results.h5 without cameras would claim a "
                "pipeline ran. The operator's labels persist in labels.h5 instead"
            )
        with h5py.File(path, "w") as f:
            f.attrs["meta"] = json.dumps(meta)
            _write_skeleton(f.create_group("skeleton"), self.skeleton)
            g2d = f.create_group("pose2d")
            _put(g2d, "points", self.pts2d)
            if self.conf is not None:
                _put(g2d, "conf", self.conf)
            _write_cameras(g2d.create_group("cameras"), self.cameras)
            if self.pts3d is not None or self.reproj_error is not None:
                # ``points`` stays whole here rather than going through the v3 reduction:
                # an assembled result's 2D is whatever stage produced it, so it is an
                # independent measurement as far as this file can tell, and the one-shot
                # writer has no upstream group to reconstruct it from.
                g3d = f.create_group("triangulation")
                _put(g3d, "points", self.pts2d)
                g3d.attrs["points2d_storage"] = "full"
                if self.pts3d is not None:
                    _put(g3d, "points3d", self.pts3d)
                if self.reproj_error is not None:
                    _put(g3d, "reproj_error", self.reproj_error)
            _write_animal(f, absent=self.absent, subject_id=self.subject_id)

    @classmethod
    def load(cls, path: str | Path) -> PoseResult:
        """Read the assembled :class:`PoseResult` back from an HDF5 file.

        Assembly prefers the most-derived data present: ``pts2d`` from
        postprocess, else eks, else triangulation, else pictorial_structures,
        else pose2d; ``pts3d`` / ``reproj_error`` from the same order (minus
        pose2d, which has no 3D); cameras from bundle_adjustment, else the pose2d
        config rig.

        Reads every version in :data:`READABLE_VERSIONS`. A v2 file stores every array,
        so it simply never takes the reconstruction path -- which is why this needs no
        version branch, and why the 2D and error a v2 and a repacked v3 copy of the same
        recording hand back agree to the float32 storage step.

        Arrays come back as float64 whatever they were stored as, so a caller cannot
        acquire a float32 pose by reading a newer file.

        Parameters
        ----------
        path
            Path to a ``.h5`` file written by :meth:`save` or the staged run.

        Returns
        -------
        PoseResult
            The assembled result (cameras, skeleton, points and ``meta``).
        """
        with h5py.File(path, "r") as f:
            # ``.get``, not ``[...]``: a file that is not a deeperfly result at all must
            # produce this function's own diagnosis, not a raw KeyError on 'meta'.
            meta = json.loads(f.attrs.get("meta", "{}"))  # type: ignore[arg-type]
            version = meta.pop("deeperfly_format_version", None)
            if _is_newer(version):
                raise ValueError(_newer_message(path, version))
            if version not in READABLE_VERSIONS:
                raise ValueError(
                    f"{path} has deeperfly format version {version!r}, expected one of "
                    f"{', '.join(str(v) for v in READABLE_VERSIONS)}; re-run the "
                    "pipeline to regenerate it"
                )
            skeleton = _read_skeleton(f["skeleton"])  # type: ignore[arg-type]
            absent, subject_id = _read_animal(f)
            cameras_group = (
                f["bundle_adjustment/cameras"]
                if "bundle_adjustment/cameras" in f
                else f["pose2d/cameras"]
            )
            cameras = _read_cameras(cameras_group)  # type: ignore[arg-type]
            pts2d = pts3d = reproj = None
            obs2d = (
                np.asarray(f["pose2d/points"][()], dtype=float)  # type: ignore[index]
                if "pose2d/points" in f
                else None
            )
            # Most-derived first: the correction chain supersedes the smoother's
            # output, which supersedes the triangulation it was seeded from, which
            # supersedes the raw detections. A stage counts as having a 2D when one can
            # be *rebuilt*, not only when one is stored -- v3 stores neither the
            # smoother's nor (all of) the chain's, and skipping past them here would
            # quietly hand back a less-derived pose than the file holds.
            for stage in (
                "postprocess",
                "eks",
                "triangulation",
                "pictorial_structures",
                "pose2d",
            ):
                if stage not in f:
                    continue
                g = f[stage]
                if pts2d is None:
                    pts2d = _rebuild_pts2d(g, cameras)  # type: ignore[arg-type]
                if pts3d is None and "points3d" in g:
                    pts3d = np.asarray(g["points3d"][()], dtype=float)  # type: ignore[index]
                if reproj is None:
                    reproj = _rebuild_reproj_error(g, obs2d, cameras)  # type: ignore[arg-type]
            conf = (
                np.asarray(f["pose2d/conf"][()], dtype=float)  # type: ignore[index]
                if "pose2d/conf" in f
                else None
            )
            nmf = nmf_angles = nmf_angle_names = nmf_body_plan = None
            nmf_chain_scales: dict[str, float] = {}
            nmf_chain_offsets: dict[str, np.ndarray] = {}
            nmf_body_scale = 1.0
            if "inverse_kinematics/body_plan" in f:
                raw = f["inverse_kinematics/body_plan"][()]  # type: ignore[index]
                nmf_body_plan = raw.decode() if isinstance(raw, bytes) else str(raw)
            if "inverse_kinematics/points3d" in f:
                nmf = np.asarray(f["inverse_kinematics/points3d"][()], dtype=float)  # type: ignore[index]
            if "inverse_kinematics/angles" in f:
                nmf_angles = np.asarray(f["inverse_kinematics/angles"][()], dtype=float)  # type: ignore[index]
                nmf_angle_names = [
                    n.decode() if isinstance(n, bytes) else n
                    for n in f["inverse_kinematics/angle_names"][()]  # type: ignore[index]
                ]
            if "inverse_kinematics" in f and "meta" in f["inverse_kinematics"].attrs:
                ik_meta = json.loads(f["inverse_kinematics"].attrs["meta"])  # type: ignore[arg-type]
                nmf_chain_scales = {
                    str(k): float(v)
                    for k, v in (ik_meta.get("chain_scales") or {}).items()
                }
                nmf_chain_offsets = {
                    str(k): np.asarray(v, dtype=float).reshape(3)
                    for k, v in (ik_meta.get("chain_offsets") or {}).items()
                }
                if ik_meta.get("body_scale") is not None:
                    nmf_body_scale = float(ik_meta["body_scale"])
        if pts2d is None:
            raise ValueError(f"{path} has no 2D points (no pose2d group)")
        return cls(
            cameras=cameras,
            skeleton=skeleton,
            pts2d=pts2d,  # type: ignore[arg-type]
            conf=conf,  # type: ignore[arg-type]
            pts3d=pts3d,  # type: ignore[arg-type]
            reproj_error=reproj,  # type: ignore[arg-type]
            nmf_pts3d=nmf,  # type: ignore[arg-type]
            nmf_angles=nmf_angles,  # type: ignore[arg-type]
            nmf_angle_names=nmf_angle_names,
            nmf_chain_scales=nmf_chain_scales,
            nmf_chain_offsets=nmf_chain_offsets,
            nmf_body_scale=nmf_body_scale,
            nmf_body_plan=nmf_body_plan,
            absent=absent,  # type: ignore[arg-type]
            subject_id=subject_id,
            meta=meta,
        )


# -- per-stage store ----------------------------------------------------------


class StageStore:
    """Per-stage read/write access to one recording's ``results.h5``.

    Used by the staged run: ``pose2d`` truncates and recreates the file
    (:meth:`write_pose2d`), every later stage replaces only its own group, and
    :meth:`truncate_from` drops a stage's group together with every later one
    (their inputs changed). All reads return ``None`` when the file or the
    requested data is absent (including files in an older schema version, which
    simply read as empty and get recomputed).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    # -- presence -------------------------------------------------------------

    def has(self, stage: str) -> bool:
        """Whether ``stage``'s output is complete **and reusable** in the store.

        Deliberately stricter than :meth:`_open`, which reads any version in
        :data:`READABLE_VERSIONS`. This is the question the run asks before *skipping* a
        stage, and skipping one on an older file would leave the run appending
        current-schema groups beside older-schema ones, in a file whose recorded version
        names only one of them. Reading an old file is safe; extending one is not. So an
        older file always reports incomplete, the run recomputes from ``pose2d``, and
        :meth:`write_pose2d`'s truncation makes the whole file current. ``deeperfly
        repack`` is the way to keep an old file's contents without recomputing.

        Parameters
        ----------
        stage
            A pose stage name (``visualization`` keeps no h5 group and is
            always ``False`` here).

        Returns
        -------
        bool
            ``True`` if the stage's marker dataset is present in a current-schema file.
        """
        marker = _STAGE_MARKER.get(stage)
        if marker is None:
            return False
        # Through :meth:`_open` and not a bare version check, so a *newer* file still
        # raises here. Answering False for one would restart the run and let
        # :meth:`write_pose2d` truncate it -- the exact destruction _open documents.
        with self._open() as f:
            if f is None:
                return False
            return marker in f and _version_in(f) == FORMAT_VERSION

    def has_candidates(self) -> bool:
        """Whether the detector's top-K candidates were cached by ``pose2d``."""
        with self._open() as f:
            return f is not None and "pose2d/candidates/xy" in f

    # -- writes ---------------------------------------------------------------

    def write_pose2d(
        self,
        *,
        cameras: CameraGroup,
        skeleton: Skeleton,
        pts2d,
        conf,
        image_sizes: dict[str, tuple[int, int]],
        footage: dict[str, list[Path]] | None = None,
        candidates: "Candidates | None" = None,
        meta: dict | None = None,
    ) -> None:
        """Write a fresh ``pose2d`` output, truncating the whole file.

        ``pose2d`` is the pipeline root: recomputing it invalidates everything
        downstream, so the file restarts from scratch (which also disposes of
        files in an older schema).

        Parameters
        ----------
        cameras
            The config rig the detection ran with.
        skeleton
            The skeleton (written run-wide).
        pts2d, conf
            The detections, ``(V, T, P, 2)`` and ``(V, T, P)``.
        image_sizes
            ``camera_name -> (height, width)`` of the raw footage frames (lets
            a later run rebuild the config rig without re-reading footage).
        footage
            ``camera_name -> footage files`` for the recording. Each camera's
            paths are recorded both resolved-absolute and relative to this file's
            directory, so a viewer handed only the result can find the videos
            (see :meth:`read_footage`). ``None`` records no footage.
        candidates
            The top-K candidate peaks to cache (when pictorial_structures is
            enabled), or ``None``.
        meta
            Extra free-form metadata merged into ``attrs["meta"]``.

        Notes
        -----
        Any existing ``animal/`` group (which keypoints are not on this animal) is read
        *before* the truncation and written back after. It is an operator-authored fact
        about the specimen, not a pipeline product, so re-running detection must not erase
        it -- and this method's ``mode="w"`` would otherwise do exactly that.
        """
        carried_absent, carried_subject = (None, None)
        if self.path.exists():
            try:
                with h5py.File(self.path, "r") as f:
                    carried_absent, carried_subject = _read_animal(f)
            except (OSError, KeyError):  # unreadable/older file: nothing to carry
                pass
        full_meta = {
            "deeperfly_format_version": FORMAT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            **(meta or {}),
        }
        with h5py.File(self.path, "w") as f:
            f.attrs["meta"] = json.dumps(full_meta)
            _write_skeleton(f.create_group("skeleton"), skeleton)
            g = f.create_group("pose2d")
            _put(g, "points", np.asarray(pts2d, dtype=float))
            if conf is not None:
                _put(g, "conf", np.asarray(conf, dtype=float))
            _write_animal(f, absent=carried_absent, subject_id=carried_subject)
            # The image sizes go in BOTH places: in the camera group (so the rig carries its
            # own pixel frame, like a calibration.toml does) and in the legacy sibling attr,
            # which every existing reader uses and which stays authoritative for now.
            _write_cameras(g.create_group("cameras"), cameras, image_sizes=image_sizes)
            g.attrs["image_sizes"] = json.dumps(
                {name: [int(h), int(w)] for name, (h, w) in image_sizes.items()}
            )
            if footage:
                from .footage import write_pointer

                outdir = self.path.parent
                g.attrs["footage"] = json.dumps(
                    {
                        name: write_pointer(files, outdir)
                        for name, files in footage.items()
                    }
                )
            if candidates is not None:
                gc = g.create_group("candidates")
                _put(gc, "xy", candidates.xy)
                _put(gc, "score", candidates.score)

    def write_cameras(
        self,
        stage: str,
        cameras: CameraGroup,
        *,
        image_sizes: dict | None = None,
        meta: dict | None = None,
    ) -> None:
        """Replace ``<stage>/cameras`` (used by ``bundle_adjustment``).

        ``image_sizes`` and ``meta`` (``units``/``scale_source``/``provenance``/``quality``)
        are what make this the same record ``calibration.toml`` carries, so an exporter can
        pass the rig's real provenance through instead of inventing one.
        """
        with h5py.File(self.path, "a") as f:
            if stage in f:
                del f[stage]
            _write_cameras(
                f.create_group(f"{stage}/cameras"),
                cameras,
                image_sizes=image_sizes,
                meta=meta,
            )

    def write_points(
        self,
        stage: str,
        *,
        pts2d,
        pts3d,
        reproj_error,
        extra: dict | None = None,
        meta: dict | None = None,
    ) -> None:
        """Replace a points group (pictorial_structures / triangulation / eks /
        postprocess).

        Parameters
        ----------
        stage
            The stage whose group is replaced. Must be a :data:`STAGES` name, or
            :meth:`truncate_from` will not know to drop it.
        pts2d, pts3d, reproj_error
            The stage's arrays; a ``None`` is simply not written.
        extra
            Further named float datasets to store alongside them -- what a stage
            measured that does not fit the three-array shape (the smoother's
            posterior variance, for instance). Read back with
            :meth:`read_point_extra`.
        meta
            Small free-form metadata for the group's ``attrs``, JSON-encoded. The
            three-array stages record no provenance without it.

        Notes
        -----
        What actually lands on disk is decided by :func:`_write_points_group`: a 2D that
        the stored 3D reprojects to is not written, and neither is a reprojection error a
        reader can recompute. The rig and the detections that decide it are read back from
        this same file rather than passed in, so no stage had to learn about the policy.
        """
        with h5py.File(self.path, "a") as f:
            cameras = _cameras_from(f)
            obs2d = (
                np.asarray(f["pose2d/points"][()], dtype=float)  # type: ignore[index]
                if "pose2d/points" in f
                else None
            )
            if stage in f:
                del f[stage]
            _write_points_group(
                f.create_group(stage),
                pts2d=pts2d,
                pts3d=pts3d,
                reproj_error=reproj_error,
                extra=extra,
                meta=meta,
                cameras=cameras,
                obs2d=obs2d,
            )

    def write_ik(
        self,
        *,
        angles,
        angle_names,
        model_pts3d,
        body_plan: str | None = None,
        meta: dict | None = None,
    ) -> None:
        """Replace the ``inverse_kinematics`` group (joint angles + fitted model joints).

        Parameters
        ----------
        angles
            Fitted joint angles ``(T, D)`` in radians.
        angle_names
            The ``D`` angle names, in column order.
        model_pts3d
            The fitted model joints ``(T, P, 3)`` in world coordinates (skeleton
            point order), for reprojection / the overlay.
        body_plan
            The solved body plan, as JSON. Stored as its own **dataset** rather than in
            ``meta``: it runs to tens of kilobytes, which is uncomfortably close to the
            64 KB an HDF5 attribute allows, and ``meta`` is encoded with a ``default=``
            fallback that would silently stringify a stray numpy value instead of
            raising.
        meta
            Small free-form metadata stored on the group's ``attrs`` (the template name,
            the registration, the estimated scales), JSON-encoded.
        """
        with h5py.File(self.path, "a") as f:
            if "inverse_kinematics" in f:
                del f["inverse_kinematics"]
            g = f.create_group("inverse_kinematics")
            _put(g, "angles", np.asarray(angles, dtype=float))
            g.create_dataset(
                "angle_names",
                data=np.array(list(angle_names), dtype=object),
                dtype=_STR,
            )
            _put(g, "points3d", np.asarray(model_pts3d, dtype=float))
            if body_plan is not None:
                g.create_dataset("body_plan", data=str(body_plan), dtype=_STR)
            if meta:
                g.attrs["meta"] = json.dumps(meta, default=str)

    def read_ik(
        self,
    ) -> tuple[np.ndarray, list[str], np.ndarray] | None:
        """``(angles, angle_names, model_pts3d)`` of the IK stage, or ``None``."""
        with self._open() as f:
            if f is None or "inverse_kinematics/angles" not in f:
                return None
            g = f["inverse_kinematics"]
            names = [
                n.decode() if isinstance(n, bytes) else n
                for n in g["angle_names"][()]  # type: ignore[index]
            ]
            return (
                g["angles"][()],  # type: ignore[index]
                names,
                g["points3d"][()],  # type: ignore[index]
            )

    def read_ik_meta(self) -> dict:
        """The IK stage's metadata, ``{}`` when absent.

        Separate from :meth:`read_ik` (which returns only the arrays) because the
        estimated ``chain_scales`` / ``body_scale`` live here and the mesh overlay needs
        them: rendering from the arrays alone silently drew the model at size 1.0. The
        stored ``body_plan`` is folded in under that key.
        """
        with self._open() as f:
            if f is None or "inverse_kinematics" not in f:
                return {}
            g = f["inverse_kinematics"]
            meta = json.loads(g.attrs["meta"]) if "meta" in g.attrs else {}  # type: ignore[arg-type]
            if "body_plan" in g:
                plan = g["body_plan"][()]  # type: ignore[index]
                meta["body_plan"] = (
                    plan.decode() if isinstance(plan, bytes) else str(plan)
                )
            return meta

    def truncate_from(self, stage: str) -> None:
        """Delete ``stage``'s group and every later stage's group.

        Called before a mid-pipeline stage recomputes, so a stale downstream
        group (possibly of a now-disabled stage) can never feed a later run.
        """
        if not self.path.exists():
            return
        drop = STAGES[STAGES.index(stage) :]
        with h5py.File(self.path, "a") as f:
            for name in drop:
                if name in f:
                    del f[name]

    # -- reads ----------------------------------------------------------------

    def write_animal(self, *, absent=None, subject_id: str | None = None) -> None:
        """Record which keypoints are not on this animal, **in place**.

        Deliberately an ``mode="a"`` patch rather than a rewrite: this is called from the
        editor's Save, and it must not disturb any pipeline stage's output. ``animal/`` is
        not in :data:`STAGES`, so :meth:`truncate_from` never deletes it and a
        mid-pipeline recompute keeps the declaration.

        This is also the seam by which the *pipeline* learns about absence: it reads
        ``animal/`` from ``results.h5`` and never opens the GUI's ``labels.h5``, so the two
        stay decoupled and a stale or unreadable sidecar cannot stop a run.
        """
        if not self.path.exists():
            return
        with h5py.File(self.path, "a") as f:
            if "animal" in f:
                del f["animal"]
            _write_animal(f, absent=absent, subject_id=subject_id)

    def read_animal(self) -> "tuple[np.ndarray | None, str | None]":
        """``(absent (P,) | None, subject_id | None)``; ``(None, None)`` when unrecorded."""
        with self._open() as f:
            return _read_animal(f) if f else (None, None)

    def read_skeleton(self) -> Skeleton | None:
        with self._open() as f:
            return _read_skeleton(f["skeleton"]) if f and "skeleton" in f else None  # type: ignore[arg-type]

    def read_pose2d(self) -> tuple[np.ndarray, np.ndarray | None] | None:
        """The pristine ``pose2d`` detections ``(pts2d, conf)``, or ``None``."""
        with self._open() as f:
            if f is None or "pose2d/points" not in f:
                return None
            conf = f["pose2d/conf"][()] if "pose2d/conf" in f else None  # type: ignore[index]
            return f["pose2d/points"][()], conf  # type: ignore[index, return-value]

    def read_camera_meta(self, stage: str) -> dict:
        """``units``/``scale_source``/``provenance``/``quality``/``image_sizes`` for a rig.

        ``{}`` when the stage or the metadata is absent, which is what lets a caller pass
        *through* what it finds rather than assert a value of its own.
        """
        with self._open() as f:
            if f is None or f"{stage}/cameras" not in f:
                return {}
            return _read_camera_meta(f[f"{stage}/cameras"])  # type: ignore[arg-type]

    def read_cameras(self, stage: str) -> CameraGroup | None:
        """The rig stored by ``stage`` (``pose2d`` or ``bundle_adjustment``)."""
        with self._open() as f:
            if f is None or f"{stage}/cameras" not in f:
                return None
            return _read_cameras(f[f"{stage}/cameras"])  # type: ignore[arg-type]

    def read_points(
        self, stage: str
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None] | None:
        """``(pts2d, pts3d, reproj_error)`` of a points stage, or ``None``.

        The 2D and the error are reconstructed when v3 chose not to store them, so a
        downstream stage reading its input cannot tell the difference -- which is the
        whole point: ``select_postprocess_input`` asks the smoother for a 2D that is no
        longer on disk, and must still get the array the smoother produced.

        Returns float64 whatever the file stores.
        """
        with self._open() as f:
            if f is None or stage not in f:
                return None
            g = f[stage]
            cameras = _cameras_from(f)
            obs2d = (
                np.asarray(f["pose2d/points"][()], dtype=float)  # type: ignore[index]
                if "pose2d/points" in f
                else None
            )
            pts3d = (
                np.asarray(g["points3d"][()], dtype=float)  # type: ignore[index]
                if "points3d" in g
                else None
            )
            return (
                _rebuild_pts2d(g, cameras),  # type: ignore[arg-type]
                pts3d,
                _rebuild_reproj_error(g, obs2d, cameras),  # type: ignore[arg-type]
            )

    def read_point_extra(self, stage: str, name: str) -> np.ndarray | None:
        """One of a points stage's :meth:`write_points` ``extra`` datasets, or ``None``."""
        with self._open() as f:
            if f is None or f"{stage}/{name}" not in f:
                return None
            return f[f"{stage}/{name}"][()]  # type: ignore[index, return-value]

    def read_point_meta(self, stage: str) -> dict:
        """A points stage's :meth:`write_points` ``meta``, ``{}`` when absent."""
        with self._open() as f:
            if f is None or stage not in f or "meta" not in f[stage].attrs:
                return {}
            return json.loads(f[stage].attrs["meta"])  # type: ignore[arg-type]

    def read_candidates(self) -> "Candidates | None":
        """The cached top-K candidate peaks, or ``None`` if not stored."""
        from .pictorial import Candidates

        with self._open() as f:
            if f is None or "pose2d/candidates/xy" not in f:
                return None
            return Candidates(
                xy=f["pose2d/candidates/xy"][()],  # type: ignore[index]
                score=f["pose2d/candidates/score"][()],  # type: ignore[index]
            )

    def read_image_sizes(self) -> dict[str, tuple[int, int]] | None:
        """``camera_name -> (height, width)`` recorded by ``pose2d``, or ``None``."""
        with self._open() as f:
            if f is None or "pose2d" not in f:
                return None
            raw = f["pose2d"].attrs.get("image_sizes")
            if raw is None:
                return None
            return {name: (h, w) for name, (h, w) in json.loads(raw).items()}

    def read_footage(self) -> dict[str, dict[str, list[str]]] | None:
        """``camera_name -> {"abs": [...], "rel": [...]}`` footage paths, or ``None``.

        The absolute paths are as resolved when ``pose2d`` ran; the relative
        paths are relative to this file's directory. ``None`` for files written
        before footage was recorded (a viewer then falls back to asking for the
        directory). See :meth:`write_pose2d`.
        """
        with self._open() as f:
            if f is None or "pose2d" not in f:
                return None
            raw = f["pose2d"].attrs.get("footage")
            if raw is None:
                return None
            return json.loads(raw)

    # -- internals -------------------------------------------------------------

    def _open(self):
        """Open the file read-only iff it exists in a schema version this build reads.

        An **unrecognized** file reads as absent: every stage is regenerable, so the run
        simply recomputes it (and :meth:`write_pose2d` truncates it on the way).

        A **newer** file is refused, because that same treat-as-absent would be silent data
        destruction: every ``has(stage)`` would report false, the run would recompute
        ``pose2d``, and the truncating write would take the newer file with it -- reporting
        success. Regenerable is not the same as disposable, and only the older direction is
        the former.

        An **older but readable** file (:data:`READABLE_VERSIONS`) opens normally, so a
        viewer or an annotation session can still get at what it holds. Deciding whether
        to *recompute* it is :meth:`has`'s job, not this one's -- see there for why the two
        answers differ.
        """
        import contextlib

        if not self.path.exists():
            return contextlib.nullcontext(None)
        f = h5py.File(self.path, "r")
        try:
            meta = json.loads(f.attrs.get("meta", "{}"))
        except (TypeError, ValueError):
            meta = {}
        version = meta.get("deeperfly_format_version")
        if _is_newer(version):
            f.close()
            raise ValueError(_newer_message(self.path, version))
        if version not in READABLE_VERSIONS:
            f.close()
            return contextlib.nullcontext(None)
        return contextlib.closing(f)


# -- repack -------------------------------------------------------------------


#: The stages :func:`repack` re-reduces. Every other group is copied through untouched
#: (bar the dtype policy), including ones this module knows nothing about: a
#: ``dfpose_predict/`` group is somebody else's record of what they did to this file, and
#: dropping it during a *space* optimization would be the same silent loss of provenance
#: the reprojection-error rule exists to prevent.
_POINT_STAGES = ("pictorial_structures", "triangulation", "eks", "postprocess")


def repack(path: str | Path, *, dst: str | Path | None = None) -> tuple[int, int]:
    """Rewrite a ``results.h5`` in the current schema, in place by default.

    Reads whatever :data:`READABLE_VERSIONS` allows and writes v3: point arrays narrowed
    to float32 and deflated, and any 2D or reprojection error a reader can rebuild left
    out. Nothing is recomputed -- the pose in the file is the pose that comes out, to
    within the float32 storage step -- so this is the way to shrink an existing recording
    without re-running the pipeline over it.

    The rewrite goes to a sibling temporary file and is moved into place only once it is
    complete, so an interrupted repack leaves the original intact.

    Parameters
    ----------
    path
        The ``results.h5`` to repack.
    dst
        Write here instead of over ``path``.

    Returns
    -------
    tuple[int, int]
        ``(bytes_before, bytes_after)``.

    Raises
    ------
    ValueError
        If the file was written by a newer build, or is not a deeperfly result.
    """
    import os
    import tempfile

    src = Path(path)
    out = Path(dst) if dst is not None else src
    before = src.stat().st_size
    with h5py.File(src, "r") as f:
        try:
            meta = json.loads(f.attrs.get("meta", "{}") or "{}")  # type: ignore[arg-type]
        except (TypeError, ValueError) as e:
            raise ValueError(f"{src} has no readable deeperfly metadata") from e
        version = _version_in(f)
        if _is_newer(version):
            raise ValueError(_newer_message(src, version))
        if version not in READABLE_VERSIONS:
            raise ValueError(
                f"{src} has deeperfly format version {version!r}, which this build "
                f"cannot read (expected one of "
                f"{', '.join(str(v) for v in READABLE_VERSIONS)})"
            )
        cameras = _cameras_from(f)
        obs2d = (
            np.asarray(f["pose2d/points"][()], dtype=float)  # type: ignore[index]
            if "pose2d/points" in f
            else None
        )
        # The reduced groups are rebuilt wholesale below; everything else is copied, so
        # name their datasets here rather than deciding per dataset during the walk.
        rebuilt = {
            s
            for s in _POINT_STAGES
            if s in f and "points3d" in f[s]  # type: ignore[operator]
        }
        fd, tmp = tempfile.mkstemp(
            dir=str(out.parent), prefix=f".{out.name}.", suffix=".repack"
        )
        os.close(fd)
        tmp_path = Path(tmp)
        try:
            with h5py.File(tmp_path, "w") as d:
                for k, v in f.attrs.items():
                    d.attrs[k] = v
                d.attrs["meta"] = json.dumps(
                    {**meta, "deeperfly_format_version": FORMAT_VERSION}
                )

                def visit(name: str, obj) -> None:
                    top = name.split("/")[0]
                    if top in rebuilt:
                        return
                    if isinstance(obj, h5py.Group):
                        g = d.require_group(name)
                        for k, v in obj.attrs.items():
                            g.attrs[k] = v
                        return
                    parent = (
                        d.require_group(name.rsplit("/", 1)[0]) if "/" in name else d
                    )
                    leaf = name.rsplit("/", 1)[-1]
                    # Strings and other non-numeric data are copied as they are: the
                    # dtype policy is about float precision and has nothing to say here.
                    if obj.dtype.kind in "OSU" or h5py.check_string_dtype(obj.dtype):
                        ds = parent.create_dataset(leaf, data=obj[()], dtype=obj.dtype)
                    else:
                        ds = _put(parent, leaf, obj[()])
                    for k, v in obj.attrs.items():
                        ds.attrs[k] = v

                f.visititems(visit)
                for stage in _POINT_STAGES:
                    if stage not in rebuilt:
                        continue
                    g_src = f[stage]
                    pts3d = np.asarray(g_src["points3d"][()], dtype=float)  # type: ignore[index]
                    known = {
                        "points",
                        "points3d",
                        "reproj_error",
                        "points2d_override",
                        "points2d_override_cols",
                    }
                    g_out = d.create_group(stage)
                    for k, v in g_src.attrs.items():  # type: ignore[union-attr]
                        g_out.attrs[k] = v
                    _write_points_group(
                        g_out,
                        pts2d=_rebuild_pts2d(g_src, cameras),  # type: ignore[arg-type]
                        pts3d=pts3d,
                        reproj_error=_rebuild_reproj_error(g_src, obs2d, cameras),  # type: ignore[arg-type]
                        extra={
                            k: g_src[k][()]  # type: ignore[index]
                            for k in g_src  # type: ignore[union-attr]
                            if k not in known and isinstance(g_src[k], h5py.Dataset)  # type: ignore[index]
                        },
                        cameras=cameras,
                        obs2d=obs2d,
                    )
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    os.replace(tmp_path, out)
    return before, out.stat().st_size


# -- camera (de)serialization ------------------------------------------------


#: The rig facts a ``calibration.toml`` carries and the HDF5 camera group had no slot for.
#: Stored as group attrs so the two artifacts describe a rig the *same* way -- and, more
#: importantly, so the exporters stop **inventing** them. Both
#: ``deeperfly calibration export`` and the pipeline's own writer hardcoded
#: ``units="config", scale_source="orbit_prior"``, which re-labels a millimeter board
#: calibration as an arbitrary-scale orbit guess: a false provenance claim on the one field
#: that tells a reader whether the numbers mean anything physical.
CAMERA_META_KEYS = ("units", "scale_source", "provenance", "quality")


def _write_cameras(
    g: h5py.Group,
    cameras: CameraGroup,
    *,
    image_sizes: dict | None = None,
    meta: dict | None = None,
) -> None:
    """Write a rig, plus the metadata that makes it the same record ``calibration.toml`` is.

    Everything beyond the five original datasets is **additive**: an older file has none of
    it and :func:`_read_cameras` returns ``None`` for what is missing, so no format version
    moves and no file needs regenerating.
    """
    g.create_dataset("names", data=np.array(cameras.names, dtype=object), dtype=_STR)
    g.create_dataset("rvecs", data=cameras.rvecs)
    g.create_dataset("tvecs", data=cameras.tvecs)
    g.create_dataset("intrs", data=cameras.intrs)
    g.create_dataset("dists", data=cameras.dists)
    # ``CameraGroup.dists`` zero-pads every camera to the group-wide max K, by contract, for
    # the JAX call sites. Harmless numerically (all-zero coefficients are the identity) but a
    # textual round-trip failure: a camera authored `dist = []` reads back as five zeros. The
    # true lengths make the round trip exact without changing the padded array anyone reads.
    g.create_dataset(
        "dist_lengths",
        data=np.asarray([len(np.atleast_1d(c.dist)) for c in cameras], dtype=np.int32),
    )
    if image_sizes:
        # In the group, not in a sibling attr on `pose2d` only -- which is why the BA rig had
        # no pixel frame of its own and `check_image_sizes` silently stopped guarding it.
        g.create_dataset(
            "image_sizes",
            data=np.asarray(
                [
                    [int(h), int(w)]
                    for (h, w) in (image_sizes.get(n, (-1, -1)) for n in cameras.names)
                ],
                dtype=np.int32,
            ),
        )
    for key, value in (meta or {}).items():
        if key in CAMERA_META_KEYS and value is not None:
            g.attrs[key] = value if isinstance(value, str) else json.dumps(value)


def _read_cameras(g: h5py.Group) -> CameraGroup:
    names = [n.decode() if isinstance(n, bytes) else n for n in g["names"][()]]  # type: ignore[index]
    return CameraGroup.from_arrays(
        names,
        g["rvecs"][()],  # type: ignore[index]
        g["tvecs"][()],  # type: ignore[index]
        g["intrs"][()],  # type: ignore[index]
        g["dists"][()],  # type: ignore[index]
    )


def _read_camera_meta(g: h5py.Group) -> dict:
    """The rig's ``units``/``scale_source``/``provenance``/``quality``, plus image sizes.

    ``{}`` for a group written before any of it existed -- which is what makes a caller
    pass *through* what it finds instead of inventing a value.
    """
    out: dict = {}
    for key in CAMERA_META_KEYS:
        if key not in g.attrs:
            continue
        raw = g.attrs[key]
        raw = raw.decode() if isinstance(raw, bytes) else raw
        if key in ("provenance", "quality"):
            try:
                out[key] = json.loads(raw)
                continue
            except (TypeError, ValueError):
                pass
        out[key] = raw
    if "image_sizes" in g:
        names = [n.decode() if isinstance(n, bytes) else n for n in g["names"][()]]
        rows = np.asarray(g["image_sizes"][()], dtype=int).reshape(-1, 2)
        out["image_sizes"] = {
            str(n): (int(h), int(w))
            for n, (h, w) in zip(names, rows)
            if h > 0 and w > 0
        }
    if "dist_lengths" in g:
        out["dist_lengths"] = [int(k) for k in np.asarray(g["dist_lengths"][()])]
    return out


# -- skeleton (de)serialization ----------------------------------------------


def _write_skeleton(g: h5py.Group, s: Skeleton) -> None:
    g.attrs["name"] = s.name
    g.create_dataset(
        "point_names", data=np.array(s.point_names, dtype=object), dtype=_STR
    )
    g.create_dataset("bones", data=s.bones)
    # One hex per POINT, in point order. Replaces the `limb_names` / `limb_id` /
    # `palette` trio, which stored the grouping the skeleton no longer has. A file
    # written before this reads back with the colormap (below), which is acceptable
    # because colors are cosmetic -- no repack, and nothing geometric is lost.
    g.create_dataset(
        "point_colors", data=np.array(s.point_colors, dtype=object), dtype=_STR
    )
    # Additive, and read back with a default: a results.h5 written before symmetry
    # existed has no such dataset and loads as a skeleton with no pairs, which only
    # disables the pair-driven features for that file.
    g.create_dataset("symmetries", data=np.asarray(s.symmetries).reshape(-1, 2))


def _read_skeleton(g: h5py.Group) -> Skeleton:
    decode = lambda arr: tuple(  # noqa: E731
        x.decode() if isinstance(x, bytes) else x for x in arr
    )
    sym = g.get("symmetries")
    stored = g.get("point_colors")
    return Skeleton(
        name=str(g.attrs["name"]),
        point_names=decode(g["point_names"][()]),  # type: ignore[index]
        bones=g["bones"][()],  # type: ignore[index]
        # Absent in a file written before colors were per point: the skeleton then
        # defaults to the colormap by index rather than guessing at a limb palette.
        point_colors=() if stored is None else decode(stored[()]),
        symmetries=(
            np.empty((0, 2), np.int64)
            if sym is None
            else np.asarray(sym[()]).reshape(-1, 2)
        ),
    )
