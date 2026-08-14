"""Self-contained HDF5 result container for the pose pipeline.

``results.h5`` (schema v2) stores each pipeline stage's output in its own group,
so a stage never overwrites another stage's data and any downstream stage can
be re-run later from pristine upstream outputs:

.. code-block:: text

    attrs["meta"]            json: {deeperfly_format_version: 2, created_utc, ...}
    skeleton/                the skeleton (point names, bones, visibility, palette)
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
        points               (V, T, P, 2) the smoothed trajectory, reprojected
        points3d             (T, P, 3) the smoothed 3D
        reproj_error         (V, T, P) against the pose2d observations
        posterior_var        (T, P) the smoother's posterior variance
        smooth_param         (P,) the fitted per-keypoint process-noise scale
    postprocess/
        points               (V, T, P, 2) 2D after the correction chain
        points3d             (T, P, 3) 3D after the correction chain
        reproj_error         (V, T, P) against the pose2d observations
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
un-triangulated points and is preserved by the float64 datasets.
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

__all__ = ["PoseResult", "StageStore"]

FORMAT_VERSION = 2
_STR = h5py.string_dtype("utf-8")


def _is_newer(version) -> bool:
    """Whether a stored ``deeperfly_format_version`` is from a *later* build than this one.

    Tolerant of junk: an unparseable version is not "newer", it is unrecognized, and the
    caller's older-file path (regenerate) is the right answer for it.
    """
    try:
        return int(version) > FORMAT_VERSION
    except (TypeError, ValueError):
        return False


def _newer_message(path, version) -> str:
    """The refusal every deeperfly artifact shares, so it reads the same wherever it lands."""
    return (
        f"{path} was written by a newer deeperfly (result format v{version}, this build "
        f"understands v{FORMAT_VERSION}); refusing to read it rather than silently "
        "dropping state it carries"
    )


#: The dataset whose presence means a stage's output is complete, per stage.
_STAGE_MARKER = {
    "pose2d": "pose2d/points",
    "bundle_adjustment": "bundle_adjustment/cameras",
    "pictorial_structures": "pictorial_structures/points",
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
            g2d.create_dataset("points", data=self.pts2d)
            if self.conf is not None:
                g2d.create_dataset("conf", data=self.conf)
            _write_cameras(g2d.create_group("cameras"), self.cameras)
            if self.pts3d is not None or self.reproj_error is not None:
                g3d = f.create_group("triangulation")
                g3d.create_dataset("points", data=self.pts2d)
                if self.pts3d is not None:
                    g3d.create_dataset("points3d", data=self.pts3d)
                if self.reproj_error is not None:
                    g3d.create_dataset("reproj_error", data=self.reproj_error)
            _write_animal(f, absent=self.absent, subject_id=self.subject_id)

    @classmethod
    def load(cls, path: str | Path) -> PoseResult:
        """Read the assembled :class:`PoseResult` back from an HDF5 file.

        Assembly prefers the most-derived data present: ``pts2d`` from
        postprocess, else eks, else triangulation, else pictorial_structures,
        else pose2d; ``pts3d`` / ``reproj_error`` from the same order (minus
        pose2d, which has no 3D); cameras from bundle_adjustment, else the pose2d
        config rig.

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
            if version != FORMAT_VERSION:
                raise ValueError(
                    f"{path} has deeperfly format version {version!r}, expected "
                    f"{FORMAT_VERSION}; re-run the pipeline to regenerate it"
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
            # Most-derived first: the correction chain supersedes the smoother's
            # output, which supersedes the triangulation it was seeded from, which
            # supersedes the raw detections.
            for stage in (
                "postprocess",
                "eks",
                "triangulation",
                "pictorial_structures",
                "pose2d",
            ):
                if pts2d is None and f"{stage}/points" in f:
                    pts2d = f[f"{stage}/points"][()]  # type: ignore[index]
                if pts3d is None and f"{stage}/points3d" in f:
                    pts3d = f[f"{stage}/points3d"][()]  # type: ignore[index]
                if reproj is None and f"{stage}/reproj_error" in f:
                    reproj = f[f"{stage}/reproj_error"][()]  # type: ignore[index]
            conf = f["pose2d/conf"][()] if "pose2d/conf" in f else None  # type: ignore[index]
            nmf = nmf_angles = nmf_angle_names = nmf_body_plan = None
            nmf_chain_scales: dict[str, float] = {}
            nmf_body_scale = 1.0
            if "inverse_kinematics/body_plan" in f:
                raw = f["inverse_kinematics/body_plan"][()]  # type: ignore[index]
                nmf_body_plan = raw.decode() if isinstance(raw, bytes) else str(raw)
            if "inverse_kinematics/points3d" in f:
                nmf = f["inverse_kinematics/points3d"][()]  # type: ignore[index]
            if "inverse_kinematics/angles" in f:
                nmf_angles = f["inverse_kinematics/angles"][()]  # type: ignore[index]
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
        """Whether ``stage``'s output is complete in the store.

        Parameters
        ----------
        stage
            A pose stage name (``visualization`` keeps no h5 group and is
            always ``False`` here).

        Returns
        -------
        bool
            ``True`` if the stage's marker dataset is present (schema v2 only).
        """
        marker = _STAGE_MARKER.get(stage)
        if marker is None:
            return False
        with self._open() as f:
            return f is not None and marker in f

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
            g.create_dataset("points", data=np.asarray(pts2d, dtype=float))
            if conf is not None:
                g.create_dataset("conf", data=np.asarray(conf, dtype=float))
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
                gc.create_dataset("xy", data=candidates.xy)
                gc.create_dataset("score", data=candidates.score)

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
        """
        with h5py.File(self.path, "a") as f:
            if stage in f:
                del f[stage]
            g = f.create_group(stage)
            for name, arr in (
                ("points", pts2d),
                ("points3d", pts3d),
                ("reproj_error", reproj_error),
                *sorted((extra or {}).items()),
            ):
                if arr is not None:
                    g.create_dataset(name, data=np.asarray(arr, dtype=float))
            if meta:
                g.attrs["meta"] = json.dumps(meta, default=str)

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
            g.create_dataset("angles", data=np.asarray(angles, dtype=float))
            g.create_dataset(
                "angle_names",
                data=np.array(list(angle_names), dtype=object),
                dtype=_STR,
            )
            g.create_dataset("points3d", data=np.asarray(model_pts3d, dtype=float))
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
        """``(pts2d, pts3d, reproj_error)`` of a points stage, or ``None``."""
        with self._open() as f:
            if f is None or stage not in f:
                return None
            g = f[stage]
            return tuple(  # type: ignore[return-value]
                g[name][()] if name in g else None  # type: ignore[index, operator]
                for name in ("points", "points3d", "reproj_error")
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
        """Open the file read-only iff it exists in the current schema version.

        An **older** file reads as absent: every stage is regenerable, so the run simply
        recomputes it (and :meth:`write_pose2d` truncates it on the way).

        A **newer** file is refused, because that same treat-as-absent would be silent data
        destruction: every ``has(stage)`` would report false, the run would recompute
        ``pose2d``, and the truncating write would take the newer file with it -- reporting
        success. Regenerable is not the same as disposable, and only the older direction is
        the former.
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
        if version != FORMAT_VERSION:
            f.close()
            return contextlib.nullcontext(None)
        return contextlib.closing(f)


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
    g.create_dataset(
        "limb_names", data=np.array(s.limb_names, dtype=object), dtype=_STR
    )
    g.create_dataset("limb_id", data=s.limb_id)
    g.create_dataset("bones", data=s.bones)
    # Additive, and read back with a default: a results.h5 written before symmetry
    # existed has no such dataset and loads as a skeleton with no pairs. That only
    # disables the pair-driven features for that file (`deeperfly.chirality` falls
    # back to name inference), so no format version bump is needed.
    g.create_dataset("symmetries", data=np.asarray(s.symmetries).reshape(-1, 2))
    pal = g.create_group("palette")
    for name, color in s.palette.items():
        pal.attrs[name] = color


def _read_skeleton(g: h5py.Group) -> Skeleton:
    decode = lambda arr: tuple(  # noqa: E731
        x.decode() if isinstance(x, bytes) else x for x in arr
    )
    palette = {
        name: (v.decode() if isinstance(v, bytes) else v)
        for name, v in g["palette"].attrs.items()  # type: ignore[index]
    }
    sym = g.get("symmetries")
    return Skeleton(
        name=str(g.attrs["name"]),
        point_names=decode(g["point_names"][()]),  # type: ignore[index]
        limb_names=decode(g["limb_names"][()]),  # type: ignore[index]
        limb_id=g["limb_id"][()],  # type: ignore[index]
        bones=g["bones"][()],  # type: ignore[index]
        palette=palette,
        symmetries=(
            np.empty((0, 2), np.int64)
            if sym is None
            else np.asarray(sym[()]).reshape(-1, 2)
        ),
    )
