"""Self-contained HDF5 result container for the pose pipeline.

``poses.h5`` (schema v2) stores each pipeline stage's output in its own group,
so a stage never overwrites another stage's data and any downstream stage can
be re-run later from pristine upstream outputs:

.. code-block:: text

    attrs["meta"]            json: {deeperfly_format_version: 2, created_utc, ...}
    skeleton/                the skeleton (point names, bones, visibility, palette)
    pose2d/
        points               (V, T, N, 2) arg-max 2D detections (visibility-masked)
        conf                 (V, T, N) detection confidences
        cameras/             the config rig as built at detect time
        attrs["image_sizes"] json {camera_name: [h, w]} of the raw footage frames
        candidates/          top-K peaks (xy, score) -- present iff the
                             pictorial_structures stage was enabled at detect time
    bundle_adjustment/
        cameras/             the BA-refined rig
    pictorial_structures/
        points               (V, T, N, 2) PS-corrected 2D
        points3d             (T, N, 3) initial 3D estimate
        reproj_error         (V, T, N)
    triangulation/
        points               (V, T, N, 2) cleaned 2D (outlier-rejecting methods)
        points3d             (T, N, 3)
        reproj_error         (V, T, N)

:class:`StageStore` is the per-stage read/write access used by the staged run;
:class:`PoseResult` is the assembled in-memory view (the *best* points present:
triangulation over pictorial over pose2d, BA cameras over the config rig). The
HDF5 file fully reconstructs the cameras and skeleton, so results are portable
without the original config files.

Arrays use the view-leading layout: ``pts2d`` is ``(V, T, N, 2)``, ``conf`` is
``(V, T, N)``, ``pts3d`` is ``(T, N, 3)``. NaN encodes missing observations /
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
from jaxtyping import Float

from .cameras import CameraGroup
from .config import STAGES
from .skeleton import Skeleton

if TYPE_CHECKING:
    from .pictorial import Candidates

__all__ = ["PoseResult", "StageStore"]

FORMAT_VERSION = 2
_STR = h5py.string_dtype("utf-8")

#: The dataset whose presence means a stage's output is complete, per stage.
_STAGE_MARKER = {
    "pose2d": "pose2d/points",
    "bundle_adjustment": "bundle_adjustment/cameras",
    "pictorial_structures": "pictorial_structures/points",
    "triangulation": "triangulation/points3d",
}


@dataclass
class PoseResult:
    """A complete multi-view pose-estimation result for one recording."""

    cameras: CameraGroup
    skeleton: Skeleton
    pts2d: Float[np.ndarray, "V T N 2"]
    conf: Float[np.ndarray, "V T N"] | None = None
    pts3d: Float[np.ndarray, "T N 3"] | None = None
    reproj_error: Float[np.ndarray, "V T N"] | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pts2d = np.asarray(self.pts2d, dtype=float)
        for name in ("conf", "pts3d", "reproj_error"):
            arr = getattr(self, name)
            if arr is not None:
                setattr(self, name, np.asarray(arr, dtype=float))

    @property
    def n_views(self) -> int:
        return self.pts2d.shape[0]

    @property
    def n_frames(self) -> int:
        return self.pts2d.shape[1]

    # -- serialization -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the result to an HDF5 file (overwriting ``path``).

        The library one-shot: ``pts2d``/``conf`` go to ``pose2d/`` and, when a 3D
        pose is present, the (possibly cleaned) 2D, 3D and reprojection error go
        to ``triangulation/`` -- so :meth:`load` round-trips the assembled view.
        ``pts2d`` is duplicated into both groups in that case (it is small next
        to the footage).

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

    @classmethod
    def load(cls, path: str | Path) -> PoseResult:
        """Read the assembled :class:`PoseResult` back from an HDF5 file.

        Assembly prefers the most-derived data present: ``pts2d`` from
        triangulation, else pictorial_structures, else pose2d; ``pts3d`` /
        ``reproj_error`` from triangulation, else pictorial_structures; cameras
        from bundle_adjustment, else the pose2d config rig.

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
            meta = json.loads(f.attrs["meta"])
            version = meta.pop("deeperfly_format_version", None)
            if version != FORMAT_VERSION:
                raise ValueError(
                    f"{path} has deeperfly format version {version!r}, expected "
                    f"{FORMAT_VERSION}; re-run the pipeline to regenerate it"
                )
            skeleton = _read_skeleton(f["skeleton"])
            cameras = _read_cameras(
                f["bundle_adjustment/cameras"]
                if "bundle_adjustment/cameras" in f
                else f["pose2d/cameras"]
            )
            pts2d = pts3d = reproj = None
            for stage in ("triangulation", "pictorial_structures", "pose2d"):
                if pts2d is None and f"{stage}/points" in f:
                    pts2d = f[f"{stage}/points"][()]
                if pts3d is None and f"{stage}/points3d" in f:
                    pts3d = f[f"{stage}/points3d"][()]
                if reproj is None and f"{stage}/reproj_error" in f:
                    reproj = f[f"{stage}/reproj_error"][()]
            conf = f["pose2d/conf"][()] if "pose2d/conf" in f else None
        if pts2d is None:
            raise ValueError(f"{path} has no 2D points (no pose2d group)")
        return cls(
            cameras=cameras,
            skeleton=skeleton,
            pts2d=pts2d,
            conf=conf,
            pts3d=pts3d,
            reproj_error=reproj,
            meta=meta,
        )


# -- per-stage store ----------------------------------------------------------


class StageStore:
    """Per-stage read/write access to one recording's ``poses.h5``.

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
            The detections, ``(V, T, N, 2)`` and ``(V, T, N)``.
        image_sizes
            ``camera_name -> (height, width)`` of the raw footage frames (lets
            a later run rebuild the config rig without re-reading footage).
        candidates
            The top-K candidate peaks to cache (when pictorial_structures is
            enabled), or ``None``.
        meta
            Extra free-form metadata merged into ``attrs["meta"]``.
        """
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
            _write_cameras(g.create_group("cameras"), cameras)
            g.attrs["image_sizes"] = json.dumps(
                {name: [int(h), int(w)] for name, (h, w) in image_sizes.items()}
            )
            if candidates is not None:
                gc = g.create_group("candidates")
                gc.create_dataset("xy", data=candidates.xy)
                gc.create_dataset("score", data=candidates.score)

    def write_cameras(self, stage: str, cameras: CameraGroup) -> None:
        """Replace ``<stage>/cameras`` (used by ``bundle_adjustment``)."""
        with h5py.File(self.path, "a") as f:
            if stage in f:
                del f[stage]
            _write_cameras(f.create_group(f"{stage}/cameras"), cameras)

    def write_points(self, stage: str, *, pts2d, pts3d, reproj_error) -> None:
        """Replace ``stage``'s points group (pictorial_structures / triangulation)."""
        with h5py.File(self.path, "a") as f:
            if stage in f:
                del f[stage]
            g = f.create_group(stage)
            for name, arr in (
                ("points", pts2d),
                ("points3d", pts3d),
                ("reproj_error", reproj_error),
            ):
                if arr is not None:
                    g.create_dataset(name, data=np.asarray(arr, dtype=float))

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

    def read_skeleton(self) -> Skeleton | None:
        with self._open() as f:
            return _read_skeleton(f["skeleton"]) if f and "skeleton" in f else None

    def read_pose2d(self) -> tuple[np.ndarray, np.ndarray | None] | None:
        """The pristine ``pose2d`` detections ``(pts2d, conf)``, or ``None``."""
        with self._open() as f:
            if f is None or "pose2d/points" not in f:
                return None
            conf = f["pose2d/conf"][()] if "pose2d/conf" in f else None
            return f["pose2d/points"][()], conf

    def read_cameras(self, stage: str) -> CameraGroup | None:
        """The rig stored by ``stage`` (``pose2d`` or ``bundle_adjustment``)."""
        with self._open() as f:
            if f is None or f"{stage}/cameras" not in f:
                return None
            return _read_cameras(f[f"{stage}/cameras"])

    def read_points(
        self, stage: str
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None] | None:
        """``(pts2d, pts3d, reproj_error)`` of a points stage, or ``None``."""
        with self._open() as f:
            if f is None or stage not in f:
                return None
            g = f[stage]
            return tuple(
                g[name][()] if name in g else None
                for name in ("points", "points3d", "reproj_error")
            )

    def read_candidates(self) -> "Candidates | None":
        """The cached top-K candidate peaks, or ``None`` if not stored."""
        from .pictorial import Candidates

        with self._open() as f:
            if f is None or "pose2d/candidates/xy" not in f:
                return None
            return Candidates(
                xy=f["pose2d/candidates/xy"][()],
                score=f["pose2d/candidates/score"][()],
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

    # -- internals -------------------------------------------------------------

    def _open(self):
        """Open the file read-only iff it exists in the current schema version."""
        import contextlib

        if not self.path.exists():
            return contextlib.nullcontext(None)
        f = h5py.File(self.path, "r")
        try:
            meta = json.loads(f.attrs.get("meta", "{}"))
        except (TypeError, ValueError):
            meta = {}
        if meta.get("deeperfly_format_version") != FORMAT_VERSION:
            f.close()
            return contextlib.nullcontext(None)
        return contextlib.closing(f)


# -- camera (de)serialization ------------------------------------------------


def _write_cameras(g: h5py.Group, cameras: CameraGroup) -> None:
    g.create_dataset("names", data=np.array(cameras.names, dtype=object), dtype=_STR)
    g.create_dataset("rvecs", data=cameras.rvecs)
    g.create_dataset("tvecs", data=cameras.tvecs)
    g.create_dataset("intrs", data=cameras.intrs)
    g.create_dataset("dists", data=cameras.dists)


def _read_cameras(g: h5py.Group) -> CameraGroup:
    names = [n.decode() if isinstance(n, bytes) else n for n in g["names"][()]]
    return CameraGroup.from_arrays(
        names, g["rvecs"][()], g["tvecs"][()], g["intrs"][()], g["dists"][()]
    )


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
    pal = g.create_group("palette")
    for name, color in s.palette.items():
        pal.attrs[name] = color


def _read_skeleton(g: h5py.Group) -> Skeleton:
    decode = lambda arr: tuple(  # noqa: E731
        x.decode() if isinstance(x, bytes) else x for x in arr
    )
    palette = {
        name: (v.decode() if isinstance(v, bytes) else v)
        for name, v in g["palette"].attrs.items()
    }
    return Skeleton(
        name=g.attrs["name"],
        point_names=decode(g["point_names"][()]),
        limb_names=decode(g["limb_names"][()]),
        limb_id=g["limb_id"][()],
        bones=g["bones"][()],
        palette=palette,
    )
