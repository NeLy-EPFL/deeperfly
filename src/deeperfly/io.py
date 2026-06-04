"""Self-contained HDF5 result container for the pose pipeline.

A :class:`PoseResult` bundles everything produced for one recording -- the
calibrated cameras, the skeleton, the 2D detections + confidences, and the
triangulated (and optionally smoothed) 3D points -- plus free-form metadata. The
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

import h5py
import numpy as np
from jaxtyping import Float

from .cameras import CameraGroup
from .skeleton import Skeleton

FORMAT_VERSION = 1
_STR = h5py.string_dtype("utf-8")


@dataclass
class PoseResult:
    """A complete multi-view pose-estimation result for one recording."""

    cameras: CameraGroup
    skeleton: Skeleton
    pts2d: Float[np.ndarray, "V T N 2"]
    conf: Float[np.ndarray, "V T N"] | None = None
    pts3d: Float[np.ndarray, "T N 3"] | None = None
    pts3d_smoothed: Float[np.ndarray, "T N 3"] | None = None
    reproj_error: Float[np.ndarray, "V T N"] | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pts2d = np.asarray(self.pts2d, dtype=float)
        for name in ("conf", "pts3d", "pts3d_smoothed", "reproj_error"):
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
        """Write the result to an HDF5 file (overwriting ``path``)."""
        meta = {
            "deeperfly_format_version": FORMAT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            **self.meta,
        }
        with h5py.File(path, "w") as f:
            f.attrs["meta"] = json.dumps(meta)
            _write_cameras(f.create_group("cameras"), self.cameras)
            _write_skeleton(f.create_group("skeleton"), self.skeleton)
            g2d = f.create_group("pose2d")
            g2d.create_dataset("points", data=self.pts2d)
            if self.conf is not None:
                g2d.create_dataset("conf", data=self.conf)
            if self.pts3d is not None or self.pts3d_smoothed is not None:
                g3d = f.create_group("pose3d")
                if self.pts3d is not None:
                    g3d.create_dataset("points", data=self.pts3d)
                if self.pts3d_smoothed is not None:
                    g3d.create_dataset("points_smoothed", data=self.pts3d_smoothed)
            if self.reproj_error is not None:
                f.create_group("diagnostics").create_dataset(
                    "reproj_error", data=self.reproj_error
                )

    @classmethod
    def load(cls, path: str | Path) -> PoseResult:
        """Read a :class:`PoseResult` back from an HDF5 file."""
        with h5py.File(path, "r") as f:
            meta = json.loads(f.attrs["meta"])
            cameras = _read_cameras(f["cameras"])
            skeleton = _read_skeleton(f["skeleton"])
            pts2d = f["pose2d/points"][()]
            conf = f["pose2d/conf"][()] if "conf" in f["pose2d"] else None
            pts3d = f["pose3d/points"][()] if "pose3d/points" in f else None
            pts3d_sm = (
                f["pose3d/points_smoothed"][()]
                if "pose3d/points_smoothed" in f
                else None
            )
            reproj = (
                f["diagnostics/reproj_error"][()]
                if "diagnostics/reproj_error" in f
                else None
            )
        meta.pop("deeperfly_format_version", None)
        return cls(
            cameras=cameras,
            skeleton=skeleton,
            pts2d=pts2d,
            conf=conf,
            pts3d=pts3d,
            pts3d_smoothed=pts3d_sm,
            reproj_error=reproj,
            meta=meta,
        )


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
        "joint_names", data=np.array(s.joint_names, dtype=object), dtype=_STR
    )
    g.create_dataset(
        "limb_names", data=np.array(s.limb_names, dtype=object), dtype=_STR
    )
    g.create_dataset("limb_id", data=s.limb_id)
    g.create_dataset("bones", data=s.bones)
    pal = g.create_group("palette")
    for name, color in s.palette.items():
        pal.attrs[name] = color
    vis = g.create_group("visibility")
    for name, idx in s.visibility.items():
        vis.create_dataset(name, data=idx)


def _read_skeleton(g: h5py.Group) -> Skeleton:
    decode = lambda arr: tuple(  # noqa: E731
        x.decode() if isinstance(x, bytes) else x for x in arr
    )
    visibility = {name: g["visibility"][name][()] for name in g.get("visibility", {})}
    palette = {
        name: (v.decode() if isinstance(v, bytes) else v)
        for name, v in g["palette"].attrs.items()
    }
    return Skeleton(
        name=g.attrs["name"],
        joint_names=decode(g["joint_names"][()]),
        limb_names=decode(g["limb_names"][()]),
        limb_id=g["limb_id"][()],
        bones=g["bones"][()],
        palette=palette,
        visibility=visibility,
    )
