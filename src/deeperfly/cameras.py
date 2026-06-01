"""Cameras and camera rigs (bundle-adjustment-unaware).

A :class:`Camera` bundles the parameters needed to project a 3D world point to
a 2D image point under the conventions of :mod:`deeperfly.geometry`: world to
camera is ``R(rvec) @ X + tvec``; the camera looks down its ``+z`` row, ``+y``
is image-down and ``+x`` image-right; the camera center is ``-R.T @ tvec``.

A :class:`CameraGroup` is an ordered collection of named cameras, typically
built from a TOML config (see :meth:`CameraGroup.from_config`). The config
describes *only* the cameras -- it knows nothing about bundle adjustment. The
bundle-adjustment wrapper in :mod:`deeperfly.bundle_adjustment` consumes a
``CameraGroup`` together with a separate ``[bundle_adjustment]`` config section.

Orientation and position can be specified in whatever combination is most
convenient -- a Rodrigues vector, a rotation matrix, a forward/up axis pair, or
an orbit around a ``look_at`` target (azimuth / elevation / roll / distance) --
as long as the supplied keys are not in conflict. Everything is resolved to a
single ``(rvec, tvec)`` pair, so the rest of the pipeline only ever sees the
canonical extrinsics. See :func:`resolve_extrinsics` for the exact rules.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float

from .geometry import (
    intr_to_kmat,
    project_full,
    rmat_to_rvec,
    rvec_to_rmat,
    triangulate_dlt,
)

# Default world "up" used to disambiguate look-at / orbit orientations.
_WORLD_UP = np.array([0.0, 0.0, 1.0])

# Keys recognized by the extrinsic resolver, grouped by the quantity they fix.
_ROTATION_KEYS = ("rvec", "rotation_matrix", "forward")
_CENTER_KEYS = ("position", "center", "eye")


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _orbit_direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Unit vector from the look-at target toward the camera.

    At ``azimuth=elevation=0`` this is ``[1, 0, 0]``; azimuth rotates in the
    world xy-plane and elevation lifts toward ``+z`` -- matching the rig laid
    out by the ``get_rmat`` helper used elsewhere in the project.
    """
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def _look_rotation(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Rotation matrix (rows = camera axes) for a camera looking along ``forward``.

    ``z`` (optical axis) is ``forward``; ``x`` (image right) is
    ``normalize(cross(z, up))``; ``y`` (image down) is ``cross(z, x)``.
    """
    z = _normalize(forward)
    x = _normalize(np.cross(z, up))
    y = np.cross(z, x)
    return np.array([x, y, z])


def _roll_matrix(roll_deg: float) -> np.ndarray:
    """Rotation about the optical (``z``) axis by ``roll_deg``."""
    r = np.deg2rad(roll_deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def resolve_extrinsics(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a camera spec dict to ``(rvec, tvec)``.

    The rotation is taken from at most one explicit source -- ``rvec``,
    ``rotation_matrix``, or ``forward`` (+ optional ``up``) -- otherwise from a
    look-at orientation implied by ``look_at`` / ``azimuth_deg`` /
    ``elevation_deg``. The camera center is taken from at most one of an
    explicit ``position`` (aka ``center`` / ``eye``) or an orbit
    ``look_at + distance * dir(azimuth, elevation)``; alternatively ``tvec`` may
    be given directly. ``roll_deg`` (default 0) composes on top of any rotation
    as a turn about the optical axis. Conflicting sources raise ``ValueError``.

    Returns
    -------
    ``(rvec, tvec)`` as ``(3,)`` float arrays.
    """
    up = np.asarray(spec.get("up", _WORLD_UP), dtype=float)

    # --- camera center (world position) -------------------------------------
    center_keys = [k for k in _CENTER_KEYS if k in spec]
    has_orbit = "distance" in spec
    has_tvec = "tvec" in spec
    if len(center_keys) > 1:
        raise ValueError(f"conflicting position keys: {center_keys}")
    if sum([bool(center_keys), has_orbit, has_tvec]) > 1:
        given = (
            center_keys
            + (["distance"] if has_orbit else [])
            + (["tvec"] if has_tvec else [])
        )
        raise ValueError(f"conflicting position/translation keys: {given}")

    center = None
    if center_keys:
        center = np.asarray(spec[center_keys[0]], dtype=float)
    elif has_orbit:
        look_at = np.asarray(spec.get("look_at", [0.0, 0.0, 0.0]), dtype=float)
        direction = _orbit_direction(
            spec.get("azimuth_deg", 0.0), spec.get("elevation_deg", 0.0)
        )
        center = look_at + float(spec["distance"]) * direction

    # --- rotation -----------------------------------------------------------
    rot_keys = [k for k in _ROTATION_KEYS if k in spec]
    if len(rot_keys) > 1:
        raise ValueError(f"conflicting rotation keys: {rot_keys}")
    implies_look_at = any(
        k in spec for k in ("look_at", "azimuth_deg", "elevation_deg")
    )

    if rot_keys == ["rvec"]:
        rmat = np.asarray(rvec_to_rmat(np.asarray(spec["rvec"], dtype=float)))
    elif rot_keys == ["rotation_matrix"]:
        rmat = np.asarray(spec["rotation_matrix"], dtype=float).reshape(3, 3)
    elif rot_keys == ["forward"]:
        rmat = _look_rotation(np.asarray(spec["forward"], dtype=float), up)
    elif implies_look_at:
        if center is None:
            raise ValueError(
                "look-at orientation needs a camera position; give 'distance' "
                "(orbit) or 'position', or specify orientation directly via "
                "'rvec' / 'rotation_matrix' / 'forward'"
            )
        target = np.asarray(spec.get("look_at", [0.0, 0.0, 0.0]), dtype=float)
        rmat = _look_rotation(target - center, up)
    else:
        raise ValueError(
            "no rotation source: provide one of 'rvec', 'rotation_matrix', "
            "'forward', or a look-at ('look_at' / 'azimuth_deg' / "
            "'elevation_deg')"
        )

    if "roll_deg" in spec:
        rmat = _roll_matrix(spec["roll_deg"]) @ rmat

    # --- translation --------------------------------------------------------
    if has_tvec:
        tvec = np.asarray(spec["tvec"], dtype=float)
    else:
        if center is None:
            raise ValueError(
                "no position source: provide 'tvec', 'position', or an orbit 'distance'"
            )
        tvec = -rmat @ center

    rvec = np.asarray(rmat_to_rvec(rmat), dtype=float)
    return rvec, tvec


def _parse_intrinsics(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a spec dict to packed ``intr = [fx, fy, cx, cy]`` and ``dist``."""
    try:
        focal = np.atleast_1d(np.asarray(spec["focal_length_px"], dtype=float))
        cx, cy = (float(v) for v in spec["principal_point_px"])
    except KeyError as exc:
        raise ValueError(f"camera spec missing intrinsic {exc}") from exc
    if focal.size == 1:
        fx = fy = float(focal[0])
    elif focal.size == 2:
        fx, fy = float(focal[0]), float(focal[1])
    else:
        raise ValueError("focal_length_px must be a scalar or [fx, fy]")
    intr = np.array([fx, fy, cx, cy])
    dist = np.asarray(spec.get("distortion_coefficients", []), dtype=float)
    return intr, dist


@dataclass
class Camera:
    """A single camera: extrinsics, intrinsics, and lens distortion.

    ``intr`` is always the 4-vector ``[fx, fy, cx, cy]`` (so every camera in a
    group has the same intrinsic layout); ``dist`` holds OpenCV-ordered
    distortion coefficients (possibly empty).
    """

    rvec: Float[np.ndarray, "3"]
    tvec: Float[np.ndarray, "3"]
    intr: Float[np.ndarray, "4"]
    dist: Float[np.ndarray, "K"]
    name: str | None = None

    @classmethod
    def from_spec(cls, spec: dict, name: str | None = None) -> Camera:
        """Build a camera from a config dict (see :func:`resolve_extrinsics`)."""
        rvec, tvec = resolve_extrinsics(spec)
        intr, dist = _parse_intrinsics(spec)
        return cls(rvec=rvec, tvec=tvec, intr=intr, dist=dist, name=name)

    @property
    def rmat(self) -> Float[np.ndarray, "3 3"]:
        return np.asarray(rvec_to_rmat(self.rvec))

    @property
    def kmat(self) -> Float[np.ndarray, "3 3"]:
        return np.asarray(intr_to_kmat(self.intr))

    @property
    def position(self) -> Float[np.ndarray, "3"]:
        """Camera center in world coordinates, ``-R.T @ tvec``."""
        return -self.rmat.T @ self.tvec

    def project(
        self, pts3d: Float[np.ndarray, "*pts 3"]
    ) -> Float[np.ndarray, "*pts 2"]:
        """Project world points to this camera's image plane."""
        out = project_full(
            np.asarray(pts3d),
            self.rvec[None],
            self.tvec[None],
            self.intr[None],
            self.dist[None],
        )
        return np.asarray(out)[0]


class CameraGroup:
    """An ordered collection of named :class:`Camera` objects."""

    def __init__(self, cameras: dict[str, Camera]):
        self.cameras = dict(cameras)

    def __len__(self) -> int:
        return len(self.cameras)

    def __getitem__(self, name: str) -> Camera:
        return self.cameras[name]

    def __iter__(self):
        return iter(self.cameras.values())

    @property
    def names(self) -> list[str]:
        return list(self.cameras)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict | str | Path) -> CameraGroup:
        """Build a group from a config dict or a path to a TOML file.

        Reads ``[camera_defaults]`` and ``[cameras.<name>]``; per-camera keys
        override the defaults. Any ``[bundle_adjustment]`` section is ignored
        here -- this class is bundle-adjustment-unaware.
        """
        if not isinstance(config, dict):
            with open(config, "rb") as f:
                config = tomllib.load(f)
        defaults = config.get("camera_defaults", {})
        cameras = {
            name: Camera.from_spec({**defaults, **spec}, name=name)
            for name, spec in config.get("cameras", {}).items()
        }
        if not cameras:
            raise ValueError("config defines no cameras")
        return cls(cameras)

    @classmethod
    def from_arrays(
        cls,
        names: list[str],
        rvecs: Float[np.ndarray, "V 3"],
        tvecs: Float[np.ndarray, "V 3"],
        intrs: Float[np.ndarray, "V 4"],
        dists: Float[np.ndarray, "V K"],
    ) -> CameraGroup:
        """Build a group from stacked per-camera arrays (e.g. BA output)."""
        rvecs, tvecs, intrs, dists = map(np.asarray, (rvecs, tvecs, intrs, dists))
        cameras = {
            name: Camera(
                rvec=rvecs[i], tvec=tvecs[i], intr=intrs[i], dist=dists[i], name=name
            )
            for i, name in enumerate(names)
        }
        return cls(cameras)

    # -- stacked parameter views --------------------------------------------

    @property
    def rvecs(self) -> Float[np.ndarray, "V 3"]:
        return np.stack([c.rvec for c in self])

    @property
    def tvecs(self) -> Float[np.ndarray, "V 3"]:
        return np.stack([c.tvec for c in self])

    @property
    def intrs(self) -> Float[np.ndarray, "V 4"]:
        return np.stack([c.intr for c in self])

    @property
    def dists(self) -> Float[np.ndarray, "V K"]:
        """Per-camera distortion, zero-padded to the group-wide max length."""
        k = max((c.dist.size for c in self), default=0)
        out = np.zeros((len(self), k))
        for i, c in enumerate(self):
            out[i, : c.dist.size] = c.dist
        return out

    # -- geometry ------------------------------------------------------------

    def project(
        self, pts3d: Float[np.ndarray, "*pts 3"]
    ) -> Float[np.ndarray, "V *pts 2"]:
        """Project world points through every camera (shape ``(V, *pts, 2)``)."""
        out = project_full(
            np.asarray(pts3d), self.rvecs, self.tvecs, self.intrs, self.dists
        )
        return np.asarray(out)

    def triangulate(
        self, pts2d: Float[np.ndarray, "V *pts 2"]
    ) -> Float[np.ndarray, "*pts 3"]:
        """Triangulate 3D points from 2D observations and this group's cameras."""
        rtmat = np.concatenate(
            (np.asarray(rvec_to_rmat(self.rvecs)), self.tvecs[..., None]), axis=-1
        )
        pmats = np.asarray(intr_to_kmat(self.intrs)) @ rtmat
        return np.asarray(triangulate_dlt(np.asarray(pts2d), pmats))
