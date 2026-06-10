"""Classes for cameras and camera rigs.

A :class:`Camera` bundles the parameters to project a 3D world point to a 2D
image point under :mod:`deeperfly.geometry`'s conventions: world to camera is
``R(rvec) @ X + tvec``; the rows of ``R`` are the +x (image-right), +y
(image-down), and +z (camera-forward) axes; and intrinsics packed as
``[fx, fy, cx, cy]`` or ``[f, cx, cy]`` with ``fx = fy = f``.

A :class:`CameraGroup` is an ordered collection of named cameras, typically built
from a TOML config (see :meth:`CameraGroup.from_config`). The config describes
*only* the cameras; the wrapper in :mod:`deeperfly.bundle_adjustment` pairs a
``CameraGroup`` with a separate ``[pipeline.bundle_adjustment]`` section.

Extrinsics are specified as an orbit around a ``look_at`` target: the camera
sits ``distance`` away in the direction given by ``azimuth_deg`` /
``elevation_deg``, looks back at the target with world ``+z`` up, and
``roll_deg`` turns it about the optical axis. See :func:`resolve_extrinsics`.
(Cameras with known raw extrinsics -- e.g. bundle-adjustment output -- are built
via :meth:`CameraGroup.from_arrays`, not a config spec.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Float

from .geometry import (
    intr_to_kmat,
    project_full,
    rmat_to_rvec,
    rvec_to_rmat,
    triangulate_dlt,
)

if TYPE_CHECKING:
    from .config import Config
    from .preprocessing import FrameTransform

# World "up" that fixes the camera roll in the look-at orientation.
_WORLD_UP = np.array([0.0, 0.0, 1.0])

# The orbit spec: the only way a config specifies extrinsics. ``distance`` is
# required; the rest default to the origin / zero angles.
_ORBIT_KEYS = ("look_at", "distance", "azimuth_deg", "elevation_deg", "roll_deg")

# Extrinsics keys from the old free-form spec grammar, rejected with a pointer
# to the orbit keys rather than silently ignored.
_REMOVED_KEYS = (
    "rvec",
    "tvec",
    "rotation_matrix",
    "forward",
    "up",
    "position",
    "center",
    "eye",
)

# Per-camera keys owned by other stages (footage glob, frame preprocessing), not
# the rig geometry -- dropped before a spec reaches :meth:`Camera.from_spec`.
_NON_RIG_KEYS = ("input", "preprocess")


def _rig_keys(spec: dict) -> dict:
    """A camera spec with the non-rig keys (``input`` / ``preprocess``) removed."""
    return {k: v for k, v in spec.items() if k not in _NON_RIG_KEYS}


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _orbit_direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Unit vector from the look-at target toward the camera.

    At ``azimuth=elevation=0`` this is ``[1, 0, 0]``; azimuth rotates in the
    world xy-plane and elevation lifts toward ``+z`` -- matching the rig laid
    out by the ``get_rmat`` helper used elsewhere in the project.

    Parameters
    ----------
    azimuth_deg
        Rotation in the world xy-plane, in degrees.
    elevation_deg
        Lift toward ``+z``, in degrees.

    Returns
    -------
    np.ndarray
        A unit direction vector of shape ``(3,)``.
    """
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def _look_rotation(forward: np.ndarray) -> np.ndarray:
    """Rotation matrix (rows = camera axes) for a camera looking along ``forward``.

    ``z`` (optical axis) is ``forward``; ``x`` (image right) is
    ``normalize(cross(z, _WORLD_UP))``; ``y`` (image down) is ``cross(z, x)``.

    Parameters
    ----------
    forward
        Optical-axis direction (need not be normalized).

    Returns
    -------
    np.ndarray
        A ``(3, 3)`` rotation matrix whose rows are the camera axes.

    Raises
    ------
    ValueError
        If ``forward`` is parallel to the world up axis, which leaves the
        camera roll undefined.
    """
    z = _normalize(forward)
    x = np.cross(z, _WORLD_UP)
    norm = np.linalg.norm(x)
    if norm < 1e-9:
        raise ValueError(
            "camera looks straight along the world up axis (elevation_deg of "
            "+/-90); its orientation is ambiguous"
        )
    x = x / norm
    y = np.cross(z, x)
    return np.array([x, y, z])


def _roll_matrix(roll_deg: float) -> np.ndarray:
    """Rotation about the optical (``z``) axis by ``roll_deg``."""
    r = np.deg2rad(roll_deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def resolve_extrinsics(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Resolve an orbit camera spec to ``(rvec, tvec)``.

    The camera sits at ``look_at + distance * dir(azimuth_deg, elevation_deg)``
    and looks back at ``look_at`` with world ``+z`` up; ``roll_deg`` then turns
    it about the optical axis. ``distance`` is required; ``look_at`` defaults
    to the origin and the angles to zero.

    Parameters
    ----------
    spec
        Camera spec dict with the orbit keys above (other keys -- intrinsics,
        footage -- are ignored here).

    Returns
    -------
    rvec, tvec : np.ndarray
        The axis-angle rotation and translation, each a ``(3,)`` float array.

    Raises
    ------
    ValueError
        If ``distance`` is missing, a removed key from the old free-form spec
        grammar is given, or ``elevation_deg`` is +/-90 (camera roll undefined).
    """
    removed = [k for k in _REMOVED_KEYS if k in spec]
    if removed:
        raise ValueError(
            f"unsupported extrinsics keys {removed}: cameras are specified as "
            f"an orbit via {list(_ORBIT_KEYS)}"
        )
    if "distance" not in spec:
        raise ValueError(
            f"camera spec needs an orbit 'distance' (orbit keys: {list(_ORBIT_KEYS)})"
        )
    look_at = np.asarray(spec.get("look_at", [0.0, 0.0, 0.0]), dtype=float)
    direction = _orbit_direction(
        spec.get("azimuth_deg", 0.0), spec.get("elevation_deg", 0.0)
    )
    center = look_at + float(spec["distance"]) * direction
    rmat = _roll_matrix(spec.get("roll_deg", 0.0)) @ _look_rotation(-direction)
    rvec = np.asarray(rmat_to_rvec(rmat), dtype=float)
    return rvec, -rmat @ center


def _parse_intrinsics(
    spec: dict,
    image_size: tuple[int, int] | None = None,
    transform: "FrameTransform | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a spec dict to packed ``intr = [fx, fy, cx, cy]`` and ``dist``.

    The spec's intrinsics describe the *raw* footage frame. ``principal_point_px``
    is optional: when the spec omits it, the principal point is placed at the
    raw image center ``((w - 1) / 2, (h - 1) / 2)`` using ``image_size``. When
    the camera has a preprocess ``transform``, the resolved intrinsics are then
    mapped through it into the canonical (transformed) frame (see
    :meth:`~deeperfly.preprocessing.FrameTransform.map_intrinsics`).

    Parameters
    ----------
    spec
        Camera spec with ``focal_length_px`` (scalar or ``[fx, fy]``) and an
        optional ``principal_point_px`` / ``distortion_coefficients``, all in
        raw-frame pixels.
    image_size
        Raw footage ``(height, width)`` (as in a NumPy image array) used to
        infer the principal point when ``principal_point_px`` is absent, and to
        anchor the preprocess affine.
    transform
        The camera's preprocess :class:`~deeperfly.preprocessing.FrameTransform`
        (or ``None`` for the identity).

    Returns
    -------
    intr, dist : np.ndarray
        Packed canonical-frame intrinsics ``[fx, fy, cx, cy]`` and the
        distortion coefficients.

    Raises
    ------
    ValueError
        If a required intrinsic is missing (and cannot be inferred),
        ``focal_length_px`` is not a scalar or 2-vector, or a non-identity
        ``transform`` comes without an ``image_size`` to anchor it.
    """
    try:
        focal = np.atleast_1d(np.asarray(spec["focal_length_px"], dtype=float))
    except KeyError as exc:
        raise ValueError(f"camera spec missing intrinsic {exc}") from exc
    if "principal_point_px" in spec:
        cx, cy = (float(v) for v in spec["principal_point_px"])
    elif image_size is not None:
        height, width = image_size
        cx, cy = (width - 1) / 2, (height - 1) / 2
    else:
        raise ValueError(
            "camera spec missing intrinsic 'principal_point_px' and no image "
            "size is available to infer it from"
        )
    if focal.size == 1:
        fx = fy = float(focal[0])
    elif focal.size == 2:
        fx, fy = float(focal[0]), float(focal[1])
    else:
        raise ValueError("focal_length_px must be a scalar or [fx, fy]")
    intr = np.array([fx, fy, cx, cy])
    dist = np.asarray(spec.get("distortion_coefficients", []), dtype=float)
    if transform is not None and not transform.is_identity():
        if image_size is None:
            raise ValueError(
                "camera has a preprocess transform but no raw image size to "
                "map its intrinsics through (the op affines need the frame's "
                "height/width)"
            )
        intr = transform.map_intrinsics(intr, dist, image_size)
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
    def from_spec(
        cls,
        spec: dict,
        name: str | None = None,
        image_size: tuple[int, int] | None = None,
        transform: "FrameTransform | None" = None,
    ) -> Camera:
        """Build a camera from a config dict (see :func:`resolve_extrinsics`).

        Parameters
        ----------
        spec
            Camera spec dict (extrinsics + intrinsics keys; intrinsics in
            raw-footage pixels).
        name
            Optional camera name stored on the result.
        image_size
            Optional raw-footage ``(height, width)`` pair used to infer the
            principal point (image center) when the spec omits
            ``principal_point_px``, and to anchor ``transform``'s affine.
        transform
            The camera's preprocess transform; when given and non-identity, the
            spec's raw-frame intrinsics are mapped through it (see
            :func:`_parse_intrinsics`).

        Returns
        -------
        Camera
            The constructed camera.
        """
        rvec, tvec = resolve_extrinsics(spec)
        intr, dist = _parse_intrinsics(spec, image_size=image_size, transform=transform)
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
        """Project world points to this camera's image plane.

        Parameters
        ----------
        pts3d
            World points of shape ``(*pts, 3)``.

        Returns
        -------
        np.ndarray
            Image points of shape ``(*pts, 2)``.
        """
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
    def from_config(
        cls,
        config: "Config",
        image_sizes: dict[str, tuple[int, int]] | None = None,
    ) -> CameraGroup:
        """Build a group from a config.

        Reads ``[cameras.defaults]`` and ``[cameras.<name>]``; per-camera keys
        override the defaults. A camera here is a geometric *view*: its
        intrinsics describe its source's raw footage frame, the frame a pathway
        maps its detections back into (see
        :mod:`deeperfly.pose2d.pathways`). Detector-input geometry (mirror,
        crop, resize) lives in the pathways, not on the view.

        Parameters
        ----------
        config
            A :class:`~deeperfly.config.Config`.
        image_sizes
            Maps a view name to its source's raw footage ``(height, width)``,
            used to infer that view's principal point (image center) when
            neither the camera spec nor ``[cameras.defaults]`` specifies
            ``principal_point_px``.

        Returns
        -------
        CameraGroup
            The configured rig.

        Raises
        ------
        ValueError
            If the config defines no cameras.
        """
        defaults, specs = config.camera_table()
        image_sizes = image_sizes or {}
        cameras = {
            name: Camera.from_spec(
                _rig_keys({**defaults, **spec}),
                name=name,
                image_size=image_sizes.get(name),
            )
            for name, spec in specs.items()
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
        """Build a group from stacked per-camera arrays (e.g. BA output).

        Parameters
        ----------
        names
            Camera names, in order, labelling the leading axis of the arrays.
        rvecs, tvecs
            Stacked extrinsics of shape ``(V, 3)``.
        intrs
            Stacked packed intrinsics of shape ``(V, 4)``.
        dists
            Stacked distortion coefficients of shape ``(V, K)``.

        Returns
        -------
        CameraGroup
            The rig assembled from the arrays.
        """
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
        """Project world points through every camera.

        Parameters
        ----------
        pts3d
            World points of shape ``(*pts, 3)``.

        Returns
        -------
        np.ndarray
            Image points of shape ``(V, *pts, 2)``.
        """
        out = project_full(
            np.asarray(pts3d), self.rvecs, self.tvecs, self.intrs, self.dists
        )
        return np.asarray(out)

    def triangulate(
        self, pts2d: Float[np.ndarray, "V *pts 2"]
    ) -> Float[np.ndarray, "*pts 3"]:
        """Triangulate 3D points from 2D observations and this group's cameras.

        Parameters
        ----------
        pts2d
            2D observations of shape ``(V, *pts, 2)``, NaN for missing.

        Returns
        -------
        np.ndarray
            Triangulated points of shape ``(*pts, 3)`` (NaN below two views).
        """
        rtmat = np.concatenate(
            (np.asarray(rvec_to_rmat(self.rvecs)), self.tvecs[..., None]), axis=-1
        )
        pmats = np.asarray(intr_to_kmat(self.intrs)) @ rtmat
        return np.asarray(triangulate_dlt(np.asarray(pts2d), pmats))
