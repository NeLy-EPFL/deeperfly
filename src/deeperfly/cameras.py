"""Classes for cameras and camera rigs.

A :class:`Camera` bundles the parameters to project a 3D world point to a 2D
image point under :mod:`deeperfly.geometry`'s conventions: world to camera is
``R(rvec) @ X + tvec``; the rows of ``R`` are the +x (image-right), +y
(image-down), and +z (camera-forward) axes; and intrinsics packed as
``[fx, fy, cx, cy]`` or ``[f, cx, cy]`` with ``fx = fy = f``.

A :class:`CameraGroup` is an ordered collection of named cameras, typically built
from a TOML config (see :meth:`CameraGroup.from_config`). The config describes
*only* the cameras; the wrapper in :mod:`deeperfly.bundle_adjustment` pairs a
``CameraGroup`` with a separate ``[bundle_adjustment]`` section.

Extrinsics are specified as an orbit around a ``look_at`` target: the camera
sits ``distance`` away in the direction given by ``azimuth_deg`` /
``elevation_deg``, looks back at the target with world ``+z`` up, and
``roll_deg`` turns it about the optical axis. See :func:`resolve_extrinsics`.
(Cameras with known raw extrinsics -- e.g. bundle-adjustment output -- are built
via :meth:`CameraGroup.from_arrays`, not a config spec.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jaxtyping import Float

from .geometry import (
    backproject_ray_one,
    intr_to_kmat,
    project_full,
    rmat_to_rvec,
    rvec_to_rmat,
    triangulate_dlt,
)

if TYPE_CHECKING:
    from .config import Config

__all__ = ["Camera", "CameraGroup", "resolve_extrinsics"]

log = logging.getLogger("deeperfly")

# World "up" that fixes the camera roll in the look-at orientation.
_WORLD_UP = np.array([0.0, 0.0, 1.0])

# The orbit spec: the only way a config specifies extrinsics. ``distance`` is
# required; the rest default to the origin / zero angles.
_ORBIT_KEYS = ("look_at", "distance", "azimuth_deg", "elevation_deg", "roll_deg")

# Extrinsics keys a config might reach for but that are not supported; rejected
# with a pointer to the orbit keys rather than silently ignored.
_UNSUPPORTED_EXTRINSICS_KEYS = (
    "rvec",
    "tvec",
    "rotation_matrix",
    "forward",
    "up",
    "position",
    "center",
    "eye",
)

# The one per-camera key that is not rig geometry: the footage pattern, which belongs to
# discovery. Dropped before a spec reaches :meth:`Camera.from_spec`.
_NON_RIG_KEYS = ("video",)


def _rig_keys(spec: dict) -> dict:
    """A camera spec with the non-rig keys removed (see :data:`_NON_RIG_KEYS`)."""
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
        If ``distance`` is missing, an unsupported extrinsics key is given, or
        ``elevation_deg`` is +/-90 (camera roll undefined).
    """
    unsupported = [k for k in _UNSUPPORTED_EXTRINSICS_KEYS if k in spec]
    if unsupported:
        raise ValueError(
            f"unsupported extrinsics keys {unsupported}: a [cameras.<name>] table "
            f"describes an orbit ({list(_ORBIT_KEYS)}). To use raw, solved extrinsics "
            "-- from bundle adjustment or a calibration board -- point the config at a "
            'calibration file instead ([cameras] calibration = "calibration.toml"), or '
            "load one with CameraGroup.from_calibration"
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
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a spec dict to packed ``intr = [fx, fy, cx, cy]`` and ``dist``.

    The spec's intrinsics describe the *raw* footage frame, and stay there. That is the
    whole reason a camera needs no notion of the detector's cropping: a pathway's frame ops
    are inverted on the way back (:class:`~deeperfly.preprocessing.FrameTransform`), so
    every detection meets a camera in raw pixels however it was windowed to reach the
    model. ``principal_point_px`` is optional: when the spec omits it, the principal point
    is placed at the raw image center ``((w - 1) / 2, (h - 1) / 2)`` using ``image_size``.

    Parameters
    ----------
    spec
        Camera spec with ``focal_length_px`` (scalar or ``[fx, fy]``) and an
        optional ``principal_point_px`` / ``distortion_coefficients``, all in
        raw-frame pixels.
    image_size
        Raw footage ``(height, width)`` (as in a NumPy image array) used to
        infer the principal point when ``principal_point_px`` is absent.

    Returns
    -------
    intr, dist : np.ndarray
        Packed raw-frame intrinsics ``[fx, fy, cx, cy]`` and the distortion
        coefficients.

    Raises
    ------
    ValueError
        If a required intrinsic is missing (and cannot be inferred), or
        ``focal_length_px`` is not a scalar or 2-vector.
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
            ``principal_point_px``.

        Returns
        -------
        Camera
            The constructed camera.
        """
        rvec, tvec = resolve_extrinsics(spec)
        intr, dist = _parse_intrinsics(spec, image_size=image_size)
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

    def backproject_ray(
        self, pixel: Float[np.ndarray, "2"]
    ) -> tuple[Float[np.ndarray, "3"], Float[np.ndarray, "3"]]:
        """The world-frame viewing ray of an image ``pixel`` through this camera.

        Inverse of :meth:`project`: returns ``(origin, direction)`` such that
        every world point ``origin + s * direction`` projects back onto
        ``pixel``. ``origin`` is the camera center. See
        :func:`deeperfly.geometry.backproject_ray_one`.

        Parameters
        ----------
        pixel
            Image point of shape ``(2,)`` in pixels.

        Returns
        -------
        origin, direction : np.ndarray
            The camera center and the (unnormalized) world-frame ray direction,
            each of shape ``(3,)``.
        """
        origin, direction = backproject_ray_one(
            jnp.asarray(pixel, dtype=float),
            jnp.asarray(self.rvec),
            jnp.asarray(self.tvec),
            jnp.asarray(self.intr),
            jnp.asarray(self.dist),
        )
        return np.asarray(origin), np.asarray(direction)


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

        Reads ``[default_camera]`` and ``[cameras.<name>]``; per-camera keys
        override the shared ones. A camera here is a geometric *view*: its
        intrinsics describe its own raw footage frame, the frame detections are
        mapped back into (see :mod:`deeperfly.pose2d.pathways`), so a detection
        window (``[pose2d.crops]``) never moves the principal point.

        Parameters
        ----------
        config
            A :class:`~deeperfly.config.Config`.
        image_sizes
            Maps a view name to its source's raw footage ``(height, width)``,
            used to infer that view's principal point (image center) when
            neither the camera spec nor ``[default_camera]`` specifies
            ``principal_point_px``.

        Returns
        -------
        CameraGroup
            The configured rig.

        Raises
        ------
        ValueError
            If the config defines no cameras.

        Notes
        -----
        ``[calibration].path`` wins over the orbit specs when present: an orbit is a
        human's description of the rig they built, a calibration is a solver's
        measurement of it, and the measurement is the better rig. The ``[cameras.<name>]`` tables
        are then read only for their **order** (the ``V`` axis of every points array is
        positional), and the geometry keys in them are ignored -- logged once, naming
        which source won, so a config that carries both cannot mislead silently.
        """
        defaults, specs = config.camera_table()
        image_sizes = image_sizes or {}
        calibration = config.calibration_path()
        if calibration is not None:
            log.info(
                "cameras: using the calibration %s (the [cameras.<name>] orbit specs "
                "are read only for their order)",
                calibration,
            )
            return cls.from_calibration(
                calibration,
                names=list(specs) or None,
                image_sizes=image_sizes or None,
            )
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
    def from_calibration(
        cls,
        path,
        *,
        names: list[str] | None = None,
        image_sizes: dict[str, tuple[int, int]] | None = None,
    ) -> CameraGroup:
        """Build a group from a solved calibration file.

        The third construction path, alongside the orbit config
        (:meth:`from_config`) and raw arrays (:meth:`from_arrays`). It exists because
        :func:`resolve_extrinsics` deliberately refuses ``rvec``/``tvec`` in a camera
        spec -- a half-specified orbit must not silently carry raw extrinsics -- which
        left a *solved* rig with nowhere to live. See :mod:`deeperfly.calibration`.

        Parameters
        ----------
        path
            A ``calibration.toml``, or a directory holding one.
        names
            Optional camera names this rig must cover, in order. When given, the
            calibration is checked against them and the group is returned in *this*
            order -- the ``V`` axis of every points array is positional, so a
            calibration written in a different order must be reordered, not
            reinterpreted.
        image_sizes
            Optional ``camera_name -> (height, width)`` of the footage in hand,
            checked against the frame the intrinsics describe.

        Returns
        -------
        CameraGroup
            The calibrated rig.

        Raises
        ------
        ValueError
            If the calibration does not cover ``names``, or describes differently
            sized footage than ``image_sizes``.
        """
        from .calibration import Calibration

        cal = Calibration.load(path)
        cal.check_image_sizes(image_sizes)
        if names is None:
            return cal.cameras
        covered = cal.check_camera_names(names)
        return cls({name: cal.cameras[name] for name in covered})

    def to_calibration(
        self,
        *,
        name: str = "calibration",
        image_sizes: dict[str, tuple[int, int]] | None = None,
        units: str = "arbitrary",
        scale_source: str = "none",
        provenance: dict | None = None,
        quality: dict | None = None,
    ):
        """Wrap this rig as a saveable :class:`~deeperfly.calibration.Calibration`.

        Parameters
        ----------
        name
            Human-facing label for the calibration.
        image_sizes
            ``camera_name -> (height, width)`` of the raw footage the intrinsics
            describe (recorded so a later consumer can be refused).
        units, scale_source, provenance, quality
            See :class:`~deeperfly.calibration.Calibration`.

        Returns
        -------
        deeperfly.calibration.Calibration
            The artifact; call ``.save(path)`` to write it.
        """
        from .calibration import Calibration

        return Calibration.from_camera_group(
            self,
            name=name,
            image_sizes=image_sizes,
            units=units,
            scale_source=scale_source,
            provenance=provenance,
            quality=quality,
        )

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
        self,
        pts2d: Float[np.ndarray, "V *pts 2"],
        weights: Float[np.ndarray, "V *pts"] | None = None,
    ) -> Float[np.ndarray, "*pts 3"]:
        """Triangulate 3D points from 2D observations and this group's cameras.

        Parameters
        ----------
        pts2d
            2D observations of shape ``(V, *pts, 2)``, NaN for missing.
        weights
            Optional per-(view, point) weights of shape ``(V, *pts)`` for a
            confidence-weighted DLT; ``None`` (default) is plain DLT. See
            :func:`deeperfly.geometry.triangulate_dlt`.

        Returns
        -------
        np.ndarray
            Triangulated points of shape ``(*pts, 3)`` (NaN below two views).
        """
        rtmat = np.concatenate(
            (np.asarray(rvec_to_rmat(self.rvecs)), self.tvecs[..., None]), axis=-1
        )
        pmats = np.asarray(intr_to_kmat(self.intrs)) @ rtmat
        w = None if weights is None else np.asarray(weights)
        return np.asarray(triangulate_dlt(np.asarray(pts2d), pmats, w))
