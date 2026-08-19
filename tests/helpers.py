"""Shared constants and helpers for the test suite.

Importable as a top-level module thanks to ``pythonpath = ["tests"]`` in
``pyproject.toml``. Pure helpers live here; pytest fixtures live in
``conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from deeperfly import geometry as geom
from deeperfly.cameras import CameraGroup
from deeperfly.config import Config

# Reference rig parameters.
FOCAL_PX = 22388.125
DISTANCE_MM = 107.463
WIDTH, HEIGHT = 1024, 512
AZIMUTHS_DEG = [-120, -90, -45, 0, 45, 90, 120]
CAMERA_NAMES = ["rh", "rm", "rf", "f", "lf", "lm", "lh"]


def leg_indices(skeleton, side: str) -> np.ndarray:
    """Point indices of one body side's leg points (``side`` is ``"r"`` or ``"l"``).

    Leg points are named ``"{side}{f|m|h}_..."`` (front / mid / hind leg); the
    antennae and abdominal markers (``"{side}_..."``) are excluded.
    """
    return np.array(
        [
            i
            for i, name in enumerate(skeleton.point_names)
            if name[:1] == side and name[1:2] in "fmh"
        ],
        dtype=np.int64,
    )


def output_points_table(point_names, specs):
    """Build a ``[pose2d.output_points.<view>]`` mapping from per-pathway channel lists.

    Parameters
    ----------
    point_names
        The skeleton's ordered point names.
    specs
        Iterable of ``(view, pathway, points)`` where ``points[i]`` is the point
        index output channel ``i`` of ``pathway`` fills in ``view`` (``-1`` drops
        the channel).

    Returns
    -------
    dict
        ``{view: {point_name: {"pathway": ..., "out_channel": i}}}``.
    """
    table: dict[str, dict] = {}
    for view, pathway, points in specs:
        entries = table.setdefault(view, {})
        for ch, p in enumerate(points):
            if p >= 0:
                entries[point_names[p]] = {"pathway": pathway, "out_channel": ch}
    return table


def reference_rmat(yaw_rad: float) -> np.ndarray:
    """Reference world->camera rotation for a camera at azimuth ``yaw_rad``.

    This is the project's ground-truth convention that :mod:`deeperfly.cameras`
    must reproduce from an orbit (``look_at`` / ``azimuth`` / ``distance``) spec.
    The camera looks toward the origin with image-down along world ``-z``.
    """
    y = np.array([0.0, 0.0, -1.0])
    z = np.array([-np.cos(yaw_rad), -np.sin(yaw_rad), 0.0])
    return np.array([np.cross(y, z), y, z])


def rig_arrays() -> dict:
    """The canonical 7-camera orbit rig as plain arrays.

    Returns a dict with ``names``, ``rvecs``, ``tvecs``, ``intrs`` (4-vector
    ``[fx, fy, cx, cy]``) and ``dists`` (empty, i.e. no distortion).

    Lives here rather than only in the ``rig`` fixture so that module- or
    session-scoped fixtures can build the rig too: a widened fixture may not depend
    on a function-scoped one, and duplicating the construction would let the copy
    drift away from the reference convention.
    """
    cx, cy = (WIDTH - 1) / 2, (HEIGHT - 1) / 2
    rmats = np.array([reference_rmat(t) for t in np.deg2rad(AZIMUTHS_DEG)])
    rvecs = np.asarray(geom.rmat_to_rvec(rmats))
    tvecs = np.array([[0.0, 0.0, DISTANCE_MM]] * len(rmats))
    intrs = np.tile([FOCAL_PX, FOCAL_PX, cx, cy], (len(rmats), 1))
    dists = np.zeros((len(rmats), 0))
    return {
        "names": CAMERA_NAMES,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "intrs": intrs,
        "dists": dists,
    }


def make_cameras() -> CameraGroup:
    """:func:`rig_arrays` as a :class:`~deeperfly.cameras.CameraGroup`."""
    r = rig_arrays()
    return CameraGroup.from_arrays(
        r["names"], r["rvecs"], r["tvecs"], r["intrs"], r["dists"]
    )


def small_rotation(sigma: float, seed: int) -> np.ndarray:
    """A random rotation matrix close to identity (axis-angle std ``sigma``)."""
    from scipy.linalg import expm

    o = np.random.default_rng(seed).normal(scale=sigma, size=3)
    skew = np.array([[0, -o[2], o[1]], [o[2], 0, -o[0]], [-o[1], o[0], 0]])
    return expm(skew)


#: A 19-channel, one-side-per-pass plan over ``fly38`` -- the shape deeperfly shipped
#: before the dense detectors. Kept as test data rather than read from the packaged
#: config, because the packaged plan is now DENSE: every view sees every point, so its
#: visibility mask is all-True and cannot express the thing these tests are about.
SPARSE_CONFIG_PATH = Path(__file__).parent / "data" / "fly38_sparse_config.toml"


def sparse_config() -> Config:
    """The 19-channel ``fly38`` plan, for tests about PARTIAL per-view visibility."""
    return Config.from_toml(SPARSE_CONFIG_PATH)


#: The historical DeepFly3D 38-point set, retired as a packaged skeleton by the 0.2
#: release and kept here as TEST DATA (the file says why).
DEEPFLY3D_SKELETON_PATH = Path(__file__).parent / "data" / "fly38_deepfly3d.toml"


def fly38_skeleton():
    """The packaged ``fly38`` preset, loaded by name.

    The same skeleton :func:`deeperfly.skeleton.Skeleton.fly` returns, but reached through
    the config layer -- so a test using this one is also asserting that the preset
    reference resolves.
    """
    from deeperfly.skeleton import Skeleton

    return Skeleton.from_config(Config.from_dict({"skeleton": {"name": "fly38"}}))


def deepfly3d_skeleton():
    """The retired DeepFly3D point set, for tests defined against ITS layout.

    Two things in the suite are: the recorded IK baseline
    (:data:`IK_BASELINE_PATH`, whose 38 columns have no names beside them and mean what
    this order says) and the chirality QC tests, which build a deliberately-mirrored pose
    as ``concatenate([left19, right19])`` -- expressible only on a skeleton whose two
    halves are contiguous index blocks.

    Loaded by PATH: it is no longer a packaged preset, and ``fly38`` now names a different
    point set.
    """
    from deeperfly.skeleton import Skeleton

    return Skeleton.from_config(
        Config.from_dict({"skeleton": {"file": str(DEEPFLY3D_SKELETON_PATH)}})
    )


def fly_masked(pts2d: np.ndarray) -> np.ndarray:
    """NaN-out the ``(view, point)`` pairs a **one-side** detector does not observe.

    Visibility is the union of :func:`sparse_config`'s pathway maps: each side camera
    sees only its own body half, and the front view is bridged by two pathways. That
    partial coverage is the premise of every test that calls this -- a joint seen by two
    or three cameras behaves differently under an outlier than one seen by seven.

    The leading axis must be the 7 fly views in order (rh, rm, rf, f, lf, lm, lh).
    """
    mask = sparse_config().detection_plan().visibility_mask()  # (7, 38)
    m = mask.reshape((mask.shape[0], *([1] * (pts2d.ndim - 3)), mask.shape[1]))
    return np.where(m[..., None], pts2d, np.nan)


#: The recorded pre-QuickIK solver output, and the two input poses it was measured
#: on (see ``test_ik_baseline.py``). Also the source of a real, fully-triangulated pose
#: for tests that want one -- ``examples/data/**/results.h5`` is git-ignored, so it is
#: not available to the suite.
IK_BASELINE_PATH = Path(__file__).parent / "data" / "ik_baseline_scipy.npz"


# -- inverse kinematics: synthetic poses --------------------------------------
#
# Shared by the solver-free geometry tests and the QuickIK solver tests, so both
# exercise the same construction. Deliberately numpy-only (no QuickIK, no JAX): a
# pose built here is exactly reachable by the model, which is what makes it usable as
# ground truth for a solve.

#: Front-leg segment lengths (NeuroMechFly units), shared by all six synthetic legs.
IK_SEGLENS = np.array([0.0, 0.40, 0.69, 0.54, 0.63])

#: Plausible coxa positions for the six legs, in a body-aligned world frame.
IK_COXAE = {
    "lf": [1.0, 0.5, 0.0],
    "rf": [1.0, -0.5, 0.0],
    "lm": [0.0, 0.6, 0.0],
    "rm": [0.0, -0.6, 0.0],
    "lh": [-1.0, 0.5, 0.0],
    "rh": [-1.0, -0.5, 0.0],
}


def bent_angles(chain, rng, frac=(0.3, 0.7)) -> np.ndarray:
    """Random joint angles within ``frac`` of each DOF's range (a non-singular pose)."""
    lo, hi = chain.bounds
    return lo + (hi - lo) * rng.uniform(frac[0], frac[1], size=len(lo))


def synth_leg_pose(template, skeleton, rng, r_body=None, n_frames=3):
    """A synthetic 3D pose: each leg placed by forward kinematics from known angles.

    Returns ``(pts3d (T, P, 3), truth {leg_name: angles})``. Only leg points are
    filled; every other skeleton point stays NaN.
    """
    from deeperfly.inverse_kinematics.forward import leg_fk

    r_body = np.eye(3) if r_body is None else np.asarray(r_body, dtype=float)
    index = {n: i for i, n in enumerate(skeleton.point_names)}
    pts3d = np.full((n_frames, skeleton.n_points, 3), np.nan)
    truth = {}
    for leg in template.legs:
        angles = bent_angles(leg, rng)
        truth[leg.name] = angles
        local = leg_fk(angles, leg.axes, IK_SEGLENS, leg.dof_counts)
        world = local @ r_body.T + np.array(IK_COXAE[leg.name])
        for j, name in enumerate(leg.point_names):
            pts3d[:, index[name]] = world[j]
    return pts3d, truth


def place_chain_markers(chain, theta, sim, pts, index, size=1.0, shift=None):
    """Fill ``pts`` with a chain's markers, FK'd by ``theta`` then placed by ``sim``.

    ``size`` grows the chain about its base before the forward kinematics -- the same
    transform the body plan bakes in and the overlay mesh applies -- so a recording can
    be synthesized for a head/abdomen that differs in size from the model geometry.
    ``shift`` then translates the grown chain in the model frame, synthesizing an animal
    whose chain base does not sit where the model's does -- which is the normal case for
    the head, and what the chain's base marker measures.

    The **anchors scale with the markers**, which is what ``_chain_joints`` does and is
    load-bearing for a serial chain: rotating a grown marker about an un-grown anchor is
    a different pose, so a synthesized abdomen would then be unreachable by the plan that
    is supposed to reproduce it. The head cannot show this (its three anchors all sit on
    its base, which scaling leaves alone) and neither can a straight chain (no rotation
    to be about the wrong point), so only a **bent** resized abdomen catches it.

    A marker the skeleton does not label is skipped rather than raising: the packaged
    model and the run's skeleton are allowed to disagree about which points exist (the
    abdomen markers are fly38b's, and ``fly38`` labels neither them nor the ``neck``),
    and the body plan itself handles that by leaving the joint untracked.
    """
    from dataclasses import replace

    from deeperfly.inverse_kinematics.forward import chain_affine

    rot, scale, trans = sim
    base = np.asarray(chain.anchors[0], dtype=float)
    delta = np.zeros(3) if shift is None else np.asarray(shift, dtype=float)
    grown = replace(
        chain,
        anchors=base + size * (np.asarray(chain.anchors, dtype=float) - base),
        marker_neutral=base
        + size * (np.asarray(chain.marker_neutral, dtype=float) - base),
    )
    for k, (name, depth) in enumerate(zip(grown.marker_names, grown.marker_depth)):
        if name not in index:
            continue
        a, b = chain_affine(grown, depth, theta)
        pts[:, index[name]] = (
            scale * (rot @ (a @ grown.marker_neutral[k] + b + delta)) + trans
        )


def rot_z(angle: float) -> np.ndarray:
    """Rotation about the world z axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
