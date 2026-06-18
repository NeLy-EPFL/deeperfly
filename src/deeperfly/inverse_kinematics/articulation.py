"""The baked head + abdomen articulation: chains the IK fits beyond the legs.

The legs are serial chains whose joints *are* tracked keypoints, so they are fit
with measured segment lengths (:mod:`deeperfly.inverse_kinematics.align`). The head
and abdomen instead articulate joints that are **not** keypoints (a neck pivot; the
abdominal segment hinges), with fixed geometry that must come from the model. This
module loads that geometry -- baked once from the NeuroMechFly MJCF into
``data/nmf_articulation.json`` by ``scripts/build_nmf_mesh_asset.py`` -- as a set of
:class:`Chain` objects:

- the **head** is a 3-DOF chain (yaw / pitch / roll) at one anchor, carrying the two
  antenna-tip markers;
- the **abdomen** is a 5-DOF sagittal pitch chain, carrying the six abdomen markers
  at the chain depths where they attach.

A recording is registered to the model by one similarity transform
(:func:`body_similarity`) fit from the six thorax-coxa keypoints (which are
body-fixed); the chain angles are then solved in the model frame
(:func:`deeperfly.inverse_kinematics.core.solve_chain`). The same chains pose the
head/abdomen of the overlay mesh, so the angles and the mesh stay consistent.
"""

from __future__ import annotations

import functools
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "Chain",
    "Articulation",
    "load_articulation",
    "body_similarity",
    "estimate_chain_scale",
    "chain_affine",
    "DEFAULT_ARTICULATION_PATH",
]

#: The packaged baked articulation asset (built by ``scripts/build_nmf_mesh_asset.py``).
DEFAULT_ARTICULATION_PATH = (
    Path(__file__).parent.parent / "data" / "nmf_articulation.json"
)

_EPS = 1e-9


@dataclass(frozen=True)
class Chain:
    """One baked articulation chain (the head or the abdomen).

    Attributes
    ----------
    name
        ``"head"`` or ``"abdomen"``.
    dof_names
        ``(D,)`` the fitted joint-angle names (the flygym joint names), in chain order
        (e.g. ``("c_thorax-c_head-yaw", "c_thorax-c_head-pitch", "c_thorax-c_head-roll")``).
    anchors, axes
        ``(D, 3)`` neutral world anchor and unit axis of each joint.
    bounds
        ``(lo, hi)`` radian bound arrays of shape ``(D,)``.
    marker_names
        ``(M,)`` the tracked keypoints this chain predicts (skeleton point names).
    marker_neutral
        ``(M, 3)`` each marker's neutral model position.
    marker_depth
        ``(M,)`` each marker's chain depth (number of proximal joints that move it).
    """

    name: str
    dof_names: tuple[str, ...]
    anchors: np.ndarray
    axes: np.ndarray
    bounds: tuple[np.ndarray, np.ndarray]
    marker_names: tuple[str, ...]
    marker_neutral: np.ndarray
    marker_depth: tuple[int, ...]


@dataclass(frozen=True)
class Articulation:
    """The loaded head/abdomen chains plus the neutral coxae that register the body.

    ``bodies`` maps each model body a marker may attach to (the abdomen segments and
    the head subtree) to its neutral world frame and chain/depth -- used to recompute
    a marker's neutral position when the run config overrides its offset (see
    :meth:`load`). Empty for an older asset baked before body frames were stored.
    """

    chains: tuple[Chain, ...]
    coxa_points: tuple[str, ...]
    coxa_neutral: np.ndarray  # (6, 3) neutral thorax-coxa positions, in coxa order
    bodies: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        ref: str | Path = DEFAULT_ARTICULATION_PATH,
        *,
        fit: tuple[str, ...] | None = None,
        bounds_overrides: dict[str, tuple[float, float]] | None = None,
        marker_overrides: dict[str, dict[str, dict]] | None = None,
    ) -> "Articulation":
        """Load the baked articulation, optionally restricting chains / overriding bounds.

        Parameters
        ----------
        ref
            Path to the articulation JSON (defaults to the packaged asset).
        fit
            Which chains to keep (subset of ``{"head", "abdomen"}``); ``None`` = all.
        bounds_overrides
            ``"<dof>" -> (lo_deg, hi_deg)`` degree overrides keyed by the flygym joint
            angle name (e.g. ``{"c_thorax-c_head-pitch": (-30, 30)}``),
            case-insensitive.
        marker_overrides
            ``chain_name -> {point_name: {"body", "offset", "depth"?}}`` redefining a
            chain's markers -- *where* each tracked keypoint sits relative to the model
            (the labeling-scheme choice). When present for a chain, the table **replaces**
            that chain's baked markers: each marker's neutral position is recomputed as
            ``body_frame @ offset`` from the baked :attr:`bodies` frame, and its depth is
            the attachment body's chain depth (or an explicit ``depth``). Lets the IK be
            retargeted to a different labeling scheme without re-running MuJoCo.
        """
        spec = json.loads(Path(ref).read_text())
        overrides = {k.lower(): v for k, v in (bounds_overrides or {}).items()}
        bodies = spec.get("bodies", {})
        markers = marker_overrides or {}
        if markers and not bodies:
            raise ValueError(
                "[inverse_kinematics] marker overrides need the attachment-body frames, "
                "but this articulation asset predates them; rebuild it with "
                "scripts/build_nmf_mesh_asset.py"
            )
        chains = []
        for c in spec["chains"]:
            if fit is not None and c["name"] not in fit:
                continue
            resolved = _resolve_markers(c, markers.get(c["name"]), bodies)
            chains.append(_build_chain(c, overrides, resolved))
        return cls(
            chains=tuple(chains),
            coxa_points=tuple(spec["coxa_points"]),
            coxa_neutral=np.asarray(spec["coxa_neutral"], dtype=float),
            bodies=bodies,
        )

    def chain(self, name: str) -> Chain | None:
        return next((c for c in self.chains if c.name == name), None)

    @property
    def dof_names(self) -> list[str]:
        return [n for c in self.chains for n in c.dof_names]


def _build_chain(
    c: dict, overrides: dict[str, tuple[float, float]], markers: list[dict]
) -> Chain:
    joints = c["joints"]
    dof_names = tuple(j["angle"] for j in joints)
    anchors = np.asarray([j["anchor"] for j in joints], dtype=float)
    axes = np.asarray([j["axis"] for j in joints], dtype=float)
    lo, hi = [], []
    for j in joints:
        key = j["angle"].lower()
        b = overrides.get(key, j["bounds_deg"])
        lo.append(np.deg2rad(float(b[0])))
        hi.append(np.deg2rad(float(b[1])))
    return Chain(
        name=c["name"],
        dof_names=dof_names,
        anchors=anchors,
        axes=axes,
        bounds=(np.asarray(lo), np.asarray(hi)),
        marker_names=tuple(m["point"] for m in markers),
        marker_neutral=np.asarray([m["neutral"] for m in markers], dtype=float),
        marker_depth=tuple(int(m["depth"]) for m in markers),
    )


def _resolve_markers(
    c: dict, override: dict[str, dict] | None, bodies: dict[str, dict]
) -> list[dict]:
    """The chain's markers: the baked set, or a config override's set.

    Without an override the chain keeps its baked markers. With one, the table
    *replaces* them: each marker's neutral world position is recomputed from the
    attachment body's baked frame (``pos + mat @ offset``) and its depth is the body's
    chain depth (or an explicit ``depth``). Used to retarget the IK to a different
    labeling scheme -- e.g. abdomen points placed at different offsets.
    """
    if not override:
        return list(c["markers"])
    out: list[dict] = []
    for point, spec in override.items():
        body = spec.get("body")
        if body not in bodies:
            raise ValueError(
                f"[inverse_kinematics.{c['name']}] marker {point!r} attaches to unknown "
                f"body {body!r}; the {c['name']} chain's bodies are "
                f"{sorted(b for b, v in bodies.items() if v.get('chain') == c['name'])}"
            )
        frame = bodies[body]
        if frame.get("chain") != c["name"] and spec.get("depth") is None:
            raise ValueError(
                f"[inverse_kinematics.{c['name']}] marker {point!r} attaches to body "
                f"{body!r}, which belongs to the {frame.get('chain')!r} chain; give an "
                f"explicit depth to use it on the {c['name']} chain anyway"
            )
        offset = np.asarray(spec.get("offset", (0.0, 0.0, 0.0)), dtype=float)
        pos = np.asarray(frame["pos"], dtype=float)
        mat = np.asarray(frame["mat"], dtype=float).reshape(3, 3)
        neutral = pos + mat @ offset
        depth = spec.get("depth")
        out.append(
            {
                "point": point,
                "neutral": neutral.tolist(),
                "depth": int(frame["depth"] if depth is None else depth),
            }
        )
    return out


@functools.lru_cache(maxsize=4)
def load_articulation(
    ref: str | Path = DEFAULT_ARTICULATION_PATH,
    fit: tuple[str, ...] | None = None,
) -> Articulation:
    """Load (and cache) the packaged head/abdomen articulation (no bounds overrides)."""
    return Articulation.load(ref, fit=fit)


def body_similarity(
    neutral_coxae: np.ndarray, measured_coxae: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Similarity transform (R, s, t) mapping the neutral coxae onto the measured ones.

    ``measured_coxae`` is ``(6, 3)`` (NaN where a coxa was not seen); at least three
    finite coxae are needed. Returns ``None`` when too few are available.
    """
    good = np.isfinite(measured_coxae).all(axis=1)
    if int(good.sum()) < 3:
        return None
    return _umeyama(neutral_coxae[good], measured_coxae[good])


def estimate_chain_scale(
    local_markers: np.ndarray,
    chain: Chain,
    *,
    clamp: tuple[float, float] = (0.3, 3.0),
) -> float:
    """Isotropic size of a chain relative to the model, from its contour length.

    The body registration (coxa Umeyama) sets the overall fly scale, but the head and
    abdomen are fixed model geometry that need not match this fly (a longer abdomen,
    a bigger head). The size is estimated from the chain's **contour length**: the
    sum of distances ``base -> centroid(depth_0) -> centroid(depth_1) -> ...`` over
    the marker groups at each chain depth. That polyline runs *along* the chain, so it
    is dominated by the chain's long axis (the meaningful size) and -- being a sum of
    rigid inter-segment lengths -- is invariant to the chain's articulation, exactly
    the length-ratio idea the coxa body scale uses.

    This deliberately avoids same-depth (left/right) marker separations: those span
    the chain's *short* axis (the model's abdomen markers sit ~0.1 apart, near the
    dorsal midline), so their measured/model ratio is both noisy and a poor proxy for
    the overall size -- it badly over-estimates the scale.

    Parameters
    ----------
    local_markers
        ``(T, M, 3)`` the chain's measured markers mapped into the model frame
        (i.e. ``R^T (world - t) / s`` with the body similarity ``(R, s, t)``); NaN
        where a marker was not observed.
    chain
        The articulation chain (its neutral markers, their depths, and base anchor).
    clamp
        ``(lo, hi)`` bounds on the returned scale, to reject outlier frames.

    Returns
    -------
    float
        The median measured/model contour-length ratio over the valid frames, clamped
        to ``clamp``. ``1.0`` (model size) when the contour cannot be measured (a
        degenerate model, or no frame with the contour fully observed).
    """
    local = np.asarray(local_markers, dtype=float)
    neutral = np.asarray(chain.marker_neutral, dtype=float)
    depth = np.asarray(chain.marker_depth)
    base = np.asarray(chain.anchors[0], dtype=float)
    depths = sorted(set(int(d) for d in depth.tolist()))

    def _contour(points: np.ndarray) -> np.ndarray:
        """Polyline length ``base -> centroid(depth_0) -> ...`` (last axis is xyz)."""
        with warnings.catch_warnings():
            # A frame missing a whole depth group gives a NaN centroid (a NaN length,
            # dropped below); nanmean warns on that empty slice, so quiet it here.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cents = [np.nanmean(points[..., depth == d, :], axis=-2) for d in depths]
        chain_pts = np.stack(
            [np.broadcast_to(base, cents[0].shape), *cents], axis=-2
        )  # (..., len(depths)+1, 3)
        return np.linalg.norm(np.diff(chain_pts, axis=-2), axis=-1).sum(axis=-1)

    model_len = float(_contour(neutral))
    if model_len < _EPS:
        return 1.0  # a single marker group / degenerate model: no length to compare
    meas_len = np.atleast_1d(_contour(local))
    meas_len = meas_len[np.isfinite(meas_len)]
    if meas_len.size == 0:
        return 1.0
    return float(np.clip(np.median(meas_len) / model_len, clamp[0], clamp[1]))


def chain_affine(
    chain: Chain, depth: int, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The model-frame affine ``(A, b)`` carrying a depth-``d`` point: ``A p + b``.

    Mirrors :func:`deeperfly.inverse_kinematics.kinematics.make_chain_fk` in numpy
    (used to pose the head/abdomen mesh nodes). ``theta`` is the chain's joint angles.
    """
    a_cum = np.eye(3)
    b_cum = np.zeros(3)
    for i in range(int(depth)):
        r = _axis_rmat(chain.axes[i], float(theta[i]))
        c = chain.anchors[i]
        b_cum = a_cum @ (c - r @ c) + b_cum
        a_cum = a_cum @ r
    return a_cum, b_cum


def _axis_rmat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation by ``angle`` about the unit ``axis`` (Rodrigues, sin/cos form)."""
    ax, ay, az = axis
    k = np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Least-squares similarity transform (rotation, scale, translation) ``src -> dst``."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = (d0.T @ s0) / len(src)
    u, sigma, vt = np.linalg.svd(cov)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1.0
    rot = u @ correction @ vt
    var = (s0**2).sum() / len(src)
    scale = float(np.trace(np.diag(sigma) @ correction) / var) if var > _EPS else 1.0
    trans = mu_d - scale * (rot @ mu_s)
    return rot, scale, trans
