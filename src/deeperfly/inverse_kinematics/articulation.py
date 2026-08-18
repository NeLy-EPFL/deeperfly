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
body-fixed). The same chains both enter the solved body plan
(:mod:`deeperfly.inverse_kinematics.bodyplan`) and pose the head/abdomen of the overlay
mesh, so the angles and the mesh stay consistent.

This module is pure model *geometry*: it loads the baked asset and measures a chain's
size, and holds no kinematics of its own. The chain forward kinematics lives in
:mod:`deeperfly.inverse_kinematics.forward` (:func:`~deeperfly.inverse_kinematics.forward.chain_affine`
is re-exported here, where its callers have always looked for it).
"""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .forward import chain_affine

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

log = logging.getLogger("deeperfly")

_EPS = 1e-9

#: How far apart two markers' neutral positions may be, after mirroring one across the
#: sagittal plane, and still count as a left/right pair to fold (:func:`_midline_groups`).
#: The baked neutrals are stored to 8 decimals, so this is loose enough to survive that
#: rounding and far tighter than any real marker spacing.
_MIRROR_TOL = 1e-6

#: Interior joint-angle draws used to test a marker separation for articulation-invariance
#: (:func:`_rigid_ruler`), on top of the bound extremes, the midpoint and the neutral pose.
_RULER_PROBES = 6

#: Relative slack allowed in that test. The classes it separates differ by ~1e14, so the
#: exact value is immaterial; it exists only to absorb floating-point noise.
_RULER_TOL = 1e-9


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
    base_point
        The marker that **measures where this chain's base sits**, or ``None``. The
        legs place each chain root at its measured median thorax-coxa; a baked chain
        has no such landmark unless the skeleton labels one, and without it the base
        comes from the coxa registration alone -- an extrapolation off six
        near-coplanar points, which puts the head pivot well off the animal. Naming a
        base point lets :func:`~deeperfly.inverse_kinematics.bodyplan.build_body_plan`
        shift the whole chain onto it, and lets :func:`estimate_chain_scale` measure
        the chain against its own base instead of the registered one.

        It is a *landmark*, not a constraint: a point at the chain's base is on the
        rotation center, so no chain DOF can move it and it tells the solver nothing
        about the angles (see :func:`deeperfly.inverse_kinematics.unfittable_branches`,
        which declines to count it as evidence the chain was observed).
    """

    name: str
    dof_names: tuple[str, ...]
    anchors: np.ndarray
    axes: np.ndarray
    bounds: tuple[np.ndarray, np.ndarray]
    marker_names: tuple[str, ...]
    marker_neutral: np.ndarray
    marker_depth: tuple[int, ...]
    base_point: str | None = None

    def marker_index(self, point: str) -> int | None:
        """The column of ``point`` in this chain's marker arrays, or ``None``."""
        return self.marker_names.index(point) if point in self.marker_names else None


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
            resolved, base_point = _resolve_markers(c, markers.get(c["name"]), bodies)
            chains.append(_build_chain(c, overrides, resolved, base_point))
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
    c: dict,
    overrides: dict[str, tuple[float, float]],
    markers: list[dict],
    base_point: str | None,
) -> Chain:
    joints = c["joints"]
    if base_point is not None:
        names = [m["point"] for m in markers]
        if base_point not in names:
            raise ValueError(
                f"[inverse_kinematics.{c['name']}] base marker {base_point!r} is not "
                f"one of the chain's markers ({sorted(names)}); the base point has to "
                "be a tracked keypoint for the recording to measure it"
            )
        # Forced, not merely defaulted: the base sits at the chain's own origin, so no
        # chain joint moves it -- and a depth read off the attachment body would say
        # otherwise. `c_head`'s baked depth is 3, which would advertise the neck as
        # carried by all three head DOFs when rotation about a point leaves that point
        # exactly where it was. The position is identical either way; what depends on
        # it is whether the solver is told this marker is evidence the chain was
        # observed (`deeperfly.inverse_kinematics.unfittable_branches`), and it is not.
        markers = [
            {**m, "depth": 0} if m["point"] == base_point else m for m in markers
        ]
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
        base_point=base_point,
    )


def _resolve_markers(
    c: dict, override: dict[str, dict] | None, bodies: dict[str, dict]
) -> tuple[list[dict], str | None]:
    """The chain's ``(markers, base_point)``: the baked set, or a config override's set.

    Without an override the chain keeps its baked markers and base point. With one, the
    table *replaces* them: each marker's neutral world position is recomputed from the
    attachment body's baked frame (``pos + mat @ offset``) and its depth is the body's
    chain depth (or an explicit ``depth``). Used to retarget the IK to a different
    labeling scheme -- e.g. abdomen points placed at different offsets.

    A marker entry may carry ``base = true`` to nominate it as the chain's base point
    (:attr:`Chain.base_point`). Because the override replaces the whole table, a config
    that redeclares a chain would otherwise silently drop the baked nomination and put
    the chain back on the registered base -- so the flag has to be expressible here.
    """
    if not override:
        return list(c["markers"]), c.get("base_point")
    out: list[dict] = []
    base_point: str | None = None
    for point, spec in override.items():
        if spec.get("base"):
            if base_point is not None:
                raise ValueError(
                    f"[inverse_kinematics.{c['name']}] declares two base markers "
                    f"({base_point!r} and {point!r}); a chain has one base"
                )
            base_point = point
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
    return out, base_point


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
    """Isotropic size of a chain relative to the model, from lengths its joints cannot change.

    The body registration (coxa Umeyama) sets the overall fly scale, but the head and
    abdomen are fixed model geometry that need not match this fly (a longer abdomen, a
    bigger head), so each chain gets one uniform multiplier of its own. It is measured
    from the chain's **rigid marker separations**: the distances between its own markers
    that no chain DOF can change, compared with the same distances on the model.

    Two properties make a separation usable, and both are checked rather than assumed:

    *Articulation-invariance.* A ruler that the chain's own joints can stretch measures
    posture, not size. Almost none of the abdomen's marker distances survive this: its
    five keypoints sit on the dorsal *surface*, which is the outside of a ventral bend,
    so a polyline through them lengthens by 57% over the joints' 30-degrees-per-hinge
    range -- more than the size differences being measured. The invariant pairs are found
    by probing the forward kinematics across the chain's own bounds (:func:`_rigid_ruler`),
    which separates them by fourteen orders of magnitude and needs no case analysis: for
    the head, the neck-to-antennae radius (rotation about a point cannot change a
    distance from that point); for the abdomen, the one pair sharing a body.

    *Immunity to the left/right split.* Mirror-symmetric markers are folded to their
    midpoint first (:func:`_midline_groups`). A left-to-right distance inherits the two
    sides' disagreement at full strength, and that disagreement is large: across the
    corpus the ``l_antenna``-``r_antenna`` span reads 1.515x the model where either
    antenna's distance to the ``neck`` reads 1.196x -- 26% wider, the same sign in all 30
    recordings, because each antenna is triangulated from its own side's cameras. The
    midpoint is unmoved by a symmetric outward push, so folding removes it.

    Multiple rulers are combined by least squares, ``s = sum(D d) / sum(D^2)`` over the
    pairs observed in that frame, which is the maximum-likelihood isotropic scale when
    the per-point error does not depend on the pair -- so a long ruler counts for more
    than a short one without any hand-set weight. The per-frame estimates are then
    reduced by their **median** over the recording: a chain's size is a constant of the
    animal, so everything the estimate does over time is noise.

    Parameters
    ----------
    local_markers
        ``(T, M, 3)`` the chain's measured markers mapped into the model frame
        (i.e. ``R^T (world - t) / s`` with the body similarity ``(R, s, t)``); NaN
        where a marker was not observed.
    chain
        The articulation chain (its neutral markers, their depths, axes and bounds).
    clamp
        ``(lo, hi)`` bounds on the returned scale, to reject outlier frames.

    Returns
    -------
    float
        The median measured/model length ratio over the valid frames, clamped to
        ``clamp``. ``1.0`` (model size) when the chain has no invariant ruler at all, or
        none that this recording observed -- warned about, because a silent ``1.0`` would
        read as a *measurement* that the chain matches the model and the overlay would
        then draw a confidently mis-sized head.
    """
    local = np.asarray(local_markers, dtype=float)
    groups = _midline_groups(chain)
    pairs = _rigid_ruler(chain, groups)
    if not pairs:
        log.warning(
            "inverse_kinematics: the %s chain has no articulation-invariant marker "
            "separation (markers %s at depths %s), so its size cannot be measured; "
            "leaving it at model size",
            chain.name,
            list(chain.marker_names),
            list(chain.marker_depth),
        )
        return 1.0
    # Mean, not nanmean: a mirror pair with one side missing is the bare contralateral
    # point, which is exactly the biased quantity the fold exists to remove. Let it go
    # NaN and drop that ruler for the frame.
    folded = np.stack(
        [local[..., list(g), :].mean(axis=-2) for g in groups], axis=-2
    )  # (T, G, 3)
    model_len = np.asarray([d for _, _, d in pairs], dtype=float)
    measured = np.stack(
        [
            np.linalg.norm(folded[..., i, :] - folded[..., j, :], axis=-1)
            for i, j, _ in pairs
        ],
        axis=-1,
    )  # (T, P)
    weight = np.where(np.isfinite(measured), model_len, 0.0)
    num = np.nansum(weight * measured, axis=-1)
    den = (weight * model_len).sum(axis=-1)
    ratio = np.divide(num, den, out=np.full(np.shape(num), np.nan), where=den > _EPS)
    ratio = np.atleast_1d(ratio)
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        log.warning(
            "inverse_kinematics: the %s chain's size ruler (%s) was never observed, so "
            "its size cannot be measured; leaving it at model size",
            chain.name,
            ", ".join(
                f"{chain.marker_names[groups[i][0]]}-{chain.marker_names[groups[j][0]]}"
                for i, j, _ in pairs
            ),
        )
        return 1.0
    return float(np.clip(np.median(ratio), clamp[0], clamp[1]))


def _midline_groups(chain: Chain) -> list[tuple[int, ...]]:
    """Marker columns grouped so that each group is one point on the animal's midline.

    A left/right mirror pair becomes one group and is averaged to its midpoint before
    anything is measured; every other marker is its own group. See
    :func:`estimate_chain_scale` for why the fold matters -- a left-to-right distance
    carries the two sides' triangulation disagreement at full strength.

    A pair qualifies only when the two markers sit at the same chain depth, which is what
    makes their midpoint a rigid point of one body rather than a moving average of two.
    """
    neutral = np.asarray(chain.marker_neutral, dtype=float)
    mirror = neutral * np.array([1.0, -1.0, 1.0])
    depth = chain.marker_depth
    taken: set[int] = set()
    groups: list[tuple[int, ...]] = []
    for i in range(len(chain.marker_names)):
        if i in taken:
            continue
        taken.add(i)
        partner = None
        if (
            abs(neutral[i, 1]) > _MIRROR_TOL
        ):  # a midline marker has nothing to fold with
            partner = next(
                (
                    j
                    for j in range(i + 1, len(chain.marker_names))
                    if j not in taken
                    and depth[j] == depth[i]
                    and np.allclose(neutral[i], mirror[j], atol=_MIRROR_TOL)
                ),
                None,
            )
        if partner is None:
            groups.append((i,))
        else:
            taken.add(partner)
            groups.append((i, partner))
    return groups


def _rigid_ruler(
    chain: Chain, groups: list[tuple[int, ...]]
) -> list[tuple[int, int, float]]:
    """``(i, j, model_length)`` for each group pair whose distance no chain DOF changes.

    Decided by *probing* the chain's own forward kinematics -- the bound extremes, the
    neutral pose and a few fixed-seed interior draws -- rather than by reasoning about
    which axes pass through which markers. The two classes are not close: over the
    abdomen's range an articulation-dependent pair swings ~1e-1 model units while an
    invariant one holds to ~1e-16, so a handful of probes settles every pair with
    fourteen orders of magnitude of margin, and a marker set retargeted in a run config
    gets the right ruler with no code change.
    """
    lo, hi = chain.bounds
    rng = np.random.default_rng(0)
    probes = [lo, hi, 0.5 * (lo + hi), np.zeros_like(lo)] + [
        lo + (hi - lo) * rng.random(len(lo)) for _ in range(_RULER_PROBES)
    ]
    spans = np.stack(
        [
            np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
            for p in (_group_positions(chain, groups, theta) for theta in probes)
        ]
    )  # (probes, G, G)
    swing = spans.max(axis=0) - spans.min(axis=0)
    model = _group_positions(chain, groups, np.zeros_like(lo))
    out: list[tuple[int, int, float]] = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            length = float(np.linalg.norm(model[i] - model[j]))
            if length > _EPS and swing[i, j] <= _RULER_TOL * max(length, 1.0):
                out.append((i, j, length))
    return out


def _group_positions(
    chain: Chain, groups: list[tuple[int, ...]], theta: np.ndarray
) -> np.ndarray:
    """``(G, 3)`` each marker group's forward-kinematic position at joint angles ``theta``."""
    out = np.empty((len(groups), 3))
    for g, cols in enumerate(groups):
        rot, trans = chain_affine(chain, int(chain.marker_depth[cols[0]]), theta)
        out[g] = np.mean([rot @ chain.marker_neutral[c] + trans for c in cols], axis=0)
    return out


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
