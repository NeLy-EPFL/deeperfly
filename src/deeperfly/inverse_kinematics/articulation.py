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

#: How many frames the whole-chain calibration fits, evenly spaced over the recording.
#: A chain's size and root are constants of the animal, so this only has to out-average
#: the per-frame detection noise; 60 costs well under a second and the median over them
#: moves by <1% against every frame.
_CALIBRATION_FRAMES = 60

#: The root shift is searched inside this fraction of the chain's own neutral extent.
#: Relative rather than absolute so a short chain cannot be translated off its own body.
_ROOT_SHIFT_FRACTION = 0.5

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
    warn_if_unmeasurable: bool = True,
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

    A chain left with **one** ruler is measured only as well as that one pair -- see
    :func:`calibrate_chain`, which measures size from the whole marker set instead and is
    what the pipeline now uses; this function remains its seed, and the two agree to ~1%
    on the head.

    The abdomen now has **no** ruler at all, and that is the point. Its one invariant pair
    used to be ``abdomen3``-``abdomen4``, which existed only because both markers hung off
    the same body (``c_abdomen6``) -- so their separation was rigid in the model while a
    real fly's is not, 18% short of a measured animal and unreachable by any joint angle.
    Re-anchoring each stripe one segment proximal put a hinge between them, which removes
    the residual and the ruler together. So a chain reporting no ruler is not a defect
    here: pass ``warn_if_unmeasurable=False`` wherever a fitted size follows.

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
    warn_if_unmeasurable
        Whether "no usable ruler" is worth a warning. True for a standalone measurement,
        where a silent ``1.0`` would read as a *measurement* that the chain matches the
        model and the overlay would then draw a confidently mis-sized head. False when
        this is only a seed for :func:`calibrate_chain`, which measures the size without a
        ruler -- the abdomen's has been deliberately removed (see above), so a warning
        there would fire on every run and say something untrue.

    Returns
    -------
    float
        The median measured/model length ratio over the valid frames, clamped to
        ``clamp``. ``1.0`` (model size) when the chain has no invariant ruler at all, or
        none that this recording observed.
    """
    local = np.asarray(local_markers, dtype=float)
    groups = _midline_groups(chain)
    pairs = _rigid_ruler(chain, groups)
    if not pairs:
        log.log(
            logging.WARNING if warn_if_unmeasurable else logging.DEBUG,
            "inverse_kinematics: the %s chain has no articulation-invariant marker "
            "separation (markers %s at depths %s), so this ruler cannot measure its "
            "size; seeding from model size",
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
        log.log(
            logging.WARNING if warn_if_unmeasurable else logging.DEBUG,
            "inverse_kinematics: the %s chain's size ruler (%s) was never observed, so "
            "this ruler cannot measure its size; seeding from model size",
            chain.name,
            ", ".join(
                f"{chain.marker_names[groups[i][0]]}-{chain.marker_names[groups[j][0]]}"
                for i, j, _ in pairs
            ),
        )
        return 1.0
    return float(np.clip(np.median(ratio), clamp[0], clamp[1]))


def _scaled_chain(chain: "Chain", scale: float):
    """A chain's ``(anchors, markers)`` grown by ``scale`` about its own base anchor.

    The same growth :func:`~deeperfly.inverse_kinematics.bodyplan._chain_joints` builds
    into the solver's plan and the mesh overlay applies to its node affines, so a size
    measured here means the same thing in all three.
    """
    anchors = np.asarray(chain.anchors, dtype=float)
    markers = np.asarray(chain.marker_neutral, dtype=float)
    base = anchors[0]
    f = float(scale)
    return base + f * (anchors - base), base + f * (markers - base)


def chain_markers(
    chain: "Chain", angles: np.ndarray, scale: float, shift: np.ndarray | None = None
) -> np.ndarray:
    """``(M, 3)`` model-frame marker positions of a chain at ``angles``, grown and shifted.

    Thin wrapper over :func:`~deeperfly.inverse_kinematics.forward.chain_fk` that applies
    the size and root conventions in one place, so the calibration below cannot drift from
    the forward kinematics the fit and the overlay use. Rolling this by hand is a live
    trap: composing a chain's rotations in the opposite order is invisible on the abdomen,
    whose five axes are all ``+Y`` and therefore commute, and wrong by ~0.9 model units on
    the head's three-axis ball joint.
    """
    from .forward import chain_fk

    anchors, markers = _scaled_chain(chain, scale)
    posed = chain_fk(
        anchors,
        np.asarray(chain.axes, dtype=float),
        tuple(int(d) for d in chain.marker_depth),
        np.asarray(angles, dtype=float),
        markers,
    )
    return posed if shift is None else posed + np.asarray(shift, dtype=float)


def _is_midline(chain: "Chain", tol: float = _MIRROR_TOL) -> bool:
    """Whether the chain lies entirely in the sagittal plane (every ``y`` is zero).

    True of the abdomen, whose five keypoints are midline dorsal points, and false of the
    head, which carries the two antennae. It decides which components of a fitted root
    shift are meaningful: see :func:`calibrate_chain`.
    """
    return bool(
        np.abs(np.asarray(chain.anchors, dtype=float)[:, 1]).max() <= tol
        and np.abs(np.asarray(chain.marker_neutral, dtype=float)[:, 1]).max() <= tol
    )


def calibrate_chain(
    local_markers: np.ndarray,
    chain: "Chain",
    *,
    seed_scale: float,
    base_shift: np.ndarray | None = None,
    fit_root: bool = False,
    n_frames: int = _CALIBRATION_FRAMES,
    clamp: tuple[float, float] = (0.3, 3.0),
) -> tuple[float, np.ndarray]:
    """A chain's size (and, optionally, its root) from its WHOLE marker set.

    :func:`estimate_chain_scale` measures size from separations no joint can change,
    which is exact but leaves the abdomen resting on a **single** ruler --
    ``abdomen3``-``abdomen4``, its two most distal keypoints, 0.234 model units apart.
    That is the shortest baseline in the chain between its least reliable points, so a
    0.05 localisation error reads as a 21% size error, and it does: measured on three
    animals across two rigs the ruler over-reads by 10.8% / 14.8% / 18.5%, always the
    same sign, and always making the abdomen come out *larger* than the same fly's head
    (1.34-1.49 against 1.20-1.25) -- one animal cannot have those two sizes.

    So this measures size the other way: fit the chain's angles and its size *together*
    against every marker, treating posture as the nuisance parameter rather than
    something to avoid. Posture is then explicitly fitted out instead of dodged, which is
    why no invariance requirement is needed and why the whole marker set can be used.
    Seeded from ``seed_scale`` (the ruler), reduced by the **median** over sampled frames
    because a chain's size is a constant of the animal.

    Measured against the ruler, on the head -- where the ruler is a long, well-defined
    ``neck``-to-antennae radius and is trusted -- the two agree to **+0.7% / +1.6% /
    +0.9%**, the marker residual is unchanged and the fitted angle ranges are identical.
    That agreement is what licenses the abdomen answer, where they differ by 15%.

    ``fit_root`` additionally frees the chain's root position, for a chain that has no
    base landmark to be placed on (the abdomen: no keypoint sits on its root). Left on the
    model's own anchor, that root inherits the coxa registration's worst-conditioned
    direction. The three shift components are **not** equally determined -- the Jacobian
    has one exact null direction, ``(-0.68, +0.68, 0, 0, 0 | scale 0.000 | dx -0.07, dy 0,
    dz +0.25)``, i.e. hinge 0 trading against hinge 1 and a root translation -- so:

    * **Size is unaffected by it.** Its coefficient in that null vector is 0.000, so
      ``seed_scale``'s replacement is well posed whether or not the root is fitted.
    * **The root is resolved once per recording**, as the median over frames, and then
      held fixed while the angles are solved per frame -- exactly how a base landmark's
      shift is already treated, and for the same reason: it is one body landmark, not a
      per-frame quantity. Fixing it removes the null direction from the per-frame solve.

    Only ``dz`` is repeatable across animals (-0.176 / -0.137 / -0.107, same sign);
    ``dx`` scatters about zero, which is the null direction showing through. It is still
    returned, because the residual and not the decomposition is what the pose is read
    from, and the per-frame scatter of the fit is 1-2% of the chain's length. ``dy`` is
    **not** fitted at all for a midline chain (:func:`_is_midline`) -- see below.

    An earlier grid search over the abdomen root reported its optimum at *positive* ``dz``
    and rejected the idea. That search held the size at the ruler's value; size and root
    are coupled through this same null direction, so with the size free the optimum moves
    and changes sign. Neither result is wrong about what it measured.

    Parameters
    ----------
    local_markers
        ``(T, M, 3)`` the chain's measured markers in the model frame, as
        :func:`estimate_chain_scale` takes them.
    chain
        The articulation chain.
    seed_scale
        Starting size, normally :func:`estimate_chain_scale`'s ruler answer.
    base_shift
        A root shift already known from a base landmark, held fixed and added to.
    fit_root
        Whether to free the root position on top of ``base_shift``.
    n_frames
        How many evenly spaced frames to fit.
    clamp
        ``(lo, hi)`` bounds on the returned size.

    Returns
    -------
    scale : float
        The median fitted size, clamped.
    shift : np.ndarray
        ``(3,)`` the chain's root shift (``base_shift`` when ``fit_root`` is false).
    """
    from scipy.optimize import least_squares

    local = np.asarray(local_markers, dtype=float)
    base = np.zeros(3) if base_shift is None else np.asarray(base_shift, dtype=float)
    lo, hi = chain.bounds
    n_dof = len(lo)
    if local.ndim != 3 or local.shape[0] == 0 or n_dof == 0:
        return float(np.clip(seed_scale, *clamp)), base
    usable = np.flatnonzero(np.isfinite(local).all(axis=(1, 2)))
    if usable.size == 0:
        log.warning(
            "inverse_kinematics: the %s chain was never fully observed, so its size "
            "cannot be fitted; keeping the ruler estimate %.4f",
            chain.name,
            seed_scale,
        )
        return float(np.clip(seed_scale, *clamp)), base
    take = usable[np.unique(np.linspace(0, usable.size - 1, n_frames).astype(int))]

    anchors = np.asarray(chain.anchors, dtype=float)
    span = float(np.linalg.norm(anchors[-1] - anchors[0])) or 1.0
    reach = _ROOT_SHIFT_FRACTION * max(span, float(np.abs(chain.marker_neutral).max()))
    # A midline chain's root is a landmark on the animal's own plane of symmetry, and the
    # registration error that displaces it is a pitch about the coxa centroid -- which is
    # in that plane too. So its lateral component is not something to measure: freeing it
    # only lets the root drift sideways in place of the chain's own lateral joints, which
    # is the same displacement attributed to the wrong thing.
    free_axes = (0, 2) if (fit_root and _is_midline(chain)) else (0, 1, 2)
    n_extra = len(free_axes) if fit_root else 0
    mid = 0.5 * (lo + hi)
    x0 = np.concatenate([mid, [seed_scale], np.zeros(n_extra)])
    blo = np.concatenate([lo, [clamp[0]], np.full(n_extra, -reach)])
    bhi = np.concatenate([hi, [clamp[1]], np.full(n_extra, reach)])

    def _shift(p: np.ndarray) -> np.ndarray:
        """The root shift a parameter vector encodes: the base plus its free axes."""
        if not fit_root:
            return base
        out = np.array(base, dtype=float)
        out[list(free_axes)] += p[n_dof + 1 :]
        return out

    scales: list[float] = []
    shifts: list[np.ndarray] = []
    for t in take:
        target = local[t]

        def residual(p, target=target):
            return (
                chain_markers(chain, p[:n_dof], p[n_dof], _shift(p)) - target
            ).ravel()

        fit = least_squares(residual, x0, bounds=(blo, bhi), xtol=1e-10, ftol=1e-10)
        scales.append(float(fit.x[n_dof]))
        shifts.append(_shift(fit.x))
    scale = float(np.clip(np.median(scales), *clamp))
    shift = np.median(np.stack(shifts), axis=0) if fit_root else base
    return scale, shift


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
