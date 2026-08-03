"""Left/right swap detection over a skeleton's symmetry pairs.

A swapped pair -- the operator dragged the left claw onto the right leg and vice versa --
is the one labeling error that costs *nothing* in every metric that looks at a point
cloud. Both pixels are on a real joint, the reprojection error is small, triangulation
converges, and the only thing that is wrong is the identity. So it has to be looked for
directly, on the structure that defines it: the symmetry pairs
(``[skeleton].symmetries``).

**The axis comes from the pairs themselves.** SLEAP's chirality feature estimates a body
midline from the nodes that are *not* in a symmetric pair and measures each pair against
it. That cannot work here: fly38 has no unpaired points at all -- every one of the 38 is
half of a pair -- so there is nothing left to fit a midline to. The formulation that does
work uses the pair **displacement** vectors ``p_right - p_left``. On a bilaterally
symmetric animal these are all roughly parallel to the left-right axis, whatever the
posture, so their principal direction *is* that axis:

* A second-moment (PCA) fit is **sign-blind**, so the axis is recovered correctly even
  when some pairs are swapped -- which is the whole point, since a method that needed
  correct pairs to find the axis could not then use the axis to find the wrong pairs.
* Along that axis every pair should have the *same* sign. The majority sign defines the
  convention for this sample and the minority pairs are the swap candidates.

No canonical side has to be learned from a training corpus, and no midline is needed.

**Run this on 3D.** :func:`check` is dimension-agnostic -- give it ``(P, 2)`` or ``(P, 3)``
-- but the two are not equally sound, and the difference is geometry rather than tuning.
In 3D "left" is a fixed halfspace: every left point sits on one side of the sagittal
plane whatever the animal is doing, so a pair's displacement always points the same way
and a disagreement really is a swap. A 2D view loses exactly the coordinate that carries
the side, *unless the view's image plane contains the left-right axis*. Measured over 300
random postures with the two body sides posed **independently** (a walking fly, not a
mirrored one), reporting false swap flags per pose at p95:

.. code-block:: text

    3D                                     0
    front camera (looks down the midline)  0
    oblique side cameras (+-45, +-120)     4 - 7
    lateral cameras (+-90)                 10 - 11

The side-view failures are not noise to be filtered: an asymmetric posture genuinely puts
the left claw right-of the right claw *in projection*, and that is true 3D structure, not
a labeling error. So per-view 2D is offered as a diagnostic -- useful for an uncalibrated
project that has nothing else, and for a front camera, where it works -- while the editor
judges the derived 3D pose.

The same measurement disqualified the obvious conditioning statistic. The displacement
cloud's **anisotropy** (first over second principal value) does not discriminate on
realistic postures: 3D scores the *lowest* of everything at 1.61 while being the only
exact test, against 2.1 for the useless lateral views. It is reported for inspection and
deliberately **not** gated on.

Nor does any scalar rescue 2D, and the gate this module does apply is worth stating
narrowly so it is not mistaken for one that does. On a *mirror-symmetric* posture the +-90
lateral views collapse outright -- the pairs separate by 0.0019 of the animal's extent
against 0.39 in 3D and 0.54 in the front view, a 200x gap -- and
:data:`MIN_SEPARATION_FRAC` refuses those. On an *asymmetric* posture nothing collapses:
those same lateral views score a healthy 0.32 while still emitting 10-11 false flags per
pose. That is precisely why the fix is to judge 3D rather than to keep tightening a
threshold: the side views' errors are structural, not ill-conditioned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int

__all__ = [
    "Chirality",
    "SwapCandidate",
    "check",
    "check_views",
    "sagittal_axis",
]

#: Minimum co-visible pairs needed to fit an axis at all. Two pairs already define a
#: direction, but with two a single swapped pair is indistinguishable from a swapped
#: *convention* -- the majority vote has no majority. Three is the first count where a
#: lone swap is outvoted.
MIN_PAIRS = 3

#: The gate, and it has one narrow job: refuse a sample in which the left-right axis has
#: **collapsed**, which is what a camera looking straight down that axis sees. The median
#: pair separation along the fitted axis, as a fraction of the point cloud's diameter, must
#: reach this value. Set from measurement: on mirror-symmetric postures the statistic is
#: 0.0019 in the +-90 lateral views against 0.39 in 3D and 0.54 in the front view, so 0.05
#: sits ~8x below the good regime and ~25x above the degenerate one.
#:
#: What it explicitly does **not** do is decide whether a 2D view is trustworthy. On an
#: asymmetric posture no view collapses -- the lateral views score 0.32 and still emit
#: 10-11 false flags per pose -- so no value of this constant makes per-view 2D sound. That
#: is a structural limit (module docstring), answered by judging 3D, not by tuning.
MIN_SEPARATION_FRAC = 0.05

#: A pair is only judged when its own separation along the axis is at least this fraction
#: of the sample's median separation. A pair whose two points nearly coincide -- a joint on
#: the midline, or one seen edge-on -- has no meaningful order, and calling it swapped
#: because it landed 0.3 px on the wrong side is how a QC signal becomes noise operators
#: learn to ignore.
MIN_MARGIN_FRAC = 0.25


@dataclass(frozen=True)
class SwapCandidate:
    """One symmetry pair whose left/right order disagrees with the sample's majority.

    Attributes
    ----------
    points
        The pair as ``(low_index, high_index)`` into the skeleton's points -- the same
        order :attr:`deeperfly.skeleton.Skeleton.symmetries` stores.
    margin
        How far the pair sits on the wrong side, along the fitted axis, in the input's own
        units (pixels for 2D, world units for 3D). Large means confidently swapped; near
        the threshold means the pair is nearly midline and the call is weak.
    relative_margin
        ``margin`` over the median separation of the judged pairs, so it is comparable
        across frames, views and zoom levels. This is the number to rank by.
    """

    points: tuple[int, int]
    margin: float
    relative_margin: float


@dataclass(frozen=True)
class Chirality:
    """The verdict for one sample (one view of one frame, or one 3D pose).

    Attributes
    ----------
    swapped
        The suspect pairs, worst :attr:`SwapCandidate.relative_margin` first. Empty when
        the sample is consistent *or* when it could not be judged -- read
        :attr:`decided` to tell those apart.
    decided
        Whether the sample was judged at all. ``False`` when too few pairs are co-visible
        or the axis is ill-conditioned; :attr:`reason` says which.
    reason
        Why the sample was not judged, or ``""`` when it was. Surfaced verbatim, because
        "the front camera cannot see this" and "these pairs are swapped" must never look
        alike to an operator.
    axis
        The fitted left-right axis (unit, pointing toward the *majority* right side), or
        ``None`` when undecided.
    n_pairs
        How many pairs were co-visible and separated enough to vote.
    separation_frac
        Median pair separation along the axis over the point cloud's diameter -- the
        statistic :data:`MIN_SEPARATION_FRAC` gates on, and the number that says how well
        this sample resolves the left-right axis at all.
    anisotropy
        First over second principal value of the displacement cloud. **Reported, never
        gated on**: measured on realistic postures it does not discriminate (see the module
        docstring). Useful only for spotting a sample whose pairs point in no common
        direction, which is labels scrambled rather than merely swapped.
    """

    swapped: tuple[SwapCandidate, ...] = ()
    decided: bool = False
    reason: str = ""
    axis: Float[np.ndarray, "D"] | None = None
    n_pairs: int = 0
    separation_frac: float = 0.0
    anisotropy: float = 0.0

    def __bool__(self) -> bool:
        """Truthy when something is suspect -- so ``if chirality:`` reads as intended."""
        return bool(self.swapped)


def sagittal_axis(
    displacements: Float[np.ndarray, "S D"],
) -> tuple[Float[np.ndarray, "D"], float]:
    """The dominant direction of a pair-displacement cloud, and its anisotropy.

    Parameters
    ----------
    displacements
        ``(S, D)`` finite vectors ``p_high - p_low``, one per co-visible pair.

    Returns
    -------
    axis : np.ndarray
        Unit ``(D,)`` first principal direction. Its **sign is arbitrary** -- fixing it is
        the caller's job (:func:`check` points it at the majority side), because a second
        moment cannot distinguish a direction from its negation. That sign-blindness is
        the property that makes the axis recoverable from partly-swapped data.
    anisotropy : float
        First singular value over the second (``inf`` when the second is ~0, i.e. a
        perfectly collinear cloud). 1.0 means "no dominant direction".

    Notes
    -----
    Uncentered on purpose. The cloud is a set of *vectors*, not positions: for an unswapped
    sample they all point the same way, so their mean is the signal and subtracting it
    would throw away exactly what is being measured.
    """
    d = np.asarray(displacements, dtype=np.float64).reshape(-1, displacements.shape[-1])
    if d.shape[0] == 0:
        raise ValueError("need at least one displacement to fit an axis")
    sv = np.linalg.svd(d, compute_uv=True)
    axis = sv[2][0]
    s = sv[1]
    second = float(s[1]) if s.size > 1 else 0.0
    aniso = float("inf") if second <= 1e-12 else float(s[0]) / second
    return axis / np.linalg.norm(axis), aniso


def check(
    points: Float[np.ndarray, "P D"],
    symmetries: Int[np.ndarray, "S 2"],
    *,
    min_pairs: int = MIN_PAIRS,
    min_separation_frac: float = MIN_SEPARATION_FRAC,
    min_margin_frac: float = MIN_MARGIN_FRAC,
) -> Chirality:
    """Judge one sample's left/right consistency. Prefer 3D -- see the module docstring.

    Parameters
    ----------
    points
        ``(P, D)`` positions, ``D`` being 3 (world -- the sound case) or 2 (one view's
        pixels -- a diagnostic). Unobserved points are ``NaN``, which is how this package
        encodes visibility everywhere; a pair with either member missing simply does not
        vote.
    symmetries
        ``(S, 2)`` pairs, i.e. :attr:`deeperfly.skeleton.Skeleton.symmetries` (or
        :meth:`~deeperfly.skeleton.Skeleton.symmetries_or_inferred` when the skeleton may
        predate the field).
    min_pairs, min_separation_frac, min_margin_frac
        The gates documented at :data:`MIN_PAIRS`, :data:`MIN_SEPARATION_FRAC` and
        :data:`MIN_MARGIN_FRAC`.

    Returns
    -------
    Chirality
        With ``decided=False`` and a ``reason`` when the sample cannot be judged -- never
        a confident empty verdict standing in for "I could not tell".

    Notes
    -----
    The majority vote means this reports pairs that disagree with *the rest of this
    sample*, not with a global convention. A sample where **every** pair is swapped is
    therefore self-consistent and reports nothing -- correctly, because with no unpaired
    landmark and no calibration there is nothing in the sample to say which side is which.
    Catching a wholesale flip needs an outside reference: the camera's own handedness, or
    the 3D reconstruction. That limit is real and is why this is a *labeling* check.
    """
    pts = np.asarray(points, dtype=np.float64)
    pairs = np.asarray(symmetries, dtype=np.int64).reshape(-1, 2)
    if pairs.size == 0:
        return Chirality(reason="the skeleton declares no symmetry pairs")

    lo, hi = pairs[:, 0], pairs[:, 1]
    covisible = np.isfinite(pts[lo]).all(axis=-1) & np.isfinite(pts[hi]).all(axis=-1)
    if int(covisible.sum()) < min_pairs:
        return Chirality(
            reason=(
                f"only {int(covisible.sum())} symmetry pairs are co-visible "
                f"(need {min_pairs})"
            ),
            n_pairs=int(covisible.sum()),
        )

    used = pairs[covisible]
    disp = pts[used[:, 1]] - pts[used[:, 0]]
    axis, aniso = sagittal_axis(disp)
    proj = np.asarray(disp @ axis)  # signed separation of each pair along the axis
    # Point the axis at the majority side, so `axis` is reportable and the majority sign
    # is positive by construction. An exact zero sum would leave the sign arbitrary, which
    # is harmless: the verdict below is computed from the counts, not from the sign.
    if proj.sum() < 0:
        axis, proj = -axis, -proj

    scale = float(np.median(np.abs(proj)))
    # The animal's own size in whatever units the points came in, so the gate is
    # scale-free: the same threshold works at 1024 px, at 480 px, and in world mm.
    finite = pts[np.isfinite(pts).all(axis=-1)]
    diameter = float(np.linalg.norm(finite.max(axis=0) - finite.min(axis=0)))
    separation_frac = scale / diameter if diameter > 0 else 0.0
    if separation_frac < min_separation_frac:
        return Chirality(
            reason=(
                f"the left-right axis has collapsed in this sample: the pairs separate by "
                f"{separation_frac:.1%} of the animal's extent "
                f"(need {min_separation_frac:.0%}), so their order carries no information "
                "-- this is what a camera looking straight down the left-right axis sees"
            ),
            n_pairs=len(used),
            separation_frac=separation_frac,
            anisotropy=aniso,
        )
    strong = np.abs(proj) >= min_margin_frac * scale
    if int(strong.sum()) < min_pairs:
        return Chirality(
            reason=(
                f"only {int(strong.sum())} pairs are separated enough to judge "
                f"(need {min_pairs}); the rest sit within "
                f"{min_margin_frac:.0%} of the midline"
            ),
            n_pairs=int(strong.sum()),
            separation_frac=separation_frac,
            anisotropy=aniso,
        )

    # The majority sign among the pairs that are actually separated. A minority pair is a
    # swap candidate; a pair below the margin is not judged either way.
    majority = 1.0 if float(proj[strong].sum()) >= 0 else -1.0
    wrong = strong & (np.sign(proj) != majority)
    candidates = [
        SwapCandidate(
            points=(int(used[k, 0]), int(used[k, 1])),
            margin=float(abs(proj[k])),
            relative_margin=float(abs(proj[k]) / scale),
        )
        for k in np.nonzero(wrong)[0]
    ]
    candidates.sort(key=lambda c: -c.relative_margin)
    return Chirality(
        swapped=tuple(candidates),
        decided=True,
        axis=axis,
        n_pairs=int(strong.sum()),
        separation_frac=separation_frac,
        anisotropy=aniso,
    )


def check_views(
    points: Float[np.ndarray, "V P 2"],
    symmetries: Int[np.ndarray, "S 2"],
    **kwargs,
) -> list[Chirality]:
    """:func:`check` per view of a ``(V, P, 2)`` frame, in view order -- a **diagnostic**.

    Each view is judged independently, because they resolve the axis differently and a
    pooled verdict would let one good view be outvoted by six that cannot see the axis.

    Read the module docstring before trusting an entry: only a view whose image plane
    contains the left-right axis gives a sound answer, and a rig's oblique side cameras
    produce several false flags per pose from real posture asymmetry that no threshold can
    separate from a swap. This exists for the uncalibrated case, where there is no 3D to
    judge instead. When a 3D pose is available, call :func:`check` on it.
    """
    arr = np.asarray(points, dtype=np.float64)
    return [check(arr[v], symmetries, **kwargs) for v in range(arr.shape[0])]
