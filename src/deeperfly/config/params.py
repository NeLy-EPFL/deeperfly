"""The ``*Params`` dataclasses: every stage's parameters, and every default.

One frozen dataclass per config section. They are the single source of truth for the
defaults -- a default is written here exactly once, :class:`deeperfly.config.core.Config`
hands the dataclass out through a typed accessor, and
:mod:`deeperfly.config.schema` reads these same classes (and these docstrings) to
describe the config to the CLI and to the editor's settings forms. Nothing else may
carry a parallel default.

Only the *closed* sections live here. The open-ended ones -- cameras, the skeleton, the
detection plan, the video specs -- are returned as the domain objects their own parsers
build, because their shape is not a fixed set of fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

__all__ = [
    "Pose2dParams",
    "AutoCropParams",
    "TriangulationParams",
    "EksParams",
    "PostprocessParams",
    "StaticPointsParams",
    "SymmetrizeParams",
    "PictorialParams",
    "IoParams",
    "BundleAdjustmentParams",
    "InverseKinematicsParams",
    "AnnotationParams",
    "GuiParams",
]


@dataclass(frozen=True)
class Pose2dParams:
    """``[pose2d]`` -- the 2D detector performance knobs.

    ``batch_size`` is the GPU forward batch (images/forward); ``decode_buffer`` is
    the decode queue depth in multiples of it. Both are clamped to ``>= 1``, and
    both are performance-only (never fingerprinted). ``precision`` is the forward
    precision *default*: a per-model ``[[pose2d.models]].precision`` overrides it,
    falling back here when a model omits it (see
    :class:`~deeperfly.pose2d.models.ModelSpec`); it is result-affecting, so the
    resolved per-model value is fingerprinted. The *what to detect* (preprocessors,
    models, pathways, output points) is the detection plan that shares the
    ``[pose2d]`` table (:meth:`Config.detection_plan`), not these knobs.
    """

    precision: str = "float16"
    batch_size: int = 16
    decode_buffer: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "batch_size", max(1, int(self.batch_size)))
        object.__setattr__(self, "decode_buffer", max(1, int(self.decode_buffer)))


@dataclass(frozen=True)
class AutoCropParams:
    """``[pose2d.crop_search]`` -- how an ``auto_crops`` window is searched.

    The knobs are the ones a recording can genuinely need to differ on; the stencil's shape
    (how many widths, how many centres, how many narrowing rounds) is measured and lives as
    constants in :mod:`deeperfly.pose2d.autocrop`.

    ``search_frames`` are the frames the objective is scored on and ``gate_frames`` the
    disjoint set the accept gate uses; both are spread over the whole recording. Raise
    ``search_frames`` when the animal's distance drifts a lot through a run -- that is
    exactly the case a single static box serves worst, so a wider sample is the honest
    answer. ``gate = false`` takes whatever confidence proposed, which is measurably
    unsafe on its own (a box can get more confident *and* less accurate) and exists for
    rigs with no usable calibration. ``gate_candidates`` is how many confidence-shortlisted
    boxes the gate ranks: confidence is what can afford to cover a 3-D space, but its
    arg-max sits a little too wide, so a handful are re-scored by the better signal, and the
    best of them is where geometry starts its own search. ``gate_evals`` caps that search's
    detection passes -- the only real cost here (~350 ms each against ~11 ms for a
    confidence probe), so it is what bounds the wall clock.
    ``agreement_warn_px`` is the agreement above which a view is reported as *still* not
    framing the animal. ``probe_batch`` is the forward batch in probes and is
    performance-only.
    """

    search_frames: int = 3
    gate_frames: int = 8
    gate: bool = True
    gate_candidates: int = 4
    gate_evals: int = 30
    probe_batch: int = 8
    agreement_warn_px: float = 20.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "search_frames", max(1, int(self.search_frames)))
        object.__setattr__(self, "gate_frames", max(0, int(self.gate_frames)))
        object.__setattr__(self, "gate_candidates", max(1, int(self.gate_candidates)))
        object.__setattr__(self, "gate_evals", max(2, int(self.gate_evals)))
        object.__setattr__(self, "probe_batch", max(1, int(self.probe_batch)))


@dataclass(frozen=True)
class TriangulationParams:
    """``[triangulation]`` -- method + per-method thresholds."""

    method: str = "ransac"
    ransac_threshold: float = 15.0
    min_inliers: int = 2
    reproj_threshold: float = 40.0
    max_drops: int = 5
    weigh_by_confidence: bool = False


@dataclass(frozen=True)
class EksParams:
    """``[eks]`` -- the ensemble Kalman smoother, as a post-process over the 3D pose.

    The nonlinear multi-view EKS of Lightning Pose 3D (see :mod:`deeperfly.eks`).
    It fits one temporally-coherent 3D trajectory per keypoint directly against
    every view's pixels -- the observation model is the rig's own projection -- so
    it both de-jitters the pose and pulls a view whose detection blew up back onto
    the animal. It runs after triangulation and uses that stage's 3D as its
    starting point.

    ``smooth_param`` is the process-noise scale, the one knob that trades
    smoothness against responsiveness: smaller follows the dynamics and smooths
    harder, larger follows the detector. Omitted (the default), it is *fitted* per
    keypoint by maximum marginal likelihood, which is the right answer far more
    often than a hand-set value -- a fly's pretarsus and its thorax do not move alike.
    Set a number to override every keypoint at once.

    ``inflate_vars`` is the cross-view consistency check: each view's prediction is
    tested against what the other views say, and the ones that disagree beyond
    ``inflate_threshold`` have their observation variance multiplied by
    ``inflate_factor`` until they stop disagreeing, so the smoother down-weights
    them instead of following them. It needs two *views*, not two models, so on a
    calibrated rig it is fully active with a single detector -- and it is the
    component that repairs blown detections. Leave it on.

    ``inflate_threshold`` is the Mahalanobis distance above which a view is called
    inconsistent (the paper's example value is 5). Lower it to be more suspicious.
    Note that the test only means what its variances mean: with one detector the
    observation variance is ``1 / confidence``, whose absolute scale is arbitrary,
    so the fraction of observations it flags moves with the detector's calibration
    rather than with any fixed false-positive rate.

    ``inflate_factor`` multiplies an offending view's variance each round (the
    paper describes doubling; the reference CLI ships 10, which is the default
    here).

    ``ensemble`` lists other recordings' ``results.h5`` files holding a *different*
    detector's 2D for this same recording. With them the observation noise becomes
    a measured across-model spread instead of the ``1 / confidence`` stand-in --
    the half of "ensemble Kalman smoother" a single detector cannot supply. The
    ensembles in this literature are the same recipe trained differently (data
    subsets, initialization, ordering), not different architectures.

    ``avg_mode`` is how the ensemble members are combined into one center,
    ``"median"`` (robust, the default) or ``"mean"``. It does nothing with a single
    member.

    ``var_mode`` is how their spread becomes a variance: ``"var"`` is the plain
    across-model variance and ``"confidence_weighted_var"`` (default) divides it by
    the mean confidence, so a cell every model is unsure about is treated as
    noisier than its agreement alone suggests.

    ``fill_unobserved`` reports the smoothed 3D reprojected into views that never
    observed the keypoint, instead of leaving those cells NaN. Off by default:
    deeperfly reads NaN in the 2D as "not observed", and filling it hands every
    downstream consumer a prediction dressed as a measurement. Turn it on to use
    the smoother as a completion step for occluded joints.

    ``fit_frames`` caps how many leading frames the ``smooth_param`` fit may use
    (0 = all). Smoothing always runs over every frame; this only bounds the cost of
    estimating one scalar per keypoint, which a few thousand frames settle.

    ``fit_iterations`` is the number of golden-section steps per keypoint. Each
    shrinks the bracket by ~0.618, so the default resolves ``log(s)`` to about 1e-4.
    """

    smooth_param: float | None = None
    inflate_vars: bool = True
    inflate_threshold: float = 5.0
    inflate_factor: float = 10.0
    ensemble: list[str] = field(default_factory=list)
    avg_mode: str = "median"
    var_mode: str = "confidence_weighted_var"
    fill_unobserved: bool = False
    fit_frames: int = 2000
    fit_iterations: int = 24


@dataclass(frozen=True)
class PostprocessParams:
    """``[postprocess]`` -- an ordered chain of corrections applied to the 3D pose.

    Everything upstream estimates the pose from *pixels*. This stage applies what is
    known about the **animal** instead: that some keypoints do not move, that the body is
    bilaterally symmetric. Those are priors, not measurements, so they are kept out of
    the estimating stages and applied in one explicit, ordered place.

    ``ops`` is that chain -- a list of inline tables, each naming an ``op`` plus its own
    keys, exactly as ``[[pose2d.preprocessors]].ops`` describes a frame-op chain. Ordered
    because the ops do not commute, and one list because a new correction should cost a
    line here rather than a new pipeline stage::

        [postprocess]
        ops = [
            { op = "static", points = ["neck", "*_thorax_coxa"], method = "median" },
            { op = "symmetrize", points = ["l*_thorax_coxa"], midline = ["neck"] },
        ]

    Empty (the default) passes the pose through unchanged -- the stage still writes its
    output, so a downstream stage's input does not depend on whether the list happened
    to be filled in. See :mod:`deeperfly.pipeline.postprocess` for the ops and what each one
    assumes; an unknown ``op`` name, or an unknown key inside one, is a hard error.

    The stage runs **after** the smoother and before inverse kinematics, and that order
    is forced rather than chosen: the smoother re-derives the 3D from the 2D
    observations, so a correction applied before it is followed straight back off by the
    fit chasing its pixels.
    """

    ops: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class StaticPointsParams:
    """``{ op = "static" }`` -- keypoints held static over the whole recording.

    Some keypoints are not moving. On a *tethered* fly the six thorax-coxa joints and
    the neck sit on the sclerotized thorax, so their position is a constant of the
    recording and everything the estimate does over time is per-frame noise. The op
    replaces each listed point with a single **temporal center** in 3D. The 2D is left
    alone: after triangulation a stage's 2D is ``project(points3d)``, so the freeze
    reaches it through the projection rather than beside it.

    ``points`` names the skeleton points to hold static. An entry may be a point name or
    a ``*`` pattern (``["neck", "*_thorax_coxa"]`` is the packaged seven). Empty leaves
    the pose untouched; a name that is not in the skeleton, or a pattern matching nothing,
    is an error rather than a silent no-op.

    ``method`` picks *which* center. They differ only in what they assume about the
    contamination, and the right choice follows from what the estimate's error
    distribution actually looks like:

    - ``"median"`` (default) -- the per-axis temporal median. Breakdown point 50%, so a
      minority of blown frames cannot move it, and it is within ~2% of the mean's
      efficiency at the sample sizes a recording gives. Right until you have a reason.
    - ``"mean"`` -- the per-axis arithmetic mean. Minimum-variance if the residual really
      is clean Gaussian noise, and worth having for exactly that check; a single blown
      frame drags it, so it is the wrong default for detector output.
    - ``"trimmed_mean"`` -- the mean of the values inside ``[trim, 1 - trim]``, tunable
      between the two above. Trimming is by *value* threshold, not by exact order
      statistic, so ties may keep slightly more or fewer than the nominal fraction.
    - ``"mode"`` -- the half-sample mode (Robertson-Cryer): recursively keep the half of
      the sample with the smallest range. Parameter-free, unlike a histogram or KDE mode.
      It follows *density* where the median follows *count*, which matters when the
      contaminant is the **majority**: correct detections are tight and wrong ones are
      scattered, so a detector that locks onto a nearby wrong feature more often than not
      leaves a median out in the middle -- on a position the point never occupied -- and
      a mode still on the true peak. Below a 50% contaminant the median is already inside
      the true cluster and this buys nothing, at the cost of more variance.
    - ``"geometric_median"`` -- the multivariate L1 center (Weiszfeld), the only option
      here that is **rotation-equivariant**. The other four work axis by axis, so their
      answer depends on the orientation of the world frame -- an arbitrary choice of the
      rig. The difference is small for near-symmetric noise and real when the outliers
      are directional.

    ``trim`` is the fraction discarded from *each* tail by ``"trimmed_mean"``; it is
    ignored by every other method. It must be in ``[0, 0.5)``: at 0 the trimmed mean is
    the mean, and at 0.5 it would be the median with nothing left in between.
    """

    points: list[str] = field(default_factory=list)
    method: str = field(
        default="median",
        metadata={
            "choices": (
                "median",
                "mean",
                "trimmed_mean",
                "mode",
                "geometric_median",
            )
        },
    )
    trim: float = 0.1


@dataclass(frozen=True)
class SymmetrizeParams:
    """``{ op = "symmetrize" }`` -- impose bilateral symmetry on the body-fixed points.

    A fly is bilaterally symmetric, so a left keypoint and its right partner should be
    mirror images across the animal's **sagittal plane**. The estimate does not know
    that: each side is triangulated from its own cameras, so the pair drifts apart by
    whatever the two sides' errors differ by. This op fits the plane and enforces the
    relationship.

    ``pairs`` is the RESOLVED left/right pairs, and it is deliberately **not** defaulted
    from ``[skeleton] point_symmetries``. That list pairs every point including the legs, and at
    any instant a fly's left and right legs are in *different gait phases* -- that
    asymmetry is the behavior being measured. Symmetrizing it would be a serious
    corruption wearing the costume of a correction. Only body-fixed pairs belong here; on
    a tethered fly that is the six thorax-coxa joints.

    The config names one half of each pair in ``points`` (a selector, so
    ``["l*_thorax_coxa"]`` is the three) and :attr:`Config.postprocess` looks the partner
    up in the skeleton -- so which pairs are corrected stays a deliberate choice while the
    pairing itself is stated once, in the skeleton.

    ``midline`` names points that lie *on* the plane and are projected onto it. Also not
    inferred: "has no mirror partner" does not imply "on the midline". A fly's abdomen
    bends laterally, so ``abdomen0..4`` are unpaired and still off-plane; on ``fly38`` the
    honest midline set is ``neck`` alone.

    The plane is fitted from ``pairs`` only -- each pair's midpoint lies on it and each
    pair's difference vector is normal to it -- so an off-plane point named in ``midline``
    can never contaminate the fit that is about to move it.

    ``per_frame`` fits a fresh plane for every frame instead of one for the recording.
    Off by default, and that default is load-bearing: a single plane is a *fixed* map, so
    it leaves an already-static point static, and ``{ op = "static" }`` followed by this
    one satisfies both properties exactly. Fitted per frame the plane wobbles with the
    estimate, so symmetrizing after a freeze un-freezes it. Turn it on only for a
    preparation whose body genuinely moves in the world frame.

    ``strength`` scales the correction: 1.0 (default) makes each pair exactly symmetric,
    0.5 moves each side half-way to the mirror of the other, 0.0 is a no-op. Below 1.0 is
    the honest setting when you believe the symmetry only approximately -- a real animal
    is not perfectly symmetric, and the residual asymmetry may be signal.
    """

    pairs: list[list[str]] = field(default_factory=list)
    midline: list[str] = field(default_factory=list)
    per_frame: bool = False
    strength: float = 1.0


@dataclass(frozen=True)
class PictorialParams:
    """``[pictorial_structures]`` -- top-K peak-recovery knobs.

    ``k`` is the accuracy/cost dial, and the only one: the hypothesis pool is
    ``C(V, 2) * k**2`` per joint, so cost is quadratic in it (163.6 ms/frame at ``k = 5``
    against 97.5 at 3 and V=8, P=38). Measured on held-out animals, ``k = 3`` keeps
    88-100% of ``k = 5``'s improvement and ``k = 2`` keeps 54-91%, all still separating
    from zero -- so 3 is the setting to reach for when a long recording makes 5 too slow.

    Wherever recovery commits nothing, the arg-max is kept rather than a NaN. That is
    unconditional and not a knob: measured, recovery abstains on 1.5-2.2% of labeled
    cells, and filling them is 3D-NEUTRAL on this rig -- of the (frame, joint) pairs an
    abstention touches, a finite 3D came back for 95/95 and 146/146 either way, because
    the abstention drops ONE VIEW's observation and eight views trivially outvote its
    absence. So the choice only decides whether the stored 2D layer may be sparser than
    the detector that produced it, and a *correction* stage that deletes data is
    surprising. On a less redundant rig -- seven cameras, or a joint only two views see --
    the same fill stops being free and starts being protective.
    """

    k: int = 5
    temporal: bool = False
    lam: float = 1.0
    #: Ignore heatmap peaks weaker than this in RAW field units. The shipped 0.05 was set
    #: on a detector whose heatmaps peak near 1.0; the multiview transformer's peak near
    #: 0.08, where it admits a second candidate in 0.04% of cells and recovery can only
    #: return its own input. Absolute, so it is a claim about one detector's output scale.
    peak_threshold: float = 5e-2
    #: ... or than this fraction of the channel's OWN peak -- the scale-free form of the
    #: same gate, and the portable one. The effective threshold is the larger of the two,
    #: so leaving this at 0 is exactly today's behavior.
    peak_threshold_rel: float = 0.0


@dataclass(frozen=True)
class IoParams:
    """``[io.image]`` -- image-sequence decode parallelism (video I/O uses PyAV)."""

    image_workers: int | None = None


@dataclass(frozen=True)
class BundleAdjustmentParams:
    """``[bundle_adjustment]`` -- bundle adjustment over scipy ``least_squares``.

    ``points_to_use`` (``None`` = all) is the RESOLVED set of skeleton points that drive
    bundle adjustment -- the config writes ``points``, a selector whose entries may be
    ``*`` patterns, and :attr:`Config.bundle_adjustment` resolves it (the names are then
    mapped to indices in
    :func:`deeperfly.pipeline.stages.stage_bundle_adjustment`);
    ``fixed`` / ``shared`` hold or tie camera parameters; ``weigh_by_confidence``
    scales each reprojection residual by ``sqrt(confidence)``; ``max_frames`` /
    ``frame_sampling`` choose how many frames to bundle-adjust on and which (see
    :func:`deeperfly.pipeline.core._subsample`); ``least_squares`` is the leftover
    flat keys (``max_nfev``, ``loss``, ``f_scale``, ...) forwarded straight to
    :func:`scipy.optimize.least_squares`.
    """

    points_to_use: list[str] | None = None
    fixed: list[str] = field(default_factory=list)
    shared: list[list[str]] = field(default_factory=list)
    weigh_by_confidence: bool = True
    max_frames: int | None = 100
    frame_sampling: str = "even"
    least_squares: dict = field(default_factory=dict)


@dataclass(frozen=True)
class InverseKinematicsParams:
    """``[inverse_kinematics]`` -- fit a NeuroMechFly model's joint angles to the 3D pose.

    The solve is QuickIK's (an optional extra, ``deeperfly[ik]``): deeperfly builds a
    body plan for the recording (:mod:`deeperfly.inverse_kinematics.bodyplan`) and
    QuickIK fits the whole body -- every leg plus the head and abdomen -- against all
    the tracked keypoints at once.

    ``model`` names a packaged model **pack** (``"neuromechfly"``) or a path to one's
    ``model.toml``: the leg template, the baked articulation and the overlay mesh,
    selected as one unit, plus the model's rest axis and registration anchors
    (:mod:`deeperfly.inverse_kinematics.pack`). ``binding`` names the artifact saying
    where each tracked point sits on that model; it defaults to
    ``"<skeleton>@<model>"``, so a normal run writes nothing and an unbound pair is a
    load error naming both halves (:mod:`deeperfly.inverse_kinematics.binding`).
    ``legs`` restricts which legs are fit (``None`` = all);
    ``chains`` restricts which of the model's non-leg chains are fit (``None`` = every
    chain it defines, ``[]`` = legs only) -- one list rather than a boolean per chain,
    so a model that articulates something other than a head and an abdomen needs no new
    key; ``bounds`` holds per-DOF degree overrides keyed by the flygym joint angle name
    ``"<parent>-<child>-<dof>"`` (e.g. ``{"rf_trochanterfemur-rf_tibia-pitch": [10,
    160]}``).

    ``n_iterations``, ``neutral_weight``, ``damping``, ``position_tolerance`` and
    ``angle_tolerance`` are QuickIK's solver knobs (its ``SolverConfig``):
    Gauss-Newton steps per frame, the weight of the pull toward each DOF's neutral
    value (which is what pins DOFs the keypoints do not determine), the
    Levenberg-Marquardt damping, and the early-stop thresholds. The plan is solved in
    *model* units, so the two tolerances mean the same thing on any rig.

    ``damping`` defaults far above QuickIK's own suggested ~1e-6, and deliberately: the
    abdomen is five near-collinear hinges, so its Jacobian is ill-conditioned and a
    lightly-damped Gauss-Newton step overshoots into the joint limits. QuickIK enforces
    limits by clamping the step rather than projecting the gradient, so once an angle
    clamps it stays there and the solve deadlocks at a feasible but wrong pose --
    measurably: at 1e-2 an exactly-straight synthetic abdomen converges to a full curl.
    Damping is the remedy for the conditioning. It cannot go much higher either --
    the head, three DOFs about one pivot, is well conditioned and simply
    under-converges when over-damped -- so 0.1 is where every chain fits.

    ``fixed_body`` fixes the body in the model frame, which suits a tethered fly: the
    leg roots then sit at their measured medians and only the joint angles move. Set it
    false for a freely-moving preparation to give QuickIK a 6-DOF root to fit per frame.

    ``symmetric_segments`` gives each leg and its mirror image one shared length per
    segment -- the mean of the two sides' measurements -- instead of measuring the two
    sides independently. A fly's left and right femurs are the same bone, so at most one
    of the two measured lengths can be anatomy, and on the eight-view example rig they
    disagree by 4-6% on every femur. It constrains the *animal*, not its pose: the two
    sides' joint angles stay independent, which they must, because a leg's left/right
    asymmetry at any instant is the behavior.

    **On by default, and it costs a little accuracy against your own keypoints.** Every
    point of a leg chain is tracked, so the chain is over-determined and the per-leg
    lengths already are the best fit to those keypoints -- on the 8-view example recording,
    sharing them raised the 3D residual 22%, the reprojection 0.14 px in 7 of 8 views, and
    the left/right gap in each DOF's median angle from 4.7 to 6.4 degrees. It is the
    default anyway because the fitted model is a statement about an ANIMAL: one fly with
    two of each bone, whose angles are comparable across sides and whose morphology can be
    reported. Set it false to fit whatever the keypoints say, two half-animals included.
    See :func:`~deeperfly.inverse_kinematics.align.symmetrize_seglens`.

    ``weigh_by_confidence`` feeds the detector's per-keypoint confidence to the solver
    as observation weights instead of weighting every observed point equally.

    ``parallel`` solves long recordings in overlapping segments on worker threads
    (``segment_len`` frames each, sharing ``overlap_len``). Off by default: each segment
    restarts from the neutral pose and only warm-starts within itself, so the angle
    traces can step at a segment seam -- a poor trade for a joint-angle time series
    unless the recording is long enough to need it.

    ``markers`` redefines a chain's markers -- *where* each tracked keypoint sits
    relative to the model, the labeling-scheme choice. Written as
    ``[inverse_kinematics.markers.<chain>]``, each holding ``point -> {"body", "offset",
    "depth"?, "base"?}``; see
    :meth:`~deeperfly.inverse_kinematics.articulation.Articulation.load`. Under its own
    table rather than as two fixed chain names beside the knobs, so a chain can never
    collide with one.

    ``base = true`` nominates a marker as its chain's **base landmark** -- the point that
    says where the chain sits, the way a leg's thorax-coxa does. The packaged head chain
    nominates ``neck``, which sits on the head's own rotation center: it constrains none
    of the three head angles, and is not counted as evidence the head was observed, but
    it places the pivot far better than the coxa registration can (that fit extrapolates
    4.7x along the coxae's worst-determined axis, and lands ~11 degrees of pitch off).
    Because a table replaces its chain's whole marker set, a config that redeclares the
    head has to carry the nomination over or the chain silently reverts to the
    registered base.
    """

    model: str = "neuromechfly"
    binding: str | None = None
    legs: list[str] | None = None
    chains: list[str] | None = None
    n_iterations: int = 60
    neutral_weight: float = 1e-3
    damping: float = 0.1
    position_tolerance: float = 1e-3
    angle_tolerance: float = 1e-3
    fixed_body: bool = True
    symmetric_segments: bool = True
    weigh_by_confidence: bool = False
    parallel: bool = False
    segment_len: int = 200
    overlap_len: int = 10
    bounds: dict[str, list[float]] = field(default_factory=dict)
    markers: dict[str, dict] = field(default_factory=dict)


#: Every key ``[inverse_kinematics]`` accepts -- its own fields and nothing else, now
#: that the marker tables live under ``markers`` rather than beside the knobs.
IK_KEYS: frozenset[str] = frozenset(f.name for f in fields(InverseKinematicsParams))


@dataclass(frozen=True)
class AnnotationParams:
    """``[annotation]`` -- how the GUI turns 2D labels into a live 3D estimate.

    The keypoint editor is a *ground-truth annotation* tool: per ``(frame, point,
    view)`` the operator authors a GT 2D pixel and, on an independent axis, a "hidden" flag
    ("hold this cell out of the training loss", which the solve never reads), and
    the 3D point is a pure function of those labels plus the detector's predictions
    (``triangulate(active 2D, cameras, method, hyperparams)``). These knobs govern
    that function; the triangulation *method* + thresholds are shared with the batch
    pipeline (``[triangulation]``), so a point with no GT re-solves to the run's
    cached 3D exactly.

    ``solve_policy`` selects how GT and predictions combine in the live 3D solve:

    - ``"gt_wins"`` (default) -- GT is authoritative wherever it has an opinion, and
      the remaining evidence supplies only what GT cannot determine. With one GT view
      that is a *depth* along the GT's viewing ray
      (:func:`~deeperfly.gui.solve.solve_depth_on_ray`); with
      ``>= min_gt_for_exclusive`` GT views it is the ill-conditioned direction of the
      GT pair (:func:`~deeperfly.gui.solve.solve_point_3d_stabilized`, controlled by
      ``gt_wins_keep_stabilizers``); with no GT, use the configured
      ``[triangulation]`` method (so it matches the run). No policy ever discards a GT
      observation.
    - ``"equal_weight"`` -- per view use GT if present else the prediction, feed all
      to the configured method. ``equal_weight_protect_gt`` (default true) forces GT
      views to stay inliers so a prediction consensus cannot vote a human label out.
    - ``"weighted_blend"`` -- one weighted DLT, GT rows at ``gt_weight`` and
      prediction rows at ``prediction_weight`` (no RANSAC voting).

    ``prediction_weight`` is ``"uniform"`` (default, matches the batch
    ``weigh_by_confidence=false`` and the finding that peak confidence does not
    track correctness), ``"confidence"``, or a fixed float. ``undistort_before_solve``
    undistorts GT/prediction pixels before the linear DLT so a placed GT reprojects
    onto itself -- off by default because the batch pipeline does not undistort, so
    enabling it improves GT accuracy at the cost of a zero-GT re-solve no longer
    matching the run's cached 3D exactly (the nonlinear ``gt_wins`` paths are
    distortion-exact either way and ignore the flag).

    ``gt_wins_keep_stabilizers`` (**on by default**) keeps the remaining views in the
    solve once GT is exclusive, because two GT views do not determine a 3D point
    equally well in every direction. Two cameras facing each other -- ``rm`` and ``lm``
    of the standard rig are exactly anti-parallel -- have viewing rays that are nearly
    the same line, so the depth along that line is barely constrained: measured on the
    test rig, 0.5 px of click noise becomes 255 um mean / 372 um p90 of 3D error
    (p90 5.3 um at a 90-degree pair, a 70x penalty) and the point lands 37 px off in
    the views the operator did not label. Turning this off restores the older behavior
    of triangulating from the GT views alone.

    ``gt_sigma_px`` is the operator's click precision in pixels and is the *only*
    parameter of that stabilized solve: together with the detector's pixel scale
    (``[triangulation].ransac_threshold``, reused rather than duplicated) it sets how
    much more a click is trusted than a prediction. It replaces the unitless
    ``gt_weight`` in this path deliberately -- the best fixed weight was measured to
    move by 10x with the noise regime, while a ratio of measurable sigmas does not.
    """

    solve_policy: str = "gt_wins"
    min_gt_for_exclusive: int = 2
    gt_weight: float = 1000.0
    prediction_weight: str | float = "uniform"
    undistort_before_solve: bool = False
    equal_weight_protect_gt: bool = True
    gt_wins_keep_stabilizers: bool = True
    gt_sigma_px: float = 0.5


@dataclass(frozen=True)
class GuiParams:
    """``[gui]`` -- the correction GUI's display settings (no effect on the pipeline).

    ``mesh_hide`` lists the NeuroMechFly overlay body parts to hide in the editor
    (default ``["wings"]``); choose from ``wings`` / ``halteres`` / ``eyes`` /
    ``antennae`` / ``head`` / ``thorax`` / ``abdomen`` / ``legs``. The rendered
    videos carry their own ``[visualization].mesh_hide`` list.
    """

    mesh_hide: list[str] = field(default_factory=lambda: ["wings"])
