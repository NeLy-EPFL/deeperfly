"""The deeperfly run configuration: one ``Config`` class over the merged TOML.

A run is driven by a single TOML file (``deeperfly init`` writes the packaged
template :data:`DEFAULT_CONFIG_PATH`). :class:`Config` is the one place that file
is loaded, validated and handed to the code: every stage reads its parameters
through a typed accessor (:attr:`Config.pose2d`, :attr:`Config.triangulation`, ...)
whose defaults live in the small frozen ``*Params`` dataclasses below -- the single
source of truth, so a default is written exactly once.

The dynamic sections (cameras, skeleton, visualization, the detection plan)
are returned as the domain objects their own parsers already build
(:class:`~deeperfly.cameras.CameraGroup`, :class:`~deeperfly.skeleton.Skeleton`,
``list[VideoSpec]``, :class:`~deeperfly.pose2d.pathways.DetectionPlan`); only open-ended leaf
kwargs (a draw op's options, scipy's ``least_squares`` kwargs) stay dicts, carried
inside their typed parent.

``Config`` keeps the original TOML *text* so a run can snapshot it into the
output dir byte-for-byte (see :meth:`Config.save_snapshot`); a later run without
``-c`` picks the snapshot back up (:meth:`Config.read_for_run`), and the
per-stage fingerprints (:mod:`deeperfly.pipeline.fingerprint`) recompute exactly
the stages whose parameters changed.
"""

from __future__ import annotations

import dataclasses
import logging
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cameras import CameraGroup
    from .pose2d.pathways import DetectionPlan
    from .skeleton import Skeleton
    from .visualization.compose import VideoSpec

__all__ = [
    "Config",
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
    "DEFAULT_CONFIG_PATH",
    "SKELETON_PRESET_DIR",
    "skeleton_presets",
]

#: Packaged template emitted by ``deeperfly init`` (also the run-config example).
DEFAULT_CONFIG_PATH = Path(__file__).parent / "data" / "default_config.toml"

#: Packaged skeletons, referenced by ``[skeleton] name``. Each file holds a complete
#: ``[skeleton]`` table in the same format a project's own ``skeleton.toml`` is written in, so
#: a preset and a project's own ``skeleton.toml`` are interchangeable.
SKELETON_PRESET_DIR = Path(__file__).parent / "data" / "skeletons"

log = logging.getLogger("deeperfly")


#: The linear pipeline stages, in run order. Each is independently toggled by a
#: ``[pipeline].do_<stage>`` boolean (see :meth:`Config.stage_flags`) and
#: parameterized by its own top-level ``[<stage>]`` table.
STAGES = (
    "pose2d",
    "bundle_adjustment",
    "pictorial_structures",
    "triangulation",
    "eks",
    "postprocess",
    "inverse_kinematics",
    "visualization",
)

#: Default for each ``do_<stage>`` when the key is omitted: detection,
#: bundle adjustment, triangulation and visualization run by default; pictorial
#: structures, the ensemble Kalman smoother, the postprocess chain and
#: inverse kinematics are opt-in.
STAGE_DEFAULTS = {
    "pose2d": True,
    "bundle_adjustment": True,
    "pictorial_structures": False,
    "triangulation": True,
    "eks": False,
    "postprocess": False,
    "inverse_kinematics": False,
    "visualization": True,
}


# -- typed per-stage params: the single source of truth for every default ----


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
    """``[pose2d.autocrop]`` -- how a ``{ op = "crop", auto = true }`` window is searched.

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
    often than a hand-set value -- a fly's claw and its thorax do not move alike.
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
            { op = "static", points = ["neck", "lf_thorax_coxa"], method = "median" },
            { op = "symmetrize", pairs = [["lf_thorax_coxa", "rf_thorax_coxa"]],
              midline = ["neck"] },
        ]

    Empty (the default) passes the pose through unchanged -- the stage still writes its
    output, so a downstream stage's input does not depend on whether the list happened
    to be filled in. See :mod:`deeperfly.postprocess` for the ops and what each one
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
    replaces each listed point with a single **temporal center**, computed independently
    in each space the result carries: once in 3D, and once per view in 2D.

    ``points`` names the skeleton points to hold static, by their
    ``[skeleton].point_names``. Empty leaves the pose untouched. A name that is not in
    the skeleton is an error, not a silent no-op.

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

    ``pairs`` names the left/right pairs to symmetrize, and it is deliberately **not**
    defaulted from ``[skeleton].symmetries``. That list pairs every point including the
    legs, and at any instant a fly's left and right legs are in *different gait phases* --
    that asymmetry is the behavior being measured. Symmetrizing it would be a serious
    corruption wearing the costume of a correction. Only body-fixed pairs belong here; on
    a tethered fly that is the six thorax-coxa joints.

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
    """``[pictorial_structures]`` -- DeepFly3D peak-recovery knobs."""

    k: int = 5
    temporal: bool = False
    lam: float = 1.0


@dataclass(frozen=True)
class IoParams:
    """``[io.image]`` -- image-sequence decode parallelism (video I/O uses PyAV)."""

    image_workers: int | None = None


@dataclass(frozen=True)
class BundleAdjustmentParams:
    """``[bundle_adjustment]`` -- bundle adjustment over scipy ``least_squares``.

    ``points_to_use`` (``None`` = all) names which skeleton points drive bundle adjustment
    (resolved to indices against the skeleton in :func:`deeperfly.pipeline.stages.stage_bundle_adjustment`);
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

    ``template`` names a packaged kinematic template (``"neuromechfly"``) or a path
    to a template TOML; ``legs`` restricts which legs are fit (``None`` = all);
    ``bounds`` holds per-DOF degree overrides keyed by the flygym joint angle name
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

    Off by default because it is a prior that **costs**, measurably. Every point of a leg
    chain is tracked, so the chain is over-determined and the per-leg lengths already are
    the best fit to the keypoints -- on that recording, sharing them raised the 3D
    residual 22%, the reprojection 0.14 px in 7 of 8 views, and the left/right gap in
    each DOF's median angle from 4.7 to 6.4 degrees. Turn it on for what it makes true of
    the model, not for accuracy; see
    :func:`~deeperfly.inverse_kinematics.align.symmetrize_seglens`.

    ``weigh_by_confidence`` feeds the detector's per-keypoint confidence to the solver
    as observation weights instead of weighting every observed point equally.

    ``parallel`` solves long recordings in overlapping segments on worker threads
    (``segment_len`` frames each, sharing ``overlap_len``). Off by default: each segment
    restarts from the neutral pose and only warm-starts within itself, so the angle
    traces can step at a segment seam -- a poor trade for a joint-angle time series
    unless the recording is long enough to need it.

    ``markers`` redefines the head/abdomen chain markers -- *where* each tracked
    keypoint sits relative to the model, the labeling-scheme choice. It is keyed by
    chain name (``"head"`` / ``"abdomen"``), each holding ``point -> {"body", "offset",
    "depth"?, "base"?}`` (from the ``[inverse_kinematics.head]`` /
    ``[inverse_kinematics.abdomen]`` config tables); see
    :meth:`~deeperfly.inverse_kinematics.articulation.Articulation.load`.

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

    template: str = "neuromechfly"
    legs: list[str] | None = None
    fit_head: bool = True
    fit_abdomen: bool = True
    n_iterations: int = 60
    neutral_weight: float = 1e-3
    damping: float = 0.1
    position_tolerance: float = 1e-3
    angle_tolerance: float = 1e-3
    fixed_body: bool = True
    symmetric_segments: bool = False
    weigh_by_confidence: bool = False
    parallel: bool = False
    segment_len: int = 200
    overlap_len: int = 10
    bounds: dict[str, list[float]] = field(default_factory=dict)
    markers: dict[str, dict] = field(default_factory=dict)


#: Every key ``[inverse_kinematics]`` accepts. Derived from
#: :class:`InverseKinematicsParams` plus the two marker sub-tables (which are read into
#: its ``markers`` field), so the strict-validation error message cannot drift from the
#: keys actually parsed.
IK_KEYS: frozenset[str] = frozenset(
    {f.name for f in fields(InverseKinematicsParams) if f.name != "markers"}
    | {"head", "abdomen"}
)


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


# -- helpers -----------------------------------------------------------------


#: The detection-plan sub-tables that share the ``[pose2d]`` table with its
#: runtime knobs (see :meth:`Config.detection_plan`). They are parsed separately
#: (:meth:`deeperfly.pose2d.pathways.DetectionPlan.from_config`), so the strict
#: :func:`_params` validator ignores them when building :class:`Pose2dParams`.
_POSE2D_PLAN_KEYS = frozenset(
    {"preprocessors", "model", "models", "pathways", "output_points", "autocrop"}
)


def _dig(data: dict, path: tuple[str, ...]) -> dict:
    """The nested sub-table at ``path`` (e.g. ``("pose2d",)``), or ``{}``."""
    for key in path:
        data = data.get(key, {}) if isinstance(data, dict) else {}
    return data if isinstance(data, dict) else {}


def _params(data: dict, path: tuple[str, ...], cls, *, ignore: frozenset = frozenset()):
    """Build a frozen ``*Params`` from the sub-table at ``path``.

    Keys absent from the table fall through to the dataclass field defaults (the
    single source of truth).

    Parameters
    ----------
    data
        The parsed config mapping.
    path
        Key path to the sub-table, e.g. ``("pose2d",)``.
    cls
        The frozen ``*Params`` dataclass to build.
    ignore
        Keys to skip -- neither validated nor passed to ``cls``. Used for the
        ``[pose2d]`` table, which also holds the detection-plan sub-tables
        (:data:`_POSE2D_PLAN_KEYS`).

    Returns
    -------
    The populated ``cls`` instance.

    Raises
    ------
    ValueError
        If the sub-table holds a key the dataclass does not define (a typo);
        the message names the section and the allowed keys.
    """
    sub = _dig(data, path)
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(sub) - fields - ignore)
    if unknown:
        loc = "[" + ".".join(path) + "]"
        raise ValueError(
            f"{loc} has unknown key(s) {unknown}; allowed: {sorted(fields)}"
        )
    return cls(**{k: v for k, v in sub.items() if k in fields})


def _source_filename(filename, name: str) -> str | list[str]:
    """Validate a ``[[sources]]`` ``filename`` value (a glob or list of globs).

    Parameters
    ----------
    filename
        The raw ``filename`` value from the config (a string, or a list of
        strings -- alternate globs tried in order).
    name
        The source's name, for the error message.

    Returns
    -------
    str or list of str
        The validated ``filename`` value, unchanged.

    Raises
    ------
    ValueError
        If ``filename`` is not a string or a list of strings.
    """
    if isinstance(filename, str):
        return filename
    if isinstance(filename, list) and all(isinstance(f, str) for f in filename):
        return filename
    raise ValueError(
        f"[[sources]] {name!r} 'filename' must be a string or list of strings, "
        f"got {filename!r}"
    )


#: Per-camera keys that a config may no longer carry, ``key -> what to write instead``.
#:
#: A key that was quietly dropped is worse than one that errors. ``preprocess`` cropped and
#: turned a view's frames, and was superseded by the detection pathway's own op chain --
#: which is the better mechanism for a reason worth stating, because it decides which of the
#: two a run can have: a pathway's ops are **inverted on the way back**, so a detection
#: reaches its camera in RAW footage pixels however it was windowed to get to the model, and
#: the camera's intrinsics go on describing the raw frame. The retired key instead moved the
#: CAMERA into cropped-pixel space. Both live at once is a silent double correction -- a fly
#: reprojecting off by exactly the crop offset, with nothing to point at -- so there is no
#: version of this where the key is honored alongside the pathway.
#:
#: It survived as accepted-and-ignored, which is the worst place for it to be: a crop is
#: exactly what a wrongly-framed axial camera needs, so the key that did nothing was the one
#: a reader of the docs reached for in the situation where being wrong costs most.
RETIRED_CAMERA_KEYS = {
    "preprocess": (
        "frame ops moved to the detection pathway, where the transform is inverted on the "
        "way back so the detections still land in raw footage pixels:\n"
        "    [pose2d]\n"
        '    preprocessors = [{ name = "crop_f", ops = [{ op = "crop", x = 400, y = 290, '
        "width = 800, height = 400 }] }]\n"
        '    pathways = [{ name = "f", source = "vid_f", preprocessor = "crop_f" }]\n'
        "  ...or `auto = true` in place of the box to search one per recording; see "
        "docs/reference/configuration.md#autocrop"
    ),
}


def _refuse_retired_camera_keys(defaults: dict, views: dict[str, dict]) -> None:
    """Raise on a ``[cameras.*]`` key this release no longer honors.

    Checked here rather than at the rig parser because the rig parser never sees these:
    they are stripped as non-geometry before a spec reaches
    :meth:`~deeperfly.cameras.Camera.from_spec`, which is exactly how one went on being
    accepted and ignored.
    """
    for where, spec in [("defaults", defaults), *views.items()]:
        for key, advice in RETIRED_CAMERA_KEYS.items():
            if key in spec:
                raise ValueError(
                    f"[cameras.{where}] carries {key!r}, which this release no longer "
                    f"honors -- it was accepted and silently ignored before, so a config "
                    f"relying on it was already running without it.\n  {advice}"
                )


# -- skeleton presets ---------------------------------------------------------


#: Retired preset names that still resolve, ``old name -> packaged name``.
#:
#: A rename is not a data migration: ``fly38b`` became ``fly38`` when it was left as the
#: only packaged skeleton, and not one coordinate moved. But a *reference* to it is on
#: disk in every config and every output-directory snapshot written before that, and
#: those would otherwise fail to load at all -- so the old spelling keeps working. It
#: resolves to the same file, and the config's own ``name`` still wins, so such a run
#: goes on calling its skeleton ``fly38b`` and computes exactly what it always did (the
#: skeleton's name is not in any fingerprint -- see
#: :func:`deeperfly.pipeline.fingerprint._skeleton_digest`).
#:
#: This is deliberately not symmetric. ``fly38`` now means the midline-abdomen set, and
#: an old config naming it means the DeepFly3D one -- a different point order under the
#: same word, which no alias can disentangle. What stands there instead is a check on the
#: points themselves: the detector refuses a checkpoint whose channels are another
#: skeleton's (:func:`deeperfly.pose2d.stream._check_channel_names`) and a run refuses an
#: output directory whose stored pose is
#: (:func:`deeperfly.pipeline.run._refuse_a_foreign_skeleton`).
SKELETON_ALIASES = {"fly38b": "fly38"}


def skeleton_presets() -> dict[str, Path]:
    """The packaged skeletons, ``name -> file`` (see :data:`SKELETON_PRESET_DIR`).

    Retired spellings from :data:`SKELETON_ALIASES` are not listed -- they resolve, but
    an error message offering them as choices would be advertising a name the release no
    longer has.
    """
    if not SKELETON_PRESET_DIR.is_dir():
        return {}
    return {p.stem: p for p in sorted(SKELETON_PRESET_DIR.glob("*.toml"))}


def _resolve_skeleton(data: dict, source: Path | None) -> dict:
    """Expand a ``[skeleton]`` table that *references* a skeleton instead of spelling one.

    A skeleton is 80-odd lines of point names, mirror pairs, limb chains and colors that
    almost never differ between recordings of the same animal -- and when it does differ it
    differs completely (the DeepFly3D set and ``fly38`` are both 38 points and share 32 of
    them, in a different order). So it belongs in a file that a config *names*, not in
    every config:

    * ``name = "fly38"`` -- a packaged preset (:func:`skeleton_presets`), or a retired
      spelling of one (:data:`SKELETON_ALIASES`).
    * ``file = "skeleton.toml"`` -- a path, resolved next to the config that wrote it.
      The format is the same, which is what makes a preset and a project's own
      ``skeleton.toml`` interchangeable.

    A table that spells out ``point_names`` is already self-contained and is left exactly
    as it is -- so every config written before presets existed keeps its own meaning, and
    ``name`` goes on being a free-text label for those. Resolution happens once, at
    :class:`Config` construction, so everything downstream (the skeleton itself, the
    fingerprints, ``deeperfly config show``) sees one fully-populated table and needs to
    know nothing about presets.

    Keys written in the config win **wholesale, per key**: a ``limb_palette`` there
    replaces the referenced one rather than merging entry by entry, because a half-merged
    palette keyed on limbs that the override renamed is not a thing anyone means.

    Returns
    -------
    dict
        ``data`` with ``[skeleton]`` expanded. The input is not mutated.

    Raises
    ------
    ValueError
        If the reference names no packaged preset / no readable file, or if the
        referenced file carries no ``[skeleton]`` table.
    """
    skel = data.get("skeleton")
    if not isinstance(skel, dict) or "point_names" in skel:
        return data

    ref, presets = skel.get("file"), skeleton_presets()
    if ref:
        path = Path(ref)
        if not path.is_absolute() and source is not None:
            path = source.parent / path
        if not path.is_file():
            raise ValueError(
                f"[skeleton] file = {ref!r} does not exist (looked in {path})"
            )
    elif skel.get("name") in presets:
        path = presets[skel["name"]]
    elif SKELETON_ALIASES.get(skel.get("name")) in presets:
        current = SKELETON_ALIASES[skel["name"]]
        path = presets[current]
        log.info(
            "[skeleton] name = %r is the former spelling of %r and resolves to it; the "
            "points are unchanged",
            skel["name"],
            current,
        )
    else:
        raise ValueError(
            f"[skeleton] declares no 'point_names' and name = {skel.get('name')!r} is not "
            f"a packaged skeleton (have: {sorted(presets)}). Either spell the skeleton out "
            "in this table, name a packaged one, or point 'file' at a skeleton.toml."
        )

    loaded = tomllib.loads(path.read_text()).get("skeleton")
    if not isinstance(loaded, dict) or "point_names" not in loaded:
        raise ValueError(
            f"{path} carries no [skeleton] table with 'point_names'; it is not a skeleton "
            "file (the format is a project's own skeleton.toml, or a packaged preset)"
        )
    log.info("skeleton %r from %s", loaded.get("name", path.stem), path)
    return {**data, "skeleton": {**loaded, **skel}}


# -- the Config class --------------------------------------------------------


class Config:
    """A loaded, validated deeperfly run configuration.

    Construct via :meth:`from_toml` (from a TOML file) or :meth:`from_dict` (a
    parsed mapping). The parsed mapping is :attr:`data`; :attr:`text` is the original
    TOML text when read from a file (``None`` for a dict), used to snapshot the config
    byte-for-byte.
    """

    def __init__(
        self, data: dict, *, text: str | None = None, source: Path | None = None
    ):
        #: The parsed config. A ``[skeleton]`` table that merely *references* a skeleton
        #: (see :func:`_resolve_skeleton`) is expanded here, once, so nothing downstream
        #: has to know presets exist. :attr:`text` keeps the reference, which is what the
        #: snapshot records -- the resolved point names reach the pose2d fingerprint, so a
        #: preset that changes under a cached run invalidates it rather than passing.
        self.data = _resolve_skeleton(data, source)
        self.text = text
        self.source = source
        #: Windows for the config's ``{ op = "crop", auto = true }`` preprocessors, once
        #: something has decided them -- ``preprocessor name -> (x, y, width, height)``.
        #: :meth:`detection_plan` applies them, so every stage of a run sees the same box
        #: the detector looked through. Populated by :meth:`read_for_run` from the recorded
        #: sidecar, and by the pose2d stage when it searches (see
        #: :mod:`deeperfly.pose2d.autocrop`). Not part of :attr:`data`, so it never enters
        #: the config snapshot or a fingerprint -- the declaration is what those describe.
        self.auto_crops: dict[str, tuple[int, int, int, int]] = {}

    # -- construction --------------------------------------------------------

    @classmethod
    def from_toml(cls, path: str | Path) -> "Config":
        """Load a config from a TOML file (preserving its text for snapshots).

        Parameters
        ----------
        path
            Path to a config TOML file.

        Returns
        -------
        Config
            The loaded, validated config.
        """
        p = Path(path)
        text = p.read_text()
        return cls(tomllib.loads(text), text=text, source=p)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Wrap an already-parsed mapping (library use, tests).

        Parameters
        ----------
        data
            A parsed config mapping.

        Returns
        -------
        Config
            The validated config wrapping ``data`` (no source text).
        """
        return cls(data)

    @classmethod
    def default(cls) -> "Config":
        """The packaged default config (:data:`DEFAULT_CONFIG_PATH`)."""
        return cls.from_toml(DEFAULT_CONFIG_PATH)

    @classmethod
    def read_for_run(cls, cli_config: str | None, outdir: Path) -> "Config":
        """Pick the config for one run: ``-c`` wins, then the ``outdir`` snapshot.

        An explicit ``-c`` always drives the run (and refreshes the snapshot --
        see :meth:`save_snapshot`). Without ``-c``, the snapshot a previous run
        left in ``<outdir>/config.toml`` is reused -- so both "pass a new ``-c``"
        and "edit the snapshot and re-run" work; either way the per-stage
        fingerprints (:mod:`deeperfly.pipeline.fingerprint`) recompute exactly
        the stages whose parameters changed. With neither, the packaged default
        is used.

        Parameters
        ----------
        cli_config
            The ``-c`` config path, or ``None``.
        outdir
            The run's output directory, which may already hold a ``config.toml``
            snapshot.

        Any automatic crop window a previous run recorded in ``outdir`` is loaded into
        :attr:`auto_crops`, so a resume that reuses the cached 2D pose still knows which
        window the detector looked through -- the visualization panels that borrow it, in
        particular, are drawn in a later process than the search.

        Returns
        -------
        Config
            The config that drives this run.
        """
        snapshot = Path(outdir) / "config.toml"
        if cli_config:
            path = Path(cli_config)
            log.info("using config %s (from -c)", path)
        elif snapshot.exists():
            path = snapshot
            log.info("using config %s (snapshot in the output dir)", path)
        else:
            path = DEFAULT_CONFIG_PATH
            log.info("using config %s (packaged default; pass -c to override)", path)
        config = cls.from_toml(path)
        from .pose2d.autocrop import read_sidecar

        config.auto_crops = dict(read_sidecar(outdir))
        return config

    # -- typed per-stage subgroups ------------------------------------------

    @property
    def pose2d(self) -> Pose2dParams:
        return _params(self.data, ("pose2d",), Pose2dParams, ignore=_POSE2D_PLAN_KEYS)

    @property
    def autocrop(self) -> AutoCropParams:
        return _params(self.data, ("pose2d", "autocrop"), AutoCropParams)

    @property
    def triangulation(self) -> TriangulationParams:
        return _params(self.data, ("triangulation",), TriangulationParams)

    @property
    def eks(self) -> EksParams:
        return _params(self.data, ("eks",), EksParams)

    @property
    def postprocess(self) -> PostprocessParams:
        return _params(self.data, ("postprocess",), PostprocessParams)

    @property
    def pictorial(self) -> PictorialParams:
        return _params(self.data, ("pictorial_structures",), PictorialParams)

    @property
    def io(self) -> IoParams:
        im = _dig(self.data, ("io", "image"))
        present: dict = {}
        if "workers" in im:
            present["image_workers"] = int(im["workers"]) or None
        return IoParams(**present)

    @property
    def bundle_adjustment(self) -> BundleAdjustmentParams:
        ba = dict(_dig(self.data, ("bundle_adjustment",)))
        points_to_use = ba.pop("points_to_use", None)
        fixed = ba.pop("fixed", [])
        shared = ba.pop("shared", [])
        weigh_by_confidence = ba.pop("weigh_by_confidence", True)
        max_frames = ba.pop("max_frames", 100)
        frame_sampling = ba.pop("frame_sampling", "even")
        return BundleAdjustmentParams(
            points_to_use=None
            if points_to_use is None
            else [str(p) for p in points_to_use],
            fixed=list(fixed),
            shared=[list(s) for s in shared],
            weigh_by_confidence=bool(weigh_by_confidence),
            max_frames=None if max_frames is None else int(max_frames),
            frame_sampling=str(frame_sampling),
            least_squares=ba,  # leftover flat keys -> scipy.optimize.least_squares
        )

    @property
    def inverse_kinematics(self) -> InverseKinematicsParams:
        ik = dict(_dig(self.data, ("inverse_kinematics",)))
        bounds = {
            str(k): [float(b) for b in v] for k, v in ik.pop("bounds", {}).items()
        }
        # [inverse_kinematics.head] / [inverse_kinematics.abdomen]: per-chain marker
        # placement (point -> {body, offset, depth?}), the labeling-scheme choice.
        markers = {
            chain: {str(p): dict(spec) for p, spec in ik.pop(chain).items()}
            for chain in ("head", "abdomen")
            if chain in ik
        }
        defaults = InverseKinematicsParams()
        template = str(ik.pop("template", defaults.template))
        legs = ik.pop("legs", None)
        fit_head = ik.pop("fit_head", defaults.fit_head)
        fit_abdomen = ik.pop("fit_abdomen", defaults.fit_abdomen)
        n_iterations = ik.pop("n_iterations", defaults.n_iterations)
        neutral_weight = ik.pop("neutral_weight", defaults.neutral_weight)
        damping = ik.pop("damping", defaults.damping)
        position_tolerance = ik.pop("position_tolerance", defaults.position_tolerance)
        angle_tolerance = ik.pop("angle_tolerance", defaults.angle_tolerance)
        fixed_body = ik.pop("fixed_body", defaults.fixed_body)
        symmetric_segments = ik.pop("symmetric_segments", defaults.symmetric_segments)
        weigh_by_confidence = ik.pop(
            "weigh_by_confidence", defaults.weigh_by_confidence
        )
        parallel = ik.pop("parallel", defaults.parallel)
        segment_len = ik.pop("segment_len", defaults.segment_len)
        overlap_len = ik.pop("overlap_len", defaults.overlap_len)
        if ik:  # any leftover key is a typo -- match _params' strict validation
            raise ValueError(
                f"[inverse_kinematics] has unknown key(s) {sorted(ik)}; "
                f"allowed: {sorted(IK_KEYS)}"
            )
        return InverseKinematicsParams(
            template=template,
            legs=None if legs is None else [str(leg) for leg in legs],
            fit_head=bool(fit_head),
            fit_abdomen=bool(fit_abdomen),
            n_iterations=int(n_iterations),
            neutral_weight=float(neutral_weight),
            damping=float(damping),
            position_tolerance=float(position_tolerance),
            angle_tolerance=float(angle_tolerance),
            fixed_body=bool(fixed_body),
            symmetric_segments=bool(symmetric_segments),
            weigh_by_confidence=bool(weigh_by_confidence),
            parallel=bool(parallel),
            segment_len=int(segment_len),
            overlap_len=int(overlap_len),
            bounds=bounds,
            markers=markers,
        )

    @property
    def gui(self) -> GuiParams:
        return _params(self.data, ("gui",), GuiParams)

    @property
    def annotation(self) -> AnnotationParams:
        return _params(self.data, ("annotation",), AnnotationParams)

    def ik_template(self):
        """The configured kinematic template (``[inverse_kinematics].template`` + bounds).

        Returns
        -------
        deeperfly.inverse_kinematics.template.KinematicTemplate
            The packaged or path-loaded template, with the legs restricted and the
            ``[inverse_kinematics.bounds]`` degree overrides applied.
        """
        from .inverse_kinematics.template import KinematicTemplate

        p = self.inverse_kinematics
        overrides = {k: (v[0], v[1]) for k, v in p.bounds.items()}
        return KinematicTemplate.load(
            p.template, legs=p.legs, bounds_overrides=overrides
        )

    def ik_articulation(self):
        """The configured head/abdomen articulation, or ``None`` if neither is fit.

        Returns
        -------
        deeperfly.inverse_kinematics.articulation.Articulation or None
            The baked chains selected by ``[inverse_kinematics].fit_head`` /
            ``fit_abdomen``, with ``[inverse_kinematics.bounds]`` degree overrides
            (keys like ``c_thorax-c_head-pitch`` / ``c_abdomen12-c_abdomen3-pitch``) and any
            ``[inverse_kinematics.head]`` / ``[inverse_kinematics.abdomen]`` marker
            placement overrides applied.
        """
        from .inverse_kinematics.articulation import Articulation

        p = self.inverse_kinematics
        fit = tuple(
            name
            for name, on in (("head", p.fit_head), ("abdomen", p.fit_abdomen))
            if on
        )
        if not fit:
            return None
        overrides = {k: (v[0], v[1]) for k, v in p.bounds.items()}
        return Articulation.load(
            fit=fit, bounds_overrides=overrides, marker_overrides=p.markers
        )

    # -- pipeline orchestration ---------------------------------------------

    def stage_flags(self) -> dict[str, bool]:
        """Which stages are enabled, from the ``[pipeline].do_<stage>`` booleans.

        Returns
        -------
        dict of str to bool
            ``stage_name -> enabled`` for every stage in :data:`STAGES`, each
            defaulting to :data:`STAGE_DEFAULTS`.
        """
        pipe = self.data.get("pipeline", {})
        return {n: bool(pipe.get(f"do_{n}", STAGE_DEFAULTS[n])) for n in STAGES}

    # -- structured sections: the domain objects their parsers build --------

    @property
    def visualization(self) -> dict:
        """The raw ``[visualization]`` table (consumed by :attr:`videos`)."""
        return self.data.get("visualization", {})

    @property
    def videos(self) -> "list[VideoSpec]":
        """The output-video specs (``[[visualization.videos]]``)."""
        from .visualization.compose import read_video_specs

        return read_video_specs(self)

    def camera_group(self, image_sizes=None) -> "CameraGroup":
        """The configured camera rig (``[cameras.*]``).

        Parameters
        ----------
        image_sizes
            Optional ``camera_name -> (height, width)`` used to infer principal
            points when a camera omits ``principal_point_px``.

        Returns
        -------
        CameraGroup
            The configured rig.
        """
        from .cameras import CameraGroup

        return CameraGroup.from_config(self, image_sizes=image_sizes)

    def skeleton(self) -> "Skeleton":
        """The configured skeleton (``[skeleton]``), or the default fly skeleton."""
        from .skeleton import Skeleton

        return Skeleton.from_config(self) if "skeleton" in self.data else Skeleton.fly()

    def mirror_views(self) -> dict[str, str]:
        """``view -> the view that sees this view's mirror image`` (``[cameras.<n>].mirror``).

        Only the views that declare it; empty when none do. Consumed by flip augmentation
        (which lives outside this package): a mirrored training sample must be told which
        camera it now *looks like*, or any metric that splits by camera side -- ipsilateral
        versus contralateral error, the one that matters most on this rig -- silently calls
        every swapped channel by the wrong side.

        The mapping must be an **involution**: if ``rf`` mirrors to ``lf`` then ``lf`` must
        mirror to ``rf``, and a camera on the midline (the front view, whose mirror is still
        a front view) declares itself. Validated here, because a half-declared pairing is
        the kind of thing that reads as correct and behaves as a silent side swap on one
        camera only.

        Declared rather than derived from the extrinsics on purpose. Which camera is the
        mirror of which is a fact about how the rig was built, and it stays true before
        there is any calibration to compute it from.

        Raises
        ------
        ValueError
            If a ``mirror`` names an unknown view, or if the relation is not symmetric.
        """
        _, cams = self.camera_table()
        out = {
            name: str(spec["mirror"])
            for name, spec in cams.items()
            if spec.get("mirror")
        }
        for name, other in out.items():
            if other not in cams:
                raise ValueError(
                    f"[cameras.{name}].mirror names unknown view {other!r}; "
                    f"views: {sorted(cams)}"
                )
            back = out.get(other)
            if back != name:
                got = "nothing" if back is None else repr(back)
                raise ValueError(
                    f"[cameras.{name}].mirror = {other!r} but [cameras.{other}].mirror "
                    f"is {got}; the mirror relation must be symmetric (a midline camera "
                    f"mirrors to itself)"
                )
        return out

    def detection_plan(self) -> "DetectionPlan":
        """The 2D detection plan (``[[sources]]`` + ``[[pose2d.preprocessors]]``/``[[pose2d.models]]``/``[[pose2d.pathways]]``).

        Any window in :attr:`auto_crops` is substituted into the matching
        ``{ op = "crop", auto = true }`` preprocessor, so a stage that re-derives the plan
        (a render reusing a cached 2D pose, say) looks through the same box detection did.
        An automatic crop with no window recorded stays unresolved and raises
        :class:`~deeperfly.preprocessing.UnresolvedAutoCrop` if anything asks for its
        geometry -- which is the loud failure, not a silent full frame.

        Returns
        -------
        DetectionPlan
            The parsed, validated plan mapping footage sources through
            preprocessors and models into the skeleton (see
            :class:`deeperfly.pose2d.pathways.DetectionPlan`).
        """
        from .pose2d.pathways import DetectionPlan

        plan = DetectionPlan.from_config(self)
        if self.auto_crops:
            from .pose2d.autocrop import resolved_plan

            plan = resolved_plan(plan, self.auto_crops)
        return plan

    def camera_table(self) -> tuple[dict, dict]:
        """Split ``[cameras]`` into the shared defaults and the per-camera specs.

        Returns
        -------
        defaults, cameras : dict
            The ``[cameras.defaults]`` spec and the real per-camera specs (keyed
            by name, with ``defaults`` and the scalar ``calibration`` key excluded).
        """
        cams = dict(self.data.get("cameras", {}))
        defaults = cams.pop("defaults", {})
        cams.pop("calibration", None)  # a path, not a camera -- see calibration_path
        # A non-table value under [cameras] is a key, not a view; keeping one would
        # reach Camera.from_spec as a spec and fail with a confusing message.
        views = {k: v for k, v in cams.items() if isinstance(v, dict)}
        _refuse_retired_camera_keys(defaults, views)
        return defaults, views

    def calibration_path(self) -> Path | None:
        """The solved rig this config points at (``[cameras].calibration``), if any.

        A relative path resolves against the **config file's own directory**, so a
        config and the calibration beside it travel together (and an output-dir
        snapshot keeps finding the rig snapshotted next to it). A config built from a
        dict has no directory, so a relative path resolves against the process's
        working directory.

        Returns
        -------
        Path or None
            The calibration file/directory, or ``None`` when the config specifies the
            rig as an orbit (the default).
        """
        raw = self.data.get("cameras", {}).get("calibration")
        if raw is None:
            return None
        path = Path(str(raw))
        if path.is_absolute() or self.source is None:
            return path
        return self.source.parent / path

    def source_patterns(self) -> dict[str, str | list[str]]:
        """Map each footage source to its glob (``[[sources]]`` ``name`` -> ``filename``).

        Read directly from the ``[[sources]]`` table (without building the whole
        detection plan) so recording discovery stays cheap. A source with no
        ``filename`` key uses its own name as the glob pattern. ``filename`` may be
        a single glob or a list of alternate globs tried in order (the first that
        resolves footage wins), so a source can name both its anatomical file and a
        legacy fallback (``["camera_RH.mp4", "camera_0.mp4"]``).

        Returns
        -------
        dict of str to (str or list of str)
            ``source_name -> footage glob(s)`` in config order.

        Raises
        ------
        ValueError
            If a source entry has no string ``name``, or its ``filename`` is not a
            string or list of strings.
        """
        out: dict[str, str | list[str]] = {}
        for s in self.data.get("sources", []) or []:
            name = s.get("name")
            if not isinstance(name, str):
                raise ValueError(
                    f"[[sources]] entry needs a string 'name', got {name!r}"
                )
            out[name] = _source_filename(s.get("filename", name), name)
        return out

    # -- snapshot round-trip -------------------------------------------------

    def snapshot_text(self) -> str:
        """The exact TOML text to snapshot.

        Returns
        -------
        str
            The original file text.

        Raises
        ------
        ValueError
            If this config was built from a dict (no source text to snapshot).
        """
        if self.text is None:
            raise ValueError(
                "this Config was built from a dict; it has no source text to snapshot"
            )
        return self.text

    def save_snapshot(self, outdir: Path) -> None:
        """Snapshot the run config into ``<outdir>/config.toml`` for reproducibility.

        A no-op rewrite when the config already came from there (see
        :meth:`read_for_run`); otherwise it records the ``-c``/default config that
        drives this run, so a later run without ``-c`` reuses the very same config.

        Parameters
        ----------
        outdir
            The run's output directory; the snapshot is written to
            ``<outdir>/config.toml``.
        """
        (Path(outdir) / "config.toml").write_text(self.snapshot_text())
