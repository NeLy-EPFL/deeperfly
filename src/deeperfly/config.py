"""The deeperfly run configuration: one ``Config`` class over the merged TOML.

A run is driven by a single TOML file (``deeperfly init`` writes the packaged
template :data:`DEFAULT_CONFIG_PATH`). :class:`Config` is the one place that file
is loaded, validated and handed to the code: every stage reads its parameters
through a typed accessor (:attr:`Config.pose2d`, :attr:`Config.triangulation`, ...)
whose defaults live in the small frozen ``*Params`` dataclasses below -- the single
source of truth, so a default is written exactly once.

The dynamic sections (cameras, skeleton, visualization, per-camera preprocessing)
are returned as the domain objects their own parsers already build
(:class:`~deeperfly.cameras.CameraGroup`, :class:`~deeperfly.skeleton.Skeleton`,
``list[VideoSpec]``, ``dict[str, FrameTransform]``); only genuinely open-ended leaf
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
    from .preprocessing import FrameTransform
    from .skeleton import Skeleton
    from .visualization.compose import VideoSpec

__all__ = [
    "Config",
    "Pose2dParams",
    "TriangulationParams",
    "PictorialParams",
    "IoParams",
    "BundleAdjustmentParams",
    "InverseKinematicsParams",
    "AnnotationParams",
    "DEFAULT_CONFIG_PATH",
]

#: Packaged template emitted by ``deeperfly init`` (also the run-config example).
DEFAULT_CONFIG_PATH = Path(__file__).parent / "data" / "default_config.toml"

log = logging.getLogger("deeperfly")


#: The linear pipeline stages, in run order. Each is independently toggled by a
#: ``[pipeline].do_<stage>`` boolean (see :meth:`Config.stage_flags`) and
#: parameterized by its own top-level ``[<stage>]`` table.
STAGES = (
    "pose2d",
    "bundle_adjustment",
    "pictorial_structures",
    "triangulation",
    "inverse_kinematics",
    "visualization",
)

#: Default for each ``do_<stage>`` when the key is omitted: detection,
#: bundle adjustment, triangulation and visualization run by default; pictorial
#: structures and inverse kinematics are opt-in.
STAGE_DEFAULTS = {
    "pose2d": True,
    "bundle_adjustment": True,
    "pictorial_structures": False,
    "triangulation": True,
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
class TriangulationParams:
    """``[triangulation]`` -- method + per-method thresholds."""

    method: str = "ransac"
    ransac_threshold: float = 15.0
    min_inliers: int = 2
    reproj_threshold: float = 40.0
    max_drops: int = 5
    weigh_by_confidence: bool = False


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
    "depth"?}`` (from the ``[inverse_kinematics.head]`` / ``[inverse_kinematics.abdomen]``
    config tables); see :meth:`~deeperfly.inverse_kinematics.articulation.Articulation.load`.

    ``constant_points`` names skeleton points whose 3D position is physically fixed
    over the recording. Before the fit, each listed point is replaced by its temporal
    median across all frames, so it stops jittering with per-frame detection noise (and
    occluded frames get filled in). Empty = off. Note that under ``fixed_body`` the leg
    roots are already held at their measured medians *by construction*, so this mainly
    matters when the body is free.
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
    weigh_by_confidence: bool = False
    parallel: bool = False
    segment_len: int = 200
    overlap_len: int = 10
    bounds: dict[str, list[float]] = field(default_factory=dict)
    markers: dict[str, dict] = field(default_factory=dict)
    constant_points: list[str] = field(default_factory=list)


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
    view)`` the operator authors a GT 2D pixel and/or an "occluded" flag (orthogonal), and
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
_POSE2D_PLAN_KEYS = frozenset({"preprocessors", "models", "pathways", "output_points"})


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
        self.data = data
        self.text = text
        self.source = source

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
        return cls.from_toml(path)

    # -- typed per-stage subgroups ------------------------------------------

    @property
    def pose2d(self) -> Pose2dParams:
        return _params(self.data, ("pose2d",), Pose2dParams, ignore=_POSE2D_PLAN_KEYS)

    @property
    def triangulation(self) -> TriangulationParams:
        return _params(self.data, ("triangulation",), TriangulationParams)

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
        weigh_by_confidence = ik.pop(
            "weigh_by_confidence", defaults.weigh_by_confidence
        )
        parallel = ik.pop("parallel", defaults.parallel)
        segment_len = ik.pop("segment_len", defaults.segment_len)
        overlap_len = ik.pop("overlap_len", defaults.overlap_len)
        constant_points = ik.pop("constant_points", [])
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
            weigh_by_confidence=bool(weigh_by_confidence),
            parallel=bool(parallel),
            segment_len=int(segment_len),
            overlap_len=int(overlap_len),
            bounds=bounds,
            markers=markers,
            constant_points=[str(n) for n in constant_points],
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

    def frame_transforms(self) -> "dict[str, FrameTransform]":
        """Per-camera frame preprocessing (the ``[cameras.<name>]`` ``preprocess`` lists)."""
        from .preprocessing import parse_frame_transforms

        return parse_frame_transforms(self)

    def mirror_views(self) -> dict[str, str]:
        """``view -> the view that sees this view's mirror image`` (``[cameras.<n>].mirror``).

        Only the views that declare it; empty when none do. Consumed by flip augmentation
        (:mod:`deeperfly.training.mirror`): a mirrored training sample must be told which
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

        Returns
        -------
        DetectionPlan
            The parsed, validated plan mapping footage sources through
            preprocessors and models into the skeleton (see
            :class:`deeperfly.pose2d.pathways.DetectionPlan`).
        """
        from .pose2d.pathways import DetectionPlan

        return DetectionPlan.from_config(self)

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
        return defaults, {k: v for k, v in cams.items() if isinstance(v, dict)}

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
