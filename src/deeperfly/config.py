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
from dataclasses import dataclass, field
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
    "DEFAULT_CONFIG_PATH",
]

#: Packaged template emitted by ``deeperfly init`` (also the run-config example).
DEFAULT_CONFIG_PATH = Path(__file__).parent / "data" / "default_config.toml"

log = logging.getLogger("deeperfly")


#: The linear pipeline stages, in run order. Each is independently toggled by a
#: ``[pipeline].do_<stage>`` boolean (see :meth:`Config.stage_flags`) and
#: parameterized by its own ``[pipeline.<stage>]`` sub-table.
STAGES = (
    "pose2d",
    "bundle_adjustment",
    "pictorial_structures",
    "triangulation",
    "visualization",
)

#: Default for each ``do_<stage>`` when the key is omitted: detection,
#: calibration, triangulation and visualization run by default; pictorial
#: structures is opt-in.
STAGE_DEFAULTS = {
    "pose2d": True,
    "bundle_adjustment": True,
    "pictorial_structures": False,
    "triangulation": True,
    "visualization": True,
}


# -- typed per-stage params: the single source of truth for every default ----


@dataclass(frozen=True)
class Pose2dParams:
    """``[pipeline.pose2d]`` -- the 2D detector performance knobs.

    ``batch_size`` is the GPU forward batch (images/forward); ``decode_buffer`` is
    the decode queue depth in multiples of it. Both are clamped to ``>= 1``. The
    *what to detect* (sources, models, pathways) lives in the top-level detection
    plan (:meth:`Config.detection_plan`), not here.
    """

    precision: str = "bfloat16"
    batch_size: int = 16
    decode_buffer: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "batch_size", max(1, int(self.batch_size)))
        object.__setattr__(self, "decode_buffer", max(1, int(self.decode_buffer)))


@dataclass(frozen=True)
class TriangulationParams:
    """``[pipeline.triangulation]`` -- method + per-method thresholds."""

    method: str = "ransac"
    ransac_threshold: float = 15.0
    min_inliers: int = 2
    reproj_threshold: float = 40.0
    max_drops: int = 5


@dataclass(frozen=True)
class PictorialParams:
    """``[pipeline.pictorial_structures]`` -- DeepFly3D peak-recovery knobs."""

    k: int = 5
    temporal: bool = False
    lam: float = 1.0


@dataclass(frozen=True)
class IoParams:
    """``[io.image]`` -- image-sequence decode parallelism (video I/O uses PyAV)."""

    image_workers: int | None = None


@dataclass(frozen=True)
class BundleAdjustmentParams:
    """``[pipeline.bundle_adjustment]`` -- calibration over scipy ``least_squares``.

    ``keypoints`` (``None`` = all) restricts which skeleton points drive calibration;
    ``fixed`` / ``shared`` hold or tie camera parameters; ``least_squares`` is the
    leftover flat keys (``max_nfev``, ``loss``, ``f_scale``, ...) forwarded straight
    to :func:`scipy.optimize.least_squares`.
    """

    keypoints: list[int] | None = None
    fixed: list[str] = field(default_factory=list)
    shared: list[list[str]] = field(default_factory=list)
    least_squares: dict = field(default_factory=dict)


# -- helpers -----------------------------------------------------------------


def _dig(data: dict, path: tuple[str, ...]) -> dict:
    """The nested sub-table at ``path`` (e.g. ``("pipeline", "pose2d")``), or ``{}``."""
    for key in path:
        data = data.get(key, {}) if isinstance(data, dict) else {}
    return data if isinstance(data, dict) else {}


def _params(data: dict, path: tuple[str, ...], cls):
    """Build a frozen ``*Params`` from the sub-table at ``path``.

    Keys absent from the table fall through to the dataclass field defaults (the
    single source of truth).

    Parameters
    ----------
    data
        The parsed config mapping.
    path
        Key path to the sub-table, e.g. ``("pipeline", "pose2d")``.
    cls
        The frozen ``*Params`` dataclass to build.

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
    unknown = sorted(set(sub) - fields)
    if unknown:
        loc = "[" + ".".join(path) + "]"
        raise ValueError(
            f"{loc} has unknown key(s) {unknown}; allowed: {sorted(fields)}"
        )
    return cls(**{k: v for k, v in sub.items() if k in fields})


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
        return _params(self.data, ("pipeline", "pose2d"), Pose2dParams)

    @property
    def triangulation(self) -> TriangulationParams:
        return _params(self.data, ("pipeline", "triangulation"), TriangulationParams)

    @property
    def pictorial(self) -> PictorialParams:
        return _params(self.data, ("pipeline", "pictorial_structures"), PictorialParams)

    @property
    def io(self) -> IoParams:
        im = _dig(self.data, ("io", "image"))
        present: dict = {}
        if "workers" in im:
            present["image_workers"] = int(im["workers"]) or None
        return IoParams(**present)

    @property
    def bundle_adjustment(self) -> BundleAdjustmentParams:
        ba = dict(_dig(self.data, ("pipeline", "bundle_adjustment")))
        keypoints = ba.pop("keypoints", None)
        fixed = ba.pop("fixed", [])
        shared = ba.pop("shared", [])
        return BundleAdjustmentParams(
            keypoints=keypoints,
            fixed=list(fixed),
            shared=[list(s) for s in shared],
            least_squares=ba,  # leftover flat keys -> scipy.optimize.least_squares
        )

    # -- pipeline orchestration ---------------------------------------------

    @property
    def fps(self) -> float | None:
        """``[pipeline].fps`` if set, else ``None`` (the caller detects/falls back)."""
        fps = self.data.get("pipeline", {}).get("fps")
        return None if fps is None else float(fps)

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
        """The raw ``[pipeline.visualization]`` table (consumed by :attr:`videos`)."""
        return self.data.get("pipeline", {}).get("visualization", {})

    @property
    def videos(self) -> "list[VideoSpec]":
        """The output-video specs (``[[pipeline.visualization.videos]]``)."""
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

    def detection_plan(self) -> "DetectionPlan":
        """The 2D detection plan (``[[sources]]``/``[[preprocessors]]``/``[[models]]``/``[[pathways]]``).

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
            by name, with ``defaults`` excluded).
        """
        cams = dict(self.data.get("cameras", {}))
        defaults = cams.pop("defaults", {})
        return defaults, cams

    def source_patterns(self) -> dict[str, str]:
        """Map each footage source to its glob (``[[sources]]`` ``name`` -> ``filename``).

        Read directly from the ``[[sources]]`` table (without building the whole
        detection plan) so recording discovery stays cheap. A source with no
        ``filename`` key uses its own name as the glob pattern.

        Returns
        -------
        dict of str to str
            ``source_name -> footage glob`` in config order.

        Raises
        ------
        ValueError
            If a source entry has no string ``name``.
        """
        out: dict[str, str] = {}
        for s in self.data.get("sources", []) or []:
            name = s.get("name")
            if not isinstance(name, str):
                raise ValueError(
                    f"[[sources]] entry needs a string 'name', got {name!r}"
                )
            out[name] = s.get("filename", name)
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
