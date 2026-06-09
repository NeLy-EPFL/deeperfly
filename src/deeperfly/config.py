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

``Config`` also owns the resume contract: it keeps the original TOML *text* so a
run can snapshot it into the output dir byte-for-byte (see
:meth:`Config.save_snapshot`), and validates removed/renamed sections with a fix-it
message (:func:`_validate`).
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
    from .skeleton import Skeleton
    from .video import FrameTransform
    from .viz.compose import VideoSpec

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
    """``[pipeline.pose2d]`` -- the 2D detector knobs.

    ``batch_size`` is the GPU forward batch (images/forward); ``decode_buffer`` is
    the decode queue depth in multiples of it. Both are clamped to ``>= 1``.
    """

    precision: str = "bfloat16"
    batch_size: int = 16
    decode_buffer: int = 4
    checkpoint: str | None = None

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
    """``[io.image]`` -- image-sequence decoder choices (video I/O uses PyAV)."""

    image_reader: str = "auto"
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


#: Removed ``[pipeline]`` scalar keys (the pre-per-stage layout), mapped to their
#: new home so an old config fails with a fix-it message, not a silent ignore.
_REMOVED_PIPELINE_KEYS = {
    "calibrate": "do_bundle_adjustment",
    "do_calibrate": "do_bundle_adjustment",
    "do_pictorial": "do_pictorial_structures",
    "do_visualize": "do_visualization",
    "triangulation": "[pipeline.triangulation].method",
    "ransac_threshold": "[pipeline.triangulation].ransac_threshold",
    "min_inliers": "[pipeline.triangulation].min_inliers",
    "smooth": "nothing -- temporal smoothing was removed",
}


def _validate(data: dict) -> None:
    """Reject removed/renamed sections with a pointer to the new layout.

    Run once at construction so both the CLI and the library ``from_config`` paths
    catch a stale config the same way.

    Parameters
    ----------
    data
        The parsed config mapping to validate.

    Raises
    ------
    SystemExit
        On a removed/renamed section, with a clean fix-it message (no traceback)
        -- these are migration errors the user must fix.
    """
    # Per-camera consolidation: rig defaults, input globs and frame preprocessing
    # now live inside [cameras.<name>] (and [cameras.defaults]).
    if "camera_defaults" in data:
        raise SystemExit(
            "[camera_defaults] was renamed; put the shared camera keys under "
            "[cameras.defaults]"
        )
    if "inputs" in data:
        raise SystemExit(
            "[inputs] was removed; give each camera its footage glob as "
            '`input = "..."` inside its [cameras.<name>] table'
        )
    if "preprocess" in data:
        raise SystemExit(
            "[preprocess.<camera>] was moved; put the flip/rot90 keys under "
            "[cameras.<camera>.preprocess]"
        )

    # Bundle adjustment is scipy least_squares only -- no solver selection.
    ba = data.get("pipeline", {}).get("bundle_adjustment", {})
    if "solver" in ba or "least_squares_scipy" in ba:
        raise SystemExit(
            "[pipeline.bundle_adjustment].solver / its solver sub-table were removed; "
            "scipy least_squares is the only solver -- put its kwargs (max_nfev, loss, "
            "...) directly under [pipeline.bundle_adjustment]"
        )

    # Pipeline stage toggles.
    if "stages" in data:
        raise SystemExit(
            "[stages] was removed; the stage toggles now live in [pipeline] as "
            + ", ".join(f"do_{n}" for n in STAGES)
        )
    pipe = data.get("pipeline", {})
    # Temporal smoothing was removed wholesale -- reject its old stage toggle and
    # sub-table with a clear message rather than the generic unknown-toggle one.
    if "do_smoothing" in pipe or "smoothing" in pipe:
        raise SystemExit(
            "temporal smoothing was removed; drop do_smoothing and the "
            "[pipeline.smoothing] table from the config"
        )
    for old, new in _REMOVED_PIPELINE_KEYS.items():
        # A scalar at the old key is the removed usage; a sub-table (dict) of the
        # same name -- e.g. the new [pipeline.triangulation] -- is fine.
        if old in pipe and not isinstance(pipe[old], dict):
            raise SystemExit(f"[pipeline].{old} was removed; use {new}")
    valid = {f"do_{name}" for name in STAGES}
    unknown = {k for k in pipe if k.startswith("do_")} - valid
    if unknown:
        raise SystemExit(
            f"[pipeline] has unknown stage toggle(s) {', '.join(sorted(unknown))}; "
            f"the stages are {', '.join(STAGES)}"
        )


# -- the Config class --------------------------------------------------------


class Config:
    """A loaded, validated deeperfly run configuration.

    Construct via :meth:`read` (from a TOML file), :meth:`from_dict` (a parsed
    mapping) or :meth:`coerce` (anything the old ``from_config`` accepted). The
    parsed mapping is :attr:`data`; :attr:`text` is the original TOML text when read
    from a file (``None`` for a dict), used to snapshot the config byte-for-byte.
    """

    def __init__(
        self, data: dict, *, text: str | None = None, source: Path | None = None
    ):
        self.data = data
        self.text = text
        self.source = source
        _validate(data)

    # -- construction --------------------------------------------------------

    @classmethod
    def read(cls, path: str | Path) -> "Config":
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
    def coerce(cls, config: "Config | dict | str | Path") -> "Config":
        """Accept a ``Config`` / parsed dict / path -- the one loader for every
        ``from_config``.

        Parameters
        ----------
        config
            An existing :class:`Config` (returned unchanged), a parsed ``dict``,
            or a path to a config TOML file.

        Returns
        -------
        Config
            The corresponding config.
        """
        if isinstance(config, cls):
            return config
        if isinstance(config, dict):
            return cls.from_dict(config)
        return cls.read(config)

    @classmethod
    def default(cls) -> "Config":
        """The packaged default config (:data:`DEFAULT_CONFIG_PATH`)."""
        return cls.read(DEFAULT_CONFIG_PATH)

    @classmethod
    def read_for_run(cls, cli_config: str | None, outdir: Path) -> "Config":
        """Pick the config for one run, preferring the snapshot already in ``outdir``.

        A previous run snapshots its config to ``<outdir>/config.toml``; that snapshot
        owns the cached results and the stage toggles that drive a resume, so it wins
        even over an explicit ``-c`` (notifying that it is ignored). To change it, edit
        that file or point ``-o`` at a fresh dir. With no snapshot, ``-c`` is used if
        given, else the packaged default.

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
        if snapshot.exists():
            if cli_config is not None:
                log.warning(
                    "using the config already in %s (ignoring -c %s); edit that file to "
                    "change the run (e.g. to toggle [pipeline].do_<stage>), or point -o "
                    "at a new dir",
                    snapshot,
                    cli_config,
                )
            log.info("using config %s (snapshot in the output dir)", snapshot)
            return cls.read(snapshot)
        if cli_config:
            path = Path(cli_config)
            log.info("using config %s (from -c)", path)
        else:
            path = DEFAULT_CONFIG_PATH
            log.info("using config %s (packaged default; pass -c to override)", path)
        return cls.read(path)

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
        if "reader" in im:
            present["image_reader"] = im["reader"]
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

        Unknown or removed toggles already failed in :func:`_validate` at
        construction.

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
        from .viz.compose import read_video_specs

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
        """Per-camera frame preprocessing (``[cameras.<name>.preprocess]``)."""
        from .video import parse_frame_transforms

        return parse_frame_transforms(self)

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

    def camera_patterns(self) -> dict[str, str]:
        """Map each camera to its footage glob (``[cameras.<name>].input``).

        Returns
        -------
        dict of str to str
            ``camera_name -> footage glob`` in config order; a camera with no
            ``input`` key uses its own name as the glob pattern.
        """
        _, cams = self.camera_table()
        return {name: spec.get("input", name) for name, spec in cams.items()}

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
        produced this run's results so a later resume reuses the very same config.

        Parameters
        ----------
        outdir
            The run's output directory; the snapshot is written to
            ``<outdir>/config.toml``.
        """
        (Path(outdir) / "config.toml").write_text(self.snapshot_text())
