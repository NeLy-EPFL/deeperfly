"""The deeperfly run configuration: one ``Config`` class over the merged TOML.

A run is driven by a single TOML file (``deeperfly init`` writes the packaged
template :data:`DEFAULT_CONFIG_PATH`). :class:`Config` is the one place that file
is loaded, validated and handed to the code: every stage reads its parameters
through a typed accessor (:attr:`Config.pose2d`, :attr:`Config.triangulation`, ...)
whose defaults live in the small frozen ``*Params`` dataclasses below -- the single
source of truth, so a default is written exactly once.

The dynamic sections (cameras, skeleton, visualization, the detection plan)
are returned as the domain objects their own parsers already build
(:class:`~deeperfly.rig.cameras.CameraGroup`, :class:`~deeperfly.skeleton.Skeleton`,
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

import copy
import dataclasses
import logging
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from .params import (
    IK_KEYS,
    AnnotationParams,
    AutoCropParams,
    BundleAdjustmentParams,
    EksParams,
    GuiParams,
    InverseKinematicsParams,
    IoParams,
    PictorialParams,
    Pose2dParams,
    PostprocessParams,
    StaticPointsParams,
    SymmetrizeParams,
    TriangulationParams,
)

if TYPE_CHECKING:
    from ..pose2d.pathways import DetectionPlan
    from ..rig.cameras import CameraGroup
    from ..skeleton import Skeleton
    from ..visualization.compose import VideoSpec

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
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "data" / "default_config.toml"

#: Packaged skeletons, referenced by ``[skeleton] name``. Each file holds a complete
#: ``[skeleton]`` table in the same format a project's own ``skeleton.toml`` is written in, so
#: a preset and a project's own ``skeleton.toml`` are interchangeable.
SKELETON_PRESET_DIR = Path(__file__).parent.parent / "data" / "skeletons"

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

#: Default for each ``do_<stage>`` when the key is omitted. Everything runs except
#: ``pictorial_structures``.
#:
#: The smoother, the correction chain and the joint-angle fit are on because they are what
#: a tethered recording wants, and leaving them off meant every user rediscovering that:
#: the smoother is worth -39% 3D jitter, the chain applies what is known about the ANIMAL
#: (a thorax plate that does not move, a body that is bilaterally symmetric), and the fit is
#: the thing most downstream analysis actually reads. Each still degrades rather than
#: fails -- a stage whose input is unavailable is skipped with the reason logged, and the
#: fit skips rather than raising when its optional solver is absent.
#:
#: ``pictorial_structures`` stays off, and not for symmetry. Switching it on rewires
#: triangulation AND the smoother onto its committed 2D, and it adds a ``candidates`` key to
#: the pose2d fingerprint, re-detecting every cached tree in existence. It no longer
#: un-densifies a dense run -- the pipeline keeps the arg-max wherever recovery abstained
#: (see `PictorialParams`) -- so that is no longer among the reasons.
#:
#: It is off rather than wrong, and what it is worth was measured (2026-08-25, r28 multiview
#: transformer, 7,838 hand-labeled cells over 4 recordings of 2 animals): **-0.58 px of mean
#: hand-label error on the FINAL pose**, which is a GROSS-ERROR REPAIR and not a general
#: accuracy gain. Cells over 20 px fall 3.55% -> 2.31%; the median barely moves; and the 62%
#: of cells the detector already placed within 5 px get 0.08 px WORSE. Whether that trade is
#: worth taking depends on whether the analysis downstream is hurt more by rare large errors
#: or by small systematic ones, which is why this is a switch and not a default.
#:
#: ``peak_threshold`` is the switch that actually matters, and the shipped value disables the
#: stage in all but name: an r28 field yields a second candidate in 0.04% of cells at 0.05, so
#: recovery has nothing to choose from and returns its own input (measured end to end: 0 of
#: 983 labeled cells re-elected, -0.02 px [-0.08, +0.05]). Set ``peak_threshold_rel``.
#:
#: ``lam`` does NOT matter, which is worth knowing before anyone tunes it: swept over 0.02 to
#: 1.0 -- a 50x range -- the result is flat to three decimals (-0.989/-0.992/-0.984/-0.981 px,
#: re-election 2.454-2.457%). The bone-length prior, i.e. the thing that makes this DeepFly3D
#: *pictorial structures* rather than plain multi-view candidate election, contributes nothing
#: on this rig. The gain is the election.
#:
#: Two caveats on those numbers. The r28 export trained on all 55 corpus recordings with no
#: holdout, so every labeled frame scored here is IN-SAMPLE and none of this is evidence the
#: effect transfers to a new animal. And labeled frames over-sample fast motion, which is
#: where the gain lives -- reweighting to each recording's own motion-speed law halves it to
#: -0.32 px.
STAGE_DEFAULTS = {
    "pose2d": True,
    "bundle_adjustment": True,
    "pictorial_structures": False,
    "triangulation": True,
    "eks": True,
    "postprocess": True,
    "inverse_kinematics": True,
    "visualization": True,
}


#: Fewest views a run can reconstruct 3D from, below which it refuses rather than degrades.
#:
#: Two, and the reason is that one view fails *silently*: ``triangulate`` and
#: ``triangulate_ransac`` both return all-NaN without raising, RANSAC gives a single
#: observation ZERO inliers and then erases it (``np.where(inliers, pts2d, nan)``), and
#: bundle adjustment -- with the shipped ``points_to_use``, whose NaN initial guess is
#: replaced by zeros -- reports success at a cost near 1e-26. Nothing outside the smoother
#: has a "too few views" diagnostic, so a one-view run produces a confident-looking nothing.
MIN_VIEWS_FOR_3D = 2


# -- typed per-stage params: the single source of truth for every default ----


# -- helpers -----------------------------------------------------------------


#: The detection-plan sub-tables that share the ``[pose2d]`` table with its
#: runtime knobs (see :meth:`Config.detection_plan`). They are parsed separately
#: (:meth:`deeperfly.pose2d.pathways.DetectionPlan.from_config`), so the strict
#: :func:`_params` validator ignores them when building :class:`Pose2dParams`.
_POSE2D_PLAN_KEYS = frozenset(
    {
        "class",
        "weights",
        "auto_crops",
        "crops",
        "crop_search",
        "input_size",
        "mean",
        "n_out_channels",
    }
)

#: ``[pose2d]`` keys this release no longer honors, ``key -> what to write instead``.
#:
#: The whole declared detection plan. Detection is dense and one-to-one -- one detector,
#: run once per camera, channel ``i`` -> point ``i`` -- so every one of these was either a
#: reference to be resolved or 38 x V rows carrying no information.
RETIRED_POSE2D_KEYS = {
    "models": 'renamed and flattened: [pose2d] class = "mvt" / weights = "x.pth". One '
    "detector per run.",
    "model": "renamed: [pose2d] class (the detector CLASS, not a reference to a table)",
    "pathways": "gone: implied by the camera table -- one pathway per camera, mapping "
    "channel i to point i of that camera.",
    "preprocessors": "gone with the op grammar: a detection window is [pose2d.crops], "
    "keyed by camera, and `fliplr`/`flipud`/`rot90`/`resize` have no consumer left.",
    "output_points": "gone: channel i is point i. A 19-channel side-agnostic checkpoint "
    "is not expressible under v2 and runs under a v1 tag.",
    "autocrop": "renamed: [pose2d.crop_search] (the SEARCH's knobs). Which cameras are "
    'searched is [pose2d] auto_crops = ["f", "h"].',
}


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


#: What makes a `video` entry a PATTERN rather than a literal filename: a glob wildcard,
#: or the leading+trailing `/` that switches it to a regex (see
#: :func:`deeperfly.recordings._compile_pattern`). A list of entries none of which has
#: either is v1's "alternate names, first match wins" -- which under v2 concatenates
#: them instead.
_GLOB_WILDCARDS = set("*?[")


def _looks_like_a_pattern(entry: str) -> bool:
    is_regex = len(entry) >= 2 and entry.startswith("/") and entry.endswith("/")
    return is_regex or bool(set(entry) & _GLOB_WILDCARDS)


def _video_pattern(video, name: str) -> str | list[str]:
    """Validate a ``[cameras.<name>].video`` value: one pattern, or a list to concatenate.

    The list form is **the one v2 key whose v1 shape still parses and now means something
    else**: ``["camera_RH.mp4", "camera_0.mp4"]`` used to be alternate names with the
    first match winning, and now names two parts of one stream to be decoded back to
    back. Silently doubling a recording is the one migration failure that produces a
    plausible result instead of an error, so it gets its own check rather than the generic
    one: a list of literal filenames -- no glob wildcard and no ``/.../`` regex anywhere
    in it -- is refused by name. Alternates belong inside a regex --
    ``/camera_(RH|0)\\.mp4/``.

    Raises
    ------
    ValueError
        If ``video`` is not a string or a list of strings, or if it is a list that reads
        as v1 alternates.
    """
    if isinstance(video, str):
        return video
    if not (isinstance(video, list) and all(isinstance(f, str) for f in video)):
        raise ValueError(
            f"[cameras.{name}] 'video' must be a string or list of strings, got {video!r}"
        )
    if len(video) > 1 and not any(_looks_like_a_pattern(entry) for entry in video):
        raise ValueError(
            f"[cameras.{name}] video = {video!r} is a list of literal filenames, which "
            "under v2 CONCATENATES them into one stream -- v1 read it as alternate names "
            "with the first match winning, so honoring it would silently double the "
            "recording. Alternates go inside a regex:\n"
            f"    video = '{_alternates_hint(video)}'"
        )
    return video


def _alternates_hint(entries: list[str]) -> str:
    """The v1 alternates rewritten as one `/.../`-delimited regex, for the error above."""
    import re as _re

    return "/" + "|".join(f"({_re.escape(e)})" for e in entries) + "/"


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
#: ``[cameras]`` keys that are no longer cameras, ``key -> what to write instead``.
RETIRED_CAMERAS_KEYS = {
    "defaults": "renamed and moved out: [default_camera], a sibling table -- so [cameras] "
    "holds nothing but cameras and no camera can be called `defaults`",
    "calibration": 'moved to its own table: [calibration]\n    path = "calibration.toml"',
}

RETIRED_CAMERA_KEYS = {
    "mirror": (
        "gone: it named the camera that sees this one's mirror image, for flip "
        "augmentation during training. Its in-repo consumer went in 0.2.0 and dfpose "
        "hardcodes its own table, so nothing reads it. Should a reader appear it is "
        "derivable with no config surface -- the mirror is the camera at the negated "
        "`azimuth_deg`, which every camera declares and which, unlike the extrinsics, is "
        "known before there is a calibration."
    ),
    "preprocess": (
        "a detection window is [pose2d.crops], keyed by camera -- where it is inverted on "
        "the way back, so the detections still land in raw footage pixels and the camera "
        "keeps raw intrinsics:\n"
        "    [pose2d.crops]\n"
        "    f = [400, 290, 800, 400]\n"
        '  ...or name the camera in [pose2d] auto_crops = ["f"] to search one per '
        "recording; see docs/reference/configuration.md#autocrop"
    ),
}


def _refuse_retired_camera_keys(defaults: dict, views: dict[str, dict]) -> None:
    """Raise on a per-camera key this release no longer honors.

    Checked here rather than at the rig parser because the rig parser never sees these:
    they are stripped as non-geometry before a spec reaches
    :meth:`~deeperfly.rig.cameras.Camera.from_spec`, which is exactly how they went on being
    accepted and ignored -- which is why they are named rather than dropped now.
    """
    for where, spec in [("[default_camera]", defaults), *views.items()]:
        loc = where if where.startswith("[") else f"[cameras.{where}]"
        for key, advice in RETIRED_CAMERA_KEYS.items():
            if key in spec:
                raise ValueError(
                    f"{loc} carries {key!r}, which this release no longer honors -- it "
                    f"was accepted and silently ignored before, so a config relying on "
                    f"it was already running without it.\n  {advice}"
                )


def _narrow_videos(data: dict, dropped: set[str]) -> None:
    """Blank dropped cameras out of every video's grid, in place.

    A grid cell is set to ``""`` rather than removed, because a grid's shape is a layout:
    the montage reads as the animal from above, and closing the gap would slide every
    remaining camera into a neighbour's place. ``""`` is already the config's own spelling
    for "leave a gap here".
    """
    if not dropped:
        return
    viz = data.get("visualization")
    if not isinstance(viz, dict):
        return
    videos = viz.get("videos")
    entries = videos.values() if isinstance(videos, dict) else (videos or [])
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        grid = entry.get("grid")
        if isinstance(grid, list):
            entry["grid"] = [
                [("" if cell in dropped else cell) for cell in row]
                if isinstance(row, list)
                else row
                for row in grid
            ]


# -- skeleton presets ---------------------------------------------------------


#: Retired skeleton names that still resolve, ``old name -> packaged name``.
#:
#: A rename is not a data migration: ``fly38b`` became ``fly38`` when it was left as the
#: only packaged skeleton, and not one coordinate moved. But a *reference* to it is on
#: disk in configs and output-directory snapshots written before that, so the old
#: spelling keeps working. It resolves to the same file, and a config's own ``name``
#: still wins, so such a run goes on calling its skeleton ``fly38b`` and computes exactly
#: what it always did (the skeleton's name is not in any fingerprint -- see
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

#: ``[inverse_kinematics]`` keys this release no longer honors,
#: ``key -> what to write instead``.
#:
#: Two chain names spelled as booleans in the schema said, in the config, that a fitted
#: model has a head and an abdomen and nothing else. Which chains a model articulates is
#: the model's to declare -- and ``template`` named one of a pack's three assets while
#: leaving the other two unreachable.
RETIRED_IK_KEYS = {
    "template": 'renamed and widened: model = "neuromechfly" selects a PACK -- the leg '
    "template, the baked articulation and the overlay mesh together, which is what "
    "makes the last two selectable at all.",
    "fit_head": 'gone: chains = ["head", "abdomen"] selects them by name (omit for '
    "every chain the model defines, [] for legs only).",
    "fit_abdomen": 'gone: chains = ["head", "abdomen"] selects them by name (omit for '
    "every chain the model defines, [] for legs only).",
}


#: ``[skeleton]`` keys this release no longer honors, ``key -> what to write instead``.
#:
#: The whole grouping half of the v1 table, plus the two names v2 settled during its own
#: development. A key that was quietly ignored is worse than one that errors, and every
#: one of these would be invisible: a ``limb_palette`` or a ``colors`` still in a file
#: leaves the skeleton on the colormap, ``point_names`` leaves it on the packaged points,
#: and a ``symmetries`` table leaves it with no mirror pairs -- so flip augmentation and
#: the ``symmetrize`` correction silently switch off. All of them look like a working run.
RETIRED_SKELETON_KEYS = {
    "point_names": "renamed: points = [...] (in the skeleton FILE, not the config)",
    "limb_points": "gone: edges = [[a, b], ...] is the whole topology, with no grouping "
    "concept. Chains are derived from the edge graph where they are needed "
    "(deeperfly.pipeline.pictorial.skeleton_chains).",
    "limb_palette": 'gone: [skeleton.point_colors] is per POINT -- "lf_*" = "#0f7399" '
    "colors the five points of one leg, and the edges between them.",
    "colors": "renamed: [skeleton.point_colors], which pairs with the new "
    "[skeleton.edge_colors] instead of being the only color table.",
    "symmetries": "renamed: point_symmetries = [[a, b], ...], to say what is symmetric "
    "the way point_names and point_colors do.",
    "name": None,  # still legal; listed so the loop below can skip it
    "file": 'renamed: include = "skeleton.toml"',
}


def skeleton_presets() -> dict[str, Path]:
    """The packaged skeletons, ``name -> file`` (see :data:`SKELETON_PRESET_DIR`).

    Retired spellings from :data:`SKELETON_ALIASES` are not listed -- they resolve, but
    an error message offering them as choices would be advertising a name the release no
    longer has.
    """
    if not SKELETON_PRESET_DIR.is_dir():
        return {}
    return {p.stem: p for p in sorted(SKELETON_PRESET_DIR.glob("*.toml"))}


def default_skeleton_spec() -> dict:
    """The ``[skeleton]`` table a config that declares none resolves to.

    The **sole packaged skeleton**, and an error if there is not exactly one. A run
    normally says nothing about its skeleton because which points exist is a property of
    the detector, not of the recording -- and what makes that safe is not this default but
    the check at the other end: a checkpoint records its own point names, and
    :func:`deeperfly.pose2d.stream._check_channel_names` refuses one whose channels are
    not these points. So the default is allowed to be "the one there is", and the moment
    there are two, a config has to choose.
    """
    presets = skeleton_presets()
    if len(presets) != 1:
        raise ValueError(
            f"this build packages {len(presets)} skeletons ({sorted(presets)}), so a "
            "config cannot leave the skeleton unsaid. Name one:\n"
            '    [skeleton]\n    include = "fly38"'
        )
    return _load_skeleton_file(next(iter(presets.values())))


def _load_skeleton_file(path: Path) -> dict:
    """The ``[skeleton]`` table of a skeleton file, or an error naming what it is not."""
    loaded = tomllib.loads(path.read_text()).get("skeleton")
    if not isinstance(loaded, dict) or "points" not in loaded:
        raise ValueError(
            f"{path} carries no [skeleton] table with 'points'; it is not a skeleton "
            "file (the format is data/skeletons/fly38.toml -- points, edges, "
            "point_symmetries, colors)"
        )
    # Only the source, here. The skeleton names ITSELF (with its digest) once it is
    # built -- see `Skeleton.from_spec` -- because a dict has no digest and a config that
    # overrode keys alongside `include` would otherwise be logged as the file's contents.
    log.info("skeleton table from %s", path)
    return loaded


def _resolve_skeleton(data: dict, source: Path | None) -> dict:
    """Fill in ``[skeleton]``: the packaged skeleton, or the one ``include`` names.

    A skeleton is a hundred lines of points, edges, mirror pairs and colors that almost
    never differ between recordings of the same animal -- and when it does differ it
    differs completely (the DeepFly3D set and ``fly38`` are both 38 points and share 32 of
    them, in a different order). So it lives in a version-controlled file, and a run
    config normally says **nothing at all**:

    * no ``[skeleton]`` table -> :func:`default_skeleton_spec`, the packaged skeleton.
    * ``include = "fly38"`` -> a packaged name (:func:`skeleton_presets`), or a retired
      spelling of one (:data:`SKELETON_ALIASES`).
    * ``include = "skeleton.toml"`` -> a path, resolved next to the config that wrote it.
      Same format, which is what makes a packaged skeleton and a project's own
      interchangeable.

    Keys written alongside ``include`` win **wholesale, per key**: a ``colors`` table
    there replaces the included one rather than merging entry by entry, because a
    half-merged color table is not a thing anyone means.

    Resolution happens once, at :class:`Config` construction, so everything downstream
    (the skeleton itself, the fingerprints, ``deeperfly config show``) sees one
    fully-populated table and needs to know nothing about where it came from.

    Returns
    -------
    dict
        ``data`` with ``[skeleton]`` expanded. The input is not mutated.

    Raises
    ------
    ValueError
        If the table carries a v1 key (:data:`RETIRED_SKELETON_KEYS`), if ``include``
        names no packaged skeleton and no readable file, or if the referenced file is not
        a skeleton file.
    """
    skel = data.get("skeleton")
    if skel is None:
        return {**data, "skeleton": default_skeleton_spec()}
    if not isinstance(skel, dict):
        raise ValueError(f"[skeleton] must be a table, got {skel!r}")

    for key, advice in RETIRED_SKELETON_KEYS.items():
        if advice is not None and key in skel:
            raise ValueError(
                f"[skeleton] carries {key!r}, which this release no longer honors.\n"
                f"  {advice}\n"
                "  A skeleton is points, edges, point_symmetries and colors, in a file of its "
                "own; see data/skeletons/fly38.toml."
            )

    ref = skel.get("include")
    if ref is None:
        if "points" not in skel:
            raise ValueError(
                "[skeleton] declares neither 'include' nor 'points'"
                + (f" (it only names {skel['name']!r})" if "name" in skel else "")
                + ". Leave the table out entirely to take the packaged skeleton, or "
                'name one: include = "fly38".'
            )
        # Spelled out in the config. Legal -- it is what the editor writes into a
        # project and what a results.h5 round-trip produces -- but not the norm.
        return data
    presets = skeleton_presets()
    if ref in presets:
        path = presets[ref]
    elif SKELETON_ALIASES.get(ref) in presets:
        current = SKELETON_ALIASES[ref]
        path = presets[current]
        log.info(
            "[skeleton] include = %r is the former spelling of %r and resolves to it; "
            "the points are unchanged",
            ref,
            current,
        )
    else:
        path = Path(str(ref))
        if not path.is_absolute() and source is not None:
            path = source.parent / path
        if not path.is_file():
            raise ValueError(
                f"[skeleton] include = {ref!r} is neither a packaged skeleton "
                f"(have: {sorted(presets)}) nor a readable file (looked in {path})"
            )
    return {**data, "skeleton": {**_load_skeleton_file(path), **skel}}


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

        Any automatic crop window a previous run recorded in ``outdir`` is loaded
        into :attr:`auto_crops`, so a resume that reuses the cached 2D pose still
        knows which window the detector looked through -- the visualization panels
        that borrow it, in particular, are drawn in a later process than the
        search.

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
        config = cls.from_toml(path)
        from ..pose2d.autocrop import read_sidecar

        config.auto_crops = dict(read_sidecar(outdir))
        return config

    # -- typed per-stage subgroups ------------------------------------------

    @property
    def pose2d(self) -> Pose2dParams:
        return _params(self.data, ("pose2d",), Pose2dParams, ignore=_POSE2D_PLAN_KEYS)

    @property
    def autocrop(self) -> AutoCropParams:
        return _params(self.data, ("pose2d", "crop_search"), AutoCropParams)

    @property
    def triangulation(self) -> TriangulationParams:
        return _params(self.data, ("triangulation",), TriangulationParams)

    @property
    def eks(self) -> EksParams:
        return _params(self.data, ("eks",), EksParams)

    @property
    def postprocess(self) -> PostprocessParams:
        """The correction chain, with every op's point SELECTORS resolved to names.

        Resolved here rather than in :mod:`deeperfly.pipeline.postprocess`, which goes on reading
        the fields it always read: the ops' ``points`` / ``midline`` become plain name
        lists, and ``symmetrize``'s ``points`` becomes the ``pairs`` it takes -- one half
        of each pair is enough, because ``[skeleton] point_symmetries`` already says the
        partner. Restating the pairs was the config repeating the skeleton.
        """
        params = _params(self.data, ("postprocess",), PostprocessParams)
        if not params.ops:
            return params
        return dataclasses.replace(
            params, ops=[self._resolved_op(op) for op in params.ops]
        )

    def _resolved_point_names(self, entries, where: str) -> list[str]:
        """A point selector -> the names it matches, in point order."""
        from ..skeleton import resolve_points

        skeleton = self.skeleton()
        return [
            skeleton.point_names[i]
            for i in resolve_points(entries, skeleton.point_names, where=where)
        ]

    def _resolved_op(self, op) -> dict:
        """One ``[[postprocess.ops]]`` entry with its selectors resolved."""
        if not isinstance(op, dict):
            return op
        out = dict(op)
        name = str(out.get("op", "?"))
        where = f'{{ op = "{name}" }}'
        if "pairs" in out:
            raise ValueError(
                f"{where} carries 'pairs', which this release no longer honors: name "
                "either half of each pair in `points` ([skeleton] point_symmetries knows the "
                'partner) -- points = ["l*_thorax_coxa"] is the packaged default\'s three.'
            )
        for key in ("points", "midline"):
            if key in out:
                out[key] = self._resolved_point_names(out[key], f"{where} {key}")
        if name == "symmetrize":
            skeleton = self.skeleton()
            pairs = []
            for point in out.pop("points", []):
                partner = skeleton.partner(point)
                if partner is None:
                    raise ValueError(
                        f"{where} points names {point!r}, which has no partner in "
                        "[skeleton] point_symmetries -- a point with no mirror image cannot be "
                        "symmetrized. Midline points go in `midline`."
                    )
                pair = sorted((point, skeleton.point_names[partner]))
                if pair not in pairs:
                    pairs.append(pair)
            out["pairs"] = pairs
        return out

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
        if "points_to_use" in ba:
            raise ValueError(
                "[bundle_adjustment] carries 'points_to_use', which this release no "
                "longer honors: renamed to `points`, and an entry may be a `*` pattern "
                '-- points = ["lf_*", "lm_*", ...] is the packaged default\'s 30 names.'
            )
        raw_points = ba.pop("points", None)
        points_to_use = (
            None
            if raw_points is None
            else self._resolved_point_names(raw_points, "[bundle_adjustment] points")
        )
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
        for key, advice in RETIRED_IK_KEYS.items():
            if key in ik:
                raise ValueError(
                    f"[inverse_kinematics] carries {key!r}, which this release no "
                    f"longer honors.\n  {advice}"
                )
        bounds = {
            str(k): [float(b) for b in v] for k, v in ik.pop("bounds", {}).items()
        }
        # [inverse_kinematics.markers.<chain>]: per-chain marker placement
        # (point -> {body, offset, depth?}), the labeling-scheme choice. Under its own
        # `markers` table rather than two fixed chain names sitting in the stage's knob
        # namespace, which is also what the strict validation below had to special-case.
        for chain in ("head", "abdomen"):
            if chain in ik:
                raise ValueError(
                    f"[inverse_kinematics.{chain}] moved: it is a marker table, not a "
                    f"knob, so it lives under [inverse_kinematics.markers.{chain}]"
                )
        raw_markers = ik.pop("markers", {})
        if not isinstance(raw_markers, dict):
            raise ValueError(
                "[inverse_kinematics.markers] must be a table of chain -> markers, "
                f"got {raw_markers!r}"
            )
        markers = {
            str(chain): {str(p): dict(spec) for p, spec in table.items()}
            for chain, table in raw_markers.items()
        }
        defaults = InverseKinematicsParams()
        model = str(ik.pop("model", defaults.model))
        binding = ik.pop("binding", defaults.binding)
        legs = ik.pop("legs", None)
        chains = ik.pop("chains", defaults.chains)
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
            model=model,
            binding=None if binding is None else str(binding),
            legs=None if legs is None else [str(leg) for leg in legs],
            chains=None if chains is None else [str(c) for c in chains],
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

    def ik_model(self):
        """The configured model pack (``[inverse_kinematics].model``).

        Returns
        -------
        deeperfly.inverse_kinematics.pack.ModelPack
            The packaged or path-loaded manifest and its three asset paths.
        """
        from ..inverse_kinematics.pack import ModelPack

        return ModelPack.load(self.inverse_kinematics.model)

    def ik_binding(self):
        """The configured (skeleton, model) binding.

        Returns
        -------
        deeperfly.inverse_kinematics.binding.Binding
            The named binding, or the ``"<skeleton>@<model>"`` default for this run's
            skeleton and pack.
        """
        from ..inverse_kinematics.binding import Binding

        p = self.inverse_kinematics
        ref = p.binding or f"{self.skeleton().name}@{self.ik_model().name}"
        return Binding.load(ref)

    def ik_template(self):
        """The configured pack's kinematic template, with this run's leg restriction.

        Returns
        -------
        deeperfly.inverse_kinematics.template.KinematicTemplate
            The pack's leg template, with the legs restricted, the pack's rest axis
            applied and the ``[inverse_kinematics.bounds]`` degree overrides applied.

        Raises
        ------
        ValueError
            If the template names a leg joint the configured skeleton does not track.
            Without the check every observation of that joint is NaN and the fit is a
            plausible-looking pose fitted to nothing.
        """
        from ..inverse_kinematics.template import KinematicTemplate

        p = self.inverse_kinematics
        pack = self.ik_model()
        overrides = {k: (v[0], v[1]) for k, v in p.bounds.items()}
        template = KinematicTemplate.load(
            pack.template_path,
            legs=p.legs,
            bounds_overrides=overrides,
            rest_axis=pack.rest_axis,
            binding=self.ik_binding(),
        )
        tracked = set(self.skeleton().point_names)
        missing = [n for n in template.model_point_names if n not in tracked]
        if missing:
            raise ValueError(
                f"[inverse_kinematics] model {p.model!r} predicts point(s) {missing}, "
                f"which skeleton {self.skeleton().name!r} does not track. Every "
                "observation of them would be missing, so the fit would be pinned by "
                "its neutral pose rather than by the data."
            )
        return template

    def ik_articulation(self):
        """The configured non-leg articulation, or ``None`` if no chain is fit.

        Returns
        -------
        deeperfly.inverse_kinematics.articulation.Articulation or None
            The baked chains selected by ``[inverse_kinematics].chains``, with
            ``[inverse_kinematics.bounds]`` degree overrides (keys like
            ``c_thorax-c_head-pitch`` / ``c_abdomen12-c_abdomen3-pitch``) and any
            ``[inverse_kinematics.markers.<chain>]`` marker placement overrides applied.
        """
        from ..inverse_kinematics.articulation import Articulation

        p = self.inverse_kinematics
        fit = None if p.chains is None else tuple(p.chains)
        if fit == ():
            return None
        pack = self.ik_model()
        overrides = {k: (v[0], v[1]) for k, v in p.bounds.items()}
        return Articulation.load(
            pack.articulation_path,
            binding=self.ik_binding(),
            anchors=pack.anchors,
            fit=fit,
            bounds_overrides=overrides,
            marker_overrides=p.markers,
        )

    # -- pipeline orchestration ---------------------------------------------

    def stage_flags(self) -> dict[str, bool]:
        """Which stages are enabled, from the ``[pipeline].<stage>`` booleans.

        The key is the stage's own name. ``do_`` was a prefix on a key inside a table
        called ``pipeline``, which already said what the booleans were about; a
        ``do_<stage>`` key is refused by name rather than silently ignored, because an
        ignored one leaves the stage at its default and reads as "the flag did nothing".

        Returns
        -------
        dict of str to bool
            ``stage_name -> enabled`` for every stage in :data:`STAGES`, each
            defaulting to :data:`STAGE_DEFAULTS`.

        Raises
        ------
        ValueError
            On a ``do_<stage>`` key, or a key that is not a stage at all.
        """
        pipe = self.data.get("pipeline", {})
        retired = sorted(k for k in pipe if k.startswith("do_") and k[3:] in STAGES)
        if retired:
            raise ValueError(
                f"[pipeline] carries {retired}, which this release no longer honors: the "
                "`do_` prefix is gone, since [pipeline] already says what these are.\n"
                + "".join(f"    {k[3:]} = ...\n" for k in retired)
            )
        unknown = sorted(set(pipe) - set(STAGES))
        if unknown:
            raise ValueError(
                f"[pipeline] has unknown key(s) {unknown}; allowed: {list(STAGES)}"
            )
        return {n: bool(pipe.get(n, STAGE_DEFAULTS[n])) for n in STAGES}

    # -- structured sections: the domain objects their parsers build --------

    @property
    def visualization(self) -> dict:
        """The raw ``[visualization]`` table (consumed by :attr:`videos`)."""
        return self.data.get("visualization", {})

    @property
    def videos(self) -> "list[VideoSpec]":
        """The output-video specs (``[[visualization.videos]]``)."""
        from ..visualization.compose import read_video_specs

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
        from ..rig.cameras import CameraGroup

        return CameraGroup.from_config(self, image_sizes=image_sizes)

    def skeleton(self) -> "Skeleton":
        """The run's skeleton.

        Always present: :func:`_resolve_skeleton` filled the table in at construction,
        from ``include`` or from the packaged skeleton, so there is nothing to fall back
        to here.
        """
        from ..skeleton import Skeleton

        return Skeleton.from_config(self)

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
        from ..pose2d.pathways import DetectionPlan

        plan = DetectionPlan.from_config(self)
        if self.auto_crops:
            from ..pose2d.autocrop import resolved_plan

            plan = resolved_plan(plan, self.auto_crops)
        return plan

    def camera_table(self) -> tuple[dict, dict]:
        """``([default_camera], [cameras.*])`` -- the shared values and the cameras.

        ``[cameras]`` is a pure name -> camera map: no reserved key and no sub-table that
        could be mistaken for an entry, which is what the sibling ``[default_camera]``
        buys. A bare key or a ``defaults`` sub-table under ``[cameras]`` is therefore
        refused by name rather than skipped -- a camera called ``defaults`` used to be the
        way to write the shared values, so silence there is what a reader of a v1 config
        would misread as "still honored".

        Returns
        -------
        defaults, cameras : dict
            The ``[default_camera]`` spec and the per-camera specs, keyed by name.
        """
        cams = dict(self.data.get("cameras", {}))
        for key, advice in RETIRED_CAMERAS_KEYS.items():
            if key in cams:
                raise ValueError(
                    f"[cameras] carries {key!r}, which this release no longer honors.\n"
                    f"  {advice}\n"
                    "  [cameras] holds one table per camera and nothing else."
                )
        bare = sorted(k for k, v in cams.items() if not isinstance(v, dict))
        if bare:
            raise ValueError(
                f"[cameras] holds non-table key(s) {bare}; it is a map of camera name "
                "to camera. Shared values go in [default_camera], the solved rig in "
                "[calibration]."
            )
        defaults = dict(self.data.get("default_camera", {}))
        _refuse_retired_camera_keys(defaults, cams)
        return defaults, cams

    def calibration_path(self) -> Path | None:
        """The solved rig this config points at (``[calibration].path``), if any.

        Its own table rather than a key under ``[cameras]``, which is what lets
        ``deeperfly project compose_config`` inject it as a fragment of its own without
        overlapping the rig fragment.

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
        table = self.data.get("calibration") or {}
        if not isinstance(table, dict):
            raise ValueError(f"[calibration] must be a table, got {table!r}")
        unknown = sorted(set(table) - {"path"})
        if unknown:
            raise ValueError(
                f"[calibration] has unknown key(s) {unknown}; allowed: path"
            )
        raw = table.get("path")
        if raw is None:
            return None
        path = Path(str(raw))
        if path.is_absolute() or self.source is None:
            return path
        return self.source.parent / path

    def narrowed_to_sources(self, available) -> "Config":
        """A copy of this config describing only what a recording actually has.

        One config routinely describes more rig than one recording holds -- the packaged
        default declares the eight-camera rig, and a seven-camera recording under it is not
        malformed. Rather than refuse the recording or invent the missing footage, the run
        narrows itself: a source with no files invalidates the pathways that read it, a
        view no surviving pathway feeds leaves the rig, and everything keyed on that view
        follows.

        What is dropped:

        * ``[cameras.<name>]`` -- the cameras whose ``video`` pattern matched nothing.
          This is what shortens the ``V`` axis, and dropping it is the whole job now: a
          camera IS a source IS a pathway IS a view, so there is no three-way bookkeeping
          left. Leaving a camera in place with no footage would give a view whose 2D is
          all-NaN, which reads as a detected-and-empty camera rather than an absent one --
          and which bundle adjustment would then export into ``calibration.toml`` at its
          unrefined nominal pose with nothing marking it as unmeasured.
        * ``[visualization.videos]`` grid cells and ``panels`` naming a dropped camera -- a
          grid cell is blanked (``""``) rather than removed, so the montage keeps its shape
          and the remaining cameras stay where the reader expects them.

        * ``[pose2d.crops]`` entries and ``auto_crops`` names for a dropped camera. These
          have to go rather than be left inert, because the plan REFUSES a crop naming a
          camera that does not exist -- which is the right answer for a typo and the wrong
          one for the legal case narrowing exists to serve.

        Point-name sections (``[bundle_adjustment] points``, ``[postprocess]`` ops,
        ``[inverse_kinematics]``) are camera-agnostic and untouched.

        Only :attr:`data` narrows. :attr:`text` -- what the snapshot records -- keeps
        saying what was *asked for*. That split is what keeps the cache honest: the
        narrowed plan is what reaches the fingerprints, so a run that proceeded on seven
        views records a seven-view fingerprint and recomputes when the eighth camera turns
        up, while the snapshot still describes the rig the operator configured.

        Parameters
        ----------
        available
            The resolved ``camera -> footage files`` map, or any container of camera
            names. A camera absent from it, or present with an empty list, is absent.

        Returns
        -------
        Config
            A narrowed copy, or ``self`` when nothing is missing (so the common case
            allocates nothing and is byte-identical).

        Raises
        ------
        SystemExit
            If fewer than :data:`MIN_VIEWS_FOR_3D` views survive. Below two views
            nothing downstream means anything and nothing says so: triangulation returns
            all-NaN without raising, RANSAC gives a single observation zero inliers and
            *erases* it, and bundle adjustment reports success at a cost near zero. A
            refusal is the only honest answer there.
        """
        if isinstance(available, dict):
            have = {n for n, files in available.items() if files}
        else:
            have = set(available or ())
        declared = list(self.source_patterns())
        gone = [n for n in declared if n not in have]
        if not gone:
            return self

        data = copy.deepcopy(self.data)
        cams = data.get("cameras")
        if isinstance(cams, dict):
            for name in gone:
                cams.pop(name, None)
        pose2d = data.get("pose2d")
        if isinstance(pose2d, dict):
            crops = pose2d.get("crops")
            if isinstance(crops, dict):
                for name in gone:
                    crops.pop(name, None)
            searched = pose2d.get("auto_crops")
            if isinstance(searched, list):
                pose2d["auto_crops"] = [n for n in searched if n not in gone]
        _narrow_videos(data, set(gone))

        narrowed = Config(data, text=self.text, source=self.source)
        narrowed.auto_crops = dict(self.auto_crops)
        surviving = list(narrowed.data.get("cameras") or {})
        log.warning(
            "narrowing this run to the footage present: camera(s) %s matched no files "
            "and are dropped -- running on %d camera(s): %s",
            gone,
            len(surviving),
            surviving,
        )
        if len(surviving) < MIN_VIEWS_FOR_3D:
            raise SystemExit(
                f"only {len(surviving)} view(s) have footage ({surviving}), and "
                f"{MIN_VIEWS_FOR_3D} are needed for 3D -- a single view triangulates to "
                "nothing without saying so.\n"
                f"  camera(s) with no files: {gone}\n"
                "  Check the [cameras.<name>] `video` patterns, or pass the recording "
                "that holds the rest of the cameras."
            )
        return narrowed

    def narrowed_to_covered_views(self, covered) -> "Config":
        """A copy of this config with the views a solved rig does not cover dropped.

        The sibling of :meth:`narrowed_to_sources`, for the other way a view can turn out to
        be unusable. A calibration is a measurement of which cameras exist, so a view it
        never covered has no extrinsics to project through -- and one config routinely
        describes more rig than one solve covers (a project whose recordings predate a
        camera being added, say).

        Both narrowings are needed, and neither substitutes: footage without a rig cannot be
        placed, and a rig without footage has nothing to place. Dropping the view from the
        RIG alone is what left ``pts2d`` with more view rows than the rig had cameras, and
        the first thing to notice was a bare einsum shape error naming no camera at all.

        Parameters
        ----------
        covered
            The camera names the rig actually covers.

        Returns
        -------
        Config
            A narrowed copy, or ``self`` when the rig covers every declared view.
        """
        keep = set(covered)
        declared = list(self.data.get("cameras") or {})
        dropped = [v for v in declared if v not in keep]
        if not dropped:
            return self

        log.warning(
            "the rig does not cover camera(s) %s, so this run drops them: a view with no "
            "measured camera cannot be projected through",
            dropped,
        )
        # Camera, source and view are one name, so there is nothing to translate: the
        # cameras the rig covers ARE the sources that survive.
        return self.narrowed_to_sources([v for v in declared if v in keep])

    def source_patterns(self) -> dict[str, str | list[str]]:
        """``camera -> its footage pattern(s)`` (``[cameras.<name>].video``), in order.

        Read straight off the camera table, without building the detection plan, so
        recording discovery stays cheap. A camera with no ``video`` key uses its own name
        as the pattern.

        The value is one glob (or, wrapped in ``/.../``, a regex) or a **list** of them;
        the list means CONCATENATION, in written order, and everything one entry matches
        is one stream in natural order (see :func:`deeperfly.recordings.camera_files`).
        Nothing is inferred from the camera name or its index.

        Returns
        -------
        dict of str to (str or list of str)
            ``camera -> pattern(s)`` in camera order.

        Raises
        ------
        ValueError
            If a ``video`` value is not a string or list of strings, or if it is a list of
            literal filenames -- v1's alternates, which v2 would concatenate (see
            :func:`_video_pattern`).
        """
        _, cameras = self.camera_table()
        return {
            name: _video_pattern(spec.get("video", name), name)
            for name, spec in cameras.items()
        }

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
