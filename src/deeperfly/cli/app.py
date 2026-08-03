"""The Typer application: command definitions and the ``main`` entry point."""

from __future__ import annotations

import argparse
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from ..config import STAGES
from ..pipeline import _OVERWRITE_ALL
from .calibrate import _cmd_calibrate
from .calibration import _cmd_calibration_export, _cmd_calibration_show
from .config import _cmd_config_set, _cmd_config_show
from .console import _configure_logging
from .gui import _cmd_gui, _cmd_labels_absent, _cmd_labels_export
from .merge import _cmd_labels_merge
from .project import (
    _cmd_project_add,
    _cmd_project_ls,
    _cmd_project_new,
    _cmd_project_rm,
    _cmd_project_status,
)
from .report import _cmd_doctor, _cmd_init, _cmd_inspect
from .run import _cmd_run
from .suggest import _cmd_labels_suggest

# -- typer app ---------------------------------------------------------------
#
# The CLI is built with Typer: typed signatures over click, with usage/--help
# rendered through rich. Each command declares its options, configures logging,
# then hands an argparse-style namespace to the matching ``_cmd_*`` worker; the
# workers stay namespace-driven so they remain callable as a library and from the
# tests. Constrained options are ``str``-valued Enums; commands pass their
# ``.value``, so workers keep receiving plain strings.


class LogLevel(str, Enum):
    """``--log-level`` choices, shared by every subcommand. A ``str`` enum, so each
    member's ``.value`` is the name :func:`_configure_logging` expects."""

    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"
    critical = "critical"


#: The shared ``--log-level`` option, declared once as a reusable parameter
#: annotation and spread across every command. ``case_sensitive=False`` accepts
#: INFO/Info/info.
LogLevelOption = Annotated[
    LogLevel,
    typer.Option(
        case_sensitive=False,
        help="logging verbosity; 'warning' or higher hides the per-stage logs and "
        "the progress bar",
    ),
]

app = typer.Typer(
    add_completion=False,  # no shell-completion options; keep the surface minimal
    rich_markup_mode="rich",  # rich-rendered (boxed, colored) usage and --help
    no_args_is_help=True,  # bare 'deeperfly' prints help instead of a usage error
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Markerless 3D pose estimation of tethered Drosophila from a multi-camera "
    "rig. 'deeperfly init' writes a config to edit; 'deeperfly run' detects 2D "
    "pose, reconstructs 3D and renders a video; 'deeperfly inspect' summarizes a "
    "result file; 'deeperfly doctor' reports the installation/runtime.",
)


@app.command()
def init(
    output: Annotated[
        str, typer.Argument(help="destination (defaults to config.toml)")
    ] = "config.toml",
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="overwrite an existing file")
    ] = False,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Write a default config.toml to edit (destination defaults to config.toml)."""
    _configure_logging(log_level.value)
    _cmd_init(argparse.Namespace(output=output, overwrite=overwrite))


@app.command()
def run(
    inputs: Annotated[
        list[Path],
        typer.Argument(
            metavar="INPUT...",
            help="one or more recording dirs or wildcard patterns (per-camera videos "
            "or image folders); several inputs / a wildcard run as a batch",
        ),
    ],
    recursive: Annotated[
        bool,
        typer.Option(
            "-r",
            "--recursive",
            help="treat each INPUT as a parent directory and run every recording "
            "nested under it (each subdirectory holding the configured per-camera "
            "footage)",
        ),
    ] = False,
    config: Annotated[
        str | None,
        typer.Option(
            "-c",
            "--config",
            help="merged config TOML (from 'deeperfly init'); "
            "defaults to the packaged default config",
        ),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option(
            "-o",
            "--output-dir",
            help="output directory (default: <input>/deeperfly_outputs; created if "
            "missing). For a batch of several recordings: end it with '/' to "
            "collect one subdirectory per recording under it (colliding names "
            "fall back to mirroring the input tree, after confirmation); a "
            "relative name without '/' creates that directory inside each "
            "recording.",
        ),
    ] = None,
    overwrite: Annotated[
        list[str] | None,
        typer.Option(
            "--overwrite",
            help="force stages to recompute even though their config is unchanged "
            "(config changes are detected automatically). A bare --overwrite "
            "recomputes everything; name stages to recompute only those (e.g. "
            "--overwrite pose2d visualization). Recomputing a stage also "
            "refreshes the stages after it.",
        ),
    ] = None,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """detect 2D -> reconstruct 3D -> visualization (the enabled stages, reusing cache).

    INPUT is one or more recording directories (per-camera videos or image folders)
    and/or wildcards matching several (e.g. 'fly*' -> fly1/, fly2/, ...), each run
    in turn. Several inputs or a wildcard run as a batch, keeping only the valid
    recordings. With -r/--recursive, each INPUT is a parent directory and every
    recording nested under it is run in turn.

    A stage already in the output dir is reused when its config is unchanged, so
    re-running a finished recording is a cheap no-op -- and editing the config
    recomputes exactly the affected stages (tweak the triangulation or the videos
    and re-run; the slow 2D detection is reused). Pass --overwrite to force a
    recompute anyway: bare redoes every stage, or name stages to redo only those
    (plus the stages after them).

    Everything else is set in the config: the do_<stage> toggles choose which stages
    run, alongside fps, background and each stage's parameters. -c wins when given;
    with no -c, a run reuses the config.toml already in the output dir, else the
    packaged default.
    """
    _configure_logging(log_level.value)
    _cmd_run(
        argparse.Namespace(
            inputs=inputs,
            recursive=recursive,
            config=config,
            output=output,
            overwrite=overwrite,
            log_level=log_level.value,
        )
    )


@app.command()
def inspect(
    input: Annotated[str, typer.Argument(help="path to a result .h5 file")],
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Print a summary of a result .h5 file."""
    _configure_logging(log_level.value)
    _cmd_inspect(argparse.Namespace(input=input))


@app.command()
def doctor(log_level: LogLevelOption = LogLevel.info) -> None:
    """Report installation/runtime: accelerators, frame I/O, weights."""
    _configure_logging(log_level.value)
    _cmd_doctor(argparse.Namespace())


@app.command()
def gui(
    path: Annotated[
        str,
        typer.Argument(
            help="a project directory, a results.h5 file, or a directory containing "
            "one (e.g. <recording>/deeperfly_outputs)"
        ),
    ],
    recording: Annotated[
        str | None,
        typer.Option(
            "--recording",
            help="which recording to open when PATH is a project (slug, id, or id "
            "prefix). A project holding exactly one recording needs no --recording; "
            "otherwise they are listed",
        ),
    ] = None,
    footage_dir: Annotated[
        str | None,
        typer.Option(
            "--footage-dir",
            help="directory to search for the footage if the paths recorded in "
            "results.h5 no longer resolve",
        ),
    ] = None,
    host: Annotated[
        str,
        typer.Option(
            "--host",
            help="address to bind the server to; the loopback default keeps the "
            "editor private (bind a routable address only behind a trusted "
            "network -- it is unauthenticated; prefer an 'ssh -L' tunnel)",
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", help="TCP port to serve on (0 picks a free one)"),
    ] = 8000,
    no_browser: Annotated[
        bool,
        typer.Option("--no-browser", help="do not open a browser on startup"),
    ] = False,
    keep_alive: Annotated[
        bool,
        typer.Option(
            "--keep-alive",
            help="keep the server running after the browser is closed (by default "
            "it stops a few seconds after the last tab closes; a refresh reconnects)",
        ),
    ] = False,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Serve the interactive web viewer/corrector for a result.

    Starts a local server and opens a browser editor. View every camera with its
    2D skeleton overlay and drag keypoints to annotate the ground-truth 2D pose;
    the 3D point is re-derived live from your labels and every view updates.
    Ground-truth labels are written to a labels.h5 sidecar and never modify
    results.h5 (an older corrections.h5 is migrated on open). It runs headless and
    can be reached from another machine's browser (default-bound to localhost;
    tunnel with 'ssh -L' for remote use).
    """
    _configure_logging(log_level.value)
    _cmd_gui(
        argparse.Namespace(
            path=path,
            recording=recording,
            footage_dir=footage_dir,
            host=host,
            port=port,
            no_browser=no_browser,
            keep_alive=keep_alive,
        )
    )


@app.command(name="labels-export")
def labels_export(
    path: Annotated[
        str,
        typer.Argument(
            help="a results.h5 file, or a directory containing one "
            "(the labels.h5 beside it is exported)"
        ),
    ],
    output: Annotated[
        str | None,
        typer.Option(
            "-o",
            "--output",
            help="output .npz (default: labels_gt.npz beside results.h5)",
        ),
    ] = None,
    include_projection: Annotated[
        bool,
        typer.Option(
            "--include-projection",
            help="also export GT confirmed from the 3D reprojection (the model's own "
            "guess); excluded by default so the export is human-placed pixels only",
        ),
    ] = False,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Export saved ground-truth labels (labels.h5) as a training/eval dataset (.npz).

    Writes the provenance-filtered GT pixels + occluded mask in footage pixel space
    (arrays ``gt_xy`` (V,T,P,2), ``gt_mask`` (V,T,P), ``occluded`` (V,T,P), ``absent``
    (P,), plus ``point_names`` / ``camera_names``). Annotate and Save in 'deeperfly gui'
    first.

    Keypoints declared **absent** (not on this animal -- an amputated leg) are excluded
    from *both* ``gt_mask`` and ``occluded``, and reported separately in ``absent``: such a
    keypoint is not ground truth, and it is not "occluded in every view" either, so it must
    not be supervised in either direction. Mask it in training.
    """
    _configure_logging(log_level.value)
    _cmd_labels_export(
        argparse.Namespace(
            path=path, output=output, include_projection=include_projection
        )
    )


@app.command(name="labels-absent")
def labels_absent(
    paths: Annotated[
        list[str],
        typer.Argument(
            help="one or more results.h5 files, or directories containing one "
            "(e.g. <recording>/deeperfly_outputs). Pass every clip of the same animal."
        ),
    ],
    points: Annotated[
        str,
        typer.Option(
            "--points",
            help="comma-separated keypoint names or fnmatch globs, e.g. "
            "'lf_femur_tibia,lf_tibia_tarsus,lf_claw' or 'lf_*'. An unmatched name is an "
            "error, so a typo cannot silently declare nothing.",
        ),
    ],
    subject: Annotated[
        str | None,
        typer.Option(
            "--subject",
            help="optional animal identifier stamped into the sidecar, so one animal's "
            "several recordings can be grouped later",
        ),
    ] = None,
    frames: Annotated[
        str | None,
        typer.Option(
            "--frames",
            help="restrict to a frame or half-open range: '900' (that frame), '900:' "
            "(from 900 to the end -- a leg lost mid-recording), '0:900', ':900'. "
            "Omit for the whole recording, which is the usual case.",
        ),
    ] = None,
    clear: Annotated[
        bool,
        typer.Option(
            "--clear",
            help="un-declare instead of declare. Nothing is lost either way: the labels "
            "an absence declaration hides are quarantined, not deleted.",
        ),
    ] = False,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Mark keypoints as absent -- not on this animal -- in one or more labels.h5.

    For an amputated leg or an ablated antenna: the keypoint does not exist, which is
    different from "occluded" (it exists but no camera can see it) and from "unlabeled".
    The declaration is per keypoint and, by default, covers the whole recording -- one
    command replaces marking every frame and every view by hand. Pass ``--frames`` for a
    limb lost part-way through (``--frames 900:``).

    Downstream, an absent keypoint is dropped from the 3D solve, excluded from the
    training export in *both* directions (neither ground truth nor occluded), and removed
    from labeling-progress denominators.

    The editor has the same gesture (select a joint, press ``x``). Close any running
    'deeperfly gui' on these directories first: saving is a whole-file rewrite, so an open
    session would overwrite what this writes.
    """
    _configure_logging(log_level.value)
    _cmd_labels_absent(
        argparse.Namespace(
            paths=paths, points=points, subject=subject, clear=clear, frames=frames
        )
    )


@app.command(name="labels-suggest")
def labels_suggest(
    path: Annotated[
        str,
        typer.Argument(
            help="a results.h5 file, or a directory containing one "
            "(e.g. <recording>/deeperfly_outputs)"
        ),
    ],
    count: Annotated[
        int, typer.Option("-n", "--count", help="how many frames to suggest")
    ] = 20,
    min_gap_s: Annotated[
        float,
        typer.Option(
            "--min-gap-s",
            help="HARD minimum spacing between suggestions, in seconds; also keeps "
            "them away from the frames already labeled. At 100 fps adjacent frames "
            "are near-duplicates, so without this a top-N is one hard moment "
            "sampled N times",
        ),
    ] = 2.0,
    fps: Annotated[
        float | None,
        typer.Option(
            "--fps",
            help="capture rate for --min-gap-s (default: the fps recorded in "
            "results.h5, else 100 with a warning)",
        ),
    ] = None,
    reserve_diversity: Annotated[
        float,
        typer.Option(
            "--reserve-diversity",
            help="fraction of -n taken on a uniform temporal grid instead of by "
            "score, so the round still sees typical poses and not only the tail",
        ),
    ] = 0.25,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            help="px; the RANSAC inlier gate and the 'this cell disagrees' gate in "
            "the reported reasons (a ranking knob, not an accuracy claim)",
        ),
    ] = 15.0,
    cap: Annotated[
        float,
        typer.Option(
            "--cap",
            help="px; per-cell residual saturation, so one blown view cannot turn "
            "the ranking into a single-outlier lottery",
        ),
    ] = 60.0,
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k",
            help="how many of the worst joints are averaged into a frame's score "
            "(a frame is worth a pass when several joints are wrong)",
        ),
    ] = 8,
    min_views: Annotated[
        int,
        typer.Option(
            "--min-views",
            help="observing views a joint needs to be scorable; below 3 a joint "
            "reprojects onto its own two views by construction, which reads as "
            "agreement it has not earned",
        ),
    ] = 3,
    points: Annotated[
        list[str] | None,
        typer.Option(
            "--points",
            help="glob(s) over the skeleton point names to score (repeatable; "
            "default all), e.g. --points '*tibia*'",
        ),
    ] = None,
    cameras: Annotated[
        list[str] | None,
        typer.Option(
            "--cameras",
            help="glob(s) over the camera names to score (repeatable; default all). "
            "Triangulation always uses every view",
        ),
    ] = None,
    exclude_labeled: Annotated[
        bool,
        typer.Option(
            "--exclude-labeled/--no-exclude-labeled",
            help="skip frames that already carry human work, read from the "
            "labels.h5 sidecar (and keep suggestions --min-gap-s away from them)",
        ),
    ] = True,
    output: Annotated[
        str | None,
        typer.Option(
            "-o",
            "--output",
            help="output .json (default: labels_suggest.json beside results.h5)",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="print the ranking and write nothing"),
    ] = False,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Rank the frames worth labeling next (active learning) -> labels_suggest.json.

    Scores each frame by the multi-view disagreement of the detector's own 2D: every
    joint is RANSAC-triangulated from the pristine pose2d detections and each view's
    detection is compared to the reprojection. Views cannot conspire, so a large
    residual means the model is probably wrong -- which is where a human label buys
    the most. Detector confidence is deliberately not used: it is confidently wrong
    exactly where it is wrong.

    The ranking is never the raw top-N. A hard --min-gap-s keeps the picks (and the
    frames already labeled) apart, since at 100 fps neighbouring frames are the
    same pose; --reserve-diversity spends part of the list on a uniform temporal
    grid; and every pick is reported with the joints and views that drove it, so
    the list is usable straight from this output.

    Writes a JSON sidecar beside results.h5 for 'deeperfly gui' to navigate;
    results.h5 and labels.h5 are only ever read.
    """
    _configure_logging(log_level.value)
    _cmd_labels_suggest(
        argparse.Namespace(
            path=path,
            count=count,
            min_gap_s=min_gap_s,
            fps=fps,
            reserve_diversity=reserve_diversity,
            threshold=threshold,
            cap=cap,
            top_k=top_k,
            min_views=min_views,
            points=points,
            cameras=cameras,
            exclude_labeled=exclude_labeled,
            output=output,
            dry_run=dry_run,
        )
    )


# -- project (a command group) -----------------------------------------------

project_app = typer.Typer(
    no_args_is_help=True,
    help="Group related recordings into a project: one skeleton, shared camera rigs, "
    "and one place to see what is labeled. A project INDEXES recordings -- their "
    "results.h5 / labels.h5 stay where they are and are adopted by symlink, so no "
    "label is ever copied or moved to create one.",
)
app.add_typer(project_app, name="project")

#: The optional project path shared by every verb that opens one. Omitted, the nearest
#: enclosing project is used (like git), so the root need not be retyped.
ProjectArg = Annotated[
    str | None,
    typer.Argument(
        help="the project directory (default: the nearest one enclosing the cwd)"
    ),
]


@project_app.command("new")
def project_new(
    root: Annotated[str, typer.Argument(help="directory to create the project in")],
    name: Annotated[
        str | None,
        typer.Option("--name", help="project name (default: the directory's name)"),
    ] = None,
    skeleton: Annotated[
        str,
        typer.Option(
            "--skeleton",
            help="'fly38' (the packaged 38-point Drosophila skeleton), 'blank' (define "
            "your own), or a path to a TOML file with a [skeleton] table",
        ),
    ] = "fly38",
    description: Annotated[
        str | None, typer.Option("--description", help="free-text description")
    ] = None,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Create a project: a skeleton, a place for rigs, and an empty recording index.

    Writes a project.toml (the index) and a skeleton.toml (what is tracked). Nothing
    else -- recordings are adopted afterwards with 'deeperfly project add', and a camera
    rig is either solved later or pointed at with a calibration file.

    Start from 'blank' for a new animal or rig: you then label with no calibration at
    all and solve the rig from those labels once there are enough correspondences.
    """
    _configure_logging(log_level.value)
    _cmd_project_new(
        argparse.Namespace(
            root=root, name=name, skeleton=skeleton, description=description
        )
    )


@project_app.command("add")
def project_add(
    project: Annotated[str, typer.Argument(help="the project directory to adopt into")],
    sources: Annotated[
        list[str],
        typer.Argument(
            metavar="RECORDING...",
            help="one or more recordings: a recording directory, its "
            "deeperfly_outputs/, or a results.h5",
        ),
    ],
    copy: Annotated[
        bool,
        typer.Option(
            "--copy",
            help="copy each recording's outputs into the project instead of linking "
            "them. The copy is a SNAPSHOT: labels authored in the original will not "
            "appear in the project, and vice versa",
        ),
    ] = False,
    slug: Annotated[
        str | None,
        typer.Option(
            "--slug",
            help="name for the recording inside the project (single source only; "
            "default: the recording directory's name)",
        ),
    ] = None,
    subject: Annotated[
        str | None,
        typer.Option(
            "--subject",
            help="animal identifier, so one specimen's several clips group together "
            "(read from results.h5 when it records one)",
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option(
            "-c",
            "--config",
            help="config supplying the per-camera footage globs. Without it, each "
            "video file in the recording directory becomes a camera named after the "
            "file (which is what a from-scratch recording wants)",
        ),
    ] = None,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Adopt recordings into a project, by reference.

    Each recording's deeperfly_outputs/ is SYMLINKED into the project, so the labels.h5
    the editor writes is the very file a training set reads -- adopting copies nothing
    and can lose nothing. A recording with no outputs yet (just videos) is adopted too;
    that is the from-scratch starting point.

    Recordings are identified by content, not path, so adopting the same one twice is a
    no-op and a backup copy is recognized as the same recording.
    """
    _configure_logging(log_level.value)
    _cmd_project_add(
        argparse.Namespace(
            project=project,
            sources=sources,
            copy=copy,
            slug=slug,
            subject=subject,
            config=config,
        )
    )


@project_app.command("ls")
def project_ls(
    project: ProjectArg = None,
    log_level: LogLevelOption = LogLevel.warning,
) -> None:
    """List a project's recordings."""
    _configure_logging(log_level.value)
    _cmd_project_ls(argparse.Namespace(project=project))


@project_app.command("status")
def project_status(
    project: ProjectArg = None,
    log_level: LogLevelOption = LogLevel.warning,
) -> None:
    """Report labeling progress across a project.

    Per recording: frames, frames carrying labels, frames marked reviewed, ground-truth
    points, occlusion marks, and whether its outputs are present. The counts are the
    LIVE rows of each labels.h5 -- a keypoint declared absent is not counted as ground
    truth, matching what an export and a training set will see.
    """
    _configure_logging(log_level.value)
    _cmd_project_status(argparse.Namespace(project=project))


@project_app.command("rm")
def project_rm(
    recording: Annotated[
        str, typer.Argument(help="a recording's slug, id, or unambiguous id prefix")
    ],
    project: ProjectArg = None,
    delete: Annotated[
        bool,
        typer.Option(
            "--delete",
            help="also remove the project's own directory for the recording. For a "
            "linked recording that removes only the link; it refuses when the outputs "
            "are a real directory, since that would be the only copy of the labels",
        ),
    ] = False,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Drop a recording from the project index (its files are left alone)."""
    _configure_logging(log_level.value)
    _cmd_project_rm(
        argparse.Namespace(project=project, recording=recording, delete=delete)
    )


# -- config (a command group) ------------------------------------------------

config_app = typer.Typer(
    no_args_is_help=True,
    help="Discover and set config keys without reading the whole file. Every key, its "
    "default and its documentation are derived from the code, so they cannot drift from "
    "it.",
)
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show(
    section: Annotated[
        str | None,
        typer.Argument(
            help="one section (e.g. triangulation, pipeline, annotation); omit for all"
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option("-c", "--config", help="config TOML (default: the packaged one)"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("-v", "--verbose", help="also print what each key means"),
    ] = False,
    log_level: LogLevelOption = LogLevel.warning,
) -> None:
    """Print a config section's keys, values, defaults and documentation.

    Keys you actually set are marked; everything else is a default. That distinction is
    what a config file cannot show you -- a 706-line file where 690 lines are defaults
    reads as 706 decisions.

    The detection plan, cameras, skeleton and video specs are open-ended and are not
    described here; they live in the file (or, for the skeleton and rig, in the project).
    """
    _configure_logging(log_level.value)
    _cmd_config_show(
        argparse.Namespace(section=section, config=config, verbose=verbose)
    )


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="SECTION.KEY, e.g. triangulation.method")],
    value: Annotated[str, typer.Argument(help="the new value")],
    config: Annotated[
        str,
        typer.Option("-c", "--config", help="the config TOML to edit (required)"),
    ],
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Set one config key, validated the same way a run would validate it.

    Appends rather than rewriting, so comments survive. The result is loaded through the
    same strict validator a run uses, so this cannot write a key a run would reject.
    """
    _configure_logging(log_level.value)
    _cmd_config_set(argparse.Namespace(key=key, value=value, config=config))


# -- calibration (a command group) -------------------------------------------
#
# The first sub-command group in this CLI: the older `labels-*` commands are flat and
# hyphenated. A group is used here because calibrations are about to grow verbs
# (`ls`, `use`) alongside the project layer, and `deeperfly calibration show` reads
# better than a fourth hyphenated prefix. The flat commands keep working unchanged.

calibration_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect and extract solved camera rigs (calibration.toml).",
)
app.add_typer(calibration_app, name="calibration")


@calibration_app.command("show")
def calibration_show(
    path: Annotated[
        str,
        typer.Argument(help="a calibration.toml, or a directory containing one"),
    ],
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Print a calibration's cameras, provenance and reprojection residuals.

    The residuals are the part worth reading. A rig that reprojects at 15 px is not a
    rig you want triangulating a fly, and nothing downstream will tell you so.
    """
    _configure_logging(log_level.value)
    _cmd_calibration_show(argparse.Namespace(path=path))


@calibration_app.command("export")
def calibration_export(
    path: Annotated[
        str,
        typer.Argument(
            help="a results.h5 file, or a directory containing one "
            "(e.g. <recording>/deeperfly_outputs)"
        ),
    ],
    output: Annotated[
        str | None,
        typer.Option(
            "-o",
            "--output",
            help="output .toml (default: calibration.toml beside results.h5)",
        ),
    ] = None,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Extract a portable calibration.toml from a result's stored camera rig.

    The bundle-adjusted rig in a results.h5 can only be used by the recording that
    produced it. Exporting turns it into a file a *second* recording can be pointed at
    ([cameras] calibration = "..." in its config), diffed against a later solve, or
    shared with a collaborator.

    The bundle-adjusted rig is preferred; a result with no bundle adjustment exports
    the un-refined config rig it detected with, with a warning and honest provenance.
    """
    _configure_logging(log_level.value)
    _cmd_calibration_export(argparse.Namespace(path=path, output=output))


@app.command()
def calibrate(
    project: ProjectArg = None,
    points: Annotated[
        str,
        typer.Option(
            "--points",
            help="what drives the solve: 'landmarks' (dedicated calibration points), "
            "'keypoints' (the skeleton's ground truth), or 'both' (default). Landmarks "
            "are worth far more per label -- a STATIC one is three unknowns however many "
            "frames observe it, while a keypoint is three unknowns PER FRAME because the "
            "animal moved",
        ),
    ] = "both",
    recordings: Annotated[
        list[str] | None,
        typer.Option(
            "--recording",
            help="restrict to these recordings (repeatable; default: all). A "
            "rig-scoped landmark ties every named recording into one solve",
        ),
    ] = None,
    from_calibration: Annotated[
        str | None,
        typer.Option(
            "--from-calibration",
            help="take intrinsics AND the initial extrinsics from an existing "
            "calibration (a board solve, or a previous run) instead of solving cold",
        ),
    ] = None,
    lens_mm: Annotated[
        float | None,
        typer.Option("--lens-mm", help="lens focal length in mm (with --sensor-mm)"),
    ] = None,
    sensor_mm: Annotated[
        float | None,
        typer.Option("--sensor-mm", help="sensor width in mm (with --lens-mm)"),
    ] = None,
    focal_px: Annotated[
        float | None,
        typer.Option("--focal-px", help="focal length in pixels, stated directly"),
    ] = None,
    scale_from: Annotated[
        str | None,
        typer.Option(
            "--scale-from",
            metavar="A,B=DISTANCE",
            help="pin the scale with a known distance between two landmarks, e.g. "
            "'tether_tip,coverslip_ne=1.8'. Without it the rig is valid up to scale: "
            "angles are meaningful, lengths and velocities are not",
        ),
    ] = None,
    free_focal: Annotated[
        bool,
        typer.Option(
            "--free-focal",
            help="let the solver adjust focal length. Off by default and rarely right: "
            "focal error trades against depth, so the residual improves while the rig "
            "gets worse",
        ),
    ] = False,
    free_k1: Annotated[
        bool,
        typer.Option(
            "--free-k1",
            help="let the solver adjust the first radial distortion coefficient. Needs "
            "plenty of well-spread labels to be identifiable",
        ),
    ] = False,
    include_unreviewed: Annotated[
        bool,
        typer.Option(
            "--include-unreviewed",
            help="use frames that are not marked reviewed. Off by default: a "
            "half-labeled frame contributes a systematically biased point, and no "
            "residual will reveal that afterwards",
        ),
    ] = False,
    loss: Annotated[
        str,
        typer.Option("--loss", help="robust loss: linear / huber / cauchy / arctan"),
    ] = "cauchy",
    f_scale: Annotated[
        float, typer.Option("--f-scale", help="robust loss scale, in pixels")
    ] = 4.0,
    name: Annotated[
        str | None,
        typer.Option("--name", help="calibration name (default: 'from-labels')"),
    ] = None,
    accept: Annotated[
        bool,
        typer.Option(
            "--accept",
            help="make the solved rig the project's current calibration. Without it the "
            "file is written but nothing switches over, so a bad solve cannot silently "
            "replace a good one",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="report readiness only: what is labeled, what is still weak, and what "
            "labeling would fix it. Solves nothing and writes nothing",
        ),
    ] = False,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Solve a project's camera rig from its hand labels.

    The from-scratch path: label 2D by hand with no calibration, then recover the rig from
    those labels. Run it with --dry-run while you are still labeling -- it reports exactly
    what is still weak and what labeling would fix it.

    Intrinsics are never derived from the labels. Extrinsics can be recovered from
    correspondences; focal length essentially cannot, and a solve permitted to guess it
    reports a small residual for a wrong rig. Supply --from-calibration (best),
    --lens-mm/--sensor-mm, or --focal-px.

    Nothing becomes the project's calibration without --accept.
    """
    _configure_logging(log_level.value)
    _cmd_calibrate(
        argparse.Namespace(
            project=project,
            points=points,
            recordings=recordings,
            from_calibration=from_calibration,
            lens_mm=lens_mm,
            sensor_mm=sensor_mm,
            focal_px=focal_px,
            scale_from=scale_from,
            free_focal=free_focal,
            free_k1=free_k1,
            include_unreviewed=include_unreviewed,
            loss=loss,
            f_scale=f_scale,
            name=name,
            accept=accept,
            dry_run=dry_run,
        )
    )


@app.command(name="labels-merge")
def labels_merge(
    recording: Annotated[
        str, typer.Argument(help="the project recording to merge INTO (slug or id)")
    ],
    source: Annotated[
        str,
        typer.Argument(
            metavar="SOURCE",
            help="the labels.h5 to merge FROM, or a directory containing one",
        ),
    ],
    project: ProjectArg = None,
    on_conflict: Annotated[
        str,
        typer.Option(
            "--on-conflict",
            help="how to settle a cell both sides authored differently AND with the same "
            "provenance: 'manual' (default -- leave it and queue it for review), 'ours', "
            "'theirs', or 'newest'. Provenance decides first regardless: a human's drag "
            "always beats a bulk-confirmed reprojection, which is the model's own guess",
        ),
    ] = "manual",
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="actually write. Without it this is a dry run that changes nothing. "
            "Applying always snapshots the destination labels.h5 first",
        ),
    ] = False,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Merge a second label set into a project recording's labels.

    For ground truth that ended up in two places -- an earlier round, another annotator's
    directory, the same recording under a different tree. A project indexes a recording
    once (by content), so the other copy's labels are otherwise invisible.

    Points and cameras are matched BY NAME, never by index: two same-sized skeletons in
    different orders are the one case where an index-based copy would silently corrupt
    every label while leaving each cell looking plausible. A same-named camera whose
    footage size differs is refused outright -- ground truth is stored in footage pixels.

    Dry run by default.
    """
    _configure_logging(log_level.value)
    _cmd_labels_merge(
        argparse.Namespace(
            project=project,
            recording=recording,
            source=source,
            on_conflict=on_conflict,
            apply=apply,
        )
    )


def _normalize_overwrite_argv(argv: list[str]) -> list[str]:
    """Let ``run``'s ``--overwrite`` take zero or more space-separated stage names.

    click options can't be variadic, so rewrite a bare ``--overwrite`` into
    ``--overwrite <_OVERWRITE_ALL>`` and ``--overwrite a b`` into the repeated
    ``--overwrite a --overwrite b`` the ``multiple=True`` option accepts. Only known
    stage names (:data:`STAGES`) are consumed after the flag, leaving the positional
    argument and later options untouched; ``--overwrite=...`` passes through as-is.

    Parameters
    ----------
    argv
        The raw argument vector (the subcommand first).

    Returns
    -------
    list of str
        ``argv`` with ``run``'s variadic ``--overwrite`` rewritten (unchanged for
        other subcommands).
    """
    if not (argv and argv[0] == "run"):
        return argv
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok != "--overwrite":
            out.append(tok)
            i += 1
            continue
        j = i + 1
        picked: list[str] = []
        while j < len(argv) and argv[j] in STAGES:
            picked.append(argv[j])
            j += 1
        for stage in picked or [_OVERWRITE_ALL]:
            out += ["--overwrite", stage]
        i = j
    return out


def main(argv: list[str] | None = None) -> None:
    """Entry point: parse ``argv`` (default ``sys.argv``) and dispatch a subcommand.

    Runs the Typer app in standalone mode so usage errors and ``--help`` render
    through rich, but swallows the ``SystemExit(0)`` of a clean exit so
    ``main([...])`` returns normally as a library / from the tests. Real failures
    still propagate. ``argv`` is normalized first
    (:func:`_normalize_overwrite_argv`).

    Parameters
    ----------
    argv
        The argument vector; defaults to ``sys.argv[1:]``.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    argv = _normalize_overwrite_argv(argv)
    command = typer.main.get_command(app)
    try:
        command(args=argv, prog_name="deeperfly")
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
