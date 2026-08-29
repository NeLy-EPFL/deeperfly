"""The ``gui`` command worker: launch the interactive web viewer/corrector.

Kept thin and free of any web import at module load -- the FastAPI/uvicorn
import happens inside :func:`deeperfly.gui.serve`, so importing ``deeperfly``
(and this module) stays cheap for every command other than ``gui``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from .console import LogLevel, LogLevelOption, _configure_logging

log = logging.getLogger("deeperfly")

#: Bind addresses that keep the editor on the local machine (no warning).
_LOOPBACK = ("127.0.0.1", "localhost", "::1")


def _find_results(path: Path) -> Path:
    """Resolve ``path`` to a ``results.h5`` file.

    Accepts the file directly, or a directory containing ``results.h5`` (or a
    ``deeperfly_outputs/results.h5`` beneath it, i.e. a recording directory).

    Parameters
    ----------
    path
        A ``results.h5`` file or a directory to search.

    Returns
    -------
    Path
        The resolved ``results.h5`` path.

    Raises
    ------
    SystemExit
        If no ``results.h5`` can be found at ``path``.
    """
    if path.is_file():
        return path
    if path.is_dir():
        for candidate in (
            path / "results.h5",
            path / "deeperfly_outputs" / "results.h5",
        ):
            if candidate.exists():
                return candidate
    raise SystemExit(
        f"no results.h5 found at {path} -- pass a results.h5 file or a directory "
        "containing one (e.g. <recording>/deeperfly_outputs)"
    )


def gui(
    path: Annotated[
        str,
        typer.Argument(
            help="a project directory, a results.h5 file, or a directory containing "
            "one (e.g. <recording>/deeperfly_outputs)"
        ),
    ],
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
    results.h5. It runs headless and
    can be reached from another machine's browser (default-bound to localhost;
    tunnel with 'ssh -L' for remote use).

    Point it at a PROJECT to get all of its recordings: the first one opens, and the
    editor's recording picker (the toolbar button, or 'b') switches between them
    without restarting.
    """
    _configure_logging(log_level.value)
    # A project directory resolves inside `serve` (it opens the first of its recordings,
    # and one with no results.h5 opens uncalibrated); anything else is resolved here so
    # a bad path fails before the server starts. `recording` is not a CLI option: which
    # recording to open is a question the editor's own picker answers, live.
    from ..project import PROJECT_FILENAME

    target = Path(path)
    is_project = (target / PROJECT_FILENAME).exists()
    results_path = target if is_project else _find_results(target)
    if host not in _LOOPBACK:
        log.warning(
            "binding %s exposes the editor on the network without authentication; "
            "prefer the default localhost and an `ssh -L` tunnel for remote use",
            host,
        )
    try:
        from ..gui import serve
    except ImportError as exc:  # pragma: no cover -- exercised manually
        raise SystemExit(str(exc)) from exc
    try:
        serve(
            results_path,
            footage_dir=footage_dir,
            recording=None,
            host=host,
            port=port,
            open_browser=not no_browser,
            exit_on_close=not keep_alive,
        )
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc


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
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Export saved ground-truth labels (labels.h5) as a training/eval dataset (.npz).

    Writes the GT pixels + the hidden mask in footage pixel space
    (arrays ``gt_xy`` (V,T,P,2), ``gt_mask`` (V,T,P), ``occluded`` (V,T,P), ``absent``
    (P,), plus ``point_names`` / ``camera_names``). Annotate and Save in 'deeperfly gui'
    first.

    ``occluded`` is the editor's **Hidden** flag: "hold this cell out of the training loss".
    It keeps its array name for compatibility, and it is a *separate* axis from ``gt_mask``,
    not a filter already applied to it -- a cell can carry a pixel and be held out. Your loss
    mask is ``gt_mask & ~occluded``.

    Keypoints declared **absent** (not on this animal -- an amputated leg) are excluded
    from *both* ``gt_mask`` and ``occluded``, and reported separately in ``absent``: such a
    keypoint is not ground truth, and a hold-out mark on something already unsupervised is
    not a decision anyone made. Mask it in training.
    """
    _configure_logging(log_level.value)
    import numpy as np

    from ..labels import export_absent, export_gt, labels_identity, load_labels
    from ..results import PoseResult, StageStore

    results_path = _find_results(Path(path))
    result = PoseResult.load(results_path)
    store = StageStore(results_path)
    identity = labels_identity(
        point_names=list(result.skeleton.point_names),
        camera_names=list(result.cameras.names),
        n_frames=result.n_frames,
        image_sizes=store.read_image_sizes(),
        footage=store.read_footage(),
    )
    labels_path = results_path.parent / "labels.h5"
    labels = load_labels(labels_path, identity=identity)
    if labels is None:
        raise SystemExit(
            f"no labels.h5 next to {results_path} -- annotate with 'deeperfly gui' "
            "and Save first"
        )
    gt_xy, gt_mask, occluded = export_gt(labels)
    absent = export_absent(labels)
    out = Path(output) if output else results_path.parent / "labels_gt.npz"
    np.savez(
        out,
        gt_xy=gt_xy,
        gt_mask=gt_mask,
        occluded=occluded,
        # (T, P) "not on this animal", per frame (a limb can be lost part-way through a
        # recording). Strictly additive: a consumer that predates this key should default
        # it to all-False. The intended training semantics is MASKING -- an absent keypoint
        # contributes no gradient -- not negative supervision.
        absent=absent,
        point_names=np.array(list(result.skeleton.point_names)),
        camera_names=np.array(list(result.cameras.names)),
    )
    whole = absent.all(axis=0) if absent.size else absent
    names = [str(n) for n, a in zip(result.skeleton.point_names, whole) if a]
    n_partial = int((absent.any(axis=0) & ~whole).sum()) if absent.size else 0
    log.info(
        "exported %d ground-truth point(s), %d held out of the loss, %d keypoint(s) not on "
        "this animal%s%s (footage space) -> %s",
        int(gt_mask.sum()),
        int(occluded.sum()),
        int(whole.sum()),
        f" ({', '.join(names)})" if names else "",
        f", plus {n_partial} absent in some frames only" if n_partial else "",
        out,
    )


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
            "'lf_femur_tibia,lf_tibia_tarsus,lf_pretarsus' or 'lf_*'. An unmatched name is an "
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
    different from **Hidden** (it exists and keeps its position; only the training loss
    skips that cell) and from "unlabeled". The declaration is per keypoint and, by default,
    covers the whole recording -- one command replaces marking every frame and every view by
    hand. Pass ``--frames`` for a limb lost part-way through (``--frames 900:``).

    Downstream, an absent keypoint is dropped from the 3D solve, excluded from the
    training export in *both* directions (neither ground truth nor hidden), and removed
    from labeling-progress denominators.

    The editor has the same gesture (select a joint, press ``x``). Close any running
    'deeperfly gui' on these directories first: saving is a whole-file rewrite, so an open
    session would overwrite what this writes.
    """
    _configure_logging(log_level.value)
    import numpy as np

    from ..labels import (
        Labels,
        labels_identity,
        load_labels,
        resolve_point_names,
        save_labels,
    )
    from ..results import PoseResult, StageStore

    names = [n.strip() for n in str(points).split(",") if n.strip()]
    if not names:
        raise SystemExit(
            "--points is empty; pass e.g. --points 'lf_femur_tibia,lf_pretarsus'"
        )

    frame_spec = frames
    span: tuple[int, int] | None = None
    if frame_spec:
        # "900:" (from 900 to the end), "0:900", ":900", "900" (that frame alone).
        try:
            if ":" in str(frame_spec):
                lo_s, _, hi_s = str(frame_spec).partition(":")
                span = (int(lo_s) if lo_s else 0, int(hi_s) if hi_s else -1)
            else:
                span = (int(frame_spec), int(frame_spec) + 1)
        except ValueError:
            raise SystemExit(
                f"--frames {frame_spec!r} is not a frame or a RANGE like '900:' or "
                "'0:900' (end exclusive)"
            ) from None

    for raw in paths:
        results_path = _find_results(Path(raw))
        result = PoseResult.load(results_path)
        store = StageStore(results_path)
        identity = labels_identity(
            point_names=list(result.skeleton.point_names),
            camera_names=list(result.cameras.names),
            n_frames=result.n_frames,
            image_sizes=store.read_image_sizes(),
            footage=store.read_footage(),
        )
        try:
            idx = resolve_point_names(names, list(result.skeleton.point_names))
        except ValueError as exc:
            raise SystemExit(f"{results_path}: {exc}") from exc

        labels_path = results_path.parent / "labels.h5"
        labels = load_labels(labels_path, identity=identity)
        if labels is None:
            labels = Labels.empty(
                result.n_views, result.n_frames, len(result.skeleton.point_names)
            )
        before = labels.absent.copy()
        if span is None:
            labels.set_absent(idx, not clear)  # the whole recording
        else:
            lo, hi = span
            hi = result.n_frames if hi < 0 else min(hi, result.n_frames)
            if not 0 <= lo < hi:
                raise SystemExit(
                    f"{results_path}: --frames {frame_spec!r} is empty or out of range "
                    f"for a {result.n_frames}-frame recording"
                )
            labels.set_absent(idx, not clear, frames=range(lo, hi))
        after = labels.absent
        if np.array_equal(before, after):
            log.info("%s: already as requested, not rewritten", labels_path)
            continue

        # Quarantine counts, so the operator sees exactly what the declaration displaced.
        vetoed = after[None, :, :]
        vetoed_gt = int((labels.gt_authored & vetoed).sum())
        vetoed_occ = int((labels.occluded & vetoed).sum())
        save_labels(
            labels_path,
            labels,
            identity=identity,
            subject_id=subject or labels.subject_id,
        )
        whole = after.all(axis=0)
        partial = after.any(axis=0) & ~whole
        declared = [str(n) for n, a in zip(result.skeleton.point_names, whole) if a]
        some = [str(n) for n, a in zip(result.skeleton.point_names, partial) if a]
        log.info(
            "%s: %d keypoint(s) absent in every frame (%s)%s%s",
            labels_path,
            len(declared),
            ", ".join(declared) if declared else "none",
            f"; {len(some)} in some frames only ({', '.join(some)})" if some else "",
            (
                f"; quarantined {vetoed_gt} GT row(s) and {vetoed_occ} hidden mark(s) "
                "(restored if you un-declare)"
                if vetoed_gt or vetoed_occ
                else ""
            ),
        )
