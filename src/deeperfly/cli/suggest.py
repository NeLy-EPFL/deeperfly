"""The ``labels-suggest`` command worker: rank the frames worth labeling next.

Presentation only -- the ranking itself lives in :mod:`deeperfly.labels.suggest`, so
the GUI (which reads the JSON sidecar) and the tests share exactly the code the
CLI runs. The printed report is the *whole* feature without the GUI: the ranked
frames, why each was picked, and the two facts most likely to mislead (a reseeded
source, and a list that came up short of ``-n``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from .console import LogLevel, LogLevelOption, _configure_logging, _info_line, console
from .gui import _find_results

log = logging.getLogger("deeperfly")


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
    from ..labels.suggest import (
        SCORE_DESCRIPTION,
        SUGGESTIONS_FILENAME,
        build_suggestions,
        glob_mask,
        prepare_inputs,
        read_labeled_frames,
        score_frames,
        select_frames,
        stored_vs_pose2d,
        write_suggestions,
    )

    results_path = _find_results(Path(path))
    try:
        inputs = prepare_inputs(results_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    fps, stamped = inputs.fps(fps)
    if not stamped:
        log.warning(
            "results.h5 records no fps; assuming %g fps for the >= %g s spacing "
            "(pass --fps to be exact)",
            fps,
            min_gap_s,
        )
    min_gap_frames = max(1, int(round(min_gap_s * fps)))

    try:
        point_mask = glob_mask(inputs.point_names, points)
        camera_mask = glob_mask(inputs.camera_names, cameras)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    labels = None
    labels_path = results_path.parent / "labels.h5"
    # Absence shapes the RANKING, so it is read whether or not labeled frames are excluded:
    # it says which keypoints exist on this animal, which is not labeling progress. A broken
    # sidecar here is not fatal -- the ranking just cannot know, and says so.
    absent_mask = None
    absent_points: list[int] = []
    try:
        _lab = read_labeled_frames(labels_path, identity=inputs.identity)
    except ValueError:
        _lab = None
    if _lab and _lab.get("absent_spans"):
        import numpy as _np

        from ..labels import spans_to_absent

        absent_mask = spans_to_absent(
            _np.asarray(_lab["absent_spans"], dtype=int).reshape(-1, 3),
            inputs.n_frames,
            len(inputs.point_names),
        )
        # Report the whole-recording subset: "absent in some frames" is a different
        # statement and belongs in the per-frame accounting, not the headline.
        absent_points = [int(i) for i in _np.nonzero(absent_mask.all(axis=0))[0]]
    if exclude_labeled:
        try:
            labels = read_labeled_frames(labels_path, identity=inputs.identity)
        except ValueError as exc:
            raise SystemExit(
                f"{exc} -- pass --no-exclude-labeled to rank anyway (the queue would "
                "then re-offer frames a human has already worked on)"
            ) from exc
    excluded = sorted(
        set((labels or {}).get("labeled_frames", []))
        | set((labels or {}).get("reviewed_frames", []))
    )

    scores = score_frames(
        inputs.cameras,
        inputs.pts2d,
        threshold=threshold,
        cap=cap,
        top_k=top_k,
        min_views=min_views,
        point_mask=point_mask,
        camera_mask=camera_mask,
        absent_mask=absent_mask,
    )
    picks, shortfall = select_frames(
        scores,
        count=count,
        min_gap_frames=min_gap_frames,
        reserve_diversity=reserve_diversity,
        exclude=excluded,
    )

    params = {
        "count": int(count),
        "min_gap_s": float(min_gap_s),
        "fps": float(fps),
        "fps_from": "meta"
        if stamped and fps is None
        else ("option" if fps else "default"),
        "min_gap_frames": int(min_gap_frames),
        "reserve_diversity": float(reserve_diversity),
        "threshold_px": float(threshold),
        "cap_px": float(cap),
        "top_k": int(top_k),
        "min_views": int(min_views),
        "points": list(points) if points else None,
        "cameras": list(cameras) if cameras else None,
        "exclude_labeled": bool(exclude_labeled),
        "absent_points": absent_points,
        "absent_point_names": [
            str(inputs.point_names[i])
            for i in absent_points
            if i < len(inputs.point_names)
        ],
        "score": SCORE_DESCRIPTION,
    }
    out = Path(output) if output else results_path.parent / SUGGESTIONS_FILENAME
    doc = build_suggestions(
        inputs,
        scores,
        picks,
        params=params,
        shortfall=shortfall,
        labels=labels,
        output_dir=out.parent,
    )

    _report(
        inputs,
        scores,
        doc,
        results_path=results_path,
        labels=labels,
        excluded=excluded,
        trap=stored_vs_pose2d(inputs, scores, threshold=threshold),
    )

    if dry_run:
        console.print("[yellow]--dry-run[/yellow]: nothing written")
        return
    written = write_suggestions(out, doc)
    console.print(f"[green]wrote[/green] {written}")
    console.print(
        "next: open 'deeperfly gui' on this directory and walk the Suggested list",
        highlight=False,
    )


def _report(
    inputs,
    scores,
    doc: dict,
    *,
    results_path: Path,
    labels: dict | None,
    excluded: list[int],
    trap: dict | None,
) -> None:
    """Print the summary + the ranked list with per-frame reasons.

    Mirrors ``inspect``'s ``label   value`` style. The reseeded note and the
    stored-vs-``pose2d`` comparison are printed unconditionally on a reseeded file:
    that table *is* the degenerate-signal trap, and having it in the output is what
    stops it being silently reintroduced.
    """
    params, coverage, shortfall = doc["params"], doc["coverage"], doc["shortfall"]
    _info_line("file:       ", results_path)
    _info_line("views:      ", f"{inputs.n_views}  {inputs.camera_names}")
    _info_line("frames:     ", f"{inputs.n_frames}  @ {params['fps']:g} fps")
    _info_line("points:     ", inputs.n_points)
    _info_line(
        "scored:     ",
        f"{doc['source']['scored_array']} (the pristine detector output; "
        "triangulation/* is never scored)",
    )
    _info_line("cameras:    ", f"{inputs.cameras_from}/cameras")
    if inputs.reseeded:
        reseed = doc["source"].get("reseed", {})
        console.print(
            "[bold yellow]reseeded:   [/bold yellow]produced by dfpose.predict "
            f"({', '.join(reseed.get('detected_by', []))}) -- the contralateral cells "
            "of triangulation/points are reprojection seeds, so their stored residual "
            "is ~0 by construction and carries no information",
            highlight=False,
        )
        if reseed.get("model"):
            _info_line(
                "predictor:  ",
                f"{reseed['model']} ckpt md5 "
                f"{str(reseed.get('checkpoint_md5', '?'))[:12]}",
            )
    if trap is not None:
        _info_line(
            "the trap:   ",
            f"on the {trap['n_cells']} substituted cell(s), the STORED "
            f"reproj_error says median {trap['far_cells_median_px']:g} px / "
            f"{trap['far_cells_frac_over_thresh']:.3f} over threshold; "
            f"pose2d/points says {trap['pose2d_far_median_px']:g} px / "
            f"{trap['pose2d_far_frac_over_thresh']:.3f}",
        )
    if params["points"] or params["cameras"]:
        _info_line(
            "selection:  ",
            f"points={params['points'] or 'all'} cameras={params['cameras'] or 'all'} "
            "(triangulation still uses every view)",
        )
    n_exist = coverage.get("n_existing_points", inputs.n_points)
    denom = (
        f"of the {n_exist} existing joints"
        if n_exist != inputs.n_points
        else "of joints"
    )
    _info_line(
        "coverage:   ",
        f"{coverage['scorable_cell_frac']:.1%} of cells fired, "
        f"{coverage['scorable_joint_frac']:.1%} {denom} scorable "
        f"(>= {params['min_views']} views; median "
        f"{coverage['median_observing_views']:g})",
    )
    _info_line(
        "residual:   ",
        f"median {coverage['global_residual_median_px']:.2f} px over every scored cell",
    )
    if coverage["global_residual_median_px"] > params["threshold_px"]:
        console.print(
            f"[yellow]NOTE[/yellow] the global median residual "
            f"({coverage['global_residual_median_px']:.2f} px) already exceeds "
            f"--threshold {params['threshold_px']:g} px: the score may be ranking "
            "calibration/decode error rather than mistakes. Check "
            f"{inputs.cameras_from}/cameras.",
            highlight=False,
        )
    pct = coverage.get("score_percentiles", {})
    if pct:
        _info_line(
            "score:      ",
            f"p10 {pct['p10']:.4f}  p50 {pct['p50']:.4f}  p90 {pct['p90']:.4f}  "
            f"p99 {pct['p99']:.4f}  ({coverage['n_distinct_scores']} distinct of "
            f"{inputs.n_frames}; within-recording rank only)",
        )
    if labels:
        _info_line(
            "labels:     ",
            f"{labels['n_gt']} GT point(s), {labels['n_occluded']} occlusion(s) in "
            f"{len(labels['labeled_frames'])} frame(s)",
        )
    else:
        _info_line("labels:     ", "none yet (no labels.h5 beside results.h5)")
    if params.get("absent_point_names"):
        names = ", ".join(params["absent_point_names"])
        _info_line(
            "absent:     ",
            f"{len(params['absent_point_names'])} keypoint(s) not on this animal "
            f"({names}) -- excluded from the score and from every denominator above",
        )
    _info_line(
        "excluded:   ",
        f"{len(excluded)} already-labeled frame(s), and everything within "
        f"{params['min_gap_frames']} frames of them",
    )
    _info_line(
        "spacing:    ",
        f">= {params['min_gap_s']:g} s ({params['min_gap_frames']} frames); at most "
        f"{shortfall['spacing_slots']} picks fit in this recording",
    )
    _info_line(
        "selected:   ",
        f"{shortfall['selected']} of {shortfall['requested']} = "
        f"{shortfall['most_wrong']} most-wrong + {shortfall['diversity']} diversity",
    )
    console.print()

    # soft_wrap: the ranked list is a table, so let long rows run off the edge
    # (as `ls -l` does) instead of rich re-flowing them into unreadable blocks.
    console.print(
        f"{'rank':>4} {'frame':>6} {'t (s)':>8} {'score':>7} {'pct':>5}  "
        f"{'kind':<11} why",
        highlight=False,
        soft_wrap=True,
    )
    for entry in doc["frames"]:
        reason = entry["reason"]
        console.print(
            f"{entry['rank']:>4} {entry['frame']:>6} {entry['t_s']:>8.2f} "
            f"{entry['score']:>7.4f} {entry['percentile']:>5.1f}  "
            f"{entry['kind']:<11} {reason['summary']}",
            highlight=False,
            markup=False,
            soft_wrap=True,
        )
        for d in reason.get("drivers", []):
            console.print(
                f"{'':>35}  {d['point_name']:<18} mean {d['disagreement_px']:>6.1f} px"
                f"  worst {d['worst_px']:>6.1f} px in {d['worst_camera']:<2}"
                f"  {d['views_over_threshold']}/{d['n_observing_views']} views over"
                f" threshold, {d['relation']} side",
                highlight=False,
                markup=False,
                soft_wrap=True,
            )
    if shortfall["reason"]:
        console.print(
            f"\n[bold yellow]NOTE[/bold yellow] only {shortfall['selected']} of "
            f"{shortfall['requested']} requested: {shortfall['reason']}. Lower "
            "--min-gap-s, ask for fewer, or add more recordings.",
            highlight=False,
        )
    console.print()
    span = [e["frame"] for e in doc["frames"]]
    if span:
        _info_line(
            "span:       ",
            f"frames {min(span)}-{max(span)} "
            f"({(max(span) - min(span)) / params['fps']:.1f} s of the recording)",
        )
    saturation = _saturation(inputs.n_frames, excluded + span, params["min_gap_frames"])
    _info_line(
        "saturation: ",
        f"{saturation:.0%} of the recording now lies within "
        f"{params['min_gap_frames']} frames of labeled or suggested work"
        + (
            " -- the next round wants another recording, not a smaller gap"
            if saturation > 0.95
            else ""
        ),
    )


def _saturation(n_frames: int, frames: list[int], min_gap: int) -> float:
    """Fraction of the recording within ``min_gap`` of any frame in ``frames``.

    Surfaces a saturating recording: when this approaches 1.0, the next round's
    answer is another recording, not a smaller gap.
    """
    if n_frames <= 0:
        return 0.0
    covered = np.zeros(n_frames, dtype=bool)
    for t in frames:
        lo, hi = max(0, int(t) - min_gap + 1), min(n_frames, int(t) + min_gap)
        covered[lo:hi] = True
    return float(covered.mean())
