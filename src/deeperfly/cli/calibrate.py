"""``deeperfly calibrate`` -- solve a project's camera rig from its hand labels.

Gathers each recording's ground-truth keypoints and calibration landmarks, checks whether
they can determine a rig at all, solves, and prints a report the operator accepts or
discards. Nothing is made *current* without an explicit ``--accept``: a calibration that
silently replaced a good one would be the most destructive thing this feature could do.

Also serves the **readiness meter** (``--dry-run``, and
:func:`_cmd_calibration_readiness`): the same gate, reported while there is still labeling
to do, with each shortfall phrased as the labeling that would fix it. That is the
difference between a feature that gets used and one that gets abandoned -- an operator
cannot be expected to guess how many frames "enough" is.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from rich.table import Table

from ..calibration import Calibration
from ..calibration_solve import (
    build_observations,
    conditioning,
    initialize_extrinsics,
    merge_observations,
    solve_rig,
)
from ..landmarks import LandmarkSet
from ..project import Project
from .console import _info_line, console

log = logging.getLogger("deeperfly")


def _open(path: str | None) -> Project:
    """The project at ``path``, or the nearest enclosing one."""
    from .project import _open as _open_project

    return _open_project(path)


# -- gathering -----------------------------------------------------------------


def _gather(project: Project, args) -> tuple[list, LandmarkSet, dict]:
    """``(per-recording observations, landmark set, notes)`` for the selected recordings.

    Only frames the operator marked **reviewed** are eligible unless
    ``--include-unreviewed``: a half-labeled frame contributes a systematically biased 3D
    point, and no residual can reveal that after the fact.
    """
    from ..gui.labels import load_labels, load_landmark_labels

    landmark_set = LandmarkSet.load(project.root)
    rig_scoped = {lm.name for lm in landmark_set if lm.shared_across_recordings}
    wanted = args.recordings or [e.slug for e in project.recordings]
    notes: list[str] = []
    per_recording = []

    for key in wanted:
        entry = project.recording(key)
        labels_path = project.labels_path(entry)
        if not labels_path.exists():
            notes.append(f"{entry.slug}: no labels.h5 yet")
            continue
        identity = _identity_for(project, entry)
        if identity is None:
            notes.append(f"{entry.slug}: cannot determine its label identity; skipped")
            continue
        try:
            labels = load_labels(labels_path, identity=identity)
        except ValueError as exc:
            notes.append(f"{entry.slug}: {exc}")
            continue
        if labels is None:
            notes.append(f"{entry.slug}: labels.h5 is empty")
            continue
        n_views = len(identity["camera_names"])
        n_frames = int(identity["n_frames"])
        landmarks = load_landmark_labels(
            labels_path, n_views=n_views, n_frames=n_frames
        )

        frames = None
        if not args.include_unreviewed:
            frames = np.nonzero(labels.reviewed)[0].tolist()
            if not frames:
                notes.append(
                    f"{entry.slug}: no frames marked reviewed "
                    "(pass --include-unreviewed to use partly-labeled frames anyway)"
                )
                continue

        gt_xy = np.where(labels.has_gt[..., None], labels.gt, np.nan)
        per_recording.append(
            build_observations(
                view_names=list(identity["camera_names"]),
                landmark_xy=None if landmarks is None else landmarks.xy,
                landmark_names=None if landmarks is None else list(landmarks.names),
                landmark_static=None if landmarks is None else landmarks.static,
                keypoint_xy=gt_xy,
                keypoint_names=list(identity["point_names"]),
                frames=frames,
                recording=entry.slug,
                use=args.points,
            )
        )
    return per_recording, landmark_set, {"notes": notes, "rig_scoped": rig_scoped}


def _identity_for(project: Project, entry) -> dict | None:
    """The labels identity for a recording, from its results.h5 or its footage.

    A calibrated recording has one recorded in ``results.h5``; an uncalibrated one is
    reconstructed the same way the editor does it, so the two paths agree on which pixel
    space the labels live in.
    """
    import json

    import h5py

    labels_path = project.labels_path(entry)
    # The sidecar carries the identity it was written against -- the most direct source,
    # and the one that cannot disagree with the labels beside it.
    try:
        with h5py.File(labels_path, "r") as f:
            identity = json.loads(f.attrs["meta"]).get("identity")
        if identity and identity.get("camera_names"):
            return identity
    except Exception:
        pass
    return None


# -- intrinsics ----------------------------------------------------------------


def _intrinsics(project: Project, args, obs) -> tuple[np.ndarray, np.ndarray, str]:
    """``(intrs (V,4), dists (V,K), source)`` for the solve.

    Deliberately never *derived from the labels*. Extrinsics are recoverable from
    correspondences; focal length essentially is not, and a bundle adjustment that is
    allowed to trade focal error against depth will report a beautiful residual for a rig
    that is wrong. So the source is required, and it is recorded.

    Raises
    ------
    SystemExit
        If no intrinsics source was given, listing the three ways to supply one.
    """
    n_views = obs.n_views
    if args.from_calibration:
        cal = Calibration.load(args.from_calibration)
        cal.check_camera_names(obs.view_names)
        intrs = np.stack([cal.cameras[n].intr for n in obs.view_names])
        dists = np.stack(
            [
                np.pad(
                    cal.cameras[n].dist,
                    (0, max(0, 5 - cal.cameras[n].dist.size)),
                )[:5]
                for n in obs.view_names
            ]
        )
        return intrs, dists, cal.provenance.get("intrinsics", "imported")

    if args.focal_px:
        focal = float(args.focal_px)
        source = "given"
    elif args.lens_mm and args.sensor_mm:
        # f_px = f_mm * W_px / W_mm -- two numbers off a datasheet.
        width = _widest(project, obs)
        focal = float(args.lens_mm) * width / float(args.sensor_mm)
        source = "optics"
        log.info(
            "focal from optics: %.4g mm lens / %.4g mm sensor over %d px = %.1f px",
            args.lens_mm,
            args.sensor_mm,
            width,
            focal,
        )
    else:
        raise SystemExit(
            "no intrinsics source. Extrinsics can be recovered from labels; focal length "
            "cannot, and a solve allowed to guess it will report a small residual for a "
            "wrong rig. Supply one of:\n"
            "  --from-calibration PATH   reuse a board or previous solve (best)\n"
            "  --lens-mm F --sensor-mm W  compute it from the datasheet\n"
            "  --focal-px F               state it directly"
        )

    sizes = _image_sizes(project, obs)
    intrs = np.stack(
        [
            np.array(
                [focal, focal, (sizes[n][1] - 1) / 2, (sizes[n][0] - 1) / 2],
                dtype=float,
            )
            for n in obs.view_names
        ]
    )
    return intrs, np.zeros((n_views, 5)), source


def _image_sizes(project: Project, obs) -> dict:
    """``view -> (h, w)`` from any selected recording's label identity.

    Needed for the principal point, which defaults to the image center. A view whose size
    is unknown is refused rather than guessed: a wrong principal point biases every
    extrinsic.
    """
    import json

    import h5py

    for entry in project.recordings:
        try:
            with h5py.File(project.labels_path(entry), "r") as f:
                sizes = json.loads(f.attrs["meta"])["identity"].get("image_sizes") or {}
            if all(name in sizes for name in obs.view_names):
                return {n: tuple(sizes[n]) for n in obs.view_names}
        except Exception:
            continue
    raise SystemExit(
        "cannot determine the footage size for every view, so the principal point would "
        "have to be guessed -- and a wrong principal point biases every extrinsic. Open "
        "the recording in 'deeperfly gui' once (which records the sizes), or pass "
        "--from-calibration"
    )


def _widest(project: Project, obs) -> int:
    sizes = _image_sizes(project, obs)
    return max(int(hw[1]) for hw in sizes.values())


# -- readiness -----------------------------------------------------------------


def _readiness(obs, cond: dict, notes: dict, *, scale_known: bool) -> list[tuple]:
    """``(ok, label, value, hint)`` rows -- the meter, phrased as what to label next."""
    summary = obs.summary()
    rows: list[tuple] = []
    blind = [n for n, c in cond["per_view_tracks"].items() if c == 0]
    rows.append(
        (
            not blind,
            "views with labels",
            f"{sum(1 for c in cond['per_view_tracks'].values() if c)} / {summary['views']}",
            f"label something in {', '.join(blind)}" if blind else "",
        )
    )
    connected = len(cond["components"]) <= 1
    rows.append(
        (
            connected,
            "co-visibility",
            "connected"
            if connected
            else " vs ".join("{" + ", ".join(c) + "}" for c in cond["components"]),
            ""
            if connected
            else "label the same point in a view from each group -- a static "
            "landmark both can see is ideal",
        )
    )
    rows.append(
        (
            summary["static_tracks"] > 0,
            "static landmarks",
            str(summary["static_tracks"]),
            ""
            if summary["static_tracks"]
            else "one static point (a coverslip scratch, the tether tip) is worth more "
            "than many keypoint frames -- it is 3 unknowns, not 3 per frame",
        )
    )
    rows.append((summary["frames"] > 0, "labeled frames", str(summary["frames"]), ""))
    ratio_ok = cond["ratio"] >= 1.5
    rows.append(
        (
            ratio_ok,
            "observations / unknowns",
            f"{cond['ratio']:.2f}x",
            "" if ratio_ok else "label more frames, or add a static landmark",
        )
    )
    weakest = cond["weakest_pair"]
    if weakest:
        rows.append(
            (
                weakest["shared"] >= 8,
                "weakest view pair",
                f"{'+'.join(weakest['views'])}  {weakest['shared']} shared",
                ""
                if weakest["shared"] >= 8
                else f"label a few more shared points in {' and '.join(weakest['views'])}",
            )
        )
    rows.append(
        (
            scale_known,
            "scale reference",
            "known distance" if scale_known else "none",
            ""
            if scale_known
            else "without one the rig is valid up to scale: angles yes, lengths no "
            "(--scale-from A,B=1.8)",
        )
    )
    return rows


def _print_readiness(rows: list[tuple], cond: dict) -> None:
    table = Table(title="Calibration readiness")
    table.add_column("", width=2)
    table.add_column("check", style="bold")
    table.add_column("value")
    table.add_column("what would help")
    for ok, label, value, hint in rows:
        table.add_row(
            "[green]OK[/green]" if ok else "[yellow]![/yellow]", label, value, hint
        )
    console.print(table)
    if not cond["ok"]:
        console.print("[yellow]not ready to solve[/yellow]", highlight=False)


# -- the command ---------------------------------------------------------------


def _cmd_calibration_readiness(args: argparse.Namespace) -> None:
    """Report how close a project is to a solvable rig (``deeperfly calibrate --dry-run``)."""
    args.dry_run = True
    _cmd_calibrate(args)


def _cmd_calibrate(args: argparse.Namespace) -> None:
    """Solve a project's rig from its labels."""
    project = _open(args.project)
    per_recording, landmark_set, notes = _gather(project, args)
    for note in notes["notes"]:
        log.warning("%s", note)
    if not per_recording:
        raise SystemExit(
            "nothing to calibrate from: no selected recording has usable labels. Label "
            "some frames in 'deeperfly gui' and mark them reviewed"
        )

    obs = merge_observations(per_recording, share=notes["rig_scoped"])
    if not obs.n_tracks:
        raise SystemExit(
            "no track is observed by two or more views, so nothing can be triangulated -- "
            "label the same points in at least two cameras"
        )

    scale_pair, scale_distance = _resolve_scale(args, obs)
    cond = conditioning(obs, free_focal=args.free_focal, free_k1=args.free_k1)
    _info_line("project:  ", f"{project.name}  ({project.root})")
    _info_line(
        "using:    ",
        f"{args.points}  ({obs.summary()['landmark_tracks']} landmark + "
        f"{obs.summary()['keypoint_tracks']} keypoint tracks, "
        f"{obs.n_observations} observations)",
    )
    _print_readiness(
        _readiness(obs, cond, notes, scale_known=scale_distance is not None), cond
    )
    if args.dry_run:
        return
    if not cond["ok"]:
        for reason in cond["reasons"]:
            console.print(f"[red]refusing:[/red] {reason}", highlight=False)
        raise SystemExit(
            "refusing to solve an under-determined rig: it would converge to something "
            "plausible-looking and then misproject every point downstream"
        )

    intrs, dists, intr_source = _intrinsics(project, args, obs)
    cold = not args.from_calibration
    if cold:
        console.print("initializing from scratch (essential matrix + PnP)...")
        rvecs, tvecs, init = initialize_extrinsics(obs, intrs, dists)
        if init.get("failed"):
            raise SystemExit(
                f"could not place view(s) {init['failed']} from these labels. Label more "
                "points shared with an already-placed view"
            )
        console.print(
            f"  seed pair {init['seed_pair']}; "
            + ", ".join(
                f"{r['view']}({r['inliers']}/{r['shared']})"
                for r in init["registration"]
            ),
            highlight=False,
        )
    else:
        prior = Calibration.load(args.from_calibration)
        rvecs = np.stack([prior.cameras[n].rvec for n in obs.view_names])
        tvecs = np.stack([prior.cameras[n].tvec for n in obs.view_names])
        init = {"seed_pair": None, "from": str(args.from_calibration)}

    result = solve_rig(
        obs,
        intrinsics=intrs,
        dists=dists,
        rvecs=rvecs,
        tvecs=tvecs,
        free_focal=args.free_focal,
        free_k1=args.free_k1,
        scale_pair=scale_pair,
        scale_distance=scale_distance,
        cold_start=cold,
        loss=args.loss,
        f_scale=args.f_scale,
    )
    _report(result, obs, cond)

    name = args.name or "from-labels"
    calibration = Calibration.from_camera_group(
        result.cameras,
        name=name,
        image_sizes=_image_sizes(project, obs),
        units="mm" if scale_distance else "arbitrary",
        scale_source="known_distance" if scale_distance else "none",
        provenance={
            "method": "labels_ba",
            "intrinsics": intr_source,
            "points": args.points,
            "recordings": sorted(
                {t.recording for t in obs.tracks if t.recording} or {"?"}
            ),
            "frames": obs.summary()["frames"],
            "reviewed_only": not args.include_unreviewed,
            "init": {k: str(v) for k, v in init.items() if k in ("seed_pair", "from")},
            "solver": result.report["solver"],
        },
        quality=result.quality,
    )
    path = project.root / "calibrations" / f"{name}.toml"
    calibration.save(path)
    report_path = path.with_suffix(".report.json")
    _write_report(report_path, result, cond, obs)
    console.print(f"[green]wrote[/green] {path}")
    console.print(f"[green]wrote[/green] {report_path}")

    if args.accept:
        project.calibration = str(path.relative_to(project.root))
        project.save()
        console.print(
            f"[green]accepted[/green] -- {project.name} now uses {calibration.name}; "
            "the editor will show 3D on its next open"
        )
    else:
        console.print(
            "not accepted. Review the residuals above, then re-run with --accept "
            f"(or point [project.calibration] at calibrations/{name}.toml by hand)",
            markup=False,  # the message names a TOML key; rich would eat the brackets
            highlight=False,
        )


def _resolve_scale(args, obs):
    """``(track pair, distance)`` from ``--scale-from A,B=1.8``, or ``(None, None)``.

    Raises
    ------
    SystemExit
        If the spec is malformed or names a track that is not in the solve.
    """
    if not args.scale_from:
        return None, None
    spec = str(args.scale_from)
    try:
        pair, distance = spec.split("=")
        a, b = (s.strip() for s in pair.split(","))
        value = float(distance)
    except ValueError:
        raise SystemExit(
            f"could not read --scale-from {spec!r}; expected 'LANDMARK_A,LANDMARK_B=1.8'"
        ) from None
    labels = [t.label for t in obs.tracks]
    try:
        return (labels.index(a), labels.index(b)), value
    except ValueError:
        raise SystemExit(
            f"--scale-from names {a!r} and {b!r}, but the solve's tracks are "
            f"{labels[:12]}{' ...' if len(labels) > 12 else ''}. Use landmark names "
            "(static landmarks keep their plain name; a per-frame track is 'name@frame')"
        ) from None


def _report(result, obs, cond) -> None:
    """Print the residual summary an operator accepts or discards on."""
    q = result.quality
    table = Table(title="Solved rig")
    table.add_column("camera", style="bold")
    table.add_column("rms (px)", justify="right")
    for name in obs.view_names:
        table.add_row(name, f"{q['per_camera_rms_px'].get(name, float('nan')):.3f}")
    console.print(table)
    _info_line(
        "overall:  ",
        f"rms {q['rms_reproj_px']:.3f} px   median {q['median_reproj_px']:.3f}   "
        f"p90 {q['p90_reproj_px']:.3f}   max {q['max_reproj_px']:.3f}",
    )
    scatter = result.report.get("static_scatter_px") or {}
    drifting = {k: v for k, v in scatter.items() if v > 2.0}
    if drifting:
        console.print(
            "[yellow]static landmark(s) whose pixel wanders[/yellow] "
            + ", ".join(f"{k} ({v:.1f} px)" for k, v in sorted(drifting.items()))
            + " -- either it is not actually static, or it was labelled on a different "
            "feature in different frames. Both corrupt the solve.",
            highlight=False,
        )
    worst = result.report["per_track"][:5]
    if worst:
        console.print(
            "worst tracks: "
            + ", ".join(f"{r['label']} {r['max_reproj_px']:.1f}px" for r in worst),
            highlight=False,
        )
    if not result.ok:
        console.print(
            "[yellow]the solver did not report convergence[/yellow] -- treat these "
            "numbers as provisional",
            highlight=False,
        )


def _write_report(path: Path, result, cond, obs) -> None:
    """Write the full machine-readable report beside the calibration."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "conditioning": cond,
                "quality": result.quality,
                "solver": result.report["solver"],
                "observations": result.report["observations"],
                "per_track": result.report["per_track"],
                "static_scatter_px": result.report["static_scatter_px"],
                "view_names": obs.view_names,
            },
            indent=2,
            default=str,
        )
    )
