"""Bundle adjustment driven from the editor, on the operator's own labels.

The pipeline's bundle adjustment fits the rig to the *detector's* 2D. This module fits it to
the **ground truth the operator placed**, from inside the editing session, and writes the
result as a NEW calibration that the session can then be switched onto. That closes a loop
the CLI could only run in two halves:

.. code-block:: text

    label a few frames  ->  solve the rig from those labels  ->  the editor's derived 3D
    (2D canvases)           (this module)                        now reprojects into every
                                                                 view, so one drag moves all

Two regimes, and the difference is not cosmetic:

**Refine.** A rig already exists and some of it is trusted. The motivating case is a camera
that was never calibrated at all -- a hind view seeded from an orbit prior -- where the seven
good cameras are held FIXED and the new one is left free. Six unknowns against hundreds of
correspondences, initialized from the prior: very well conditioned. Holding the good cameras
also means a handful of labels cannot damage a rig that took a whole recording to fit.

**Cold start.** No calibration exists (a fresh project: every view is an independent 2D
canvas). Bundle adjustment cannot start from nothing, so the extrinsics must be *initialized*
first -- :func:`deeperfly.rig.solve.initialize_extrinsics`, which uses an orbit prior
when the config declares one and otherwise runs incremental SfM (essential matrix + PnP). Here
the gauge is free and so is scale: without a known distance the solved rig is correct up to
scale, angles are meaningful and lengths are not, and the calibration records
``units = "arbitrary"`` so nothing downstream can pretend otherwise.

**What is deliberately not offered.** Intrinsics stay fixed by default. Focal length trades
against depth, so a solve permitted to adjust it reports a *smaller* residual for a *worse*
rig -- the one failure mode a residual cannot reveal. The tab can free them, and says this
when you do.

**The gauge is checked, not assumed.** If no camera has both its rotation and its translation
held fixed, the problem has six (or seven, with scale) free directions along which the cost is
flat. The optimizer will happily drift down them and report a fine residual for a rig whose
world frame has moved out from under every 3D point in the file. :func:`check_gauge` refuses
that case rather than silently pinning a camera the operator did not choose.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("deeperfly")

#: The per-camera parameter groups the tab exposes as fix/free. ``build_state`` also accepts
#: element references (``"rm.tvec[2]"``); those are not expressible as a checkbox and ride
#: along in :attr:`BaSettings.extra_fixed` verbatim.
PARAMS: tuple[str, ...] = ("rvec", "tvec", "intr", "dist")

#: Losses ``scipy.optimize.least_squares`` accepts, as offered in the tab.
LOSSES: tuple[str, ...] = ("linear", "soft_l1", "huber", "cauchy", "arctan")

#: How frames are subsampled when there are more than ``max_frames``.
SAMPLINGS: tuple[str, ...] = ("even", "confidence", "coverage", "diversity")

#: A track needs at least this many views to have a determined 3D position. A track seen once
#: contributes three free parameters and no information, so it is dropped and counted.
MIN_VIEWS_PER_TRACK = 2

_REF = re.compile(
    r"^(?P<cam>\*|[A-Za-z0-9_.-]+)\.(?P<param>rvec|tvec|intr|dist|kmat)(?P<idx>\[\d+\])?$"
)


# -- the fix/free matrix ------------------------------------------------------------------


def expand_fixed(refs, camera_names) -> dict[str, set[str]]:
    """``["*.intr", "rh.rvec"]`` -> ``{camera: {param, ...}}``.

    Element references (``"rm.tvec[2]"``) are NOT expanded into the matrix: a checkbox cannot
    say "only the z component", and quietly widening it to the whole vector would fix more
    than the config asked. They are preserved separately by :func:`split_refs`.
    """
    names = list(camera_names)
    out: dict[str, set[str]] = {n: set() for n in names}
    for ref in refs or ():
        m = _REF.match(str(ref).strip())
        if not m or m.group("idx"):
            continue
        param = "intr" if m.group("param") == "kmat" else m.group("param")
        cam = m.group("cam")
        for n in names if cam == "*" else ([cam] if cam in out else []):
            out[n].add(param)
    return out


def split_refs(refs, camera_names) -> tuple[dict[str, set[str]], list[str]]:
    """``(matrix, leftovers)`` -- the checkbox-expressible refs and everything else."""
    names = set(camera_names)
    leftovers = []
    for ref in refs or ():
        text = str(ref).strip()
        m = _REF.match(text)
        if (
            not m
            or m.group("idx")
            or (m.group("cam") != "*" and m.group("cam") not in names)
        ):
            leftovers.append(text)
    return expand_fixed(refs, camera_names), leftovers


def collapse_fixed(matrix: dict[str, Any], camera_names) -> list[str]:
    """``{camera: [param, ...]}`` -> the shortest reference list, ``"*.p"`` where universal.

    Round-trips with :func:`expand_fixed`, and writes the same shorthand the shipped configs
    use, so a plan that came from the config and went back unchanged reads unchanged.
    """
    names = list(camera_names)
    refs: list[str] = []
    per = {n: set(matrix.get(n) or ()) for n in names}
    for param in PARAMS:
        holders = [n for n in names if param in per[n]]
        if names and len(holders) == len(names):
            refs.append(f"*.{param}")
        else:
            refs.extend(f"{n}.{param}" for n in holders)
    return refs


def check_gauge(matrix: dict[str, Any], camera_names) -> list[str]:
    """Problems that make the solve ill-posed, worst first (empty = go ahead).

    The world frame is defined only by whatever is held still. With every camera free, the
    whole rig can rotate, translate and (with intrinsics free) rescale at no cost to the
    residual, so the fit "succeeds" while every 3D point in the file moves.
    """
    names = list(camera_names)
    per = {n: set(matrix.get(n) or ()) for n in names}
    anchored = [n for n in names if {"rvec", "tvec"} <= per[n]]
    free = [n for n in names if not ({"rvec", "tvec"} & per[n])]
    problems = []
    if not anchored:
        problems.append(
            "no camera has BOTH its rotation and position fixed, so the world frame is free: "
            "the rig can rotate and translate at no cost to the residual and every 3D point "
            "moves with it. Fix one camera's rvec and tvec (any one -- it only sets the frame)"
        )
    if not free:
        problems.append("every camera is fully fixed, so there is nothing to solve for")
    return problems


# -- settings -----------------------------------------------------------------------------


@dataclass
class BaSettings:
    """Everything the solve needs beyond the observations, as the tab presents it."""

    fixed: dict[str, list[str]] = field(default_factory=dict)
    extra_fixed: list[str] = field(default_factory=list)
    shared: list[list[str]] = field(default_factory=list)
    loss: str = "cauchy"
    f_scale: float = 4.0
    max_nfev: int = 2000
    max_frames: int | None = 100
    frame_sampling: str = "even"
    weigh_by_confidence: bool = False
    points_to_use: list[str] | None = None
    #: free the focal length. Off, and the tab warns when it is turned on.
    free_focal: bool = False

    def refs(self, camera_names) -> list[str]:
        """The full ``fixed`` list handed to the solver."""
        refs = collapse_fixed(self.fixed, camera_names)
        refs.extend(self.extra_fixed)
        if not self.free_focal and not any(r == "*.intr" for r in refs):
            # Fixed per camera above is equivalent, but a config that only listed some
            # cameras' intrinsics would otherwise leave the rest free by accident.
            missing = [
                n for n in camera_names if "intr" not in set(self.fixed.get(n) or ())
            ]
            refs.extend(f"{n}.intr" for n in missing)
        return refs

    def to_json(self) -> dict:
        return {
            "fixed": {k: sorted(v) for k, v in self.fixed.items()},
            "extra_fixed": list(self.extra_fixed),
            "shared": [list(g) for g in self.shared],
            "loss": self.loss,
            "f_scale": self.f_scale,
            "max_nfev": self.max_nfev,
            "max_frames": self.max_frames,
            "frame_sampling": self.frame_sampling,
            "weigh_by_confidence": self.weigh_by_confidence,
            "points_to_use": self.points_to_use,
            "free_focal": self.free_focal,
        }

    @classmethod
    def from_json(cls, data: dict, camera_names) -> BaSettings:
        """Build from the tab's payload, validating every choice loudly."""
        data = dict(data or {})
        loss = str(data.get("loss", "cauchy"))
        if loss not in LOSSES:
            raise ValueError(f"unknown loss {loss!r}; choose from {list(LOSSES)}")
        sampling = str(data.get("frame_sampling", "even"))
        if sampling not in SAMPLINGS:
            raise ValueError(
                f"unknown frame_sampling {sampling!r}; choose from {list(SAMPLINGS)}"
            )
        raw_fixed = data.get("fixed") or {}
        fixed: dict[str, list[str]] = {}
        for cam in camera_names:
            want = raw_fixed.get(cam) or []
            bad = [p for p in want if p not in PARAMS]
            if bad:
                raise ValueError(
                    f"camera {cam!r}: unknown parameter(s) {bad}; expected {list(PARAMS)}"
                )
            fixed[cam] = [p for p in PARAMS if p in want]
        mf = data.get("max_frames", 100)
        return cls(
            fixed=fixed,
            extra_fixed=[str(r) for r in (data.get("extra_fixed") or [])],
            shared=[[str(x) for x in g] for g in (data.get("shared") or [])],
            loss=loss,
            f_scale=float(data.get("f_scale", 4.0)),
            max_nfev=int(data.get("max_nfev", 2000)),
            max_frames=None if mf in (None, "", 0) else int(mf),
            frame_sampling=sampling,
            weigh_by_confidence=bool(data.get("weigh_by_confidence", False)),
            points_to_use=(
                None
                if data.get("points_to_use") in (None, "")
                else [str(p) for p in data["points_to_use"]]
            ),
            free_focal=bool(data.get("free_focal", False)),
        )


def settings_from_config(
    config, camera_names, *, has_rig: bool
) -> tuple[BaSettings, str | None]:
    """The tab's defaults, read from the recording/project config's ``[bundle_adjustment]``.

    With a rig present the config's own ``fixed`` list is the default; with none, everything
    is free except the first camera's pose (which sets the world frame).

    Returns the settings and, when this function had to add something the config did not say,
    a note explaining what and why -- so the tab never silently differs from the config.
    """
    names = list(camera_names)
    ba = getattr(config, "bundle_adjustment", None) if config is not None else None
    if ba is None:
        matrix = {n: [] for n in names}
        leftovers: list[str] = []
        ls: dict = {}
        shared: list[list[str]] = []
        points_to_use = None
        weigh = False
        max_frames: int | None = 100
        sampling = "even"
    else:
        expanded, leftovers = split_refs(list(ba.fixed), names)
        matrix = {n: [p for p in PARAMS if p in expanded[n]] for n in names}
        ls = dict(ba.least_squares or {})
        shared = [list(g) for g in (ba.shared or [])]
        points_to_use = None if ba.points_to_use is None else list(ba.points_to_use)
        weigh = bool(ba.weigh_by_confidence)
        max_frames = ba.max_frames
        sampling = str(ba.frame_sampling)

    note: str | None = None
    if not has_rig:
        # Cold start: nothing is trusted, so nothing is fixed but the gauge.
        matrix = {n: [] for n in names}
        if names:
            matrix[names[0]] = ["rvec", "tvec"]
            note = (
                f"no rig yet, so every camera is free and {names[0]} holds the world frame. "
                "Without a known distance the solved rig is correct only up to scale"
            )
    elif names and not any({"rvec", "tvec"} <= set(matrix[n]) for n in names):
        # The shipped configs fix only "*.intr" and leave the gauge to the optimizer, which a
        # pipeline run can afford (it starts from the orbit prior and re-solves everything).
        # Here the operator is refining a rig that already means something, so opening on a
        # refusal would be useless: anchor the first camera and say that is what happened.
        matrix[names[0]] = sorted(set(matrix[names[0]]) | {"rvec", "tvec"})
        note = (
            f"the config's fixed list does not hold any camera still, so {names[0]}'s pose is "
            "anchored here to define the world frame -- otherwise the whole rig could drift "
            "at no cost to the residual. Change it if you want a different camera anchored"
        )
    settings = BaSettings(
        fixed=matrix,
        extra_fixed=leftovers,
        shared=shared,
        loss=str(ls.get("loss", "cauchy")),
        f_scale=float(ls.get("f_scale", 4.0)),
        max_nfev=int(ls.get("max_nfev", 2000)),
        max_frames=max_frames,
        frame_sampling=sampling,
        weigh_by_confidence=weigh,
        points_to_use=points_to_use,
    )
    return settings, note


# -- observations from the operator's labels ----------------------------------------------


@dataclass
class Observations:
    """GT pixels arranged as bundle adjustment wants them: one 3D unknown per track.

    A skeleton keypoint at frame *t* is its own track, because the animal moved between
    frames. ``pts2d`` is ``(V, N, 2)`` with NaN where a view has no label for that track.
    """

    pts2d: np.ndarray
    view_names: list[str]
    #: ``(frame, point_index)`` per track, so a residual can be attributed back.
    tracks: list[tuple[int, int]]
    #: per-view count of labeled cells actually used
    per_view: dict[str, int]
    n_frames: int
    dropped_single_view: int
    notes: list[str] = field(default_factory=list)

    @property
    def n_tracks(self) -> int:
        return len(self.tracks)


def gather(
    state,
    *,
    max_frames: int | None = None,
    frame_sampling: str = "even",
    point_indices=None,
) -> Observations:
    """Collect the operator's labels into an :class:`Observations`.

    Only labels are used -- never the detector's 2D. That is the point: the rig this produces
    is traceable to pixels a human placed, and a detector bias cannot leak into it.
    """
    labels = state.labels
    names = list(state.camera_names)
    gt = np.asarray(labels.gt, dtype=float)  # (V, T, P, 2)
    has = np.asarray(labels.has_gt, dtype=bool)  # (V, T, P)
    notes: list[str] = []

    keep_points = np.ones(gt.shape[2], dtype=bool)
    if point_indices is not None:
        keep_points[:] = False
        keep_points[list(point_indices)] = True
        has = has & keep_points[None, None, :]

    # Frames that carry any label at all; a frame with none contributes nothing.
    per_frame = has.any(axis=(0, 2))
    frames = np.flatnonzero(per_frame)
    if max_frames is not None and len(frames) > max_frames:
        if frame_sampling == "even":
            pick = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        else:
            # coverage: the frames with the most multi-view cells are the best conditioned.
            score = (has[:, frames].sum(axis=0) >= MIN_VIEWS_PER_TRACK).sum(axis=1)
            pick = np.argsort(-score)[:max_frames]
            pick.sort()
        notes.append(
            f"using {max_frames} of {len(frames)} labeled frames ({frame_sampling}); raise "
            "max_frames to use them all"
        )
        frames = frames[pick]

    tracks: list[tuple[int, int]] = []
    cols: list[np.ndarray] = []
    dropped = 0
    for t in frames:
        seen = has[:, t]  # (V, P)
        counts = seen.sum(axis=0)
        for p in np.flatnonzero(counts > 0):
            if counts[p] < MIN_VIEWS_PER_TRACK:
                dropped += 1
                continue
            col = np.where(seen[:, p][:, None], gt[:, t, p], np.nan)
            tracks.append((int(t), int(p)))
            cols.append(col)

    pts2d = (
        np.stack(cols, axis=1)
        if cols
        else np.full((len(names), 0, 2), np.nan, dtype=float)
    )
    used = np.isfinite(pts2d).all(axis=-1)
    per_view = {n: int(used[i].sum()) for i, n in enumerate(names)}
    if dropped:
        notes.append(
            f"{dropped} labeled cell(s) are in only one view, so their 3D is undetermined and "
            "they cannot constrain a camera; they are not used"
        )
    return Observations(
        pts2d=pts2d,
        view_names=names,
        tracks=tracks,
        per_view=per_view,
        n_frames=int(len(frames)),
        dropped_single_view=dropped,
        notes=notes,
    )


def covisibility(obs: Observations) -> np.ndarray:
    """``(V, V)`` count of tracks seen by both views -- the graph the solve rests on."""
    seen = np.isfinite(obs.pts2d).all(axis=-1)  # (V, N)
    return seen.astype(np.int64) @ seen.astype(np.int64).T


def preflight(obs: Observations, settings: BaSettings, camera_names) -> dict:
    """What is known BEFORE solving: coverage, the co-visibility graph, and refusals.

    Reported so the operator is never shown a beautiful residual for a rig the labels could
    not have determined. A camera that must be solved for needs its own observations, and it
    needs them shared with cameras that are already placed.
    """
    names = list(camera_names)
    per = {n: set(settings.fixed.get(n) or ()) for n in names}
    free = [n for n in names if not ({"rvec", "tvec"} <= per[n])]
    co = covisibility(obs)
    problems = list(check_gauge(settings.fixed, names))
    warnings: list[str] = list(obs.notes)

    for n in free:
        i = names.index(n)
        if obs.per_view.get(n, 0) == 0:
            problems.append(
                f"{n} is free to move but has no ground truth at all, so nothing determines "
                f"its pose. Label points in {n}, or fix it"
            )
            continue
        shared = int(co[i].sum() - co[i, i])
        if shared == 0:
            problems.append(
                f"{n}'s labels share no track with any other view, so its pose cannot be "
                f"tied to the rig. Label, in {n}, points that are also labeled elsewhere"
            )
        elif shared < 12:
            warnings.append(
                f"{n} shares only {shared} track(s) with the rest of the rig; 6 unknowns "
                "against that few observations will be poorly determined"
            )
    if obs.n_tracks == 0:
        problems.insert(
            0, "no track is labeled in two or more views, so there is nothing to fit"
        )
    if settings.free_focal:
        warnings.append(
            "focal length is free: it trades against depth, so the residual can improve while "
            "the rig gets worse. Prefer fixing it unless you have many well-spread labels"
        )
    return {
        "n_tracks": obs.n_tracks,
        "n_frames": obs.n_frames,
        "per_view": obs.per_view,
        "free": free,
        "fixed_cameras": [n for n in names if n not in free],
        "covisibility": {
            names[i]: {names[j]: int(co[i, j]) for j in range(len(names)) if j != i}
            for i in range(len(names))
        },
        "problems": problems,
        "warnings": warnings,
        "ok": not problems,
    }


# -- the solve ----------------------------------------------------------------------------


def _residuals(cameras, pts2d, pts3d) -> dict:
    """Per-view and overall reprojection error, in pixels."""
    from ..rig.triangulation import reprojection_error

    err = np.asarray(reprojection_error(cameras, pts3d, pts2d), dtype=float)
    out: dict[str, Any] = {}
    with np.errstate(invalid="ignore"):
        for i, name in enumerate(cameras.names):
            v = err[i][np.isfinite(err[i])]
            out[name] = {
                "n": int(v.size),
                "median": float(np.median(v)) if v.size else None,
                "mean": float(v.mean()) if v.size else None,
                "p90": float(np.percentile(v, 90)) if v.size else None,
            }
        flat = err[np.isfinite(err)]
    return {
        "per_view": out,
        "median": float(np.median(flat)) if flat.size else None,
        "mean": float(flat.mean()) if flat.size else None,
        "p90": float(np.percentile(flat, 90)) if flat.size else None,
        "n": int(flat.size),
    }


def solve(
    cameras,
    obs: Observations,
    settings: BaSettings,
    *,
    cold_start: bool = False,
    intrinsics=None,
    dists=None,
) -> dict:
    """Bundle-adjust ``cameras`` against ``obs``, returning the new rig and a report.

    Reuses :func:`deeperfly.rig.bundle_adjustment.bundle_adjust` unchanged -- the same solver the
    pipeline runs. The only thing different here is that the observations are hand labels and
    the fix/free split came from the operator.
    """
    from ..rig.bundle_adjustment import bundle_adjust
    from ..rig.cameras import CameraGroup

    names = list(obs.view_names)
    problems = check_gauge(settings.fixed, names)
    if problems:
        raise ValueError(problems[0])
    if obs.n_tracks == 0:
        raise ValueError(
            "no track is labeled in two or more views, so there is nothing to fit"
        )

    if cold_start:
        from ..rig.solve import initialize_extrinsics

        intrs = np.asarray(intrinsics, dtype=float)
        dsts = np.asarray(dists, dtype=float)
        rvecs, tvecs, init = initialize_extrinsics(obs, intrs, dsts)
        if init.get("failed"):
            raise ValueError(
                f"could not place view(s) {init['failed']} from these labels -- label points "
                "they share with a view that is already placed"
            )
        start = CameraGroup.from_arrays(names, rvecs, tvecs, intrs, dsts)
        init_note = init
    else:
        start = cameras
        init_note = {"from": "the session's current rig"}

    before = _residuals(
        start, obs.pts2d, np.asarray(start.triangulate(obs.pts2d), dtype=float)
    )
    refs = settings.refs(names)
    result, solved, pts3d = bundle_adjust(
        start,
        obs.pts2d,
        fixed=refs,
        shared=[list(g) for g in settings.shared],
        loss=settings.loss,
        f_scale=settings.f_scale,
        max_nfev=settings.max_nfev,
    )
    after = _residuals(solved, obs.pts2d, pts3d)

    moved = {}
    for i, name in enumerate(names):
        d_r = float(
            np.linalg.norm(np.asarray(solved.rvecs[i]) - np.asarray(start.rvecs[i]))
        )
        d_t = float(
            np.linalg.norm(np.asarray(solved.tvecs[i]) - np.asarray(start.tvecs[i]))
        )
        moved[name] = {"rvec": d_r, "tvec": d_t}
    return {
        "cameras": solved,
        "pts3d": pts3d,
        "before": before,
        "after": after,
        "moved": moved,
        "fixed_refs": refs,
        "init": init_note,
        "success": bool(getattr(result, "success", True)),
        "nfev": int(getattr(result, "nfev", 0) or 0),
        "cost": float(getattr(result, "cost", float("nan"))),
        "message": str(getattr(result, "message", "") or ""),
    }


# -- persistence --------------------------------------------------------------------------


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text).strip()).strip("-")
    return s or "calibration"


def recording_dir(project_root: Path, recording: str | None) -> Path:
    """Where a recording's calibrations live: ``<project>/calibrations/<recording>/``.

    Scoped per recording because a rig belongs to one: the cameras are re-aimed between
    sessions, and a calibration solved on Fly3 says nothing about Fly4. A flat project-wide
    directory listed every recording's rigs side by side with no way to tell which applied
    here, which is exactly how you pick the wrong one.
    """
    base = Path(project_root) / "calibrations"
    return base / _slug(recording) if recording else base


def stored_meta(results_path: Path | str) -> dict[str, Any]:
    """``results.h5``'s ``meta`` mapping, or ``{}`` if it cannot be read.

    Best effort by design: an unreadable or absent file must degrade to "no recorded
    decisions", never break the caller. The two decisions read out of it here --
    ``provisional_cameras`` and ``active_calibration`` -- are facts about ONE recording that
    have to survive a restart, which is why they live in the file and not in the session.
    """
    import json

    import h5py

    try:
        with h5py.File(Path(results_path), "r") as f:
            return dict(json.loads(f.attrs.get("meta", "{}")))
    except Exception:  # noqa: BLE001 -- a missing/locked results.h5 is not an error here
        return {}


def base_provisional(results_path: Path | str, camera_names) -> list[str]:
    """Views the rig in ``results.h5`` never calibrated -- the un-promoted truth.

    Read from the file rather than from a live state, because selecting a calibration that
    solves a view narrows the EFFECTIVE set and must not destroy the record: going back to
    the pipeline rig has to make that view provisional again, and nothing else remembers
    that it was.
    """
    names = set(str(n) for n in camera_names)
    stored = stored_meta(results_path).get("provisional_cameras") or ()
    return [str(v) for v in stored if str(v) in names]


def still_provisional(path: Path, current) -> list[str]:
    """Which of ``current`` this calibration did NOT solve, so they stay display-only.

    A calibration lists every camera in the rig, so merely appearing in one proves
    nothing. What proves it is the ``fixed`` list this editor records in the provenance:
    a camera whose pose was held fixed was carried over, not solved. A calibration from
    anywhere else (a board solve, an import) carries no such list -- then the honest
    reading is that it IS a calibration for these cameras, so nothing stays provisional.
    """
    from ..rig.calibration import Calibration

    want = [str(c) for c in (current or ())]
    if not want:
        return []
    try:
        prov = dict(Calibration.load(path).provenance or {})
    except Exception:  # noqa: BLE001
        return want  # unreadable provenance: change nothing
    if "gui bundle adjustment" not in str(prov.get("solved_by", "")):
        return []
    refs = {str(r) for r in (prov.get("fixed") or ())}

    def held(cam: str) -> bool:
        return bool({f"{cam}.rvec", "*.rvec"} & refs) and bool(
            {f"{cam}.tvec", "*.tvec"} & refs
        )

    return [cam for cam in want if held(cam)]


def apply_active_calibration(
    state,
    results_path: Path | str,
    project_root: Path | str | None,
    recording_slug: str | None,
) -> Path | None:
    """Switch ``state`` onto the rig this recording was last selected onto; return its path.

    ``None`` when there is nothing to apply (no project, no recorded choice, or the file it
    names is gone) -- and then the rig in ``results.h5`` stands, which is the correct
    fallback rather than an error.

    This is module-level, and not a closure inside ``create_app``, because the editor's
    derived 3D is only reproducible OUTSIDE the editor if the rig selection is. Anything
    that recomputes what the operator saw -- a training-target export, an audit, a
    regression check -- has to make exactly this choice, and a second copy of the promotion
    rule would drift from the one the GUI actually uses.
    """
    from ..rig.cameras import CameraGroup

    name = stored_meta(results_path).get("active_calibration")
    if not name or project_root is None:
        return None
    path = recording_dir(Path(project_root), recording_slug) / str(name)
    if not path.is_file():
        log.warning(
            "%s was switched onto calibration %s, which is gone; using the rig in results.h5",
            recording_slug,
            name,
        )
        return None
    base = base_provisional(results_path, state.camera_names)
    try:
        state.result.cameras = CameraGroup.from_calibration(
            path, names=list(state.camera_names)
        )
    except (ValueError, KeyError) as exc:
        log.warning("could not re-apply calibration %s: %s", path.name, exc)
        return None
    state.invalidate_derived()
    state.set_provisional(still_provisional(path, base))
    return path


def unique_path(directory: Path, name: str) -> Path:
    """A path under ``directory`` for ``name`` that does not exist yet.

    A new calibration NEVER replaces an existing one: a rig is the thing every 3D number in
    the project is measured against, and an overwrite would silently reinterpret work already
    done. A name collision gets a numeric suffix instead.
    """
    directory.mkdir(parents=True, exist_ok=True)
    base = _slug(name)
    path = directory / f"{base}.toml"
    n = 2
    while path.exists():
        path = directory / f"{base}-{n}.toml"
        n += 1
    return path


def save_calibration(
    directory: Path,
    solved,
    *,
    name: str,
    image_sizes: dict | None,
    obs: Observations,
    settings: BaSettings,
    report: dict,
    recording: str | None,
) -> Path:
    """Write the solved rig as a new calibration file and return its path.

    ``image_sizes`` is ``camera -> (height, width)``, the convention
    :meth:`deeperfly.results.StageStore.read_image_sizes` returns and
    :class:`~deeperfly.rig.calibration.Calibration` stores.
    """
    from ..rig.calibration import Calibration

    quality = {
        "reprojection_median_px": report["after"]["median"],
        "reprojection_mean_px": report["after"]["mean"],
        "reprojection_p90_px": report["after"]["p90"],
        "n_observations": report["after"]["n"],
        "per_view_median_px": {
            k: v["median"] for k, v in report["after"]["per_view"].items()
        },
    }
    cal = Calibration.from_camera_group(
        solved,
        name=_slug(name),
        image_sizes=image_sizes or {},
        # Images cannot determine scale; physical units enter at inverse kinematics.
        units="arbitrary",
        scale_source="none",
        provenance={
            "solved_by": "deeperfly gui bundle adjustment",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "recording": recording,
            "n_tracks": obs.n_tracks,
            "n_frames": obs.n_frames,
            "per_view_labels": obs.per_view,
            "fixed": report["fixed_refs"],
            "shared": [list(g) for g in settings.shared],
            "loss": settings.loss,
            "f_scale": settings.f_scale,
            "free_focal": settings.free_focal,
            "init": {
                k: v
                for k, v in (report.get("init") or {}).items()
                if k != "registration"
            },
            "reprojection_before_px": report["before"]["median"],
            "intrinsics_source": "carried over from the rig this refined",
        },
        quality=quality,
    )
    path = unique_path(directory, name)
    cal.save(path)
    log.info(
        "wrote calibration %s (median %.2f px on %d observations)",
        path,
        quality["reprojection_median_px"] or float("nan"),
        quality["n_observations"],
    )
    return path


def delete_calibration(path: Path, *, active: Path | None = None) -> None:
    """Remove a calibration this editor produced.

    Two guards, both because a calibration is what every 3D number in the project is measured
    against. The rig in use is never deleted out from under the session, and a calibration
    this GUI did not write -- a board solve, an imported rig -- is not the editor's to throw
    away, however cluttered the list looks.
    """
    from ..rig.calibration import Calibration

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such calibration: {path}")
    if active is not None and path.resolve() == Path(active).resolve():
        raise ValueError(
            f"{path.name} is the calibration this session is using; switch to another first"
        )
    try:
        solved_by = str((Calibration.load(path).provenance or {}).get("solved_by", ""))
    except Exception:
        # Unreadable: it cannot be in use and nothing can depend on it, so let it go.
        solved_by = "deeperfly gui bundle adjustment"
    if "gui bundle adjustment" not in solved_by:
        raise ValueError(
            f"{path.name} was not solved in the editor ({solved_by or 'unknown origin'}), so "
            "the editor will not delete it -- remove it from the project directory by hand"
        )
    path.unlink()
    log.info("deleted calibration %s", path)


def list_calibrations(directory: Path, *, active: Path | None = None) -> list[dict]:
    """Every calibration in ``directory``, newest first, with enough to choose between them."""
    from ..rig.calibration import Calibration

    rows = []
    for path in sorted(Path(directory).glob("*.toml")):
        entry: dict[str, Any] = {
            "path": str(path),
            "file": path.name,
            "active": active is not None and path.resolve() == Path(active).resolve(),
        }
        try:
            cal = Calibration.load(path)
        except Exception as exc:  # a malformed file must not hide the good ones
            entry.update(name=path.stem, error=f"{type(exc).__name__}: {exc}")
            rows.append(entry)
            continue
        prov = dict(cal.provenance or {})
        entry["deletable"] = "gui bundle adjustment" in str(prov.get("solved_by", ""))
        entry.update(
            name=cal.name,
            cameras=list(cal.cameras.names),
            units=cal.units,
            scale_source=cal.scale_source,
            created_utc=prov.get("created_utc"),
            solved_by=prov.get("solved_by"),
            median_px=(cal.quality or {}).get("reprojection_median_px"),
            n_observations=(cal.quality or {}).get("n_observations"),
        )
        rows.append(entry)
    rows.sort(key=lambda r: (r.get("created_utc") or "", r["file"]), reverse=True)
    return rows
