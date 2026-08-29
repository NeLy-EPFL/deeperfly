"""Where a recording's solved rigs live on disk, and which one is active.

A recording accumulates calibrations: the one the pipeline wrote, the ones an operator
solved by hand in the editor, and the provisional stand-in a not-yet-solved camera gets.
This module is the filing system for them -- naming, listing, saving, deleting, and
recording which one the recording is currently using.

It sits here rather than in the editor because none of it is about editing. The editor
happens to be where an operator solves a rig today (see :mod:`deeperfly.gui.ba`), but the
files it writes are read by the pipeline, the CLI and the next session, so their layout
belongs to the rig.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("deeperfly")


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
