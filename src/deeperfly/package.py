"""``.dfpkg`` -- a project as one file, with the labeled frames inside it.

A project is a directory that *references* multi-gigabyte footage on a lab share. That is
right for working and wrong for sharing: a collaborator on another machine cannot resolve
those paths, and neither can a merge run six months later on an archived recording.

A package is the portable form. It carries everything the project *owns* -- the skeleton, the
rig, the calibrations, the landmark definitions, the manifest -- plus, per recording, its
``labels.h5`` verbatim and, optionally, **the frames those labels annotate**. That last part
is what makes it self-contained, and it is affordable for exactly the reason SLEAP's
``.pkg.slp`` is: only the labeled frames matter. On this project's own corpus that is 50
frames out of 4,073 -- about 40 MB of JPEG against 5.5 GB of video.

.. code-block:: text

    /meta                    attrs: format_version, project_toml, created_utc, provenance
    /skeleton                skeleton.toml text
    /rig                     rig.toml text            (when the project has one)
    /landmarks               landmarks.toml text      (when it declares any)
    /profiles/<name>         profile text
    /calibrations/<name>     calibration.toml text
    /recordings/<slug>/
        meta                 attrs: id, subject, n_frames, fps, footage basenames
        labels               the labels.h5 file, byte for byte
        frames/<camera>      vlen uint8 -- one JPEG per embedded frame
        frame_index          (F,) int32 -- which source frame each row is

``labels`` is stored as **the original bytes**, not a re-serialization. Ground truth is the
one irreplaceable thing here, and a byte copy cannot be corrupted by a schema mistake in this
module -- an import writes the file back out and the existing loader validates it, exactly as
if it had never travelled.

**Embedding policy** (``--embed``): ``user`` (default) embeds the frames carrying human
labels, ``all`` adds the suggested ones, ``none`` embeds no pixels and produces an
index-only package for collaborators who share the filesystem.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

__all__ = [
    "PACKAGE_FORMAT_VERSION",
    "PACKAGE_SUFFIX",
    "EMBED_POLICIES",
    "PackageReport",
    "export_package",
    "import_package",
    "describe_package",
]

log = logging.getLogger("deeperfly")

PACKAGE_FORMAT_VERSION = 1
PACKAGE_SUFFIX = ".dfpkg"

#: Which frames get their pixels embedded.
EMBED_POLICIES = ("user", "all", "none")

_STR = h5py.string_dtype("utf-8")
_VLEN_U8 = h5py.vlen_dtype(np.uint8)

#: JPEG quality for embedded frames. 92 is visually lossless for annotation review while
#: staying ~10x smaller than PNG. The *labels* are exact regardless -- they are stored
#: coordinates, not pixels -- so this only affects how the frame looks when reviewed.
_JPEG_QUALITY = 92


@dataclass
class PackageReport:
    """What an export or import moved."""

    recordings: int = 0
    label_files: int = 0
    embedded_frames: int = 0
    bytes_written: int = 0
    calibrations: int = 0
    skipped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "recordings": self.recordings,
            "label_files": self.label_files,
            "embedded_frames": self.embedded_frames,
            "bytes_written": self.bytes_written,
            "calibrations": self.calibrations,
            "skipped": self.skipped,
            "notes": self.notes,
        }


# -- export --------------------------------------------------------------------


def export_package(project, out: str | Path, *, embed: str = "user") -> PackageReport:
    """Write ``project`` to a ``.dfpkg``.

    Parameters
    ----------
    project
        The :class:`~deeperfly.project.Project` to package.
    out
        Destination path (``.dfpkg`` appended when absent).
    embed
        One of :data:`EMBED_POLICIES`.

    Returns
    -------
    PackageReport
        Counts, plus anything skipped -- a recording whose footage no longer resolves is
        *reported*, not silently omitted, because a package quietly missing its frames is
        indistinguishable from one that never had them.

    Raises
    ------
    ValueError
        If ``embed`` is unknown.
    """
    if embed not in EMBED_POLICIES:
        raise ValueError(f"embed must be one of {list(EMBED_POLICIES)}, got {embed!r}")
    path = Path(out)
    if path.suffix != PACKAGE_SUFFIX:
        path = path.with_name(path.name + PACKAGE_SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = PackageReport()

    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["package_format_version"] = PACKAGE_FORMAT_VERSION
        meta.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        meta.attrs["embed"] = embed
        meta.attrs["project_toml"] = (project.root / "project.toml").read_text()
        meta.attrs["provenance"] = json.dumps(
            {"exported_from": str(project.root.resolve()), "name": project.name}
        )

        _put_text(f, "skeleton", project.skeleton_path())
        _put_text(f, "rig", project.rig_path())
        _put_text(f, "landmarks", project.root / "landmarks.toml")
        for profile in sorted((project.root / "profiles").glob("*.toml")):
            _put_text(f, f"profiles/{profile.stem}", profile)
        for calibration in sorted((project.root / "calibrations").glob("*.toml")):
            _put_text(f, f"calibrations/{calibration.stem}", calibration)
            report.calibrations += 1

        for entry in project.recordings:
            group = f.create_group(f"recordings/{entry.slug}")
            group.attrs["id"] = entry.id
            group.attrs["n_frames"] = -1 if entry.n_frames is None else entry.n_frames
            if entry.subject:
                group.attrs["subject"] = entry.subject
            if entry.fps is not None:
                group.attrs["fps"] = float(entry.fps)
            group.attrs["origin"] = json.dumps(entry.origin or {})
            report.recordings += 1

            labels_path = project.labels_path(entry)
            if labels_path.exists():
                # The original bytes: ground truth is the irreplaceable thing here, and a
                # byte copy cannot be corrupted by a schema mistake in this module.
                group.create_dataset(
                    "labels",
                    data=np.frombuffer(labels_path.read_bytes(), dtype=np.uint8),
                )
                report.label_files += 1
            else:
                report.notes.append(f"{entry.slug}: no labels.h5 to package")

            if embed != "none":
                report.embedded_frames += _embed_frames(
                    project, entry, group, embed, report
                )

    report.bytes_written = path.stat().st_size
    log.info(
        "wrote %s (%d recording(s), %d label file(s), %d embedded frame(s), %.1f MB)",
        path,
        report.recordings,
        report.label_files,
        report.embedded_frames,
        report.bytes_written / 1e6,
    )
    return report


def _put_text(f, name: str, path: Path) -> None:
    """Store a project text file, if it exists, as one utf-8 dataset."""
    if path.exists():
        f.create_dataset(name, data=path.read_text(), dtype=_STR)


def _frames_to_embed(project, entry, policy: str) -> list[int]:
    """Which frame indices to embed: the labeled ones, plus suggestions under ``"all"``.

    Only the labeled frames matter, which is what keeps a package small -- on this project's
    own corpus, 50 frames out of 4,073.
    """
    from .project import _coo_rows

    frames: set[int] = set()
    labels_path = project.labels_path(entry)
    if labels_path.exists():
        try:
            with h5py.File(labels_path, "r") as lf:
                for key in ("gt/index", "occluded/index"):
                    if key in lf:
                        rows = _coo_rows(lf[key][()])
                        frames.update(int(r) for r in rows[:, 1])
                if "reviewed/index" in lf:
                    frames.update(
                        int(t) for t in np.asarray(lf["reviewed/index"][()]).reshape(-1)
                    )
                if "landmarks/index" in lf:
                    rows = np.asarray(lf["landmarks/index"][()]).reshape(-1, 3)
                    frames.update(int(r) for r in rows[:, 1])
        except Exception as exc:
            log.warning("could not read %s for frame selection: %s", labels_path, exc)
    if policy == "all":
        sidecar = project.outputs_dir(entry) / "labels_suggest.json"
        if sidecar.exists():
            try:
                data = json.loads(sidecar.read_text())
                frames.update(
                    int(row["frame"])
                    for row in (data.get("frames") or [])
                    if isinstance(row, dict) and "frame" in row
                )
            except Exception as exc:
                log.warning("could not read %s: %s", sidecar, exc)
    return sorted(frames)


def _embed_frames(project, entry, group, policy: str, report: PackageReport) -> int:
    """JPEG-encode the selected frames per camera into ``group``. Returns the count."""
    wanted = _frames_to_embed(project, entry, policy)
    if not wanted:
        return 0
    try:
        from .gui import _recording_footage
        from .gui.readers import FrameSource, resolve_camera_files
    except Exception:  # pragma: no cover -- the GUI package is a core dep
        return 0

    footage = _recording_footage(project, entry)
    resolved = {}
    for name, paths in footage.items():
        files = resolve_camera_files(
            {"abs": [str(p) for p in paths]}, project.recording_dir(entry), None
        )
        if files:
            resolved[name] = files
    if not resolved:
        report.skipped.append(
            f"{entry.slug}: footage does not resolve, so its frames could not be embedded "
            "(the labels are packaged regardless)"
        )
        return 0

    import cv2

    source = FrameSource(resolved)
    group.create_dataset("frame_index", data=np.asarray(wanted, dtype=np.int32))
    count = 0
    for name in resolved:
        blobs = []
        for t in wanted:
            frame = source.frame(name, t)
            if frame is None:
                blobs.append(np.zeros(0, dtype=np.uint8))
                continue
            bgr = frame if frame.ndim == 2 else frame[..., ::-1]
            ok, buf = cv2.imencode(
                ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]
            )
            blobs.append(
                np.frombuffer(buf.tobytes(), dtype=np.uint8)
                if ok
                else np.zeros(0, dtype=np.uint8)
            )
            count += 1
        dataset = group.create_dataset(f"frames/{name}", (len(blobs),), dtype=_VLEN_U8)
        for i, blob in enumerate(blobs):
            dataset[i] = blob
    return count


# -- import --------------------------------------------------------------------


def describe_package(path: str | Path) -> dict:
    """What a package holds, without unpacking it.

    Lets a dry-run import report the contents before anything is written -- the same
    principle as merge's dry run, for the same reason.

    Raises
    ------
    ValueError
        If the file was written by a newer deeperfly.
    """
    with h5py.File(_resolve(path), "r") as f:
        meta = f["meta"]
        version = int(meta.attrs.get("package_format_version", 1))
        if version > PACKAGE_FORMAT_VERSION:
            raise ValueError(
                f"{path} was written by a newer deeperfly (package format v{version}, "
                f"this build understands v{PACKAGE_FORMAT_VERSION}); refusing to read it"
            )
        recordings = []
        for slug in sorted(f.get("recordings", {})):
            group = f[f"recordings/{slug}"]
            embedded = (
                sum(len(group[f"frames/{c}"]) for c in group["frames"])
                if "frames" in group
                else 0
            )
            recordings.append(
                {
                    "slug": slug,
                    "id": str(group.attrs.get("id", "")),
                    "subject": group.attrs.get("subject"),
                    "n_frames": int(group.attrs.get("n_frames", -1)),
                    "has_labels": "labels" in group,
                    "embedded_frames": int(embedded),
                }
            )
        return {
            "format_version": version,
            "created_utc": str(meta.attrs.get("created_utc", "")),
            "embed": str(meta.attrs.get("embed", "")),
            "provenance": json.loads(meta.attrs.get("provenance", "{}")),
            "has_rig": "rig" in f,
            "has_landmarks": "landmarks" in f,
            "calibrations": sorted(f.get("calibrations", {})),
            "profiles": sorted(f.get("profiles", {})),
            "recordings": recordings,
        }


def import_package(
    path: str | Path, dest: str | Path, *, apply: bool = True
) -> PackageReport:
    """Unpack a ``.dfpkg`` into a new project directory.

    Deliberately only into a **new or empty** directory. Importing *into* an existing
    project is a merge -- with skeleton reconciliation, content dedup and conflict policy
    (see :mod:`deeperfly.merge`) -- and quietly overwriting files instead would be the
    destructive shortcut that looks like it worked.

    Parameters
    ----------
    path
        The package.
    dest
        Directory to create the project in.
    apply
        When false nothing is written (the dry run).

    Returns
    -------
    PackageReport
        What was (or would be) written.

    Raises
    ------
    SystemExit
        If ``dest`` already holds a project.
    """
    package = _resolve(path)
    target = Path(dest)
    if (target / "project.toml").exists():
        raise SystemExit(
            f"{target} already holds a project. Importing into an existing project is a "
            "merge, not an unpack -- import into a fresh directory, then "
            "'deeperfly labels-merge' the recordings you want"
        )
    report = PackageReport()

    with h5py.File(package, "r") as f:
        version = int(f["meta"].attrs.get("package_format_version", 1))
        if version > PACKAGE_FORMAT_VERSION:
            raise ValueError(f"{package} was written by a newer deeperfly (v{version})")
        if apply:
            (target / "recordings").mkdir(parents=True, exist_ok=True)
            (target / "calibrations").mkdir(exist_ok=True)
            (target / "profiles").mkdir(exist_ok=True)
            (target / "project.toml").write_text(_text(f["meta"].attrs["project_toml"]))
            for name, out in (
                ("skeleton", "skeleton.toml"),
                ("rig", "rig.toml"),
                ("landmarks", "landmarks.toml"),
            ):
                if name in f:
                    (target / out).write_text(_text(f[name][()]))
            for kind in ("calibrations", "profiles"):
                for stem in f.get(kind, {}):
                    (target / kind / f"{stem}.toml").write_text(
                        _text(f[f"{kind}/{stem}"][()])
                    )
        report.calibrations = len(f.get("calibrations", {}))

        for slug in sorted(f.get("recordings", {})):
            group = f[f"recordings/{slug}"]
            report.recordings += 1
            outputs = target / "recordings" / slug / "deeperfly_outputs"
            if "labels" in group:
                report.label_files += 1
                if apply:
                    outputs.mkdir(parents=True, exist_ok=True)
                    (outputs / "labels.h5").write_bytes(
                        np.asarray(group["labels"][()], dtype=np.uint8).tobytes()
                    )
            if "frames" in group:
                n = sum(len(group[f"frames/{c}"]) for c in group["frames"])
                report.embedded_frames += int(n)
                if apply:
                    _write_frames(group, target / "recordings" / slug / "frames")
            elif apply:
                report.notes.append(
                    f"{slug}: no embedded frames -- its footage must be re-pointed before "
                    "the editor can show anything"
                )
    if apply:
        log.info("imported %s into %s", package, target)
    return report


def _write_frames(group, out: Path) -> None:
    """Write embedded JPEGs as ``<out>/<camera>/<frame index>.jpg``.

    Files rather than a re-embedded container: the frame index is in the *filename*, so a
    frame is traceable to its source without opening anything -- the same reason DeepLabCut
    encodes it there.
    """
    index = (
        np.asarray(group["frame_index"][()], dtype=int)
        if "frame_index" in group
        else None
    )
    for camera in group["frames"]:
        folder = out / camera
        folder.mkdir(parents=True, exist_ok=True)
        data = group[f"frames/{camera}"]
        for i in range(len(data)):
            blob = np.asarray(data[i], dtype=np.uint8)
            if blob.size == 0:
                continue
            t = int(index[i]) if index is not None and i < len(index) else i
            (folder / f"{t:06d}.jpg").write_bytes(blob.tobytes())


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists() and p.suffix != PACKAGE_SUFFIX:
        alt = p.with_name(p.name + PACKAGE_SUFFIX)
        if alt.exists():
            return alt
    if not p.exists():
        raise FileNotFoundError(f"no package at {p}")
    return p


def _text(raw) -> str:
    return raw.decode() if isinstance(raw, bytes) else str(raw)
