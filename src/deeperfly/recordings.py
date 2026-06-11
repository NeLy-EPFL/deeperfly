"""Resolve ``run`` inputs into recordings: camera-source globbing and discovery."""

from __future__ import annotations

import glob
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .config import Config

log = logging.getLogger("deeperfly")

__all__ = [
    "camera_files",
    "source_patterns",
    "source_sources",
    "source_image_sizes",
    "default_outdir",
    "Recording",
    "find_recording",
    "OutdirPlan",
    "plan_outdirs",
    "resolve_recordings",
    "require_input_footage",
]


# -- input -> camera frame resolution ----------------------------------------


def _footage_exts() -> tuple[str, ...]:
    """Footage extensions deeperfly can read, in priority order (video before image).

    Recognizes a camera's frames and, when a folder mixes several, picks the one to
    keep (earliest wins). Imported lazily so resolving filenames doesn't pull in the
    I/O stack.

    Returns
    -------
    tuple of str
        Lowercase extensions (with the dot), video kinds before image kinds.
    """
    from .io import IMAGE_EXTS, VIDEO_EXTS

    return VIDEO_EXTS + IMAGE_EXTS


def _is_video_ext(suffix: str) -> bool:
    """Whether a file ``suffix`` (e.g. ``.mp4``) is a video extension.

    Video footage is a single file per camera; images form a sequence.

    Parameters
    ----------
    suffix
        A filename suffix including the dot (case-insensitive).

    Returns
    -------
    bool
        ``True`` for a known video extension.
    """
    from .io import VIDEO_EXTS

    return suffix.lower() in VIDEO_EXTS


def _camera_glob(pattern: str) -> str:
    """A camera's ``input`` value as a filename glob.

    A value that already names a file (a known footage suffix like
    ``camera_0.mp4``) or carries its own wildcard (``camera_0/*``, ``cam*``) is used
    verbatim; a bare name (``camera_0``) is treated as a *prefix*, so ``camera_0``
    becomes ``camera_0*`` and matches both ``camera_0.mp4`` and an image sequence
    ``camera_0_000123.jpg ...``.

    Parameters
    ----------
    pattern
        A camera's ``input`` value (a name, prefix, file or wildcard).

    Returns
    -------
    str
        The glob to match inside the recording directory.
    """
    has_wildcard = any(c in pattern for c in "*?[")
    if has_wildcard or Path(pattern).suffix.lower() in _footage_exts():
        return pattern
    return f"{pattern}*"


def camera_files(root: Path, pattern: str) -> list[Path]:
    """A camera's footage files under ``root`` matching its ``input`` ``pattern``.

    Globs ``pattern`` (see :func:`_camera_glob`), keeps files with a known footage
    extension, and -- when several extensions match -- keeps the highest-priority
    one. Naturally sorted. Video footage is a single file, so several matching
    videos keep only the first (warned); images stay as the whole sequence. Empty
    when nothing footage-like matches, so the caller can treat the camera as absent.

    Parameters
    ----------
    root
        The recording directory to glob inside.
    pattern
        The camera's ``input`` glob (see :func:`_camera_glob`).

    Returns
    -------
    list of Path
        Naturally-sorted footage files (one video, or an image sequence).
    """
    from natsort import natsorted

    exts = _footage_exts()
    files = [
        p
        for p in root.glob(_camera_glob(pattern))
        if p.is_file() and p.suffix.lower() in exts
    ]
    present = {p.suffix.lower() for p in files}
    if len(present) > 1:
        keep = min(present, key=exts.index)
        files = [p for p in files if p.suffix.lower() == keep]
    return _first_if_video(root, pattern, natsorted(files))


def _first_if_video(root: Path, name: str, files: list[Path]) -> list[Path]:
    """Reduce a camera's video footage to its first file (warning).

    A camera's video is one file, but images are a sequence, so an image sequence
    is left untouched.

    Parameters
    ----------
    root
        The recording directory (for the warning message).
    name
        The camera name (for the warning message).
    files
        The naturally-sorted matched files.

    Returns
    -------
    list of Path
        ``files[:1]`` for multi-file video footage, else ``files`` unchanged.
    """
    if len(files) > 1 and _is_video_ext(files[0].suffix):
        log.warning(
            "recording %s: camera %s matches %d video files %s; using only the first "
            "(%s) -- video footage is a single file per camera",
            root,
            name,
            len(files),
            [p.name for p in files],
            files[0].name,
        )
        return files[:1]
    return files


def source_patterns(config: Config) -> dict[str, str]:
    """``source-name -> footage glob`` (the ``[[sources]]`` ``input`` key), in order.

    A source with no ``input`` entry defaults to its own name as the pattern.

    Parameters
    ----------
    config
        A :class:`~deeperfly.config.Config`.

    Returns
    -------
    dict of str to str
        ``source_name -> footage glob`` in config order.
    """
    return config.source_patterns()


def source_sources(
    config: Config, *, sources: dict[str, list[Path]] | None = None, input=None
) -> list[tuple[str, list[Path]]]:
    """``(name, footage-files)`` per source (in ``[[sources]]`` order).

    Prefers the files ``deeperfly run`` already resolved (``sources``) so footage is
    globbed once per run; otherwise resolves each source from ``input`` with the
    per-source ``input`` globs (a library caller). With neither, every source
    resolves to an empty list.

    Parameters
    ----------
    config
        The run config (for the per-source globs).
    sources
        Optional pre-resolved ``source_name -> footage files`` map (preferred).
    input
        Optional recording root to glob each source from when ``sources`` is unset.

    Returns
    -------
    list of (str, list of Path)
        ``(name, footage-files)`` per source in ``[[sources]]`` order; each is
        the list passed to :func:`deeperfly.io.open_reader`.
    """
    patterns = config.source_patterns()
    if sources and all(name in sources for name in patterns):
        return [(name, sources[name]) for name in patterns]
    if input is None:
        return [(name, []) for name in patterns]
    return [(name, camera_files(Path(input), pat)) for name, pat in patterns.items()]


def source_image_sizes(
    config: Config, *, sources: dict[str, list[Path]] | None = None, input=None
) -> dict[str, tuple[int, int]]:
    """``name -> (height, width)`` of the raw footage, from a single frame per source.

    Used to resolve each view's intrinsics (the view's intrinsics describe its
    source's raw frame) and to anchor each pathway's coordinate inverse. Reads
    only frame 0 (host), so it is cheap and independent of the full streaming
    decode.

    Parameters
    ----------
    config
        The run config (I/O backends).
    sources
        Optional pre-resolved ``source_name -> footage files`` map.
    input
        Optional recording root (see :func:`source_sources`).

    Returns
    -------
    dict of str to tuple of int
        ``source_name -> (height, width)`` of the raw frame.
    """
    from . import io

    sizes: dict[str, tuple[int, int]] = {}
    for name, src in source_sources(config, sources=sources, input=input):
        head = io.open_reader(src)[[0]]
        sizes[name] = (int(head.shape[1]), int(head.shape[2]))
    return sizes


def default_outdir(inp: str | Path) -> Path:
    """Default output dir when ``-o`` is omitted: ``<input>/deeperfly_outputs``.

    Parameters
    ----------
    inp
        The recording directory (or a glob/file whose parent is used).

    Returns
    -------
    Path
        ``<input>/deeperfly_outputs`` (sibling of a file/glob input).
    """
    p = Path(inp)
    base = p if p.is_dir() else p.parent
    return base / "deeperfly_outputs"


def _has_glob(pattern: str) -> bool:
    """Whether ``pattern`` carries a shell wildcard (so it should be expanded).

    Parameters
    ----------
    pattern
        An input argument.

    Returns
    -------
    bool
        ``True`` if ``pattern`` contains a ``*``, ``?`` or ``[`` wildcard.
    """
    return any(c in pattern for c in "*?[")


@dataclass(frozen=True)
class Recording:
    """One unit of work: a camera -> footage-files map and where its results go.

    ``sources`` maps a camera name to its naturally-sorted footage files (a single
    video, or an image sequence), already reconciled to one extension and validated
    to share a file and frame count with the other cameras. Empty only for a
    directory kept so a resume can reuse a cached result though its footage is
    absent (see :func:`resolve_recordings`).

    ``outdir`` is this recording's output directory (see :func:`plan_outdirs`) --
    the run's durable identity, holding the config snapshot and cached ``poses.h5``.
    The input directory is not retained; a resume re-passes the recording, which
    re-resolves ``sources`` the same way.
    """

    sources: dict[str, list[Path]]
    outdir: Path


def _frame_counts_match(root: Path, sources: dict[str, list[Path]]) -> bool:
    """Whether every camera under ``root`` has the same file and frame count.

    File counts are compared directly. For image sequences equal file counts
    already imply equal frame counts, so only *video* footage is probed for its
    frame count, and only when knowable (an unreadable file's ``None`` count is
    skipped rather than falsely rejecting the recording). Warns when counts differ.

    Parameters
    ----------
    root
        The recording directory (for the warning message).
    sources
        ``camera_name -> footage files`` for the recording.

    Returns
    -------
    bool
        ``True`` if every camera has the same file (and frame) count.
    """
    file_counts = {n: len(ps) for n, ps in sources.items()}
    if len(set(file_counts.values())) > 1:
        log.warning(
            "recording %s has an uneven file count across cameras %s; skipping it",
            root,
            file_counts,
        )
        return False
    sample = next((ps for ps in sources.values() if ps), [])
    if not sample or not _is_video_ext(sample[0].suffix):
        return True  # image sequence (or empty): the file count already settled it
    from . import io

    frame_counts = {n: io.open_reader(ps).count() for n, ps in sources.items()}
    known = {c for c in frame_counts.values() if c is not None}
    if len(known) > 1:
        log.warning(
            "recording %s has an uneven frame count across cameras %s; skipping it",
            root,
            frame_counts,
        )
        return False
    return True


def find_recording(root: Path, config: Config) -> dict[str, list[Path]] | None:
    """``root``'s ``camera -> footage-files`` map if it is a recording, else ``None``.

    A *recording* is a directory holding footage for every configured camera (its
    ``input`` glob); the footage is a single video file or an image sequence. A
    directory matching *no* camera is silently not a recording (an intermediate or
    output dir); the rest warn and skip:

    - footage for only some cameras (a malformed recording, or a wrong ``input``);
    - files matched but none with a known footage extension;
    - several footage extensions in one folder (the highest-priority one is kept,
      and any camera then left with nothing counts as missing);
    - an unequal file or frame count across cameras (see :func:`_frame_counts_match`).

    Parameters
    ----------
    root
        The candidate directory.
    config
        The run config (for the per-camera globs).

    Returns
    -------
    dict of str to list of Path or None
        ``camera -> footage files`` if ``root`` is a valid recording, else
        ``None``.
    """
    if not root.is_dir():
        return None
    from natsort import natsorted

    exts = _footage_exts()
    patterns = source_patterns(config)
    # Raw matches (any file) per source, so "no match" is distinguishable from
    # "matched, but not footage".
    raw = {
        name: [p for p in root.glob(_camera_glob(pat)) if p.is_file()]
        for name, pat in patterns.items()
    }
    present = {name: ps for name, ps in raw.items() if ps}
    if not present:
        return None  # nothing here looks like a camera's files: not a recording
    missing = [name for name in patterns if name not in present]
    if missing:
        log.warning(
            "recording %s has footage for only %s (missing %s); skipping it",
            root,
            sorted(present),
            missing,
        )
        return None
    sources = {
        name: [p for p in ps if p.suffix.lower() in exts]
        for name, ps in present.items()
    }
    no_ext = sorted(name for name, ps in sources.items() if not ps)
    if no_ext:
        log.warning(
            "recording %s: camera(s) %s matched files but none with a known footage "
            "extension %s; skipping it",
            root,
            no_ext,
            list(exts),
        )
        return None
    seen = {p.suffix.lower() for ps in sources.values() for p in ps}
    if len(seen) > 1:
        keep = min(seen, key=exts.index)
        log.warning(
            "recording %s mixes footage extensions %s; using %s",
            root,
            sorted(seen, key=exts.index),
            keep,
        )
        sources = {
            name: [p for p in ps if p.suffix.lower() == keep]
            for name, ps in sources.items()
        }
        gone = sorted(name for name, ps in sources.items() if not ps)
        if gone:
            log.warning(
                "recording %s has no %s footage for %s; skipping it", root, keep, gone
            )
            return None
    sources = {
        name: _first_if_video(root, name, natsorted(ps)) for name, ps in sources.items()
    }
    if not _frame_counts_match(root, sources):
        return None
    return sources


def _expand_pattern(pattern: str) -> tuple[list[Path], bool]:
    """One ``run`` input argument -> ``(paths, is_glob)``.

    A wildcard (``fly*``, ``data/*``) expands to its sorted matches (possibly
    empty); a literal argument yields just itself. ``is_glob`` flags which it was,
    so a wildcard's incidental non-recording matches are skipped silently while a
    literal path the user typed is reported when invalid.

    Parameters
    ----------
    pattern
        One ``run`` input argument.

    Returns
    -------
    paths : list of Path
        The expansion (a wildcard's sorted matches, or just the literal path).
    is_glob : bool
        Whether ``pattern`` was a wildcard.
    """
    if _has_glob(pattern):
        return [Path(p) for p in sorted(glob.glob(pattern))], True
    return [Path(pattern)], False


def _dedup_found(
    found: Iterable[tuple[Path, dict[str, list[Path]]]],
) -> list[tuple[Path, dict[str, list[Path]]]]:
    """Drop ``(dir, sources)`` pairs whose directory repeats, keeping the first.

    Overlapping inputs/roots can match one directory twice; keeping the first
    occurrence keeps run order predictable.

    Parameters
    ----------
    found
        Discovered ``(dir, sources)`` pairs.

    Returns
    -------
    list of (Path, dict)
        The de-duplicated pairs in first-seen order.
    """
    seen: set = set()
    out: list[tuple[Path, dict[str, list[Path]]]] = []
    for d, src in found:
        key = d.resolve()
        if key not in seen:
            seen.add(key)
            out.append((d, src))
    return out


@dataclass(frozen=True)
class OutdirPlan:
    """The per-recording output directories for one run.

    ``outdirs`` is aligned with the recording directories handed to
    :func:`plan_outdirs`. ``mirror_confirm``, when set, is a human-readable
    description of a name-collision fallback (mirroring the input tree) that the
    caller must confirm with the user *before* any run starts.
    """

    outdirs: list[Path]
    mirror_confirm: str | None = None


def plan_outdirs(dirs: list[Path], output: str | None) -> OutdirPlan:
    """Resolve each recording's output directory from the raw ``-o`` string.

    A single recording uses ``-o`` as given (default: its own
    ``deeperfly_outputs``). A batch (several recordings) reads ``-o`` like
    ``rsync`` reads a trailing slash:

    - no ``-o``: each recording's own ``deeperfly_outputs``;
    - ``-o`` *ending in a path separator* ("collect"): one subdirectory per
      recording under it, ``<o>/<name>``; when recording names collide (e.g.
      ``a/rec`` and ``b/rec``), every output instead mirrors its recording's
      path from their common ancestor (``<o>/a/rec``, ``<o>/b/rec``), pending
      user confirmation (:attr:`OutdirPlan.mirror_confirm`);
    - a *relative* ``-o`` without a trailing separator: that directory inside
      each recording, ``<recording>/<o>`` (the default is effectively
      ``-o deeperfly_outputs``);
    - an *absolute* ``-o`` without a trailing separator: treated as "collect"
      (an absolute path cannot nest inside each recording), with a log note.

    Parameters
    ----------
    dirs
        The resolved recording directories.
    output
        The raw ``-o`` string (the trailing-slash distinction is lost on a
        ``Path``), or ``None``.

    Returns
    -------
    OutdirPlan
        The output directories, aligned with ``dirs``.
    """
    if len(dirs) == 1:
        return OutdirPlan([Path(output) if output else default_outdir(dirs[0])])
    if not output:
        return OutdirPlan([default_outdir(d) for d in dirs])
    collect = output.endswith(("/", os.sep))
    if not collect and os.path.isabs(output):
        log.info(
            "-o %s is absolute: collecting per-recording outputs under it (a "
            "relative name would create that directory inside each recording)",
            output,
        )
        collect = True
    if not collect:
        return OutdirPlan([d / output for d in dirs])
    base = Path(output)
    names = [d.name for d in dirs]
    if len(set(names)) == len(names):
        return OutdirPlan([base / name for name in names])
    # Names collide -> mirror each recording's path from the common ancestor, so
    # the runs can't silently share one output dir. Needs user confirmation.
    resolved = [d.resolve() for d in dirs]
    ancestor = Path(os.path.commonpath([str(p) for p in resolved]))
    outdirs = [base / p.relative_to(ancestor) for p in resolved]
    dupes = sorted({n for n in names if names.count(n) > 1})
    mapping = "\n".join(f"  {d}  ->  {o}" for d, o in zip(dirs, outdirs))
    confirm = (
        f"recording names collide under -o {output} ({', '.join(dupes)}); "
        f"mirroring the input paths from {ancestor} instead:\n{mapping}"
    )
    return OutdirPlan(outdirs, mirror_confirm=confirm)


def resolve_recordings(
    inputs: list[Path], *, recursive: bool, config: Config
) -> list[tuple[Path, dict[str, list[Path]]]]:
    """Expand the ``run`` inputs into the recordings to process.

    ``inputs`` is one or more input arguments, each a literal path or a wildcard
    pattern expanded against the filesystem (:func:`_expand_pattern`). A *recording*
    is a directory holding footage for every configured camera, resolved to a
    ``camera -> files`` map by :func:`find_recording` (which warns and skips a
    malformed one). Output directories are resolved separately
    (:func:`plan_outdirs`). The behaviors:

    - A single literal path is taken as that one recording -- kept (with empty
      sources) even when it is not valid footage, so a resume from its cached result
      still works -- with a warning naming it when it is not a valid recording.
    - Several inputs and/or a wildcard run as a batch: only the valid recordings are
      kept (a wildcard's incidental non-recording matches are dropped silently);
      nothing valid is a warned error.
    - With ``--recursive`` each input is a *parent* directory whose subtree is walked
      for recordings; an empty result is an error.

    De-duplicated by directory (overlapping inputs) keeping first-seen order.

    Parameters
    ----------
    inputs
        One or more input arguments (literal paths or wildcard patterns).
    recursive
        Whether each input is a parent directory whose subtree is searched.
    config
        The discovery config (recognizes recording directories).

    Returns
    -------
    list of (Path, dict)
        ``(recording directory, camera -> footage files)`` per recording.

    Raises
    ------
    SystemExit
        If no valid recording can be resolved from ``inputs``.
    """
    candidates: list[tuple[Path, bool]] = []
    for arg in inputs:
        paths, is_glob = _expand_pattern(str(arg))
        if is_glob and not paths:
            log.warning("input pattern %r matched no paths", str(arg))
        candidates += [(p, is_glob) for p in paths]

    if recursive:
        found: list[tuple[Path, dict[str, list[Path]]]] = []
        for root, is_glob in candidates:
            if not root.is_dir():
                if not is_glob:  # a literal parent the user named but that is absent
                    log.warning(
                        "%s is not a directory -- --recursive searches a parent "
                        "directory for recordings; skipping",
                        root.resolve(),
                    )
                continue
            for d in [root, *sorted(root.rglob("*"))]:
                if d.is_dir() and (src := find_recording(d, config)) is not None:
                    found.append((d, src))
        found = _dedup_found(found)
        if not found:
            log.warning(
                "no recordings found under %s (searched recursively); a recording is "
                "a directory holding footage for every configured camera",
                [str(p) for p, _ in candidates] or [str(a) for a in inputs],
            )
            raise SystemExit("no recordings to run")
        return found

    # Non-recursive. A single explicit path is honored as-is (resume-friendly): keep
    # it even when it is not valid footage, so resuming from its cache still works.
    if len(candidates) == 1 and not candidates[0][1]:
        path = candidates[0][0]
        src = find_recording(path, config)
        if src is None:
            log.warning(
                "%s is not a valid recording directory -- it does not hold footage "
                "for every configured camera (it can still resume from a cached "
                "result in its output dir)",
                path.resolve(),
            )
            src = {}
        return [(path, src)]

    # Several inputs and/or a wildcard: a batch. Keep only the valid recordings; only
    # warn (and error) when the inputs yield no valid recording at all.
    found = _dedup_found(
        (p, src)
        for p, _ in candidates
        if (src := find_recording(p, config)) is not None
    )
    if not found:
        log.warning(
            "none of the inputs is a valid recording directory (a directory holding "
            "footage for every configured camera)",
        )
        raise SystemExit("no valid recording directories among the inputs")
    return found


def require_input_footage(
    config: Config, *, sources: dict[str, list[Path]] | None = None, input=None
) -> None:
    """Fail (before any output dir is created) if the run's recording is unreadable.

    Checked only when ``pose2d`` will actually decode frames; a resume that reuses
    a cached 2D pose needs no footage. The footage was resolved up front by
    :func:`resolve_recordings` (``sources``); a library caller that set only
    ``input`` is validated directly. Raising here keeps a fresh run that can't
    read its input from leaving an empty ``deeperfly_outputs`` behind.

    Parameters
    ----------
    config
        The run config (for the per-camera globs).
    sources
        The pre-resolved ``camera_name -> footage files`` map, or ``None`` when a
        library caller passes only ``input``.
    input
        The recording root validated directly when ``sources`` is unset.

    Raises
    ------
    SystemExit
        If the recording is missing, not a directory, or has no footage for some
        camera.
    """
    patterns = source_patterns(config)
    if sources is None and input is not None:
        root = Path(input)
        if not root.exists():
            raise SystemExit(
                f"input recording {root} does not exist -- pass an existing directory "
                "holding the per-camera video/images for this run"
            )
        if not root.is_dir():
            raise SystemExit(
                f"input recording {root} is not a directory -- the run input is a "
                "directory of per-camera footage, not a single file"
            )
        for name, pat in patterns.items():
            if not camera_files(root, pat):
                raise SystemExit(f"no video or images for camera {name!r} under {root}")
        return

    sources = sources or {}
    missing = [name for name in patterns if not sources.get(name)]
    if missing:
        raise SystemExit(
            f"this run needs footage for pose2d but the recording resolved no files "
            f"for camera(s) {missing} (see the warning above) -- pass a recording that "
            "holds video/images for every camera, or resume from a cached poses.h5"
        )
