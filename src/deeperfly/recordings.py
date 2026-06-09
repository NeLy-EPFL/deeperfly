"""Resolve ``run`` inputs into recordings: camera-source globbing and discovery."""

from __future__ import annotations

import glob
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .config import Config

log = logging.getLogger("deeperfly")


# -- input -> camera frame resolution ----------------------------------------


def _footage_exts() -> tuple[str, ...]:
    """Footage extensions deeperfly can read, in priority order (video before image).

    Recognizes a camera's frames and, when a folder mixes several, picks the one to
    keep (earliest wins). Imported lazily so resolving filenames doesn't pull in the
    video stack.

    Returns
    -------
    tuple of str
        Lowercase extensions (with the dot), video kinds before image kinds.
    """
    from .video.io import _IMAGE_EXTS, _VIDEO_EXTS

    return _VIDEO_EXTS + _IMAGE_EXTS


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
    from .video.io import _VIDEO_EXTS

    return suffix.lower() in _VIDEO_EXTS


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


def camera_patterns(config: Config | dict) -> dict[str, str]:
    """``camera-name -> footage glob`` (the per-camera ``input`` key), in config order.

    A camera with no ``input`` entry defaults to its own name as the pattern.

    Parameters
    ----------
    config
        A :class:`~deeperfly.config.Config` or a parsed config ``dict`` (the
        recording-discovery configs in the tests do this).

    Returns
    -------
    dict of str to str
        ``camera_name -> footage glob`` in config order.
    """
    return Config.coerce(config).camera_patterns()


def camera_sources(
    config: Config, *, sources: dict[str, list[Path]] | None = None, input=None
) -> list[tuple[str, list[Path]]]:
    """``(name, footage-files)`` per camera (in ``[cameras]`` order).

    Prefers the files ``deeperfly run`` already resolved (``sources``) so footage is
    globbed once per run; otherwise resolves each camera from ``input`` with the
    per-camera ``input`` globs (a library caller). With neither, every camera
    resolves to an empty list.

    Parameters
    ----------
    config
        The run config (for the per-camera globs).
    sources
        Optional pre-resolved ``camera_name -> footage files`` map (preferred).
    input
        Optional recording root to glob each camera from when ``sources`` is unset.

    Returns
    -------
    list of (str, list of Path)
        ``(name, footage-files)`` per camera in ``[cameras]`` order; each source
        is the list passed to :func:`deeperfly.video.read_frames`.
    """
    patterns = config.camera_patterns()
    if sources and all(name in sources for name in patterns):
        return [(name, sources[name]) for name in patterns]
    if input is None:
        return [(name, []) for name in patterns]
    return [(name, camera_files(Path(input), pat)) for name, pat in patterns.items()]


def camera_image_sizes(
    config: Config, *, sources: dict[str, list[Path]] | None = None, input=None
) -> dict[str, tuple[int, int]]:
    """``name -> (height, width)`` from a single frame per camera.

    Used to infer each view's principal point. Reads only frame 0 (host), so it is
    cheap and independent of the full streaming decode.

    Parameters
    ----------
    config
        The run config (I/O backends and per-camera preprocessing).
    sources
        Optional pre-resolved ``camera_name -> footage files`` map.
    input
        Optional recording root (see :func:`camera_sources`).

    Returns
    -------
    dict of str to tuple of int
        ``camera_name -> (height, width)`` of the preprocess-transformed frame.
    """
    from . import video

    backend = config.io.video_reader
    image_backend = config.io.image_reader
    # Size the principal point on the *transformed* frame -- the detector and the
    # overlays use the preprocess-transformed footage, so a rot90 that swaps
    # H/W must swap here too.
    transforms = config.frame_transforms()
    sizes: dict[str, tuple[int, int]] = {}
    for name, src in camera_sources(config, sources=sources, input=input):
        head = video.read_frames(
            src, backend=backend, image_backend=image_backend, indices=[0]
        )
        head = transforms.get(name, video.FrameTransform()).apply(head)
        sizes[name] = tuple(int(d) for d in head.shape[1:3])
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

    ``outdir`` is this recording's output directory (see :func:`_run_outdir`) --
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
    from . import video

    frame_counts = {n: video.count_frames(ps) for n, ps in sources.items()}
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
    patterns = camera_patterns(config)
    # Raw matches (any file) per camera, so "no match" is distinguishable from
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


def _run_outdir(output: str | None, recording: Path, *, batch: bool) -> Path:
    """Output directory for one recording.

    Default (no ``-o``): the recording's own ``deeperfly_outputs``. With ``-o``:
    that directory for a single recording, or a per-recording subdirectory under it
    for a wildcard/recursive batch (so the runs don't overwrite each other).

    Parameters
    ----------
    output
        The ``-o`` value, or ``None`` for the default.
    recording
        The recording directory.
    batch
        Whether this run processes several recordings (so ``-o`` nests per name).

    Returns
    -------
    Path
        The output directory for this recording.
    """
    if not output:
        return default_outdir(recording)
    base = Path(output)
    return base / recording.name if batch else base


def _plan_recordings(
    found: list[tuple[Path, dict[str, list[Path]]]], output: str | None
) -> list[Recording]:
    """Turn discovered ``(dir, sources)`` pairs into :class:`Recording`\\ s.

    The output directory is resolved per recording (:func:`_run_outdir`); whether
    this is a *batch* (several recordings, so ``-o`` nests per recording) is known
    only here, once every input has been resolved.

    Parameters
    ----------
    found
        Discovered ``(dir, sources)`` pairs.
    output
        The ``-o`` value, or ``None``.

    Returns
    -------
    list of Recording
        One :class:`Recording` per discovered pair.
    """
    batch = len(found) > 1
    return [Recording(src, _run_outdir(output, d, batch=batch)) for d, src in found]


def resolve_recordings(
    inputs: list[Path], *, recursive: bool, config: Config, output: str | None = None
) -> list[Recording]:
    """Expand the ``run`` inputs into the recordings to process (footage + output dir).

    ``inputs`` is one or more input arguments, each a literal path or a wildcard
    pattern expanded against the filesystem (:func:`_expand_pattern`). A *recording*
    is a directory holding footage for every configured camera, resolved to a
    ``camera -> files`` map by :func:`find_recording` (which warns and skips a
    malformed one). Each kept recording is paired with its output directory
    (:func:`_run_outdir`, honoring ``output`` = ``-o``); the input directory is not
    retained past this point. The behaviors:

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
    output
        The ``-o`` value, or ``None``.

    Returns
    -------
    list of Recording
        The recordings to process.

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
        return _plan_recordings(found, output)

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
        return _plan_recordings([(path, src)], output)

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
    return _plan_recordings(found, output)


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
    patterns = camera_patterns(config)
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
