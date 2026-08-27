"""Resolve ``run`` inputs into recordings: camera-source globbing and discovery."""

from __future__ import annotations

import glob
import logging
import os
import re
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
    """Footage extensions deeperfly can read (video kinds, then image kinds).

    Imported lazily so resolving filenames does not pull in the I/O stack. Order no
    longer carries a *priority*: a pattern names the extension it wants, so a directory
    holding both ``camera_RH.mp4`` and ``camera_RH.avi`` is an error naming both rather
    than a silent pick.
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


#: Runs of digits, masked out to test whether matches are parts of ONE series.
_DIGITS = re.compile(r"\d+")


def _series_key(name: str) -> str:
    """``name`` with every run of digits replaced by ``#``.

    ``camera_RH_0.mp4`` and ``camera_RH_1.mp4`` share a key (``camera_RH_#.mp4``);
    ``camera_RH.mp4`` and ``camera_0.mp4`` do not.
    """
    return _DIGITS.sub("#", name)


def _entry_matches(root: Path, pattern: str, camera: str) -> list[Path]:
    """Every filename directly inside ``root`` that ``pattern`` fully matches.

    ``re.fullmatch``, case-insensitively, on the FILENAME only -- a pattern never
    traverses into a subdirectory. Naturally sorted, and checked to be parts of one
    series (see :func:`_series_key`), which is the whole safety of letting a pattern match
    several files: everything one entry matches is concatenated into one stream, so a
    pattern loose enough to catch two naming schemes would silently splice two recordings.

    Raises
    ------
    ValueError
        If ``pattern`` is not a valid regex, or if its matches are not one series (the
        message names the files and the camera).
    """
    from natsort import natsorted

    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(
            f"[cameras.{camera}] video = {pattern!r} is not a valid regex: {exc}. "
            "Write patterns as TOML LITERAL strings (single quotes) -- a backslash in a "
            "basic string is an escape."
        ) from exc
    exts = _footage_exts()
    files = natsorted(
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in exts and rx.fullmatch(p.name)
    )
    keys = {_series_key(p.name) for p in files}
    if len(keys) > 1:
        raise ValueError(
            f"recording {root}: [cameras.{camera}] video = {pattern!r} matches "
            f"{len(files)} files that are not parts of one series: "
            f"{[p.name for p in files]}. Everything one pattern matches is decoded as ONE "
            "stream, so these would be spliced together. Narrow the pattern, or put the "
            "alternatives in a list if they really are consecutive parts."
        )
    return list(files)


def camera_files(root: Path, pattern: str | list[str]) -> list[Path]:
    """A camera's footage under ``root``: everything its ``video`` pattern matches.

    One regex, or a list of them **concatenated in order**. Each entry contributes its
    matches in natural order, and the entries are laid end to end -- so a split recording
    (``camera_RH_0.mp4``, ``camera_RH_1.mp4``) and an image sequence are the same rule,
    and alternate NAMES go inside the regex (``camera_(RH|0)`` plus an extension) rather
    than in the list.

    Empty when nothing matches, so the caller can treat the camera as absent.

    Parameters
    ----------
    root
        The recording directory to look inside (filenames only, no recursion).
    pattern
        The camera's ``video`` value: a regex, or a list of them.
    camera
        The camera's name, for the error messages.

    Returns
    -------
    list of Path
        The camera's files, in decode order.

    Raises
    ------
    ValueError
        If a pattern is not a valid regex, if one entry's matches are not parts of one
        series, or if two entries match the same file.
    """
    entries = [pattern] if isinstance(pattern, str) else list(pattern)
    out: list[Path] = []
    for entry in entries:
        for path in _entry_matches(root, entry, _camera_hint(pattern)):
            if path in out:
                raise ValueError(
                    f"recording {root}: {path.name} is matched by two of "
                    f"{entries!r}, so it would be decoded twice"
                )
            out.append(path)
    return out


#: Set by :func:`find_recording` around a camera's resolution, so the error messages
#: above can name it. A module global rather than a parameter because ``camera_files`` is
#: called from four places that do not all have the name to hand.
_CAMERA_HINT: str = "?"


def _camera_hint(pattern) -> str:
    return _CAMERA_HINT


def _raw_matches(root: Path, pattern: str | list[str]) -> list[Path]:
    """Like :func:`camera_files`, but keeping every matched file rather than only footage.

    So a caller can tell "matched, but not footage" from "matched nothing".
    """
    entries = [pattern] if isinstance(pattern, str) else list(pattern)
    out: list[Path] = []
    for entry in entries:
        try:
            rx = re.compile(entry, re.IGNORECASE)
        except re.error:
            continue
        out += [
            p for p in sorted(root.iterdir()) if p.is_file() and rx.fullmatch(p.name)
        ]
    return out


def source_patterns(config: Config) -> dict[str, str | list[str]]:
    """``source-name -> footage glob`` (the ``[[sources]]`` ``input`` key), in order.

    A source with no ``input`` entry defaults to its own name as the pattern. A
    value may be a single glob or a list of alternate globs tried in order (see
    :func:`_as_alternates`).

    Parameters
    ----------
    config
        A :class:`~deeperfly.config.Config`.

    Returns
    -------
    dict of str to (str or list of str)
        ``source_name -> footage glob(s)`` in config order.
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

    A source the map does not mention resolves to an empty list, PER SOURCE. It used to
    take all-or-nothing -- one absent key and the whole map was discarded, so a recording
    holding seven of eight cameras resolved to *zero* footage rather than to seven. An
    empty list is already the established encoding for "this source has nothing"
    (:func:`camera_files` returns one, and says so), which is what lets a caller narrow
    the run to what is present instead of refusing the recording.

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
    if sources is not None:
        return [(name, list(sources.get(name) or [])) for name in patterns]
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

    A source with no footage is **absent from the result** rather than an error: there is
    no frame to read a size from, and the caller decides what that means. That is what
    makes the map "the sizes we know" -- which is how a run narrowed to the cameras it has
    still resolves intrinsics for those.

    Returns
    -------
    dict of str to tuple of int
        ``source_name -> (height, width)`` of the raw frame, for the sources that
        resolved footage.
    """
    from . import io

    sizes: dict[str, tuple[int, int]] = {}
    for name, src in source_sources(config, sources=sources, input=input):
        if not src:  # nothing to open; `io.open_reader([])` would raise
            continue
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
    the run's durable identity, holding the config snapshot and cached ``results.h5``.
    The input directory is not retained; a resume re-passes the recording, which
    re-resolves ``sources`` the same way.
    """

    sources: dict[str, list[Path]]
    outdir: Path


def _frame_counts_match(root: Path, sources: dict[str, list[Path]]) -> bool:
    """Whether every camera under ``root`` covers the same number of FRAMES.

    The file-count comparison this used to make first **inverts** under concatenation and
    is gone for video: the acquisition splits each camera at its own byte threshold, so one
    camera legitimately being two files while another is three is the normal case, and
    comparing file counts would silently skip every split recording. What is compared is
    the concatenated frame count, which :meth:`deeperfly.io.ConcatReader.count` returns as
    the sum over parts.

    For an image sequence the file count IS the frame count, so it stays the cheap proxy
    it always was.

    This is also the check that catches a runaway pattern: a camera whose regex matches two
    real cameras' files comes out at twice the others' T. The series check in
    :func:`_entry_matches` cannot see that (a pattern like ``camera_[0-9]`` plus an
    extension matches two cameras, and they are one series by its test), so this is the
    backstop.

    Parameters
    ----------
    root
        The recording directory (for the warning message).
    sources
        ``camera_name -> footage files`` for the recording.

    Returns
    -------
    bool
        ``True`` if every camera covers the same number of frames.
    """
    sample = next((ps for ps in sources.values() if ps), [])
    if not sample:
        return True
    if not _is_video_ext(sample[0].suffix):
        counts = {n: len(ps) for n, ps in sources.items()}
        if len(set(counts.values())) > 1:
            log.warning(
                "recording %s has an uneven image count across cameras %s; skipping it",
                root,
                counts,
            )
            return False
        return True
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

    A *recording* is a directory holding footage for at least one configured source (its
    ``input`` glob); the footage is a single video file or an image sequence. A directory
    matching *no* source is silently not a recording (an intermediate or output dir).

    **A partial recording is a recording.** Footage for only some of the configured
    sources is reported, with a warning naming what is absent, and the run narrows itself
    to what is present (:meth:`~deeperfly.config.Config.narrowed_to_sources`). One config
    describing a superset of rigs is the normal case -- the packaged default declares the
    eight-camera rig, and a seven-camera recording under it is not malformed, it is a
    seven-camera recording. Refusing the whole directory for a camera nobody has meant the
    config had to be edited per rig, and the failure named every camera rather than the one
    that was missing.

    These still warn and skip, because each is a directory that cannot be read coherently
    rather than one that is merely short a camera:

    - files matched but none with a known footage extension;
    - several footage extensions in one folder (the highest-priority one is kept,
      and any source then left with nothing counts as absent);
    - an unequal file or frame count across the sources that ARE present (see
      :func:`_frame_counts_match`).

    Parameters
    ----------
    root
        The candidate directory.
    config
        The run config (for the per-camera globs).

    Returns
    -------
    dict of str to list of Path or None
        ``source -> footage files`` for the sources that resolved, if ``root`` is a
        recording at all; else ``None``. A source with no footage is absent from the
        map, which is what every consumer reads as "this source has nothing".
    """
    if not root.is_dir():
        return None

    exts = _footage_exts()
    patterns = source_patterns(config)
    global _CAMERA_HINT
    raw: dict[str, list[Path]] = {}
    sources: dict[str, list[Path]] = {}
    for name, pat in patterns.items():
        _CAMERA_HINT = name
        try:
            raw[name] = _raw_matches(root, pat)
            files = camera_files(root, pat)
        finally:
            _CAMERA_HINT = "?"
        if files:
            sources[name] = files
    if not any(raw.values()):
        return None  # nothing here looks like a camera's files: not a recording
    missing = [name for name in patterns if name not in sources]
    if missing:
        # Not a refusal: the run narrows to the cameras that are here. Logged at WARNING
        # rather than INFO because the usual cause is a wrong `video` pattern, which looks
        # exactly like a camera that was never recorded.
        log.warning(
            "recording %s has footage for %d of %d configured camera(s) -- absent: %s. "
            "The run will use the %s it has; check the [cameras.<name>] `video` patterns "
            "if that is not what you expect",
            root,
            len(sources),
            len(patterns),
            missing,
            sorted(sources),
        )
    matched_only = sorted(n for n, ps in raw.items() if ps and n not in sources)
    if matched_only:
        log.warning(
            "recording %s: camera(s) %s matched files but none with a known footage "
            "extension %s",
            root,
            matched_only,
            list(exts),
        )
    if not sources:
        return None
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
                "%s holds footage for none of the configured sources (it can still "
                "resume from a cached result in its output dir)",
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
            "none of the inputs is a recording directory (a directory holding footage "
            "for at least one configured source)",
        )
        raise SystemExit("no valid recording directories among the inputs")
    return found


def require_input_footage(
    config: Config, *, sources: dict[str, list[Path]] | None = None, input=None
) -> None:
    """Fail (before any output dir is created) if the run's recording is unreadable.

    "Unreadable" means nothing at all resolved, not "short a camera". A recording holding
    some of the configured sources is a narrower recording, and the run adapts to it; this
    gate exists for the case where the path is simply wrong, where failing before an empty
    ``deeperfly_outputs`` is created is the whole point.

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
        If the recording is missing, not a directory, or holds footage for **no**
        configured source. Footage for *some* of them is not an error: the run narrows
        itself to what is present
        (:meth:`~deeperfly.config.Config.narrowed_to_sources`), which is also where the
        floor of two views is enforced.
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
        found = {name: camera_files(root, pat) for name, pat in patterns.items()}
        if not any(found.values()):
            raise SystemExit(
                f"no video or images for ANY of the {len(patterns)} configured "
                f"source(s) under {root}\n  looked for: {dict(patterns)}"
            )
        return

    sources = sources or {}
    if not any(sources.get(name) for name in patterns):
        raise SystemExit(
            "this run needs footage for pose2d but the recording resolved no files for "
            f"ANY of its {len(patterns)} configured source(s) (see the warning above) -- "
            "pass a recording holding the per-camera video/images, or resume from a "
            "cached results.h5.\n"
            f"  looked for: {dict(patterns)}"
        )
