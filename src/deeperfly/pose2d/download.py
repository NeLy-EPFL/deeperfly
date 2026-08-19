"""Find a detector's weights on this machine.

Every detector deeperfly runs is trained per project, so there is nothing to download:
:func:`resolve_weights` turns what a config wrote into a file on disk, and says exactly
what it searched when it cannot.

The point of the search path is that a *recording's* config should not have to carry a
machine's directory layout. ``weights = "mvt_alt8_gray_fly38.pth"`` is a fact about which
model a run used and travels with the recording; ``/mnt/upramdya/data/TL/...`` is a fact
about one mount on one machine and breaks the moment the config is opened anywhere else.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import platformdirs

log = logging.getLogger("deeperfly")

#: Environment variable naming where per-project checkpoints live: one directory, or
#: several separated by ``os.pathsep``. Searched (then the download cache) for a
#: ``weights`` value written as a bare filename.
MODELS_ENV = "DEEPERFLY_MODELS"


def cache_dir() -> Path:
    """Per-user cache directory for deeperfly weights (created on demand).

    Nothing writes here any more -- no detector auto-provisions -- but it stays on the
    search path so a checkpoint dropped in it is found, which is the one place a user can
    put one without setting an environment variable.
    """
    d = Path(platformdirs.user_cache_dir("deeperfly")) / "weights"
    d.mkdir(parents=True, exist_ok=True)
    return d


def search_path() -> list[Path]:
    """Where a bare ``weights`` filename is looked for, in order.

    ``$DEEPERFLY_MODELS`` (``os.pathsep``-separated, like ``PATH``) then the download
    cache. Read live rather than at import, so a test or a shell can set it.
    """
    raw = os.environ.get(MODELS_ENV, "")
    dirs = [Path(p).expanduser() for p in raw.split(os.pathsep) if p.strip()]
    return [*dirs, cache_dir()]


def resolve_weights(value: str | None, *, cls: str, model_name: str) -> Path | None:
    """Turn a ``[[pose2d.models]]`` ``weights`` value into a file on disk.

    Three forms, and the distinction is whether the value looks like a *path*:

    * empty / absent -> ``None``, which every class refuses (see
      :func:`missing_weights`) -- there is no checkpoint to fall back to.
    * a bare filename (``mvt_alt8_gray_fly38.pth``) -> searched along :func:`search_path`.
    * anything with a separator, or absolute, or ``~`` -> used as written.

    Parameters
    ----------
    value
        The raw ``weights`` value.
    cls, model_name
        The model's class and name, used only to make the failure readable.

    Returns
    -------
    Path or None
        The resolved checkpoint, or ``None`` when nothing was asked for.

    Raises
    ------
    SystemExit
        If a value was given and no file matches it -- naming every directory searched,
        because "which of these did you mean" is the only question at that moment.
    """
    if not value:
        return None
    raw = str(value)
    # `os.altsep` is None on POSIX, and `"" in raw` is always True -- so it has to be
    # tested for truth before it is tested for membership.
    separators = [s for s in (os.sep, os.altsep) if s]
    looks_like_path = any(s in raw for s in separators) or raw.startswith("~")
    if looks_like_path or Path(raw).is_absolute():
        path = Path(raw).expanduser()
        if not path.is_file():
            raise SystemExit(
                f"[[pose2d.models]] {model_name!r} (class {cls!r}): no detector "
                f"checkpoint at {path}"
            )
        return path

    tried = search_path()
    for d in tried:
        candidate = d / raw
        if candidate.is_file():
            return candidate
    where = "\n".join(f"    {d}" for d in tried)
    raise SystemExit(
        f"[[pose2d.models]] {model_name!r} (class {cls!r}): no checkpoint named {raw!r}.\n"
        f"  Searched (${MODELS_ENV}, then the download cache):\n{where}\n"
        f"  Set {MODELS_ENV}=/path/to/models, or write an explicit path:\n"
        f'    weights = "/path/to/{raw}"'
    )


def missing_weights(cls: str, model_name: str) -> SystemExit:
    """The failure for a per-project class whose ``weights`` is empty.

    Its own function because both loaders raise it and the message is the entire user
    experience of a fresh install: nothing downloads, so the error has to *be* the setup
    instructions.
    """
    return SystemExit(
        f"[[pose2d.models]] {model_name!r} (class {cls!r}) has no 'weights'. Every "
        "detector deeperfly runs is trained per project, so there is nothing to "
        "auto-provision -- name a checkpoint.\n"
        "  Either put it on the search path:\n"
        f"    export {MODELS_ENV}=/path/to/models\n"
        '    weights = "my_detector.pth"     # in [[pose2d.models]]\n'
        "  ...or write the path outright:\n"
        '    weights = "/path/to/my_detector.pth"\n'
        "  See docs/reference/configuration.md#weights for where the released "
        "checkpoints live."
    )
