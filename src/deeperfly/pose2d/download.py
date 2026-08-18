"""Find the detector's weights: the auto-provisioned DeepFly2D cache, and everything else.

Two jobs. The published DeepFly2D checkpoint is downloaded on first use and cached
per-user (the detector loads it directly, no conversion). Every other detector is trained
per project, so there is nothing to download -- for those, :func:`resolve_weights` turns
what a config wrote into a file on disk, and says exactly what is missing when it cannot.

The point of the search path is that a *recording's* config should not have to carry a
machine's directory layout. ``weights = "mvt_alt8_fly38b.pth"`` is a fact about which
model a run used and travels with the recording; ``/mnt/upramdya_data/TL/...`` is a fact
about one mount on one machine and breaks the moment the config is opened anywhere else.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path

import platformdirs

log = logging.getLogger("deeperfly")

#: Environment variable naming where per-project checkpoints live: one directory, or
#: several separated by ``os.pathsep``. Searched (then the download cache) for a
#: ``weights`` value written as a bare filename.
MODELS_ENV = "DEEPERFLY_MODELS"

# Original DeepFly2D stacked-hourglass weights (from df2d/inference.py). The
# upstream release is a legacy PyTorch pickle the file name calls ``.tar`` (it is
# not a tar archive); we cache it locally as ``.pth`` to match torch convention.
TORCH_WEIGHTS_URL = "https://www.dropbox.com/s/csgon8uojr3gdd9/sh8_front_j8.tar?dl=1"
TORCH_WEIGHTS_NAME = "sh8_deepfly.pth"


def cache_dir() -> Path:
    """Per-user cache directory for deeperfly weights (created on demand)."""
    d = Path(platformdirs.user_cache_dir("deeperfly")) / "weights"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_torch_weights(*, force: bool = False, sha256: str | None = None) -> Path:
    """Download the original PyTorch checkpoint to the cache and return its path.

    Parameters
    ----------
    force
        Re-download even when the cached file already exists.
    sha256
        Optional expected checksum to verify the download against.

    Returns
    -------
    Path
        The cached checkpoint path.

    Raises
    ------
    ValueError
        If ``sha256`` is given and the download fails verification.
    """
    dest = cache_dir() / TORCH_WEIGHTS_NAME
    if dest.exists() and not force:
        return dest
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(TORCH_WEIGHTS_URL, tmp)
    if sha256 is not None and _sha256(tmp) != sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError("downloaded weights failed checksum verification")
    tmp.replace(dest)
    return dest


def torch_weights_path() -> Path:
    """Expected path of the cached PyTorch checkpoint (``.pth``)."""
    return cache_dir() / TORCH_WEIGHTS_NAME


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

    * empty / absent -> ``None``. The caller's class decides what that means: the
      hourglass auto-provisions its published checkpoint, a per-project class refuses.
    * a bare filename (``mvt_alt8_fly38b.pth``) -> searched along :func:`search_path`.
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

    Its own function because all three dense loaders raise it and the message is the
    entire user experience of a fresh install: there is no checkpoint to download, so the
    error has to *be* the setup instructions.
    """
    return SystemExit(
        f"[[pose2d.models]] {model_name!r} (class {cls!r}) has no 'weights', and there "
        "is no auto-provisioned checkpoint for this class -- it is trained per project.\n"
        "  Either name a file on the search path:\n"
        f"    export {MODELS_ENV}=/path/to/models\n"
        '    weights = "my_detector.pth"     # in [[pose2d.models]]\n'
        "  ...or write the path outright:\n"
        '    weights = "/path/to/my_detector.pth"\n'
        '  (The one detector needing no checkpoint is `class = "hourglass"`, whose '
        "published DeepFly2D weights are downloaded and cached on first use -- but it "
        "predicts 19 channels, one body side per pass, so switching to it also means the "
        "`fly38` skeleton and an explicit [pose2d.output_points] table. See the "
        "configuration reference.)"
    )
