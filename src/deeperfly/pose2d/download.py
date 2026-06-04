"""Fetch and cache the pretrained DeepFly2D weights.

Downloads the original PyTorch checkpoint on first use and caches it per-user; the
detector loads it directly (no conversion).
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

import platformdirs

log = logging.getLogger("deeperfly")

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
    """Download the original PyTorch checkpoint to the cache and return its path."""
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
