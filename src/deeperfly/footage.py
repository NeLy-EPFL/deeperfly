"""One way to record where a camera's footage is, shared by every artifact that records it.

The same fact -- *these files are this camera's footage* -- was written three different ways:

.. code-block:: text

    results.h5        {"abs": [...], "rel": [...]}    both flavors, always
    recording.toml    abs = [...]  XOR  names = [...]  never both, never rel
    .dfpkg            nothing at all (its docstring claimed basenames + sizes)

and the readers disagreed to match: the frame resolver understood ``abs``/``rel`` but not
``names``, while the project's own reader understood ``abs``/``names`` but not ``rel``. Two
call sites fabricated a ``{"abs": [...]}`` wrapper purely to borrow the resolver.

One record, four keys, and the same keys whether it lands in an HDF5 attribute's JSON or a
TOML sub-table:

.. code-block:: text

    abs    resolved absolute paths, as seen at write time
    rel    paths relative to THE DIRECTORY OF THE FILE CARRYING THE POINTER
    names  basenames -- the only flavor that survives an archive
    bytes  size per file: the recording id's own input, previously stored nowhere

Three properties keep this drop-in. The anchor is always "this file's directory", which was
already true of ``results.h5`` and already the argument both project call sites passed. The
keys are **additive**, and every reader used ``.get``, so no format version moves and old
files keep loading. And :func:`basenames` is bit-identical to what the labels identity
already stored, so **no label is invalidated** by adopting this.

``bytes`` earns its place separately: :func:`deeperfly.project.recording_fingerprint` hashes
basename + byte size, and nothing on disk kept the sizes -- so a ``rec_`` id could neither be
re-derived nor explained once the footage was gone.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

__all__ = ["write_pointer", "resolve", "basenames", "sizes"]

log = logging.getLogger("deeperfly")


def write_pointer(files, anchor: str | Path) -> dict:
    """The canonical pointer for one camera's ``files``, anchored at ``anchor``.

    Parameters
    ----------
    files
        The camera's footage paths, in order.
    anchor
        The directory of the file this pointer will be stored in -- what ``rel`` is
        relative to.

    Returns
    -------
    dict
        ``{"abs", "rel", "names", "bytes"}``. Every key is always present, so a reader never
        has to guess which flavor a writer chose. ``bytes`` carries ``-1`` for a file that
        does not currently resolve, which is distinguishable from a real size of 0.
    """
    anchor = Path(anchor)
    resolved = [Path(p).resolve() for p in files]
    out_bytes = []
    for p in resolved:
        try:
            out_bytes.append(int(p.stat().st_size))
        except OSError:
            out_bytes.append(-1)
    return {
        "abs": [str(p) for p in resolved],
        "rel": [os.path.relpath(p, anchor) for p in resolved],
        "names": [p.name for p in resolved],
        "bytes": out_bytes,
    }


def resolve(
    spec, anchor: str | Path, footage_dir: str | Path | None = None
) -> list[Path] | None:
    """One camera's files, from any pointer flavor. ``None`` when none of them resolve.

    Tried in order: ``abs``; ``rel`` against ``anchor``; ``names`` under ``footage_dir``;
    and finally ``names`` against ``anchor``, which is what makes a recording adopted with
    only basenames openable at all without ``--footage-dir``.

    Accepts a bare list as well as a mapping, so a caller holding plain paths does not have
    to wrap them.
    """
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        spec = {"abs": [str(p) for p in spec]}
    anchor = Path(anchor)

    attempts = [
        [Path(p) for p in (spec.get("abs") or [])],
        [anchor / r for r in (spec.get("rel") or [])],
    ]
    names = list(spec.get("names") or [])
    if not names:
        # Older pointers carry no `names`; derive them, so the by-name fallbacks still work.
        names = [Path(p).name for p in (spec.get("abs") or spec.get("rel") or [])]
    if footage_dir is not None:
        attempts.append([Path(footage_dir) / n for n in names])
    # Beside the file carrying the pointer: an archived recording whose only surviving
    # flavor is `names` resolves here instead of against the process's working directory,
    # which is what a bare basename used to be tried against.
    attempts.append([anchor / n for n in names])

    for candidate in attempts:
        if candidate and all(p.exists() for p in candidate):
            return candidate
    return None


def basenames(spec) -> list[str]:
    """One camera's footage basenames, **sorted** -- the identity projection.

    Deliberately byte-identical to what :func:`deeperfly.labels.store.labels_identity` already
    stored, so adopting the canonical pointer cannot invalidate a single existing label.
    Prefers ``rel`` over ``abs`` for the same reason the old readers did.
    """
    if spec is None:
        return []
    if isinstance(spec, (list, tuple)):
        paths = list(spec)
    else:
        paths = []
        for key in ("rel", "abs", "names"):
            paths = list(spec.get(key) or [])
            if paths:
                break
    return sorted(os.path.basename(str(p)) for p in paths)


def sizes(spec) -> list[int]:
    """One camera's recorded file sizes, or ``[]`` when the pointer predates ``bytes``."""
    if isinstance(spec, dict):
        return [int(b) for b in (spec.get("bytes") or [])]
    return []
