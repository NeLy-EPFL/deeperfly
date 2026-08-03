"""A small, focused TOML writer -- and a section extractor -- for deeperfly's own files.

``tomllib`` reads TOML but does not write it, and the alternative is a core dependency
for a handful of fixed, shallow schemas (``calibration.toml``, ``project.toml``). So
this module writes the subset those schemas need: scalars, flat arrays of scalars, and
nested tables.

Two properties are load-bearing rather than incidental:

**Floats round-trip exactly.** Values go through :func:`repr`, which is the shortest
representation that parses back to the same double -- so ``save -> load -> save`` is
byte-stable, and a stored camera rig is not quietly rounded on every rewrite. ``repr``
also already emits TOML's own ``nan`` / ``inf`` spellings.

**Scalars are emitted before sub-tables.** TOML binds a bare ``key = value`` to the most
recent ``[table]`` header, so writing a sub-table before a sibling scalar would silently
*reparent* that scalar into it. :func:`table_lines` sorts them, which is what makes the
output round-trip at all.

This is deliberately not a general TOML serializer: it does not do inline tables, arrays
of tables, multi-line strings, or dates. Anything it cannot express should be left out of
a schema rather than worked around here -- a file that lies about its contents is worse
than one that omits them.
"""

from __future__ import annotations

import re

import numpy as np

__all__ = ["key", "quote", "scalar", "value", "table_lines", "extract_section"]

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def key(name) -> str:
    """A TOML key: bare when it can be, quoted otherwise."""
    text = str(name)
    return text if _BARE_KEY.match(text) else quote(text)


def quote(text: str) -> str:
    """A TOML basic string with the escapes the spec requires."""
    out = str(text).replace("\\", "\\\\").replace('"', '\\"')
    out = out.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return f'"{out}"'


def scalar(v) -> str:
    """One TOML scalar. ``bool`` is checked before ``int`` -- it is a subclass of it."""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        # repr() round-trips exactly and already emits TOML's `nan` / `inf` spellings.
        return repr(float(v))
    return quote(v)


def value(v) -> str:
    """One TOML value: a scalar, or a flat array of them."""
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(scalar(x) for x in v) + "]"
    return scalar(v)


def table_lines(path: list[str], mapping: dict) -> list[str]:
    """A ``[a.b]`` table and its sub-tables, scalars first so no key outlives its header.

    Parameters
    ----------
    path
        The table's key path, e.g. ``["calibration", "provenance"]``.
    mapping
        The table's contents. ``dict`` values become sub-tables; everything else is a
        scalar or a flat array.

    Returns
    -------
    list of str
        The lines to join with newlines.
    """
    scalars = {k: v for k, v in mapping.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in mapping.items() if isinstance(v, dict)}
    lines = ["[" + ".".join(key(p) for p in path) + "]"]
    lines += [f"{key(k)} = {value(v)}" for k, v in scalars.items()]
    for name, sub in tables.items():
        lines.append("")
        lines += table_lines([*path, str(name)], sub)
    return lines


def extract_section(text: str, name: str) -> str:
    """The ``[name]`` table and its ``[name.*]`` sub-tables, verbatim, from TOML ``text``.

    Used to lift a section out of the packaged config with its **comments intact** --
    which is the point: the ``[skeleton]`` block's prose explains the limb ordering and
    the palette convention, and a project seeded from it should carry that, not a
    machine-serialized equivalent.

    Header matching is anchored to the start of a line, so a section *mentioned* inside a
    comment (the packaged config discusses ``[cameras.defaults]`` before declaring it) is
    not mistaken for the declaration.

    Parameters
    ----------
    text
        TOML source.
    name
        The top-level table name, without brackets.

    Returns
    -------
    str
        The section text, ending in a single newline.

    Raises
    ------
    ValueError
        If ``text`` has no ``[name]`` table.
    """
    lines = text.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.strip() == f"[{name}]"),
        None,
    )
    if start is None:
        raise ValueError(f"no [{name}] table in this TOML text")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        stripped = lines[i].lstrip()
        # The next top-level header ends the section; `[name.sub]` belongs to it.
        if stripped.startswith("[") and not stripped.startswith(
            (f"[{name}.", f"[[{name}.")
        ):
            end = i
            break
    # The comment block sitting immediately above the next header documents *that*
    # section, not this one -- TOML has no way to say so, but the convention is
    # universal. Without this, extracting [skeleton] from the packaged config carries
    # away the banner introducing [cameras], and a project's skeleton.toml ships with a
    # paragraph about camera rigs.
    while end > start + 1 and (
        not lines[end - 1].strip() or lines[end - 1].lstrip().startswith("#")
    ):
        end -= 1
    return "\n".join(lines[start:end]) + "\n"
