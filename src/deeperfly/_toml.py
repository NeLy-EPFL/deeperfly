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

__all__ = [
    "key",
    "quote",
    "scalar",
    "value",
    "table_lines",
    "extract_section",
    "extract_tables",
    "top_level_tables",
]

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
    """One TOML value: a scalar, or an array of them -- nested to any depth.

    Recursive rather than flat-only because ``[skeleton].point_symmetries`` is an array of
    2-element arrays. A ``str`` is a scalar, never iterated.
    """
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(value(x) for x in v) + "]"
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
    comment (the packaged config discusses ``[default_camera]`` before declaring it) is
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
    headers = _header_scan(lines)
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if headers[i] == name and line.strip() == f"[{name}]"
        ),
        None,
    )
    if start is None:
        raise ValueError(f"no [{name}] table in this TOML text")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        # The next top-level header ends the section; `[name.sub]` belongs to it. Read from
        # the depth-aware scan, so a `symmetries = [ ["a","b"], ... ]` row does not read as
        # one.
        if headers[i] is not None and headers[i] != name:
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


def _header_name(line: str) -> str | None:
    """The top-level table name a header line declares, or ``None`` if it is not a header.

    Handles both ``[a.b]`` and the array-of-tables ``[[a]]``, returning the *first* path
    segment in each case -- which is the granularity a section extractor works at.

    Line-local, so it cannot tell a header from a *continuation line of a multi-line
    array* that happens to begin with ``[`` -- which ``[skeleton].point_symmetries`` does, one
    ``["lf_pretarsus", "rf_pretarsus"],`` row per line. Callers must go through
    :func:`_header_scan`, which supplies the missing bracket-depth context.
    """
    stripped = line.strip()
    if not stripped.startswith("["):
        return None
    inner = stripped.lstrip("[").rstrip("]").strip()
    if not inner:
        return None
    return inner.split(".", 1)[0].strip().strip('"')


def _bracket_delta(line: str) -> int:
    """Net ``[`` minus ``]`` on ``line``, ignoring comments and quoted strings.

    Only used to decide whether the *next* line is inside a multi-line array, so a header
    line's own balanced brackets cancel and contribute nothing.

    Single-line quoting only: TOML's multi-line basic/literal strings (``\"\"\"``) would
    need a running state this package has no use for, and the configs here do not use them.
    """
    depth = 0
    quote = ""
    for ch in line:
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
        elif ch == "#":
            break
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
    return depth


def _header_scan(lines: list[str]) -> list[str | None]:
    """Per line, the top-level table it declares -- ``None`` for everything else.

    The one place the "is this a header?" question is answered, because answering it
    per-line is wrong in a way that is invisible until some config grows a multi-line array
    of arrays: a comment line, and any line reached at nonzero bracket depth, cannot be a
    header however much it looks like one.
    """
    out: list[str | None] = []
    depth = 0
    for line in lines:
        stripped = line.lstrip()
        commented = stripped.startswith("#")
        name = None if (commented or depth > 0) else _header_name(line)
        out.append(name)
        if not commented:
            depth = max(0, depth + _bracket_delta(line))
    return out


def top_level_tables(text: str) -> list[str]:
    """Every top-level table name declared in ``text``, in order of first appearance.

    Header lines only -- a name *mentioned* in a comment is not a declaration, which is the
    distinction that makes this usable on the packaged config (whose prose discusses tables
    before declaring them).
    """
    seen: list[str] = []
    for name in _header_scan(text.splitlines()):
        if name and name not in seen:
            seen.append(name)
    return seen


def extract_tables(text: str, names) -> str:
    """Every ``[name]`` / ``[[name]]`` / ``[name.*]`` block for ``names``, verbatim and in order.

    The composition primitive behind a project's ``rig.toml``: a *fragment* holding whole
    top-level tables lifted out of a config with their comments intact. Fragments that own
    disjoint table names can then simply be concatenated -- no TOML writer involved, so
    nothing can be silently mis-serialized, and every comment survives.

    Parameters
    ----------
    text
        TOML source.
    names
        Top-level table names to keep.

    Returns
    -------
    str
        The kept blocks, newline-terminated (empty when none matched).
    """
    wanted = set(names)
    lines = text.splitlines()
    headers = _header_scan(lines)
    out: list[str] = []
    keeping = False
    # Comments immediately above a header introduce it, so they are buffered and emitted
    # with the block they belong to -- the same convention extract_section relies on.
    pending: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        name = headers[i]
        if name is not None:
            keeping = name in wanted
            if keeping:
                out.extend(pending)
                out.append(line)
            pending = []
            continue
        if stripped.startswith("#") or not stripped:
            pending.append(line)
            if keeping:
                out.extend(pending)
                pending = []
            continue
        pending = []
        if keeping:
            out.append(line)
    while out and not out[-1].strip():
        out.pop()
    return ("\n".join(out) + "\n") if out else ""
