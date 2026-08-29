"""``deeperfly config`` -- discover and set config keys without reading the whole file.

The packaged config is 706 lines across 50 tables. Someone changing one triangulation knob
should not have to read 132 detector channel mappings to find it, and should not have to
guess what the knob means. Both come from :mod:`deeperfly.config.schema`, which derives the
answer from the dataclasses that already define it -- so this cannot drift from the code.

``show`` distinguishes **set** from **default**, which a config file cannot: a 706-line file
where 690 lines are defaults reads as 706 decisions.
"""

from __future__ import annotations

import logging
import re
from typing import Annotated

import typer
from rich.table import Table
from rich.text import Text

from ..config import DEFAULT_CONFIG_PATH, Config
from ..config.schema import describe, effective, sections, stage_flags_spec
from .console import LogLevel, LogLevelOption, _configure_logging, console

log = logging.getLogger("deeperfly")


def _load(path: str | None) -> Config:
    """The config at ``path``, else the packaged default."""
    return Config.from_toml(path) if path else Config.default()


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "—"
    if isinstance(value, (list, tuple)):
        return "[]" if not value else str(list(value))
    if isinstance(value, dict):
        return "{}" if not value else str(value)
    return str(value)


config_app = typer.Typer(
    no_args_is_help=True,
    help="Discover and set config keys without reading the whole file. Every key, its "
    "default and its documentation are derived from the code, so they cannot drift from "
    "it.",
)


@config_app.command("show")
def config_show(
    section: Annotated[
        str | None,
        typer.Argument(
            help="one section (e.g. triangulation, pipeline, annotation); omit for all"
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option("-c", "--config", help="config TOML (default: the packaged one)"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("-v", "--verbose", help="also print what each key means"),
    ] = False,
    log_level: LogLevelOption = LogLevel.warning,
) -> None:
    """Print a config section's keys, values, defaults and documentation.

    Keys you actually set are marked; everything else is a default. That distinction is
    what a config file cannot show you -- a 706-line file where 690 lines are defaults
    reads as 706 decisions.

    The detection plan, cameras, skeleton and video specs are open-ended and are not
    described here; they live in the file (or, for the skeleton and rig, in the project).
    """
    _configure_logging(log_level.value)
    config = _load(config)
    wanted = [section] if section else sections()

    if section in (None, "pipeline"):
        # `[pipeline]` has no params dataclass, so its rows are built here rather than by
        # `effective()`. Each row is `(value, is_default)` rather than a bare bool;
        # getting that wrong prints an em-dash for every flag, which is what it did -- and
        # which reads as "unset" for exactly the keys a reader is most likely to be
        # checking. (The field name and the stage name are the same word now.)
        declared = config.data.get("pipeline", {}) or {}
        _print_section(
            stage_flags_spec(),
            {
                stage: (on, stage not in declared)
                for stage, on in config.stage_flags().items()
            },
        )
        if section == "pipeline":
            return
        wanted = [s for s in wanted if s != "pipeline"]

    for name in wanted:
        try:
            spec = describe(name)
            values = effective(config, name)
        except KeyError as exc:
            raise SystemExit(str(exc).strip("'")) from None
        except (
            ValueError
        ) as exc:  # an unknown key in the file -- Config's own validator
            raise SystemExit(f"[{name}] in this config is invalid: {exc}") from None
        _print_section(spec, values, verbose=verbose)


def _print_section(
    spec, values, *, verbose: bool = False, all_default: bool = False
) -> None:
    # Text, not markup: the title is a TOML table name in square brackets, which rich
    # would otherwise parse as a style tag and swallow.
    table = Table(title=Text(f"[{spec.name}]", style="bold cyan"))
    table.add_column("key", style="bold")
    table.add_column("value")
    table.add_column("", width=3)  # set / default marker
    table.add_column("type", style="dim")
    if verbose:
        table.add_column("meaning")
    for f in spec.fields:
        raw = values.get(f.name)
        if isinstance(raw, tuple):
            value, is_default = raw
        else:
            value, is_default = raw, all_default
        row = [
            f.name,
            _fmt(value),
            "" if is_default else "[green]set[/green]",
            f.type,
        ]
        if verbose:
            row.append(f.doc or "")
        table.add_row(*row)
    console.print(table)
    if spec.doc:
        console.print(f"  {spec.doc}", markup=False, highlight=False, style="dim")


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="SECTION.KEY, e.g. triangulation.method")],
    value: Annotated[str, typer.Argument(help="the new value")],
    config: Annotated[
        str,
        typer.Option("-c", "--config", help="the config TOML to edit (required)"),
    ],
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Set one config key, validated the same way a run would validate it.

    Appends rather than rewriting, so comments survive. The result is loaded through the
    same strict validator a run uses, so this cannot write a key a run would reject.
    """
    _configure_logging(log_level.value)
    import tomllib

    from .. import _toml

    if "." not in key:
        raise SystemExit(
            f"expected SECTION.KEY (e.g. triangulation.method), got {key!r}"
        )
    section, key = key.split(".", 1)
    try:
        spec = describe(section) if section != "pipeline" else stage_flags_spec()
    except KeyError as exc:
        raise SystemExit(str(exc).strip("'")) from None
    known = {f.name for f in spec.fields}
    if key not in known:
        raise SystemExit(f"[{section}] has no key {key!r}; it accepts {sorted(known)}")

    path = config
    if path is None:
        raise SystemExit(
            "pass -c/--config: refusing to edit the packaged default in place "
            f"({DEFAULT_CONFIG_PATH}). 'deeperfly init' writes a copy to edit"
        )
    text = open(path).read()
    value = _coerce(value)
    addition = (
        f"\n# set by 'deeperfly config set {key}'\n"
        f"[{section}]\n{_toml.key(key)} = {_toml.value(value)}\n"
    )
    # Three cases, and the middle one is why this is not just an append. TOML forbids
    # declaring a table twice, so a new key can only be appended when its table is absent.
    # When the table AND the key are already there, the operation is an in-place value
    # rewrite -- which reorders nothing, and is the only way to change a key the packaged
    # config states. (It states every `[pipeline]` flag, so without this there is no way to
    # turn a default-on stage off.) A table that exists without the key is still refused:
    # appending a bare key after an existing header would reparent whatever follows it.
    if _has_table(text, section):
        candidate, n = re.subn(
            rf"(?m)^(\s*{re.escape(key)}\s*=\s*)\S.*$",
            lambda m: f"{m.group(1)}{_toml.value(value)}",
            text,
            count=1,
        )
        if n != 1:
            raise SystemExit(
                f"[{section}] already exists in {path} but does not state {key!r}. "
                "Appending a bare key after an existing table header would reparent the "
                f"keys below it, so add it under [{section}] by hand -- "
                f"'deeperfly config show --config {path} {section} -v' prints what it means"
            )
    else:
        candidate = text + addition
    try:
        parsed = tomllib.loads(candidate)
        cfg = Config.from_dict(parsed)
        # `[pipeline]` is not a params dataclass -- it is the stage toggles, read through
        # `stage_flags()`. Validating it through the accessor table would have looked up a
        # `Config.pipeline` that does not exist, which is the other half of why setting a
        # stage flag could never work.
        if section == "pipeline":
            cfg.stage_flags()
        else:
            cfg.__getattribute__(
                {"pictorial_structures": "pictorial"}.get(section, section)
            )
    except Exception as exc:
        raise SystemExit(f"{key} = {value!r} is not valid: {exc}") from None

    open(path, "w").write(candidate)
    console.print(f"[green]set[/green] {key} = {_fmt(value)} in {path}")


def _has_table(text: str, section: str) -> bool:
    """Whether ``[section]`` is declared at the start of a line (not merely mentioned)."""
    return any(line.strip() == f"[{section}]" for line in text.splitlines())


def _coerce(raw: str):
    """A CLI string as the TOML scalar it looks like."""
    lowered = raw.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw
