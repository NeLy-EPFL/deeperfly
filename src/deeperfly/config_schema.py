"""The config, described from the code -- so nothing can drift from it.

The run config is comprehensive and, at 706 lines across 50 tables, overwhelming: someone
who wants to change ``[triangulation].method`` is reading a file that also declares 38
keypoints, seven footage sources and 132 detector channel mappings. The fix is not a
prettier file. It is being able to ask *what can I set, what does it default to, and what
does it mean* -- and to set one thing without reading the rest.

The load-bearing observation is that **the answer already exists in the code**. Every
knob-bearing section is a frozen ``*Params`` dataclass (:class:`~deeperfly.config.Pose2dParams`,
:class:`~deeperfly.config.TriangulationParams`, ...) whose fields carry names, types and
defaults, and whose class docstring explains them -- often better than a form's help text
ever would. ``InverseKinematicsParams`` spends two paragraphs on why ``damping`` defaults to
0.1 (the abdomen is five near-collinear hinges, so a lightly-damped Gauss-Newton step
overshoots into the joint limits and deadlocks).

So this module *derives* the schema rather than restating it:

- :func:`describe` reflects over ``dataclasses.fields`` plus the parsed docstring;
- :func:`sections` lists the describable sections;
- validation is **not** reimplemented -- ``Config``'s own strict ``_params`` loader is the
  only validator, so a rejected key is rejected identically however it arrived.

A new field therefore appears in ``deeperfly config show`` (and in the GUI forms built on
this) with no second place to update. The sections this cannot describe -- the detection
plan, the video specs -- are open-ended by nature, and pretending otherwise with a
half-schema would be worse than admitting they need the file.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field

from .config import (
    STAGE_DEFAULTS,
    AnnotationParams,
    BundleAdjustmentParams,
    Config,
    GuiParams,
    InverseKinematicsParams,
    PictorialParams,
    Pose2dParams,
    TriangulationParams,
)

__all__ = ["FieldSpec", "SectionSpec", "describe", "sections", "SECTIONS", "effective"]


#: The describable config sections: TOML table -> the dataclass that defines its keys.
#:
#: Deliberately not exhaustive. ``[pose2d]``'s *detection plan* sub-tables, ``[cameras]``,
#: ``[skeleton]`` and ``[[visualization.videos]]`` are structural or open-ended -- a schema
#: for them would be a fiction, and they belong in a file (or, for the skeleton and rig, in
#: the project's own editors).
SECTIONS: dict[str, type] = {
    "pose2d": Pose2dParams,
    "triangulation": TriangulationParams,
    "pictorial_structures": PictorialParams,
    "bundle_adjustment": BundleAdjustmentParams,
    "inverse_kinematics": InverseKinematicsParams,
    "annotation": AnnotationParams,
    "gui": GuiParams,
}


@dataclass(frozen=True)
class FieldSpec:
    """One settable key: what it is, what it defaults to, and what it means."""

    name: str
    type: str
    default: object
    doc: str = ""
    choices: tuple | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "default": self.default,
            "doc": self.doc,
            "choices": list(self.choices) if self.choices else None,
        }


@dataclass(frozen=True)
class SectionSpec:
    """One config table's fields, plus the prose that explains the table as a whole."""

    name: str
    doc: str = ""
    fields: tuple[FieldSpec, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "doc": self.doc,
            "fields": [f.as_dict() for f in self.fields],
        }


def sections() -> list[str]:
    """The section names :func:`describe` understands, in a sensible reading order."""
    return list(SECTIONS)


def describe(section: str) -> SectionSpec:
    """Reflect one config section into a :class:`SectionSpec`.

    Parameters
    ----------
    section
        A key of :data:`SECTIONS` (e.g. ``"triangulation"``).

    Returns
    -------
    SectionSpec
        Its fields, with defaults, types and per-field prose lifted from the dataclass
        docstring.

    Raises
    ------
    KeyError
        If the section has no schema, naming the ones that do -- an honest "this one needs
        the file" rather than an empty form.
    """
    try:
        cls = SECTIONS[section]
    except KeyError:
        raise KeyError(
            f"no schema for [{section}]; describable sections are {sections()}. The "
            "detection plan, cameras, skeleton and video specs are open-ended and live "
            "in the config file (or, for the skeleton and rig, in the project)"
        ) from None

    summary, per_field = _parse_doc(cls.__doc__ or "")
    specs = []
    for f in dataclasses.fields(cls):
        default = f.default
        if default is dataclasses.MISSING:
            default = (
                f.default_factory()  # type: ignore[misc]
                if f.default_factory is not dataclasses.MISSING
                else None
            )
        specs.append(
            FieldSpec(
                name=f.name,
                type=_type_name(f.type),
                default=default,
                doc=per_field.get(f.name, ""),
                choices=f.metadata.get("choices"),
            )
        )
    return SectionSpec(name=section, doc=summary, fields=tuple(specs))


def effective(config: Config, section: str) -> dict:
    """``field -> (value, is_default)`` for one section of a loaded config.

    Answers the question a config file cannot: *which of these did I actually set?* A
    706-line file where 690 lines are defaults reads as 706 decisions; this separates the
    handful that were made from the rest.

    Raises
    ------
    KeyError
        If the section has no schema.
    ValueError
        If the section holds an unknown key -- raised by ``Config``'s own strict loader,
        deliberately not reimplemented here, so a typo is reported identically however it
        arrived.
    """
    spec = describe(section)
    params = _params_of(config, section)
    raw = config.data.get(section, {}) or {}
    out = {}
    for f in spec.fields:
        value = getattr(params, f.name, f.default)
        out[f.name] = (value, f.name not in raw)
    return out


def _params_of(config: Config, section: str):
    """The typed ``*Params`` for a section, via ``Config``'s own accessors."""
    accessor = {
        "pose2d": "pose2d",
        "triangulation": "triangulation",
        "pictorial_structures": "pictorial",
        "bundle_adjustment": "bundle_adjustment",
        "inverse_kinematics": "inverse_kinematics",
        "annotation": "annotation",
        "gui": "gui",
    }[section]
    return getattr(config, accessor)


def stage_flags_spec() -> SectionSpec:
    """``[pipeline]``'s ``do_<stage>`` booleans, which are generated rather than declared.

    They are not a dataclass -- :meth:`Config.stage_flags` builds them from
    :data:`~deeperfly.config.STAGES` -- so they are described separately rather than
    omitted, since "which stages run" is the single most-changed thing in the whole config.
    """
    return SectionSpec(
        name="pipeline",
        doc="Which stages run. Each is independently toggled and reads its own "
        "[<stage>] table.",
        fields=tuple(
            FieldSpec(
                name=f"do_{stage}",
                type="bool",
                default=default,
                doc=f"Run the {stage.replace('_', ' ')} stage.",
                choices=(True, False),
            )
            for stage, default in STAGE_DEFAULTS.items()
        ),
    )


# -- docstring parsing ---------------------------------------------------------
#
# The `*Params` docstrings are prose, not a machine format, but they follow one consistent
# habit: a field is introduced as ``field_name`` (double-backtick) and discussed until the
# next such introduction. That is enough to attribute paragraphs to fields without asking
# anyone to rewrite them into a schema -- and the prose is the good part, so re-authoring it
# as terse help text would be a downgrade.

_FIELD_MENTION = re.compile(r"``([a-z_][a-z0-9_]*)``")


def _plain(text: str) -> str:
    """RST-ish docstring prose as plain text.

    The docstrings are written for Sphinx, so they carry ``literals`` and :role:`targets`.
    Both are noise in a terminal table or an HTML form field, and the double backticks in
    particular survive rich's markup pass looking like a typo.
    """
    out = re.sub(
        r":[a-z:]+:`~?([^`]+)`", r"\1", text
    )  # :func:`~pkg.thing` -> pkg.thing
    out = out.replace("``", "")
    return re.sub(r"\s+", " ", out).strip()


def _parse_doc(doc: str) -> tuple[str, dict[str, str]]:
    """``(summary, {field: prose})`` from a ``*Params`` docstring.

    The summary is the first paragraph (usually ``[section] -- one line``). Per-field prose
    is attributed by the first ``field`` mention that starts a sentence, which is the
    convention these docstrings already follow. Attribution is best-effort by design: a
    missing entry costs a field its help text, while a wrong *parse* that raised would cost
    the whole schema.
    """
    text = doc.strip()
    if not text:
        return "", {}
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    summary = _plain(paragraphs[0])
    per_field: dict[str, str] = {}
    for para in paragraphs[1:]:
        flat = " ".join(para.split())
        plain = _plain(para)
        # A paragraph "belongs" to the first field it names at (or very near) its start;
        # later mentions in the same paragraph are cross-references, not introductions.
        match = _FIELD_MENTION.search(flat)
        if match is None or match.start() > 40:
            continue
        name = match.group(1)
        per_field[name] = (
            plain if name not in per_field else f"{per_field[name]} {plain}"
        )
    return summary, per_field


def _type_name(annotation) -> str:
    """A short, human-readable type name for a dataclass field annotation."""
    text = (
        annotation
        if isinstance(annotation, str)
        else getattr(annotation, "__name__", str(annotation))
    )
    return str(text).replace("typing.", "").strip()
