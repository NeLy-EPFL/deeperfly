"""The binding: where each tracked point of one skeleton sits on one model.

A skeleton says what is tracked; a model pack
(:mod:`deeperfly.inverse_kinematics.pack`) says what is fitted. Neither can say where a
tracked point sits on the model, because that is a fact about the **pair** -- how this
skeleton was labelled against this model -- so it lives in a third artifact keyed on
both halves, ``data/bindings/<skeleton>@<model>.toml``.

That is what makes the two independent in both directions. Without it a pack has to
reach into the skeleton's namespace (a leg template composing point names from its leg
ids) and a skeleton cannot reach a second model at all; with it, binding ``fly38`` to
flybody means writing one more file and changing neither half.

**This is the only artifact where a skeleton point name and a model body name may
appear together**, which is a greppable invariant and a test
(``test_the_binding_is_the_only_place_the_two_namespaces_meet``).

One row per tracked point::

    lf_femur_tibia = { body = "lf_tibia" }
    lf_pretarsus        = { body = "lf_tarsus5", offset = [0.0, 0.0, -0.107] }
    neck           = { body = "c_head", base = true }
    abdomen0       = { body = "c_abdomen12", offset = [-0.37, 0.0, 0.34], approximate = true }

``offset`` is in the body's own frame and defaults to zero, so most rows are just a
body. The two flags are what the row cannot be without:

``base``
    This point is its chain's base landmark -- it says where the chain sits and
    constrains none of its DOFs, because it lies on the very pivot they rotate about.
    Nominated rather than derived: a point at a chain's own origin is the one case
    where "is this a joint" is not a geometric question.
``approximate``
    The placement is a modelling decision rather than a measurement -- the point has no
    exact counterpart on the model. **Carried and reported, never acted on**: the fit's
    residual splits exact rows from approximate ones so a reader can tell a bad fit from
    a bad retarget. Down-weighting them, or fitting their offsets, would absorb real
    error into the retarget.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["Binding", "BindingRow", "BINDING_DIR", "bindings"]

#: Where the packaged bindings live, one file per (skeleton, model) pair.
BINDING_DIR = Path(__file__).parent.parent / "data" / "bindings"


def bindings() -> dict[str, Path]:
    """The packaged bindings, ``"<skeleton>@<model>" -> file``."""
    if not BINDING_DIR.is_dir():
        return {}
    return {p.stem: p for p in sorted(BINDING_DIR.glob("*.toml"))}


@dataclass(frozen=True)
class BindingRow:
    """One tracked point's placement on the model."""

    point: str
    body: str
    offset: np.ndarray  # (3,) in the body's own frame
    approximate: bool = False
    base: bool = False

    @property
    def at_body_origin(self) -> bool:
        """Whether the point sits exactly on the body's origin (its parent joint)."""
        return not bool(np.any(self.offset))


@dataclass(frozen=True)
class Binding:
    """A loaded binding: the rows, in the file's order, plus both halves' names."""

    skeleton: str
    model: str
    rows: tuple[BindingRow, ...]
    #: ``body -> rows``, for the reverse lookup a template joint and an anchor need.
    #: A body may carry several points (a segment with two markers on it), so this is
    #: a list and the two lookups below say which one they mean.
    by_body: dict[str, tuple[BindingRow, ...]] = field(default_factory=dict)

    @classmethod
    def load(cls, ref: str | Path) -> "Binding":
        """Load a binding by ``"<skeleton>@<model>"`` or by path.

        Raises
        ------
        FileNotFoundError
            If the pair is not packaged and the path does not exist. The message names
            both halves, because an unbound pair is the normal way this fails and the
            fix is to write that one file.
        ValueError
            If two rows claim the same body *at its origin* -- then "which point
            observes this body" has no answer, and a leg joint or an anchor would
            resolve arbitrarily.
        """
        known = bindings()
        path = known.get(str(ref))
        if path is None:
            path = Path(ref)
        if not path.is_file():
            skeleton, _, model = str(ref).partition("@")
            raise FileNotFoundError(
                f"no binding for skeleton {skeleton!r} on model {model or '?'!r}: "
                f"nothing at data/bindings/{ref}.toml and no such path. A binding says "
                "where each tracked point sits on the model; write one (deeperfly ik "
                f"bind {skeleton} {model}) rather than fitting against no placement. "
                f"Packaged: {sorted(known)}"
            )
        spec = tomllib.loads(path.read_text())
        rows = []
        for point, r in spec.get("points", {}).items():
            rows.append(
                BindingRow(
                    point=str(point),
                    body=str(r["body"]),
                    offset=np.asarray(r.get("offset", (0.0, 0.0, 0.0)), dtype=float),
                    approximate=bool(r.get("approximate", False)),
                    base=bool(r.get("base", False)),
                )
            )
        by_body: dict[str, tuple[BindingRow, ...]] = {}
        for row in rows:
            by_body[row.body] = by_body.get(row.body, ()) + (row,)
        for body, group in by_body.items():
            at_origin = [r for r in group if r.at_body_origin]
            if len(at_origin) > 1:
                names = ", ".join(repr(r.point) for r in at_origin)
                raise ValueError(
                    f"binding {path.name} has two points at the origin of body "
                    f"{body!r} ({names}); which one observes it would then be arbitrary"
                )
        return cls(
            skeleton=str(spec.get("skeleton", "")),
            model=str(spec.get("model", "")),
            rows=tuple(rows),
            by_body=by_body,
        )

    def point_for(self, body: str) -> str | None:
        """The single point attached to ``body``, or ``None``.

        What a leg template's joint resolves through: it names the model body its joint
        sits at, and this says which tracked point observes it. A pretarsus resolves here
        despite its non-zero offset -- the offset is mesh geometry (where the tip is on
        the last tarsus), while the chain's segment length is measured from the data.
        ``None`` when the body carries no point, or more than one.
        """
        group = self.by_body.get(body, ())
        return group[0].point if len(group) == 1 else None

    def origin_point_for(self, body: str) -> str | None:
        """The point sitting exactly on ``body``'s origin, or ``None``.

        The stricter lookup, for the registration anchors: an anchor is a body whose
        *position* a tracked point reports, so a point carried at an offset on it is
        not an observation of it.
        """
        row = next((r for r in self.by_body.get(body, ()) if r.at_body_origin), None)
        return None if row is None else row.point

    def row(self, point: str) -> BindingRow | None:
        return next((r for r in self.rows if r.point == point), None)

    @property
    def approximate(self) -> tuple[str, ...]:
        """The points whose placement is a modelling decision, not a measurement."""
        return tuple(r.point for r in self.rows if r.approximate)
