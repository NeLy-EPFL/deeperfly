"""Calibration landmarks: the non-skeleton points that make a from-scratch rig solvable.

A rig can be solved from correspondences alone, but *which* correspondences decides
whether it converges. The obvious choice -- skeleton keypoints -- is the weak one, for a
reason worth stating plainly:

    A keypoint at frame *t* is a **different 3D point** from the same keypoint at frame
    *t+1*, because the animal moved. So *N* labeled frames of *P* keypoints add ``3*N*P``
    unknowns alongside the ``6*(V-1)`` camera unknowns, and every one of those points sits
    inside a 3 mm blob near the middle of the field.

A **static** landmark is the opposite: a scratch on the coverslip, the tip of the tether, a
dust speck on the glass. It is *one* 3D point observed in ``V*N`` images -- ``2*V*N``
equations against 3 unknowns -- and it is spread through the scene *volume* rather than
concentrated where the animal is. That is what conditions the solve.

So a landmark carries a ``static`` flag, and it changes the geometry rather than just the
bookkeeping (see :func:`deeperfly.calibration_solve.build_observations`):

.. code-block:: text

    static = true,  scope = "recording"   one 3D unknown per (recording, landmark)
    static = true,  scope = "rig"         one 3D unknown per landmark, shared across
                                          every recording on the rig -- the strongest
                                          constraint available, and the easiest to get
                                          wrong if the rig is bumped between sessions
    static = false                        one 3D unknown per (frame, landmark), like a
                                          skeleton point (a moving fiducial)

**Why landmarks are not skeleton points.** They live in their own namespace, keyed by name,
and never enter :class:`~deeperfly.skeleton.Skeleton`. That is deliberate and structural:
a landmark must never reach the detector's ``output_points``, the IK body plan, the
bone-length priors, the training export, or the rendered videos. Keeping them out of the
skeleton enforces that by construction, where a "landmark limb" *inside* the skeleton would
need a filter at every one of those call sites -- and would change the fingerprinted
``point_names``, invalidating every label already authored.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import _toml

__all__ = [
    "Landmark",
    "LandmarkSet",
    "LANDMARKS_FILENAME",
    "LANDMARKS_FORMAT_VERSION",
    "SCOPES",
]

log = logging.getLogger("deeperfly")

LANDMARKS_FILENAME = "landmarks.toml"

#: Bumped when this file's schema changes incompatibly. It needs a version more than the
#: other config fragments do: the landmark ORDER is an ``L`` axis that every stored
#: observation indexes into positionally, and unlike the skeleton there is no
#: ``skeleton_migrate`` equivalent for it and no fingerprint of it in ``labels.h5``. Until
#: now the axis survived only because ``labels.h5`` happens to store the landmark ``names``
#: alongside, making each file self-describing -- which is luck, not design.
LANDMARKS_FORMAT_VERSION = 1

#: How widely a static landmark's single 3D point is shared. ``"recording"`` is the safe
#: default; ``"rig"`` ties every recording on the rig into one solve.
SCOPES = ("recording", "rig")


@dataclass(frozen=True)
class Landmark:
    """One calibration landmark's definition.

    Attributes
    ----------
    name
        Unique identifier, used as the label key and in the solve report.
    static
        Whether this is one fixed 3D point over time (the useful case) rather than a
        per-frame one.
    scope
        For a static landmark, whether its point is shared across recordings
        (:data:`SCOPES`). Ignored when ``static`` is false -- a moving landmark cannot be
        shared across recordings, since there is nothing fixed to share.
    color
        Hex color for the editor overlay.
    note
        Free text: *which* scratch on the coverslip, so the next operator finds the same
        one. This is load-bearing for a rig-scoped landmark, where labeling a different
        speck in a second recording silently corrupts the solve.
    """

    name: str
    static: bool = True
    scope: str = "recording"
    color: str = "#e8a33d"
    note: str = ""

    @property
    def shared_across_recordings(self) -> bool:
        """Whether one 3D point serves every recording (static *and* rig-scoped)."""
        return bool(self.static) and self.scope == "rig"


@dataclass
class LandmarkSet:
    """A project's landmark definitions, in a stable order.

    The order is the ``L`` axis every stored landmark observation indexes into, so it is
    treated the same way as the skeleton's point order: appending is safe, reordering is a
    migration.
    """

    landmarks: list[Landmark] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.landmarks)

    def __iter__(self):
        return iter(self.landmarks)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.landmarks[key]
        return self.landmarks[self.index(key)]

    @property
    def names(self) -> list[str]:
        return [lm.name for lm in self.landmarks]

    @property
    def static_mask(self) -> list[bool]:
        return [bool(lm.static) for lm in self.landmarks]

    def index(self, name: str) -> int:
        """The ``L``-axis index of ``name``.

        Raises
        ------
        KeyError
            If no landmark has that name.
        """
        for i, lm in enumerate(self.landmarks):
            if lm.name == name:
                return i
        raise KeyError(f"no landmark named {name!r} (have {self.names})")

    # -- persistence ----------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> LandmarkSet:
        """Read a ``landmarks.toml``; an absent file is an empty set.

        Raises
        ------
        ValueError
            If an entry lacks a name, a name repeats, or a scope is unknown -- each of
            which would make a stored observation ambiguous rather than merely wrong.
        """
        p = Path(path)
        if p.is_dir():
            p = p / LANDMARKS_FILENAME
        if not p.exists():
            return cls()
        data = tomllib.loads(p.read_text())
        # An absent key means v1, matching every other versioned artifact -- so files written
        # before this existed keep loading and no migration is forced.
        version = int((data.get("landmarks") or {}).get("format_version", 1))
        if version > LANDMARKS_FORMAT_VERSION:
            raise ValueError(
                f"{p} was written by a newer deeperfly (landmarks format v{version}, this "
                f"build understands v{LANDMARKS_FORMAT_VERSION}); refusing to read it "
                "rather than silently dropping state it carries"
            )
        out: list[Landmark] = []
        seen: set[str] = set()
        for row in data.get("landmark", []) or []:
            name = row.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"{p}: a [[landmark]] entry has no 'name'")
            if name in seen:
                raise ValueError(
                    f"{p}: landmark {name!r} is defined twice -- names are the label "
                    "keys, so a repeat makes every observation of it ambiguous"
                )
            seen.add(name)
            scope = str(row.get("scope", "recording"))
            if scope not in SCOPES:
                raise ValueError(
                    f"{p}: landmark {name!r} has scope {scope!r}; expected one of "
                    f"{list(SCOPES)}"
                )
            out.append(
                Landmark(
                    name=name,
                    static=bool(row.get("static", True)),
                    scope=scope,
                    color=str(row.get("color", "#e8a33d")),
                    note=str(row.get("note", "")),
                )
            )
        return cls(out)

    def save(self, path: str | Path) -> Path:
        """Write a ``landmarks.toml`` (overwriting). Returns the path written."""
        out = Path(path)
        if out.is_dir():
            out = out / LANDMARKS_FILENAME
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Calibration landmarks -- non-skeleton points used to solve the camera rig.",
            "#",
            "# A STATIC landmark (a scratch on the coverslip, the tether tip, a dust",
            "# speck) is ONE 3D point observed in every frame that labels it, so it",
            "# contributes many equations against three unknowns and sits out in the scene",
            "# volume. That is what makes a from-scratch solve converge; skeleton",
            "# keypoints move every frame and cluster where the animal is.",
            "#",
            '# scope = "rig" shares one 3D point across every recording on this rig -- the',
            "# strongest constraint available, and wrong the moment the rig is bumped, so",
            "# the solve report always breaks its residual down per recording.",
            "#",
            "# `note` matters: it is how the next operator finds the SAME speck.",
            "",
            "[landmarks]",
            f"format_version = {LANDMARKS_FORMAT_VERSION}",
        ]
        for lm in self.landmarks:
            lines += ["", "[[landmark]]"]
            row: dict = {"name": lm.name, "static": lm.static}
            if lm.static:
                row["scope"] = lm.scope
            row["color"] = lm.color
            if lm.note:
                row["note"] = lm.note
            lines += [f"{_toml.key(k)} = {_toml.value(v)}" for k, v in row.items()]
        out.write_text("\n".join(lines) + "\n")
        return out

    def add(self, landmark: Landmark) -> LandmarkSet:
        """Append a landmark (returns self).

        Raises
        ------
        ValueError
            If the name is already taken.
        """
        if landmark.name in self.names:
            raise ValueError(f"landmark {landmark.name!r} already exists")
        self.landmarks.append(landmark)
        return self
