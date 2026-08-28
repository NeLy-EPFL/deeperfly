"""The model pack: the mechanical model the IK stage fits, selected by name.

A pack is a directory holding a ``model.toml`` manifest and the three assets it names
-- the leg template, the baked articulation, and the overlay mesh -- plus the few facts
that are true of the model as a whole: its rest axis and its registration anchors.
``[inverse_kinematics] model = "neuromechfly"`` selects the packaged one; a path to a
manifest selects any other.

**A pack names no skeleton point and carries no offset.** Where a tracked point sits on
the model is a fact about the (skeleton, model) *pair*, not about either alone, so it
lives in a binding (:mod:`deeperfly.inverse_kinematics.binding`) instead. That is what
lets one skeleton reach a second model, and it is greppable: a skeleton point name
appearing in a pack file is a bug, and a test says so.

Segment *lengths* are not in a pack either -- they are measured per recording from the
3D pose (:mod:`deeperfly.inverse_kinematics.align`), which is why a fit reprojects
tightly on any animal.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ModelPack", "MODELS", "MODEL_DIR"]

#: Where the packaged model packs live, one directory each.
MODEL_DIR = Path(__file__).parent.parent / "data" / "models"


def _packaged() -> dict[str, Path]:
    if not MODEL_DIR.is_dir():
        return {}
    return {d.name: d / "model.toml" for d in sorted(MODEL_DIR.iterdir()) if d.is_dir()}


#: Named packaged model packs, ``name -> manifest path``.
MODELS: dict[str, Path] = _packaged()


@dataclass(frozen=True)
class ModelPack:
    """A loaded model manifest: the three asset paths plus the model-wide facts."""

    name: str
    #: The manifest's own directory -- the three assets are resolved against it.
    root: Path
    template_path: Path
    articulation_path: Path
    mesh_path: Path
    #: The direction a segment extends along in its parent joint's frame.
    rest_axis: tuple[float, float, float]
    #: The model bodies whose observed positions register the recording to the model.
    anchors: tuple[str, ...]

    @classmethod
    def load(cls, ref: str | Path = "neuromechfly") -> "ModelPack":
        """Load a pack by packaged name or by path to its ``model.toml``.

        A path to the pack *directory* works too, which is the shape a user is most
        likely to type.

        Raises
        ------
        FileNotFoundError
            If the name is not packaged and the path is not a manifest, or if the
            manifest names an asset that is not beside it -- an incomplete pack is a
            load error rather than a stage that fails halfway through a run.
        """
        path = MODELS.get(str(ref))
        if path is None:
            path = Path(ref)
            if path.is_dir():
                path = path / "model.toml"
        if not path.is_file():
            raise FileNotFoundError(
                f"inverse-kinematics model {ref!r} not found "
                f"(known: {sorted(MODELS)}, or a path to a model.toml)"
            )
        spec = tomllib.loads(path.read_text())
        root = path.parent

        def asset(key: str) -> Path:
            if key not in spec:
                raise FileNotFoundError(
                    f"model pack {path} declares no {key!r}; a pack is a template, an "
                    "articulation and a mesh"
                )
            p = root / str(spec[key])
            if not p.is_file():
                raise FileNotFoundError(
                    f"model pack {path} names {key} = {spec[key]!r}, which is not "
                    f"beside it ({p})"
                )
            return p

        rest = spec.get("rest_axis", (0.0, 0.0, -1.0))
        return cls(
            name=str(spec.get("name", root.name)),
            root=root,
            template_path=asset("template"),
            articulation_path=asset("articulation"),
            mesh_path=asset("mesh"),
            rest_axis=tuple(float(v) for v in rest),  # type: ignore[arg-type]
            anchors=tuple(str(a) for a in spec.get("anchors", ())),
        )
