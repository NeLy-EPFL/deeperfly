"""The binding: the adaptor between one skeleton and one model.

A skeleton says what is tracked, a model pack says what is fitted, and neither can say
where a tracked point sits on the model -- that is a fact about the pair. These tests
pin that separation (which is greppable, and asserted below), and that the packaged
binding reproduces what used to be stated in three places and two forms.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import numpy as np
import pytest
from helpers import fly38_skeleton  # noqa: F401

from deeperfly.inverse_kinematics.articulation import Articulation, load_articulation
from deeperfly.inverse_kinematics.binding import BINDING_DIR, Binding, bindings
from deeperfly.inverse_kinematics.pack import MODEL_DIR, ModelPack
from deeperfly.inverse_kinematics.template import KinematicTemplate
from deeperfly.skeleton import Skeleton


@pytest.fixture
def binding() -> Binding:
    return Binding.load("fly38@neuromechfly")


def test_the_packaged_pair_is_bound(binding):
    assert (binding.skeleton, binding.model) == ("fly38", "neuromechfly")
    assert len(binding.rows) == Skeleton.fly().n_points
    assert {r.point for r in binding.rows} == set(Skeleton.fly().point_names)


def test_the_binding_is_the_only_place_the_two_namespaces_meet():
    """The invariant the whole decoupling rests on, and it is a grep.

    A skeleton point name inside a model pack is the pack reaching into a namespace it
    does not own -- which is what made "fly38 on flybody" unexpressible no matter how
    many packs existed. A model body name inside a skeleton file is the same violation
    mirrored.
    """
    points = set(Skeleton.fly().point_names)
    bodies = set(Binding.load("fly38@neuromechfly").by_body)

    def tokens(text: str) -> set[str]:
        # Whole identifiers, so `abdomen1` does not "appear" inside `c_abdomen12`.
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))

    for path in sorted(MODEL_DIR.rglob("*")):
        if path.suffix not in (".toml", ".json"):
            continue
        leaked = sorted(points & tokens(path.read_text()))
        assert not leaked, f"{path} names skeleton point(s) {leaked}"

    for path in sorted(Path("src/deeperfly/data/skeletons").glob("*.toml")):
        leaked = sorted(bodies & tokens(path.read_text()))
        assert not leaked, f"{path} names model body/bodies {leaked}"


def test_every_packaged_binding_names_a_packaged_pair():
    for name, path in bindings().items():
        spec = tomllib.loads(path.read_text())
        skeleton, _, model = name.partition("@")
        assert spec["skeleton"] == skeleton and spec["model"] == model
        assert (Path("src/deeperfly/data/skeletons") / f"{skeleton}.toml").is_file()
        ModelPack.load(model)  # raises if the pack is missing or incomplete


def test_an_unbound_pair_fails_by_naming_both_halves():
    """The error a new animal or a new model hits, so it has to say what to write."""
    with pytest.raises(FileNotFoundError, match="skeleton 'fly38' on model 'flybody'"):
        Binding.load("fly38@flybody")


def test_two_points_on_one_body_origin_is_refused(tmp_path):
    """Then "which point observes this body" has no answer, and a leg joint picks one."""
    p = tmp_path / "a@b.toml"
    p.write_text(
        'skeleton = "a"\nmodel = "b"\n[points]\n'
        'one = { body = "x" }\ntwo = { body = "x" }\n'
    )
    with pytest.raises(ValueError, match="two points at the origin of body 'x'"):
        Binding.load(p)


def test_a_point_carried_at_an_offset_does_not_observe_its_body(binding):
    """The pretarsus rides tarsus5 at the tip, so it is not a measurement of tarsus5's origin."""
    assert (
        binding.point_for("lf_tarsus5") == "lf_pretarsus"
    )  # the template resolves here
    assert binding.origin_point_for("lf_tarsus5") is None  # an anchor would not


def test_the_leg_template_names_no_point_until_the_binding_says_so(binding):
    """The template declares model bodies; the point names arrive from the pair."""
    bare = KinematicTemplate.load("neuromechfly", binding=None)
    assert {j.point for leg in bare.legs for j in leg.joints} == {""}
    assert [j.body for j in bare.legs[0].joints][0].endswith("_coxa")

    bound = KinematicTemplate.load("neuromechfly", binding=binding)
    assert bound.legs[0].joints[0].point == "lf_thorax_coxa"


def test_the_chains_markers_are_binding_rows(binding):
    """`articulation.json` carries no markers: they are the binding's rows on its bodies."""
    spec = tomllib.loads(
        (BINDING_DIR / "fly38@neuromechfly.toml").read_text()
    )  # the file itself, to pin the schema
    assert "offset" not in spec["points"]["neck"]  # a bare body is the common row
    assert spec["points"]["neck"]["base"] is True

    raw = ModelPack.load().articulation_path.read_text()
    assert '"markers"' not in raw and '"base_point"' not in raw

    art = Articulation.load(binding=binding)
    head = art.chain("head")
    assert head.marker_names == ("l_antenna", "r_antenna", "neck")
    assert head.base_point == "neck"
    assert head.marker_depth == (3, 3, 0)
    assert not any(head.marker_approximate)

    abdomen = art.chain("abdomen")
    assert abdomen.marker_names == tuple(f"abdomen{i}" for i in range(5))
    assert all(abdomen.marker_approximate)  # no exact counterpart on this model


def test_a_marker_neutral_is_derived_from_the_body_frame(binding):
    """Not baked: `pos + mat @ offset`, so moving a row moves the marker with no MuJoCo."""
    art = Articulation.load(binding=binding)
    bodies = art.bodies
    for chain in art.chains:
        for name, neutral in zip(chain.marker_names, chain.marker_neutral):
            row = binding.row(name)
            frame = bodies[row.body]
            want = np.asarray(frame["pos"]) + np.asarray(frame["mat"]).reshape(
                3, 3
            ) @ np.asarray(row.offset)
            np.testing.assert_allclose(neutral, want, rtol=0, atol=0)


def test_the_anchors_resolve_through_the_binding(binding):
    """A pack names bodies; which point observes each is the pair's to say."""
    art = load_articulation()
    assert art.anchors == tuple(
        f"{leg}_coxa" for leg in ("lf", "lm", "lh", "rf", "rm", "rh")
    )
    assert art.anchor_points == tuple(
        f"{leg}_thorax_coxa" for leg in ("lf", "lm", "lh", "rf", "rm", "rh")
    )
    # ...and their neutral positions come out of the asset's own body frames.
    for body, pos in zip(art.anchors, art.anchor_neutral):
        np.testing.assert_allclose(pos, art.bodies[body]["pos"], rtol=0, atol=0)
