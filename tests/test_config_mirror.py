"""Tests for ``[cameras.<name>].mirror`` -- the declared left/right rig pairing.

The key names the view that sees THIS view's mirror image. Nothing in the run path reads
it: it is metadata about how the rig was built, kept in the config because that is where a
fact about the rig belongs and because flip augmentation needs it -- a horizontally
flipped right-side view is a valid left-side view, which is what makes the augmentation
legal, and the flipped sample has to be relabeled with the *mirrored* camera or any metric
splitting ipsilateral from contralateral error calls every swapped channel by the wrong
side.

So what is pinned here is the two things :meth:`Config.mirror_views` is for: that a
half-declared pairing is refused rather than silently swapping one camera's sides, and
that the key never reaches the rig parser as though it were geometry.
"""

from __future__ import annotations

import pytest

from deeperfly.config import Config


def test_the_packaged_rig_declares_a_symmetric_pairing():
    """Every view pairs, and the pairing is an involution -- which the packaged rig is."""
    mapping = Config.default().mirror_views()
    assert mapping, "the packaged config declares a mirror for every view"
    for view, partner in mapping.items():
        assert mapping[partner] == view, f"{view} -> {partner} does not come back"
    # The midline camera is its own mirror; that is a pairing, not a missing one.
    assert mapping["f"] == "f"


def test_a_half_declared_pairing_is_refused():
    """One-sided is the dangerous state: it reads as correct and swaps sides on one camera."""
    cfg = Config.from_dict(
        {
            "cameras": {
                "defaults": {"distance": 1.0},
                "a": {"azimuth_deg": 0, "mirror": "b"},
                "b": {"azimuth_deg": 90},
            }
        }
    )
    with pytest.raises(ValueError, match="must be symmetric"):
        cfg.mirror_views()


def test_a_mirror_naming_an_unknown_view_is_refused():
    cfg = Config.from_dict(
        {
            "cameras": {
                "defaults": {"distance": 1.0},
                "a": {"azimuth_deg": 0, "mirror": "ghost"},
            }
        }
    )
    with pytest.raises(ValueError, match="unknown view"):
        cfg.mirror_views()


def test_the_key_does_not_reach_the_rig_parser():
    """It is stage metadata, not geometry, so it is stripped before ``Camera.from_spec``.

    Asserted on the packaged config because that is the one that carries a ``mirror`` on
    every view: the rig must parse with the key present, and the key must still be
    readable from where it belongs.
    """
    cfg = Config.default()
    sizes = {name: (512, 1024) for name in cfg.camera_table()[1]}
    rig = cfg.camera_group(image_sizes=sizes)
    assert list(rig.names) == list(cfg.camera_table()[1])
    assert cfg.mirror_views()


def test_a_rig_declaring_no_pairs_is_empty_not_invented():
    """No declaration means no pairing -- never a guess from the extrinsics.

    Which camera mirrors which is a fact about how the rig was built. A symmetric orbit
    makes it *look* derivable, and deriving it would be wrong the moment a rig is not
    symmetric -- so an undeclared rig reports nothing rather than something plausible.
    """
    cfg = Config.from_dict(
        {
            "cameras": {
                "defaults": {"distance": 1.0},
                "a": {"azimuth_deg": -90},
                "b": {"azimuth_deg": 90},
            }
        }
    )
    assert cfg.mirror_views() == {}
