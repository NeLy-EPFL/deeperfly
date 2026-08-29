"""The canonical footage pointer -- one record wherever "where is this camera's footage"
is stored.

The bug this replaced: three shapes for one fact (``{abs, rel}`` in results.h5, ``abs`` XOR
``names`` in recording.toml, nothing in .dfpkg) and reader populations that did not match
either shape. The two properties that make adopting it safe are pinned here: ``basenames``
is byte-identical to what the labels identity already stored, and every key is additive.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly.footage import basenames, resolve, sizes, write_pointer


@pytest.fixture
def clip(tmp_path):
    """A recording directory with one camera's two files."""
    rec = tmp_path / "rec"
    rec.mkdir()
    a = rec / "camera_RH.mp4"
    b = rec / "camera_LH.mp4"
    a.write_bytes(b"\0" * 111)
    b.write_bytes(b"\0" * 222)
    return rec, [a, b]


# -- the record -----------------------------------------------------------------


def test_a_pointer_always_carries_every_flavor(clip, tmp_path):
    """A reader must never have to guess which flavor a writer chose."""
    rec, files = clip
    anchor = tmp_path / "rec" / "deeperfly_outputs"
    anchor.mkdir()
    pointer = write_pointer(files, anchor)
    assert set(pointer) == {"abs", "rel", "names", "bytes"}
    assert pointer["names"] == ["camera_RH.mp4", "camera_LH.mp4"]
    assert pointer["bytes"] == [111, 222]
    assert all(p.startswith("/") for p in pointer["abs"])
    assert pointer["rel"] == ["../camera_RH.mp4", "../camera_LH.mp4"]


def test_the_byte_sizes_make_the_recording_id_re_derivable(clip, tmp_path):
    """`recording_fingerprint` hashes basename + byte size, and nothing kept the sizes -- so
    an id could neither be re-derived nor explained once the footage was gone.
    """
    rec, files = clip
    pointer = write_pointer(files, rec)
    assert sizes(pointer) == [111, 222]
    for f in files:
        f.unlink()
    # The sizes survive the footage: they are in the record, not on the filesystem.
    assert sizes(pointer) == [111, 222]
    assert basenames(pointer) == ["camera_LH.mp4", "camera_RH.mp4"]


def test_an_unresolvable_file_records_minus_one_not_zero(tmp_path):
    """-1 is distinguishable from a real size of 0."""
    pointer = write_pointer([tmp_path / "gone.mp4"], tmp_path)
    assert pointer["bytes"] == [-1]
    assert pointer["names"] == ["gone.mp4"]


# -- resolution -----------------------------------------------------------------


def test_abs_wins_when_it_resolves(clip, tmp_path):
    rec, files = clip
    assert resolve(write_pointer(files, rec), rec) == [f.resolve() for f in files]


def test_rel_saves_a_recording_that_moved_with_its_project(clip, tmp_path):
    """The flavor recording.toml used to omit entirely, so this case simply failed."""
    rec, files = clip
    pointer = write_pointer(files, rec)
    moved = tmp_path / "elsewhere"
    rec.rename(moved)
    got = resolve(pointer, moved)
    assert got is not None
    assert [p.name for p in got] == ["camera_RH.mp4", "camera_LH.mp4"]
    assert all(p.exists() for p in got)


def test_names_resolve_under_footage_dir(clip, tmp_path):
    rec, files = clip
    pointer = write_pointer(files, rec)
    pointer["abs"] = ["/nowhere/camera_RH.mp4", "/nowhere/camera_LH.mp4"]
    pointer["rel"] = ["nope/camera_RH.mp4", "nope/camera_LH.mp4"]
    got = resolve(pointer, tmp_path / "unrelated", footage_dir=rec)
    assert got is not None and all(p.exists() for p in got)


def test_names_resolve_beside_the_pointers_own_file(clip):
    """A names-only pointer used to be tried against the process CWD, which is a fiction."""
    rec, files = clip
    got = resolve({"names": ["camera_RH.mp4", "camera_LH.mp4"]}, rec)
    assert got is not None and all(p.exists() for p in got)


def test_nothing_resolving_is_none_not_a_partial_list(tmp_path):
    assert resolve({"abs": ["/a.mp4"], "rel": ["b.mp4"]}, tmp_path) is None


def test_a_bare_list_is_accepted_so_callers_need_no_wrapper(clip):
    """The `{"abs": [...]}` wrappers at the old call sites existed only to borrow this."""
    rec, files = clip
    assert resolve(files, rec) == [f.resolve() for f in files]


def test_an_old_pointer_with_no_names_still_uses_the_by_name_fallbacks(clip):
    """Additive keys: a file written before `names` existed must not lose a fallback."""
    rec, _ = clip
    old = {"abs": ["/nowhere/camera_RH.mp4"], "rel": ["nope/camera_RH.mp4"]}
    got = resolve(old, rec)  # `names` derived from `abs`, then tried beside the anchor
    assert got is not None and got[0].name == "camera_RH.mp4"
    assert got[0].parent == rec


# -- the identity projection ----------------------------------------------------


def test_basenames_are_sorted_and_flavor_agnostic(clip):
    """Byte-identical to what labels_identity already stored -- the safety property that
    lets this be adopted without invalidating a single existing label.
    """
    rec, files = clip
    pointer = write_pointer(files, rec)
    want = ["camera_LH.mp4", "camera_RH.mp4"]  # sorted, not file order
    assert basenames(pointer) == want
    assert basenames({"rel": pointer["rel"]}) == want
    assert basenames({"abs": pointer["abs"]}) == want
    assert basenames({"names": pointer["names"]}) == want
    assert basenames(files) == want
    assert basenames(None) == []


def test_basenames_prefers_rel_over_abs_as_it_always_did(clip):
    got = basenames({"rel": ["x/one.mp4"], "abs": ["/y/two.mp4"]})
    assert got == ["one.mp4"]


def test_sizes_of_a_pointer_that_predates_bytes_is_empty(clip):
    assert sizes({"abs": ["/a.mp4"]}) == []


# -- the labels identity must be unchanged by the refactor ----------------------


def test_the_stored_labels_identity_footage_is_unchanged(tmp_path, result):
    """The regression guard that matters: if this projection changed, every existing
    labels.h5 in every project would stop loading.
    """
    from deeperfly.labels import labels_identity

    footage = {
        name: {"rel": [f"{name}.mp4"], "abs": [f"/data/{name}.mp4"]}
        for name in result.cameras.names
    }
    identity = labels_identity(
        point_names=list(result.skeleton.point_names),
        camera_names=list(result.cameras.names),
        n_frames=result.n_frames,
        footage=footage,
    )
    assert identity["footage"] == {
        name: [f"{name}.mp4"] for name in result.cameras.names
    }


def test_results_h5_records_the_canonical_pointer(tmp_path, cameras, fly):
    """And the reader still sees the two flavors it always did."""
    from deeperfly.results import StageStore

    rec = tmp_path / "rec"
    rec.mkdir()
    for name in cameras.names:
        (rec / f"{name}.mp4").write_bytes(b"\0" * (10 + len(name)))
    outputs = rec / "deeperfly_outputs"
    outputs.mkdir()
    store = StageStore(outputs / "results.h5")
    v, t, p = len(cameras.names), 2, len(fly.point_names)
    store.write_pose2d(
        cameras=cameras,
        skeleton=fly,
        pts2d=np.zeros((v, t, p, 2)),
        conf=np.ones((v, t, p)),
        image_sizes={n: (8, 16) for n in cameras.names},
        footage={n: [rec / f"{n}.mp4"] for n in cameras.names},
    )
    got = store.read_footage()
    first = got[list(cameras.names)[0]]
    assert set(first) == {"abs", "rel", "names", "bytes"}
    assert first["rel"][0].startswith("..")
    assert first["bytes"][0] > 0
