"""Tests for the GUI's editor core: state, corrections sidecar, footage resolution.

The editor logic -- :class:`EditorState`, the corrections sidecar, footage
resolution -- carries no web/Qt dependency and is tested directly here. The
FastAPI server that drives it over HTTP/WebSocket is tested in
``test_gui_server.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly.gui import (
    Corrections,
    EditorState,
    load_corrections,
    resolve_footage,
    save_corrections,
)

# -- EditorState: 2D edits ----------------------------------------------------


def test_2d_edit_overlays_without_touching_original(result):
    original = result.pts2d.copy()
    state = EditorState.from_result(result)
    state.apply_2d_edit(view=0, point=3, xy=(12.0, 34.0), frame=1)

    disp = state.display_pts2d(1)
    assert np.allclose(disp[0, 3], [12.0, 34.0])
    # other points and views unchanged
    assert np.allclose(disp[1, 3], result.pts2d[1, 1, 3])
    # the result's own arrays are never mutated
    np.testing.assert_array_equal(result.pts2d, original)
    assert state.dirty


# -- EditorState: 3D edits ----------------------------------------------------


def test_3d_edit_lands_on_cursor_and_moves_other_views(result):
    state = EditorState.from_result(result)
    view, point, frame = 2, 5, 0
    before = state.display_pts3d_projected(frame)
    drag = before[view, point] + np.array([10.0, -8.0])

    x_new = state.apply_3d_edit(view, point, drag, frame)
    assert x_new is not None

    after = state.display_pts3d_projected(frame)
    # the dragged view's reprojection lands exactly on the cursor
    assert np.allclose(after[view, point], drag, atol=1e-4)
    # at least one other view's reprojection moved
    other = (view + 1) % state.n_views
    assert not np.allclose(after[other, point], before[other, point])
    assert state.dirty


def test_3d_edit_is_noop_for_nan_point(result):
    state = EditorState.from_result(result)
    state.result.pts3d[0, 7] = np.nan
    assert state.apply_3d_edit(0, 7, (1.0, 2.0), 0) is None
    assert not state.dirty


def test_3d_edit_unavailable_without_3d(result):
    result.pts3d = None
    state = EditorState.from_result(result)
    assert not state.has_3d
    assert state.display_pts3d_projected(0) is None
    assert state.apply_3d_edit(0, 0, (1.0, 2.0), 0) is None


def test_reset_point_clears_corrections(result):
    state = EditorState.from_result(result)
    state.apply_2d_edit(0, 4, (1.0, 2.0), frame=0)
    state.apply_3d_edit(1, 4, state.display_pts3d_projected(0)[1, 4] + 5.0, frame=0)
    assert state.corrections.any_edits

    state.reset_point(4, frame=0)
    assert not state.corrections.pts2d_edited[:, 0, 4].any()
    assert not state.corrections.pts3d_edited[0, 4]


def test_reset_point_view_clears_only_that_view(result):
    state = EditorState.from_result(result)
    state.apply_2d_edit(0, 4, (1.0, 2.0), frame=0)
    state.apply_2d_edit(1, 4, (3.0, 4.0), frame=0)

    state.reset_point_view(0, 4, frame=0)
    assert not state.corrections.pts2d_edited[0, 0, 4]  # reverted
    assert state.corrections.pts2d_edited[1, 0, 4]  # the other view is left alone


def test_reset_frame_clears_only_that_frame(result):
    state = EditorState.from_result(result)
    state.apply_2d_edit(0, 4, (1.0, 2.0), frame=0)
    state.apply_2d_edit(1, 7, (3.0, 4.0), frame=0)
    state.apply_3d_edit(1, 4, state.display_pts3d_projected(0)[1, 4] + 5.0, frame=0)
    state.apply_2d_edit(0, 4, (5.0, 6.0), frame=1)  # a different frame, left untouched

    state.reset_frame(frame=0)
    assert not state.corrections.pts2d_edited[:, 0].any()  # every point/view reverted
    assert not state.corrections.pts3d_edited[0].any()
    assert state.corrections.pts2d_edited[0, 1, 4]  # the other frame is left alone


# -- EditorState: 3D refinement (fixed/finalized constraints) -----------------


def test_3d_edit_with_nothing_fixed_lands_on_cursor(result):
    # With no fixed views the constrained re-solve must reduce to the original
    # single-ray behavior: the dragged view lands exactly under the cursor.
    state = EditorState.from_result(result)
    view, point, frame = 2, 5, 0
    drag = state.display_pts2d_refine(frame)[view, point] + np.array([10.0, -8.0])
    assert state.apply_3d_edit(view, point, drag, frame) is not None
    assert np.allclose(state.display_pts2d_refine(frame)[view, point], drag, atol=1e-4)


def test_toggle_fixed_sets_then_clears_mask_and_pixel(result):
    state = EditorState.from_result(result)
    view, point, frame = 1, 5, 0
    pix = state.display_pts2d_refine(frame)[view, point].copy()

    assert state.toggle_fixed(view, point, frame) is True
    assert state.corrections.pts2d_fixed[view, frame, point]
    assert np.allclose(state.corrections.pts2d[view, frame, point], pix)
    assert np.allclose(state.display_pts2d_refine(frame)[view, point], pix)

    assert state.toggle_fixed(view, point, frame) is False
    assert not state.corrections.pts2d_fixed[view, frame, point]
    assert not state.corrections.pts2d_edited[view, frame, point]


def test_fixed_view_pixel_is_held_when_another_view_is_dragged(result):
    state = EditorState.from_result(result)
    point, frame = 5, 0
    fixed_view, drag_view = 0, 2
    held = state.display_pts2d_refine(frame)[fixed_view, point].copy()
    state.toggle_fixed(fixed_view, point, frame)

    drag = state.display_pts2d_refine(frame)[drag_view, point] + np.array([12.0, -9.0])
    state.apply_3d_edit(drag_view, point, drag, frame)

    # the finalized view never moves, even though the 3D point was re-solved
    assert np.allclose(state.display_pts2d_refine(frame)[fixed_view, point], held)


def test_fixing_a_view_changes_how_others_follow(result):
    point, frame = 5, 0
    fixed_view, drag_view, other = 0, 2, 4
    offset = np.array([14.0, -11.0])

    free = EditorState.from_result(result)
    drag = free.display_pts2d_refine(frame)[drag_view, point] + offset
    free.apply_3d_edit(drag_view, point, drag, frame)
    free_other = free.display_pts2d_refine(frame)[other, point].copy()

    constrained = EditorState.from_result(result)
    constrained.toggle_fixed(fixed_view, point, frame)
    drag = constrained.display_pts2d_refine(frame)[drag_view, point] + offset
    constrained.apply_3d_edit(drag_view, point, drag, frame)
    constrained_other = constrained.display_pts2d_refine(frame)[other, point]

    # the same drag in view 2 moves view 4 differently depending on view 0's fix
    assert not np.allclose(free_other, constrained_other)


def test_dlt_resolve_from_two_fixed_views_recovers_point(result):
    # The synthetic 2D is the exact projection of the 3D, so fixing two views at
    # their (consistent) pixels must re-triangulate back to the true 3D point.
    state = EditorState.from_result(result)
    point, frame = 5, 0
    a, b = 0, 3
    state.toggle_fixed(a, point, frame)
    state.toggle_fixed(b, point, frame)

    assert np.allclose(
        state.display_pts3d(frame)[point], result.pts3d[frame, point], atol=1e-6
    )
    proj = state.display_pts2d_refine(frame)
    assert np.allclose(proj[a, point], result.pts2d[a, frame, point], atol=1e-4)
    assert np.allclose(proj[b, point], result.pts2d[b, frame, point], atol=1e-4)


def test_dragging_a_fixed_view_keeps_it_under_cursor(result):
    state = EditorState.from_result(result)
    view, point, frame = 0, 5, 0
    state.toggle_fixed(view, point, frame)
    state.toggle_fixed(3, point, frame)  # a second fixed view, so the DLT path runs

    drag = state.display_pts2d_refine(frame)[view, point] + np.array([8.0, 6.0])
    state.apply_3d_edit(view, point, drag, frame)
    # a fixed view's own drag is the "does not satisfy the model" gesture: it stays put
    assert np.allclose(state.display_pts2d_refine(frame)[view, point], drag)


def test_3d_drag_release_pins_the_dragged_view(result):
    # "Drag pins the view": a release (fix=True) finalizes the dragged view at the
    # drop pixel, so it stays there (no snap to the reprojection) and becomes a
    # constraint -- even when another fixed view would otherwise pull it off.
    state = EditorState.from_result(result)
    view, point, frame = 2, 5, 0
    state.toggle_fixed(0, point, frame)  # a constraint that would cause a snap
    drag = state.display_pts2d_refine(frame)[view, point] + np.array([12.0, -9.0])

    state.apply_3d_edit(view, point, drag, frame, fix=True)

    assert state.corrections.pts2d_fixed[view, frame, point]
    assert np.allclose(state.corrections.pts2d[view, frame, point], drag)
    # displayed exactly at the drop pixel -- it does not snap to project(X)
    assert np.allclose(state.display_pts2d_refine(frame)[view, point], drag, atol=1e-9)


def test_live_3d_drag_does_not_pin(result):
    # The live re-solve (fix defaults to False) must NOT finalize the view; only
    # the release does, so mid-drag updates never leave a stray pin behind.
    state = EditorState.from_result(result)
    view, point, frame = 2, 5, 0
    drag = state.display_pts2d_refine(frame)[view, point] + np.array([5.0, 5.0])
    state.apply_3d_edit(view, point, drag, frame)
    assert not state.corrections.pts2d_fixed[view, frame, point]


def test_reset_point_clears_fixed(result):
    state = EditorState.from_result(result)
    state.toggle_fixed(1, 4, frame=0)
    assert state.corrections.pts2d_fixed[1, 0, 4]
    state.reset_point(4, frame=0)
    assert not state.corrections.pts2d_fixed[:, 0, 4].any()


# -- EditorState: invisible (obscured) views ----------------------------------


def test_toggle_invisible_re_solves_3d_without_the_marked_view(result):
    # The fixture's 2D are exact projections of its 3D, so a clean subset
    # re-triangulates the true point. Corrupt one view's 2D *and* the stored 3D so
    # the displayed point starts wrong; marking the bad view invisible must drop it
    # and recover the true 3D from the remaining views (not a no-op).
    state = EditorState.from_result(result)
    bad_view, point, frame = 0, 5, 0
    true_3d = result.pts3d[frame, point].copy()
    result.pts2d[bad_view, frame, point] += np.array([40.0, -30.0])  # garbage detection
    result.pts3d[frame, point] = true_3d + np.array([0.5, -0.4, 0.3])  # wrong base 3D
    assert not np.allclose(state.display_pts3d(frame)[point], true_3d)

    assert state.toggle_invisible(bad_view, point, frame) is True
    assert state.corrections.pts2d_invisible[bad_view, frame, point]
    assert np.allclose(state.display_pts3d(frame)[point], true_3d, atol=1e-6)


def test_toggle_invisible_then_back_restores_the_view(result):
    state = EditorState.from_result(result)
    view, point, frame = 0, 5, 0
    assert state.toggle_invisible(view, point, frame) is True
    assert state.toggle_invisible(view, point, frame) is False
    assert not state.corrections.pts2d_invisible[view, frame, point]


def test_invisible_and_fixed_are_mutually_exclusive(result):
    state = EditorState.from_result(result)
    view, point, frame = 1, 5, 0

    # fixing then obscuring drops the fixed flag/pixel
    state.toggle_fixed(view, point, frame)
    assert state.corrections.pts2d_fixed[view, frame, point]
    state.toggle_invisible(view, point, frame)
    assert state.corrections.pts2d_invisible[view, frame, point]
    assert not state.corrections.pts2d_fixed[view, frame, point]
    assert not state.corrections.pts2d_edited[view, frame, point]

    # obscuring then fixing drops the invisible flag
    state.toggle_fixed(view, point, frame)
    assert state.corrections.pts2d_fixed[view, frame, point]
    assert not state.corrections.pts2d_invisible[view, frame, point]


def test_dragging_an_invisible_view_un_obscures_it(result):
    # Dragging an obscured view places it: the flag clears (back to normal) and the
    # 3D re-solves so the dragged view lands under the cursor. A release on a
    # formerly-obscured view does not pin it (fix=False), so it stays normal.
    state = EditorState.from_result(result)
    view, point, frame = 2, 5, 0
    state.toggle_invisible(view, point, frame)
    assert state.corrections.pts2d_invisible[view, frame, point]

    drag = state.display_pts2d_refine(frame)[view, point] + np.array([8.0, 6.0])
    assert state.apply_3d_edit(view, point, drag, frame, fix=False) is not None
    assert not state.corrections.pts2d_invisible[view, frame, point]  # back to normal
    assert not state.corrections.pts2d_fixed[view, frame, point]
    assert np.allclose(state.display_pts2d_refine(frame)[view, point], drag, atol=1e-4)


def test_toggle_invisible_unavailable_without_3d(result):
    result.pts3d = None
    state = EditorState.from_result(result)
    assert state.toggle_invisible(0, 0, 0) is None


def test_reset_point_clears_invisible(result):
    state = EditorState.from_result(result)
    state.toggle_invisible(1, 4, frame=0)
    assert state.corrections.pts2d_invisible[1, 0, 4]
    state.reset_point(4, frame=0)
    assert not state.corrections.pts2d_invisible[:, 0, 4].any()


def test_fresh_overlay_seeds_invisible_from_nan_detections(result):
    # A NaN 2D detection means that camera did not see the keypoint, so a fresh
    # overlay starts that per-view point obscured (and everything else visible).
    result.pts2d[0, 1, 4] = np.nan
    result.pts2d[2, 0, 5, 0] = np.nan  # a single NaN coordinate counts as missing
    state = EditorState.from_result(result)

    invisible = state.corrections.pts2d_invisible
    assert invisible[0, 1, 4]
    assert invisible[2, 0, 5]
    expected = ~np.isfinite(result.pts2d).all(axis=-1)
    np.testing.assert_array_equal(invisible, expected)
    # seeding a default is not an edit: the session opens clean
    assert not state.dirty
    assert not state.corrections.any_edits


def test_loaded_corrections_keep_their_own_invisible_mask(result):
    # A provided overlay (e.g. loaded from a sidecar) is authoritative -- its
    # saved invisible mask is not re-seeded from the result's NaNs.
    result.pts2d[0, 1, 4] = np.nan
    corrections = Corrections.empty(
        result.n_views, result.n_frames, result.pts2d.shape[2]
    )
    state = EditorState.from_result(result, corrections)
    assert not state.corrections.pts2d_invisible.any()


# -- corrections sidecar ------------------------------------------------------


def test_corrections_roundtrip(tmp_path, result):
    state = EditorState.from_result(result)
    state.apply_2d_edit(0, 1, (5.0, 6.0), frame=2)
    state.apply_3d_edit(1, 4, state.display_pts3d_projected(0)[1, 4] + 3.0, frame=0)

    path = tmp_path / "corrections.h5"
    save_corrections(path, state.corrections, source="results.h5")
    assert not state.corrections.dirty  # saving clears the dirty flag

    loaded = load_corrections(
        path, result.n_views, result.n_frames, result.pts2d.shape[2]
    )
    assert loaded is not None
    np.testing.assert_array_equal(loaded.pts2d_edited, state.corrections.pts2d_edited)
    np.testing.assert_array_equal(loaded.pts3d_edited, state.corrections.pts3d_edited)
    np.testing.assert_array_equal(
        np.nan_to_num(loaded.pts2d), np.nan_to_num(state.corrections.pts2d)
    )
    np.testing.assert_array_equal(
        np.nan_to_num(loaded.pts3d), np.nan_to_num(state.corrections.pts3d)
    )


def test_corrections_roundtrip_preserves_fixed(tmp_path, result):
    state = EditorState.from_result(result)
    state.toggle_fixed(0, 5, frame=0)
    state.toggle_fixed(2, 5, frame=0)

    path = tmp_path / "corrections.h5"
    save_corrections(path, state.corrections)
    loaded = load_corrections(
        path, result.n_views, result.n_frames, result.pts2d.shape[2]
    )
    assert loaded is not None
    np.testing.assert_array_equal(loaded.pts2d_fixed, state.corrections.pts2d_fixed)


def test_corrections_roundtrip_preserves_invisible(tmp_path, result):
    state = EditorState.from_result(result)
    state.toggle_invisible(0, 5, frame=0)
    state.toggle_invisible(3, 5, frame=0)

    path = tmp_path / "corrections.h5"
    save_corrections(path, state.corrections)
    loaded = load_corrections(
        path, result.n_views, result.n_frames, result.pts2d.shape[2]
    )
    assert loaded is not None
    np.testing.assert_array_equal(
        loaded.pts2d_invisible, state.corrections.pts2d_invisible
    )


def test_load_corrections_v1_without_fixed_defaults_to_false(tmp_path, result):
    import h5py

    state = EditorState.from_result(result)
    state.apply_2d_edit(0, 1, (5.0, 6.0), frame=0)
    path = tmp_path / "corrections.h5"
    save_corrections(path, state.corrections)
    # simulate a v1 sidecar written before the "fixed" dataset existed
    with h5py.File(path, "a") as f:
        del f["pose2d_corrections/fixed"]

    loaded = load_corrections(
        path, result.n_views, result.n_frames, result.pts2d.shape[2]
    )
    assert loaded is not None
    assert loaded.pts2d_fixed.shape == state.corrections.pts2d_edited.shape
    assert not loaded.pts2d_fixed.any()


def test_load_corrections_without_invisible_defaults_to_false(tmp_path, result):
    import h5py

    state = EditorState.from_result(result)
    state.apply_2d_edit(0, 1, (5.0, 6.0), frame=0)
    path = tmp_path / "corrections.h5"
    save_corrections(path, state.corrections)
    # simulate a v2 sidecar written before the "invisible" dataset existed
    with h5py.File(path, "a") as f:
        del f["pose2d_corrections/invisible"]

    loaded = load_corrections(
        path, result.n_views, result.n_frames, result.pts2d.shape[2]
    )
    assert loaded is not None
    assert loaded.pts2d_invisible.shape == state.corrections.pts2d_edited.shape
    assert not loaded.pts2d_invisible.any()


def test_load_corrections_missing_returns_none(tmp_path):
    assert load_corrections(tmp_path / "absent.h5", 1, 1, 1) is None


def test_load_corrections_shape_mismatch_raises(tmp_path, result):
    path = tmp_path / "corrections.h5"
    save_corrections(path, EditorState.from_result(result).corrections)
    with pytest.raises(ValueError, match="different result"):
        load_corrections(
            path, result.n_views + 1, result.n_frames, result.pts2d.shape[2]
        )


# -- footage resolution -------------------------------------------------------


def test_resolve_footage_prefers_absolute_then_relative_then_dir(tmp_path):
    import os as _os

    results_dir = tmp_path / "deeperfly_outputs"
    results_dir.mkdir()
    video = tmp_path / "camera_0.mp4"
    video.write_bytes(b"x")
    rel = _os.path.relpath(video, results_dir)

    # absolute path resolves
    footage = {"cam0": {"abs": [str(video)], "rel": [rel]}}
    resolved, missing = resolve_footage(footage, results_dir)
    assert missing == [] and resolved["cam0"] == [video]

    # absolute broken, relative resolves
    footage = {"cam0": {"abs": ["/nope/camera_0.mp4"], "rel": [rel]}}
    resolved, missing = resolve_footage(footage, results_dir)
    assert missing == [] and resolved["cam0"][0].resolve() == video.resolve()

    # both broken, footage_dir fallback resolves by file name
    footage = {"cam0": {"abs": ["/nope/camera_0.mp4"], "rel": ["../gone/camera_0.mp4"]}}
    resolved, missing = resolve_footage(footage, results_dir, footage_dir=tmp_path)
    assert missing == [] and resolved["cam0"] == [video]

    # nothing resolves -> reported missing
    resolved, missing = resolve_footage(footage, results_dir)
    assert missing == ["cam0"] and resolved == {}


def test_resolve_footage_none_is_empty(tmp_path):
    resolved, missing = resolve_footage(None, tmp_path)
    assert resolved == {} and missing == []


def test_corrections_empty_shapes(result):
    corr = Corrections.empty(result.n_views, result.n_frames, result.pts2d.shape[2])
    assert corr.pts2d.shape == (*result.pts2d.shape[:2], result.pts2d.shape[2], 2)
    assert np.isnan(corr.pts2d).all()
    assert not corr.pts2d_edited.any()
    assert not corr.pts2d_fixed.any()
    assert not corr.pts2d_invisible.any()
    assert not corr.any_edits


# -- browser launch: silence the external opener's stdio ----------------------


def test_quiet_child_output_swallows_then_restores(tmp_path):
    """`_quiet_child_output` drops OS-level writes in-block and restores fd 2 after.

    This guards the fix that keeps an external browser opener's noise (e.g. VS
    Code's `browser.sh` -> Node `url.parse()` warning) out of the CLI output.
    """
    import os

    from deeperfly.gui import _quiet_child_output

    # Stand in for the terminal: point fd 2 at a file we can read back.
    sink = tmp_path / "stderr.log"
    fd = os.open(sink, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    saved = os.dup(2)
    os.dup2(fd, 2)
    os.close(fd)
    try:
        with _quiet_child_output():
            os.write(2, b"swallowed\n")  # a noisy child inheriting our stderr
        os.write(2, b"kept\n")  # fd 2 restored -> reaches the sink again
    finally:
        os.dup2(saved, 2)
        os.close(saved)

    assert sink.read_text() == "kept\n"
