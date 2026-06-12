"""Tests for the optional GUI's Qt-free core and (headless) widgets.

The editor logic -- :class:`EditorState`, the corrections sidecar, footage
resolution -- carries no Qt dependency and is tested directly. The widget tests
are guarded by ``pytest.importorskip("PySide6")`` and run on the ``offscreen``
Qt platform (set below), so they work in CI without a display.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from deeperfly.gui import (
    Corrections,
    EditMode,
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
    assert not corr.any_edits


# -- headless widgets ---------------------------------------------------------


@pytest.fixture
def qapp():
    # The offscreen platform must be selected before the QApplication is created.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qtwidgets = pytest.importorskip("PySide6.QtWidgets")
    return qtwidgets.QApplication.instance() or qtwidgets.QApplication([])


def _blank_source(result):
    from helpers import HEIGHT, WIDTH

    from deeperfly.gui.readers import FrameSource

    image_sizes = {name: (HEIGHT, WIDTH) for name in result.cameras.names}
    return FrameSource({}, image_sizes=image_sizes)


def test_window_builds_with_blank_frames(qapp, result, tmp_path):
    from deeperfly.gui.window import MainWindow

    source = _blank_source(result)
    state = EditorState.from_result(result)
    window = MainWindow(
        state,
        source,
        results_path=str(tmp_path / "results.h5"),
        corrections_path=tmp_path / "corrections.h5",
    )
    assert len(window._views) == result.n_views
    state.corrections.dirty = False  # avoid the unsaved-changes dialog on close
    window.close()


def test_window_2d_drag_updates_state(qapp, result, tmp_path):
    from deeperfly.gui.window import MainWindow

    source = _blank_source(result)
    state = EditorState.from_result(result)
    window = MainWindow(
        state,
        source,
        results_path=str(tmp_path / "results.h5"),
        corrections_path=tmp_path / "corrections.h5",
    )
    window._mode_combo.setCurrentIndex(window._mode_combo.findData(EditMode.edit_2d))
    window._views[0].pointDragged.emit(0, 2, 100.0, 50.0)

    assert state.dirty
    assert np.allclose(state.display_pts2d(0)[0, 2], [100.0, 50.0])

    state.corrections.dirty = False
    window.close()


def test_set_points_keeps_a_writable_array(qapp):
    # Projected 3D points arrive as a (read-only) JAX buffer; the view must hold a
    # writable copy so a drag can write the dragged joint straight into it.
    import jax.numpy as jnp

    from deeperfly.gui.view import PoseView

    view = PoseView(0)
    pts = jnp.asarray(np.zeros((5, 2)))  # read-only
    assert not np.asarray(pts).flags.writeable
    view.set_points(pts)
    assert view._pts.flags.writeable
    view._pts[0] = [1.0, 2.0]  # would raise "assignment destination is read-only"
    assert np.allclose(view._pts[0], [1.0, 2.0])


def test_3d_drag_live_updates_every_view(qapp, result, tmp_path):
    from deeperfly.gui.window import MainWindow

    source = _blank_source(result)
    state = EditorState.from_result(result)
    window = MainWindow(
        state,
        source,
        results_path=str(tmp_path / "results.h5"),
        corrections_path=tmp_path / "corrections.h5",
    )
    window._mode_combo.setCurrentIndex(window._mode_combo.findData(EditMode.edit_3d))

    point = 5
    before = [np.array(v._pts[point]) for v in window._views]
    target = before[0] + np.array([10.0, -7.0])
    # A mid-drag move (not a release) must already move every view's reprojection.
    window._views[0].pointDragging.emit(0, point, float(target[0]), float(target[1]))
    after = [np.array(v._pts[point]) for v in window._views]

    assert np.allclose(after[0], target, atol=1e-3)  # dragged view lands on cursor
    assert any(  # at least one other view followed live
        not np.allclose(before[i], after[i]) for i in range(1, len(window._views))
    )
    assert state.dirty

    state.corrections.dirty = False
    window.close()


def test_window_right_click_fixes_point_in_edit_3d(qapp, result, tmp_path):
    from deeperfly.gui.window import MainWindow

    source = _blank_source(result)
    state = EditorState.from_result(result)
    window = MainWindow(
        state,
        source,
        results_path=str(tmp_path / "results.h5"),
        corrections_path=tmp_path / "corrections.h5",
    )
    window._mode_combo.setCurrentIndex(window._mode_combo.findData(EditMode.edit_3d))

    point = 5
    window._views[1].pointFixToggled.emit(1, point)
    assert state.corrections.pts2d_fixed[1, state.frame, point]
    # the ring shows on that view
    assert window._views[1]._fixed is not None and window._views[1]._fixed[point]
    # other views are not marked
    assert not window._views[0]._fixed[point]

    # a right-click outside Edit 3D is ignored
    window._mode_combo.setCurrentIndex(window._mode_combo.findData(EditMode.edit_2d))
    window._views[2].pointFixToggled.emit(2, point)
    assert not state.corrections.pts2d_fixed[2, state.frame, point]

    state.corrections.dirty = False
    window.close()


def test_dragged_joint_follows_cursor_despite_fixed_constraint(qapp, result, tmp_path):
    # Regression: while dragging, the dragged joint must track the mouse exactly.
    # With another view fixed, the constrained 3D re-solve reprojects the dragged
    # joint a hair off the cursor; that live update must not overwrite the
    # cursor-pinned joint mid-drag (it would feel like resistance).
    from deeperfly.gui.window import MainWindow

    source = _blank_source(result)
    state = EditorState.from_result(result)
    window = MainWindow(
        state,
        source,
        results_path=str(tmp_path / "results.h5"),
        corrections_path=tmp_path / "corrections.h5",
    )
    window._mode_combo.setCurrentIndex(window._mode_combo.findData(EditMode.edit_3d))

    point = 5
    # Fix view 1 so view 0's drag is genuinely constrained (2-observation DLT).
    window._views[1].pointFixToggled.emit(1, point)

    view0 = window._views[0]
    target = np.array(view0._pts[point]) + np.array([12.0, -9.0])
    # Reproduce the state a real mouseMoveEvent sets up before emitting.
    view0._dragging = point
    view0._pts[point] = target
    view0.pointDragging.emit(0, point, float(target[0]), float(target[1]))

    # The dragged joint stays exactly under the cursor (no resistance) even though
    # the constrained reprojection for view 0 lands elsewhere.
    assert np.allclose(view0._pts[point], target, atol=1e-6)
    proj = state.display_pts3d_projected()[0, point]
    assert not np.allclose(proj, target, atol=1e-3)  # the model disagrees, as expected

    view0._dragging = None
    state.corrections.dirty = False
    window.close()


def test_3d_drag_release_pins_view_in_window(qapp, result, tmp_path):
    # End-to-end: a real press/drag/release in Edit 3D pins the dropped view at
    # the cursor (fixed flag + ring) so it does not snap, even with another view
    # already fixed to constrain the solve.
    from deeperfly.gui.window import MainWindow

    source = _blank_source(result)
    state = EditorState.from_result(result)
    window = MainWindow(
        state,
        source,
        results_path=str(tmp_path / "results.h5"),
        corrections_path=tmp_path / "corrections.h5",
    )
    window._mode_combo.setCurrentIndex(window._mode_combo.findData(EditMode.edit_3d))

    point = 5
    window._views[0].pointFixToggled.emit(0, point)  # a constraining fixed view

    view2 = window._views[2]
    target = np.array(view2._pts[point]) + np.array([13.0, -10.0])
    # Reproduce the events a real mouse drag emits: live moves, then a release
    # (mouseReleaseEvent clears _dragging before emitting pointDragged).
    view2._dragging = point
    view2._pts[point] = target
    view2.pointDragging.emit(2, point, float(target[0]), float(target[1]))
    view2._dragging = None
    view2.pointDragged.emit(2, point, float(target[0]), float(target[1]))

    assert state.corrections.pts2d_fixed[2, state.frame, point]  # pinned
    assert view2._fixed is not None and view2._fixed[point]  # ring shows
    assert np.allclose(view2._pts[point], target, atol=1e-4)  # no snap

    state.corrections.dirty = False
    window.close()


def test_window_save_writes_sidecar(qapp, result, tmp_path):
    from deeperfly.gui.window import MainWindow

    source = _blank_source(result)
    state = EditorState.from_result(result)
    corrections_path = tmp_path / "corrections.h5"
    window = MainWindow(
        state,
        source,
        results_path=str(tmp_path / "results.h5"),
        corrections_path=corrections_path,
    )
    state.apply_2d_edit(0, 1, (3.0, 4.0), frame=0)
    window._on_save()

    assert corrections_path.exists()
    assert not state.dirty
    window.close()  # not dirty -> no dialog
