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
