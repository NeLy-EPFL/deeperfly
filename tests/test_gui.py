"""Tests for the GUI's editor core: state, labels overlay, footage resolution.

The editor logic -- :class:`EditorState` and the :class:`Labels` overlay -- carries
no web/Qt dependency and is tested directly here. The FastAPI server that drives it
over HTTP/WebSocket is tested in ``test_gui_server.py``; the sparse ``labels.h5``
sidecar and the per-point 3D solve have their own suites (``test_gui_labels.py`` /
``test_gui_solve.py``).

The editor is a ground-truth annotation tool: per ``(view, frame, point)`` the
operator authors a tri-state (GT pixel / occluded / nothing) and the 3D is derived.
So a "2D edit" creates a GT pixel, ``toggle_fixed`` confirms/clears a GT, and
``toggle_invisible`` toggles the occluded flag.
"""

from __future__ import annotations

import numpy as np

from deeperfly.gui import EditorState, resolve_footage
from deeperfly.gui.labels import Provenance

# -- displayed 2D: GT over prediction -----------------------------------------


def test_2d_edit_creates_gt_without_touching_original(result):
    original = result.pts2d.copy()
    state = EditorState.from_result(result)
    state.apply_2d_edit(view=0, point=3, xy=(12.0, 34.0), frame=1)

    disp = state.display_pts2d(1)
    assert np.allclose(disp[0, 3], [12.0, 34.0])  # GT shows here
    assert np.allclose(disp[1, 3], result.pts2d[1, 1, 3])  # other view: the prediction
    np.testing.assert_array_equal(result.pts2d, original)  # result never mutated
    assert state.labels.has_gt[0, 1, 3]
    assert state.labels.gt_provenance[0, 1, 3] == Provenance.DRAGGED
    assert state.dirty


def test_occluded_view_displays_nan(result):
    state = EditorState.from_result(result)
    state.toggle_invisible(0, 5, frame=0)
    disp = state.display_pts2d(0)
    assert not np.all(np.isfinite(disp[0, 5]))  # occluded -> NaN (front-end ghosts it)


# -- placeholder seeds for joints absent from a view --------------------------


def _reject_point(result, point):
    """NaN one point across every view + its 3D -- a point triangulation dropped."""
    result.pts2d[:, :, point] = np.nan
    result.pts3d[:, point] = np.nan
    result.reproj_error[:, :, point] = np.nan
    return result


def test_placeholder_seeds_only_absent_joints(result):
    # A rejected point (NaN in every view, no 3D) gets a finite draggable seed in
    # every view; nothing that is actually shown gets a seed on top of it.
    p = 5
    state = EditorState.from_result(_reject_point(result, p))
    ph = state.placeholder_pts2d(0)
    assert np.isfinite(ph[:, p]).all()
    disp = state.display_pts2d(0)
    shown = np.isfinite(disp).all(axis=-1)  # (V, P)
    seeded = np.isfinite(ph).all(axis=-1)  # (V, P)
    assert not (shown & seeded).any()  # a seed only where the joint is absent
    assert np.isnan(ph[:, 0]).all()  # a fully-observed point gets no seed


def test_placeholder_prefers_raw_detection(result):
    # The top-priority seed is the raw detector pixel triangulation dropped.
    p = 5
    raw = result.pts2d.copy()  # a finite detection everywhere, before we reject it
    state = EditorState.from_result(_reject_point(result, p), raw_pts2d=raw)
    ph = state.placeholder_pts2d(0)
    assert np.allclose(ph[:, p], raw[:, 0, p])


def test_placeholder_skips_occluded_view(result):
    # Occluding a view is the operator asserting the point cannot be placed there, so
    # that view gets no seed while the others still do.
    p = 5
    state = EditorState.from_result(_reject_point(result, p))
    state.toggle_invisible(0, p, frame=0)
    ph = state.placeholder_pts2d(0)
    assert np.isnan(ph[0, p]).all()
    assert np.isfinite(ph[1, p]).all()


def test_placeholder_falls_back_to_image_centre(result):
    # With no raw detection and no neighbour/temporal signal (a single-frame result
    # whose whole skeleton is gone), the seed is the view's image centre.
    result.pts2d[:] = np.nan
    result.pts3d[:] = np.nan
    result.reproj_error[:] = np.nan
    sizes = {name: (480, 640) for name in result.cameras.names}
    state = EditorState.from_result(result, image_sizes=sizes)
    ph = state.placeholder_pts2d(0)
    assert np.allclose(ph[:, 0], [640 / 2, 480 / 2])


def test_3d_edit_authors_gt_even_when_3d_unsolvable(result):
    # A first view dropped on an otherwise-absent point authors GT even though a single
    # usable view + NaN prior yields no 3D; a second view then triangulates it.
    p = 5
    state = EditorState.from_result(_reject_point(result, p))
    assert np.isnan(state.display_pts3d(0)[p]).any()  # rejected -> NaN 3D
    x1 = state.apply_3d_edit(view=0, point=p, xy=(100.0, 120.0), frame=0)
    assert x1 is None  # nothing to triangulate from one view
    assert state.labels.has_gt[0, 0, p]  # ... but the GT pixel still stuck
    assert np.isnan(state.display_pts3d(0)[p]).any()  # 3D still unresolved
    x2 = state.apply_3d_edit(view=1, point=p, xy=(140.0, 90.0), frame=0)
    assert x2 is not None and np.isfinite(x2).all()  # two views -> a finite 3D
    assert np.isfinite(state.display_pts3d(0)[p]).all()


# -- derived 3D ---------------------------------------------------------------


def test_derived_3d_recovers_run_pose_with_no_labels(result):
    # With no labels the point is triangulated from all predictions (the configured
    # method), recovering the run's 3D exactly on the exact-projection fixture.
    state = EditorState.from_result(result)
    assert np.allclose(state.display_pts3d(0), result.pts3d[0], atol=1e-6)


def test_3d_edit_lands_on_cursor_and_moves_other_views(result):
    state = EditorState.from_result(result)
    view, point, frame = 2, 5, 0
    before = state.display_pts3d_projected(frame).copy()
    drag = before[view, point] + np.array([10.0, -8.0])

    x_new = state.apply_3d_edit(view, point, drag, frame)
    assert x_new is not None

    # the dragged view is GT, so the displayed (refined) 2D lands exactly on the cursor
    refine = state.display_pts2d_refine(frame)
    assert np.allclose(refine[view, point], drag, atol=1e-6)
    # at least one other view's reprojection moved as the 3D re-solved
    other = (view + 1) % state.n_views
    after = state.display_pts3d_projected(frame)
    assert not np.allclose(after[other, point], before[other, point])
    assert state.dirty


def test_3d_edit_unavailable_without_3d(result):
    result.pts3d = None
    state = EditorState.from_result(result)
    assert not state.has_3d
    assert state.display_pts3d_projected(0) is None
    assert state.apply_3d_edit(0, 0, (1.0, 2.0), 0) is None


def test_reset_point_clears_labels(result):
    state = EditorState.from_result(result)
    state.apply_2d_edit(0, 4, (1.0, 2.0), frame=0)
    state.apply_3d_edit(1, 4, state.display_pts3d_projected(0)[1, 4] + 5.0, frame=0)
    assert state.labels.any_labels

    state.reset_point(4, frame=0)
    assert not state.labels.has_gt[:, 0, 4].any()
    assert not state.labels.occluded[:, 0, 4].any()


def test_reset_point_view_clears_only_that_view(result):
    state = EditorState.from_result(result)
    state.apply_2d_edit(0, 4, (1.0, 2.0), frame=0)
    state.apply_2d_edit(1, 4, (3.0, 4.0), frame=0)

    state.reset_point_view(0, 4, frame=0)
    assert not state.labels.has_gt[0, 0, 4]  # reverted
    assert state.labels.has_gt[1, 0, 4]  # the other view is left alone


def test_reset_frame_clears_only_that_frame(result):
    state = EditorState.from_result(result)
    state.apply_2d_edit(0, 4, (1.0, 2.0), frame=0)
    state.apply_2d_edit(1, 7, (3.0, 4.0), frame=0)
    state.apply_2d_edit(0, 4, (5.0, 6.0), frame=1)  # a different frame, left untouched

    state.reset_frame(frame=0)
    assert not state.labels.has_gt[:, 0].any()  # every point/view reverted
    assert state.labels.has_gt[0, 1, 4]  # the other frame is left alone


# -- corrected (labelled) frames ----------------------------------------------


def test_corrected_frames_tracks_labels_and_resets(result):
    state = EditorState.from_result(result)
    assert state.corrected_frames() == []  # a fresh overlay carries no labels

    state.apply_2d_edit(0, 3, (12.0, 34.0), frame=1)
    state.apply_2d_edit(2, 7, (5.0, 6.0), frame=1)
    state.apply_3d_edit(0, 9, state.display_pts3d_projected(4)[0, 9] + 5.0, frame=4)

    listed = state.corrected_frames()
    assert [f["frame"] for f in listed] == [1, 4]  # sorted ascending
    assert all(not f["reviewed"] for f in listed)  # nothing reviewed yet

    state.reset_frame(frame=1)  # reverting a frame drops it from the list
    assert [f["frame"] for f in state.corrected_frames()] == [4]


def test_corrected_frames_lists_occluded_toggle(result):
    state = EditorState.from_result(result)
    view, point = 1, 5
    state.toggle_invisible(view, point, frame=0)
    listed = state.corrected_frames()
    assert [f["frame"] for f in listed] == [0] and not listed[0]["reviewed"]


def test_reviewed_flag_lists_frame_and_survives_reset(result):
    state = EditorState.from_result(result)
    # A frame marked reviewed is listed even with no point labels...
    state.set_reviewed(True, frame=2)
    listed = state.corrected_frames()
    assert [f["frame"] for f in listed] == [2] and listed[0]["reviewed"]

    # ... and stays listed (still reviewed) after its point labels are reset.
    state.apply_2d_edit(0, 3, (1.0, 2.0), frame=2)
    state.reset_frame(frame=2)
    listed = state.corrected_frames()
    assert [f["frame"] for f in listed] == [2] and listed[0]["reviewed"]

    # Clearing the flag on a frame with no labels drops it from the list entirely.
    state.set_reviewed(False, frame=2)
    assert state.corrected_frames() == []


# -- GT as a 3D constraint (the old "fixed" gesture) --------------------------


def test_3d_edit_with_no_other_labels_lands_on_cursor(result):
    # The dragged view is GT, so the refined display lands exactly on the cursor
    # regardless of how the shared 3D re-solves.
    state = EditorState.from_result(result)
    view, point, frame = 2, 5, 0
    drag = state.display_pts2d_refine(frame)[view, point] + np.array([10.0, -8.0])
    assert state.apply_3d_edit(view, point, drag, frame) is not None
    assert np.allclose(state.display_pts2d_refine(frame)[view, point], drag, atol=1e-6)


def test_toggle_fixed_confirms_then_clears_gt(result):
    state = EditorState.from_result(result)
    view, point, frame = 1, 5, 0
    pix = state.display_pts2d_refine(frame)[view, point].copy()

    assert state.toggle_fixed(view, point, frame) is True
    assert state.labels.has_gt[view, frame, point]
    assert np.allclose(state.labels.gt[view, frame, point], pix)
    assert (
        state.labels.gt_provenance[view, frame, point]
        == Provenance.CONFIRMED_PREDICTION
    )
    assert np.allclose(state.display_pts2d_refine(frame)[view, point], pix)

    assert state.toggle_fixed(view, point, frame) is False
    assert not state.labels.has_gt[view, frame, point]


def test_gt_view_pixel_is_held_when_another_view_is_dragged(result):
    state = EditorState.from_result(result)
    point, frame = 5, 0
    gt_view, drag_view = 0, 2
    held = state.display_pts2d_refine(frame)[gt_view, point].copy()
    state.toggle_fixed(gt_view, point, frame)

    drag = state.display_pts2d_refine(frame)[drag_view, point] + np.array([12.0, -9.0])
    state.apply_3d_edit(drag_view, point, drag, frame)

    # the confirmed view never moves, even though the 3D point was re-solved
    assert np.allclose(state.display_pts2d_refine(frame)[gt_view, point], held)


def test_two_gt_views_recover_the_point(result):
    # The synthetic 2D is the exact projection of the 3D, so confirming two views at
    # their (consistent) pixels must re-triangulate back to the true 3D point.
    state = EditorState.from_result(result)
    point, frame = 5, 0
    a, b = 0, 3
    state.toggle_fixed(a, point, frame)
    state.toggle_fixed(b, point, frame)

    assert np.allclose(
        state.display_pts3d(frame)[point], result.pts3d[frame, point], atol=1e-6
    )


def test_dragging_a_gt_view_keeps_it_under_cursor(result):
    state = EditorState.from_result(result)
    view, point, frame = 0, 5, 0
    state.toggle_fixed(view, point, frame)
    state.toggle_fixed(3, point, frame)  # a second GT view

    drag = state.display_pts2d_refine(frame)[view, point] + np.array([8.0, 6.0])
    state.apply_3d_edit(view, point, drag, frame)
    # a GT view's own drag places it there: it stays under the cursor
    assert np.allclose(state.display_pts2d_refine(frame)[view, point], drag, atol=1e-6)


# -- occluded (obscured) views ------------------------------------------------


def test_toggle_invisible_drops_a_biased_view_from_the_solve(result):
    # A small (inlier) perturbation on one view biases the derived 3D; occluding that
    # view drops it and recovers the clean solve.
    state = EditorState.from_result(result)
    bad_view, point, frame = 0, 5, 0
    true = result.pts3d[frame, point].copy()
    result.pts2d[bad_view, frame, point] += np.array([6.0, 5.0])  # within RANSAC thresh
    before = state.display_pts3d(frame)[point].copy()

    assert state.toggle_invisible(bad_view, point, frame) is True
    assert state.labels.occluded[bad_view, frame, point]
    after = state.display_pts3d(frame)[point]
    assert np.linalg.norm(after - true) < np.linalg.norm(before - true)
    assert np.allclose(after, true, atol=1e-6)


def test_toggle_invisible_then_back(result):
    state = EditorState.from_result(result)
    view, point, frame = 0, 5, 0
    assert state.toggle_invisible(view, point, frame) is True
    assert state.toggle_invisible(view, point, frame) is False
    assert not state.labels.occluded[view, frame, point]


def test_occluded_and_gt_are_mutually_exclusive(result):
    state = EditorState.from_result(result)
    view, point, frame = 1, 5, 0

    state.toggle_fixed(view, point, frame)  # confirm GT
    assert state.labels.has_gt[view, frame, point]
    state.toggle_invisible(view, point, frame)  # then occlude -> drops the GT
    assert state.labels.occluded[view, frame, point]
    assert not state.labels.has_gt[view, frame, point]


def test_dragging_an_occluded_view_un_occludes_it(result):
    state = EditorState.from_result(result)
    view, point, frame = 2, 5, 0
    state.toggle_invisible(view, point, frame)
    assert state.labels.occluded[view, frame, point]

    drag = state.display_pts2d_refine(frame)[view, point] + np.array([8.0, 6.0])
    assert state.apply_3d_edit(view, point, drag, frame) is not None
    assert not state.labels.occluded[view, frame, point]  # placing it un-occludes it
    assert state.labels.has_gt[view, frame, point]
    assert np.allclose(state.display_pts2d_refine(frame)[view, point], drag, atol=1e-6)


def test_toggle_invisible_unavailable_without_3d(result):
    result.pts3d = None
    state = EditorState.from_result(result)
    assert state.toggle_invisible(0, 0, 0) is None


def test_reset_point_clears_occluded(result):
    state = EditorState.from_result(result)
    state.toggle_invisible(1, 4, frame=0)
    assert state.labels.occluded[1, 0, 4]
    state.reset_point(4, frame=0)
    assert not state.labels.occluded[:, 0, 4].any()


def test_fresh_overlay_seeds_nothing(result):
    # Unlike the old corrections overlay, a fresh labels overlay authors nothing:
    # a missed detection is "absent" (derived), not a stored occlusion.
    result.pts2d[0, 1, 4] = np.nan
    state = EditorState.from_result(result)
    assert not state.labels.any_labels
    assert not state.dirty


# -- undo / redo --------------------------------------------------------------


def test_undo_redo_reverts_and_reapplies_a_drag(result):
    state = EditorState.from_result(result)
    v, p, f = 2, 5, 0
    drag = state.display_pts2d_refine(f)[v, p] + np.array([9.0, -7.0])
    state.apply_3d_edit(v, p, drag, f)
    assert state.labels.has_gt[v, f, p] and state.can_undo

    assert state.undo() == f
    assert not state.labels.has_gt[v, f, p]  # reverted to no GT
    assert state.can_redo

    assert state.redo() == f
    assert state.labels.has_gt[v, f, p]  # re-applied


def test_a_streamed_drag_is_one_undo_step(result):
    state = EditorState.from_result(result)
    v, p, f = 2, 5, 0
    for dx in (1.0, 2.0, 3.0):  # a streamed drag: many messages, same point
        base = state.display_pts2d_refine(f)[v, p]
        state.apply_3d_edit(v, p, base + np.array([dx, 0.0]), f)
    assert len(state._undo) == 1  # coalesced into a single step
    state.undo()
    assert not state.labels.has_gt[v, f, p]


def test_reset_is_undoable(result):
    state = EditorState.from_result(result)
    state.apply_2d_edit(0, 4, (1.0, 2.0), frame=0)
    state.reset_point(4, frame=0)
    assert not state.labels.has_gt[0, 0, 4]
    state.undo()  # undo the reset -> the GT returns
    assert state.labels.has_gt[0, 0, 4]


# -- bulk confirm -------------------------------------------------------------


def test_confirm_predictions_promotes_to_gt(result):
    state = EditorState.from_result(result)
    f = 0
    targets = [(v, 5) for v in range(state.n_views)]
    assert state.confirm(targets, "predictions", f) is True
    for v in range(state.n_views):
        assert state.labels.has_gt[v, f, 5]
        assert state.labels.gt_provenance[v, f, 5] == Provenance.CONFIRMED_PREDICTION
    # one undo step reverts the whole bulk confirm
    assert state.undo() == f
    assert not state.labels.has_gt[:, f, 5].any()


def test_confirm_skips_occluded_and_existing_gt(result):
    state = EditorState.from_result(result)
    f = 0
    state.toggle_invisible(0, 5, f)  # occlude view 0
    state.toggle_fixed(1, 5, f)  # already GT at view 1
    state.confirm([(v, 5) for v in range(state.n_views)], "predictions", f)
    assert not state.labels.has_gt[0, f, 5]  # occluded view left untouched
    assert state.labels.occluded[0, f, 5]
    assert state.labels.has_gt[1, f, 5]  # pre-existing GT kept


def test_confirm_projections_uses_reprojection(result):
    state = EditorState.from_result(result)
    f, point = 0, 5
    proj = state.display_pts3d_projected(f)[:, point].copy()
    assert state.confirm([(v, point) for v in range(state.n_views)], "projections", f)
    for v in range(state.n_views):
        assert (
            state.labels.gt_provenance[v, f, point] == Provenance.CONFIRMED_PROJECTION
        )
        assert np.allclose(state.labels.gt[v, f, point], proj[v], atol=1e-6)


def test_clear_gt_reverts_to_prediction(result):
    state = EditorState.from_result(result)
    state.apply_2d_edit(0, 5, (10.0, 20.0), 0)
    assert state.labels.has_gt[0, 0, 5]
    state.clear_gt(0, 5, 0)
    assert not state.labels.has_gt[0, 0, 5]
    assert np.allclose(state.display_pts2d(0)[0, 5], result.pts2d[0, 0, 5])


# -- footage resolution -------------------------------------------------------


def test_resolve_footage_prefers_absolute_then_relative_then_dir(tmp_path):
    import os as _os

    results_dir = tmp_path / "deeperfly_outputs"
    results_dir.mkdir()
    video = tmp_path / "camera_0.mp4"
    video.write_bytes(b"x")
    rel = _os.path.relpath(video, results_dir)

    footage = {"cam0": {"abs": [str(video)], "rel": [rel]}}
    resolved, missing = resolve_footage(footage, results_dir)
    assert missing == [] and resolved["cam0"] == [video]

    footage = {"cam0": {"abs": ["/nope/camera_0.mp4"], "rel": [rel]}}
    resolved, missing = resolve_footage(footage, results_dir)
    assert missing == [] and resolved["cam0"][0].resolve() == video.resolve()

    footage = {"cam0": {"abs": ["/nope/camera_0.mp4"], "rel": ["../gone/camera_0.mp4"]}}
    resolved, missing = resolve_footage(footage, results_dir, footage_dir=tmp_path)
    assert missing == [] and resolved["cam0"] == [video]

    resolved, missing = resolve_footage(footage, results_dir)
    assert missing == ["cam0"] and resolved == {}


def test_resolve_footage_none_is_empty(tmp_path):
    resolved, missing = resolve_footage(None, tmp_path)
    assert resolved == {} and missing == []


# -- browser launch: silence the external opener's stdio ----------------------


def test_quiet_child_output_swallows_then_restores(tmp_path):
    """`_quiet_child_output` drops OS-level writes in-block and restores fd 2 after."""
    import os

    from deeperfly.gui import _quiet_child_output

    sink = tmp_path / "stderr.log"
    fd = os.open(sink, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    saved = os.dup(2)
    os.dup2(fd, 2)
    os.close(fd)
    try:
        with _quiet_child_output():
            os.write(2, b"swallowed\n")
        os.write(2, b"kept\n")
    finally:
        os.dup2(saved, 2)
        os.close(saved)

    assert sink.read_text() == "kept\n"
