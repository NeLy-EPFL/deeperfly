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
import pytest

from deeperfly.gui import EditorState, resolve_footage

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


def test_a_triangulation_rejected_peak_is_still_a_detection(result):
    """The detected layer is the network's output, so a rejected peak still shows.

    ``result.pts2d`` is NaN wherever the pipeline rejected a peak; the detected layer reads
    ``raw_pts2d`` instead, so the operator keeps seeing what the network actually said --
    which is the best thing to drag a GT from. The cell therefore needs no placeholder
    seed at all: it has a position of its own.
    """
    p = 5
    raw = result.pts2d.copy()  # a finite detection everywhere, before we reject it
    state = EditorState.from_result(_reject_point(result, p), raw_pts2d=raw)
    np.testing.assert_allclose(state.display_pts2d(0)[:, p], raw[:, 0, p])
    assert np.isnan(state.placeholder_pts2d(0)[:, p]).all()  # nothing to seed


def test_placeholder_seeds_a_projected_view_at_its_reprojection(result):
    # A Projected (occluded) view has no position of its OWN, so it is seeded even when a
    # reprojection exists -- and the seed sits exactly on that reprojection. The front-end
    # hides the seed underneath the visible ring (poseView.js ``placeholderPos``); it is
    # there for when the operator hides the reprojected overlay (`p`), which would
    # otherwise take the cell's only handle with it.
    p = 5
    state = EditorState.from_result(result)  # 3D intact: the reprojection exists
    state.toggle_invisible(0, p, frame=0)
    proj = state.display_pts3d_projected(0)
    assert np.isfinite(proj[0, p]).all()  # a ghost is drawn there
    ph = state.placeholder_pts2d(0)
    assert np.allclose(ph[0, p], proj[0, p])  # seeded, right where the ring is
    assert np.isnan(ph[1, p]).all()  # an unoccluded view keeps its own detection


def test_placeholder_seeds_a_projected_view_with_no_reprojection_to_follow(result):
    # The real strand, reproduced: a joint the run's triangulation rejected (finite
    # detections, NaN 3D -- so `_solve_point` has no cache to fall back on either) that
    # the operator then set Projected in EVERY view. The solve is now below two usable
    # views, so there is no 3D and hence no reprojection to follow: the detection is
    # suppressed and the ghost never appears. Every view must still get a draggable seed,
    # or the joint is unreachable -- not drawn means not hit-testable, so it could not even
    # be selected to be Reset. Regression: rh_tibia_tarsus vanished from a real labeled
    # frame exactly this way (all 6 views Projected, RANSAC had dropped its 3D).
    p = 5
    # No detections anywhere and no cached 3D: nothing can place this joint. (Excluding
    # every view would NOT do it -- an exclusion that would leave the point unsolvable
    # stands down, see test_excluding_every_view_keeps_the_projection_alive.)
    result.pts2d[:, :, p] = np.nan
    result.pts3d[:, p] = np.nan
    result.reproj_error[:, :, p] = np.nan
    state = EditorState.from_result(result)
    for v in range(state.n_views):
        state.toggle_invisible(v, p, frame=0)
    assert np.isnan(state.display_pts3d(0)[p]).any()  # nothing left to project
    assert np.isnan(state.display_pts2d(0)[:, p]).all()  # ... and nothing displayed
    ph = state.placeholder_pts2d(0)
    assert np.isfinite(ph[:, p]).all()  # every view stays correctable


def test_placeholder_seeds_a_projected_view_of_a_rejected_point(result):
    # The same strand via the other route: a point triangulation dropped (no detection,
    # no 3D) that the operator then set Projected in one view. That view has nothing to
    # follow either, so it is seeded like the rest.
    p = 5
    state = EditorState.from_result(_reject_point(result, p))
    state.toggle_invisible(0, p, frame=0)
    ph = state.placeholder_pts2d(0)
    assert np.isfinite(ph[0, p]).all()
    assert np.isfinite(ph[1, p]).all()


def _grabbable(state, frame, *, projected_visible=True):
    """``(V, P)`` bool: does the canvas draw SOMETHING grabbable for each cell?

    Mirrors the front-end's position precedence (poseView.js): the cell's own position
    -- GT, else the view's usable detection, which an occluded view has none of
    (``detPos``) -- else the reprojection while that overlay is shown
    (``shownLatentPos``), else the Missing seed (``placeholderPos``). Not drawn means not
    hit-testable and not marquee-selectable (``grabCandidates``), i.e. unreachable.
    """

    def on_image(a):
        """Finite is not enough: a position outside the canvas is drawn nowhere.

        This clause was missing, and that is precisely how a real frame lost a point --
        IN10B014_260320_Fly2_009 frame 40 reprojected `rh_claw` to x=-5.5 in view `rf`, which is
        finite, passed every grabbability test here, and was invisible in the GUI.
        """
        ok = np.isfinite(a).all(axis=-1)
        if state.image_sizes_wh is None:
            return ok
        w = state.image_sizes_wh[:, 0][:, None]
        h = state.image_sizes_wh[:, 1][:, None]
        x, y = a[..., 0], a[..., 1]
        with np.errstate(invalid="ignore"):
            return ok & (x >= 0) & (x < w) & (y >= 0) & (y < h)

    own = on_image(state.display_pts2d(frame))  # GT over detection
    seed = on_image(state.placeholder_pts2d(frame))
    proj = state.display_pts3d_projected(frame) if state.has_3d else None
    ring = (
        on_image(proj) if proj is not None and projected_visible else np.zeros_like(own)
    )
    return own | ring | seed


def test_every_joint_stays_grabbable_after_bulk_projected(result):
    # THE invariant the seed layer exists for: `a` then `3` -- select every point in every
    # view, set them all Projected -- must leave every cell with something to grab. Every
    # view is now dropped from every point's solve, so a point the run cached no 3D for has
    # no ghost either (the solve is below two usable views and the run-cache fallback is
    # empty), and the seed is its only handle. Checked with the reprojected overlay both on
    # and OFF: hiding it must not strip the last handle off an occluded cell either.
    # Regression: lm_claw vanished from a real labeled frame exactly this way.
    result.pts2d[:, :, 5] = np.nan  # a joint nothing can place: no detection, no 3D
    result.pts3d[:, 5] = np.nan
    result.reproj_error[:, :, 5] = np.nan
    state = EditorState.from_result(result)
    targets = [(v, p) for v in range(state.n_views) for p in range(state.n_points)]
    state.occlude_targets(targets, 0)

    assert np.isnan(state.display_pts2d(0)).all()  # no cell has a position of its own
    assert np.isnan(state.display_pts3d(0)[5]).any()  # ... and point 5 has no 3D left
    for projected_visible in (True, False):
        assert _grabbable(state, 0, projected_visible=projected_visible).all()


def test_every_joint_stays_grabbable_in_the_untouched_frame(result):
    # The same invariant on a frame nobody has edited, with the reprojected overlay hidden
    # -- a detector miss must not be reachable only via an overlay the operator can turn
    # off. Point 5 is missing from view 0's detections but keeps its 3D.
    result.pts2d[0, :, 5] = np.nan
    state = EditorState.from_result(result)
    for projected_visible in (True, False):
        assert _grabbable(state, 0, projected_visible=projected_visible).all()


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


# -- corrected (labeled) frames ----------------------------------------------


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


def test_marking_a_view_occluded_does_not_change_the_solve(result):
    """Occlusion is a statement about the image, not an instruction to the solve.

    It used to drop the view -- which was its whole point, back when a hand-exclusion was
    the only defence against a bad observation. The robust estimator is that defence now,
    so the flag is free to mean only what it says, and a cell can be both hand-placed and
    marked not-visible without the two contradicting each other.
    """
    state = EditorState.from_result(result)
    view, point, frame = 0, 5, 0
    before = state.display_pts3d(frame)[point].copy()

    assert state.toggle_invisible(view, point, frame) is True
    assert state.labels.occluded[view, frame, point]
    np.testing.assert_allclose(state.display_pts3d(frame)[point], before)


def test_toggle_invisible_then_back(result):
    state = EditorState.from_result(result)
    view, point, frame = 0, 5, 0
    assert state.toggle_invisible(view, point, frame) is True
    assert state.toggle_invisible(view, point, frame) is False
    assert not state.labels.occluded[view, frame, point]


def test_occlusion_and_gt_are_orthogonal(result):
    """Both at once is a legitimate state: a joint placed *through* an occluder.

    They used to be mutually exclusive because occlusion meant "drop this view from the 3D
    solve", so a pixel and an exclusion were contradictory. Occlusion no longer touches the
    solve -- it is a statement about the image -- and "I know where this is from the other
    views, and you cannot see it here" is exactly what a human labeling a contralateral leg
    is asserting. Losing either half would lose a real label.
    """
    state = EditorState.from_result(result)
    view, point, frame = 1, 5, 0

    state.toggle_fixed(view, point, frame)  # place GT
    assert state.labels.has_gt[view, frame, point]
    state.toggle_invisible(view, point, frame)  # ... and mark it not visible here
    assert state.labels.occluded[view, frame, point]
    assert state.labels.has_gt[view, frame, point], "the pixel was destroyed"


def test_dragging_an_occluded_view_places_gt_and_keeps_the_occlusion(result):
    """A drag asserts a position, not visibility, so it leaves the occlusion standing.

    It used to un-occlude, because an occlusion was an instruction to the solve and the
    drag contradicted it. Now it is a statement about the pixels, which the drag does not
    contradict -- placing a joint you cannot see, from the geometry of the views that can,
    is the normal way a contralateral leg gets labeled.
    """
    state = EditorState.from_result(result)
    view, point, frame = 2, 5, 0
    state.toggle_invisible(view, point, frame)

    drag = state.display_pts2d_refine(frame)[view, point] + np.array([8.0, 6.0])
    assert state.apply_3d_edit(view, point, drag, frame) is not None
    assert state.labels.occluded[view, frame, point]  # still not visible here
    assert state.labels.has_gt[view, frame, point]  # ... but we know where it is
    assert np.allclose(state.display_pts2d_refine(frame)[view, point], drag, atol=1e-6)


def test_toggle_invisible_available_without_3d(result):
    # Occlusion is an authored label, not a derived quantity, so it does not need a
    # solve to record. Gating it would leave a 2D-only hand-labeling pass with no way to
    # reject a view except the far stronger "this joint is not on this animal".
    result.pts3d = None
    state = EditorState.from_result(result)
    assert state.toggle_invisible(0, 0, 0) is True
    assert state.labels.occluded[0, 0, 0]
    assert state.toggle_invisible(0, 0, 0) is False


def test_occlude_targets_available_without_3d(result):
    result.pts3d = None
    state = EditorState.from_result(result)
    state.occlude_targets([(0, 4), (1, 4)], frame=0)
    assert state.labels.occluded[0, 0, 4] and state.labels.occluded[1, 0, 4]


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


def test_two_gestures_on_one_point_are_two_undo_steps(result):
    """A gesture ends at its settle, so a later nudge of the same joint is its own step.

    Coalescing used to key on "the top of the undo stack has this same (frame, point)",
    which merged two gestures the operator made minutes apart -- and in different views,
    so one ctrl-z cleared GT in a view they never touched this time.
    """
    state = EditorState.from_result(result)
    f, p = 0, 5
    state.apply_3d_edit(0, p, state.display_pts2d_refine(f)[0, p] + 5.0, f, fix=True)
    state.apply_3d_edit(3, p, state.display_pts2d_refine(f)[3, p] + 5.0, f, fix=True)
    assert len(state._undo) == 2
    state.undo()  # reverts only the second gesture
    assert state.labels.has_gt[0, f, p] and not state.labels.has_gt[3, f, p]


def test_a_new_drag_after_an_undo_does_not_rejoin_the_old_gesture(result):
    """Drag A, drag B, ctrl-z, drag A again: the new drag is its own step ...

    ... and it must not leave a live redo branch. The stale redo entry is a *whole-frame*
    label snapshot, so redoing it re-applied B's edit and clobbered the drag the operator
    had just made on A -- other points moving, from the operator's seat.
    """
    state = EditorState.from_result(result)
    f, a, b = 0, 5, 9
    state.apply_3d_edit(0, a, state.display_pts2d_refine(f)[0, a] + 5.0, f, fix=True)
    state.apply_3d_edit(0, b, state.display_pts2d_refine(f)[0, b] + 5.0, f, fix=True)
    state.undo()  # reverts b
    assert state.can_redo

    xy = state.display_pts2d_refine(f)[0, a] + np.array([11.0, -4.0])
    state.apply_3d_edit(0, a, xy, f, fix=True)  # a fresh gesture on a
    assert not state.can_redo  # the redo branch died with the new edit
    assert len(state._undo) == 2  # ... and it did not rejoin a's first gesture
    np.testing.assert_allclose(state.labels.gt[0, f, a], xy)
    assert not state.labels.has_gt[0, f, b]  # b stayed reverted


def test_reset_is_undoable(result):
    state = EditorState.from_result(result)
    state.apply_2d_edit(0, 4, (1.0, 2.0), frame=0)
    state.reset_point(4, frame=0)
    assert not state.labels.has_gt[0, 0, 4]
    state.undo()  # undo the reset -> the GT returns
    assert state.labels.has_gt[0, 0, 4]


# -- the verbs: create GT, delete GT, exclude a detection ----------------------
#
# A cell has no "state" to assign. It carries a pixel the operator created or it does not,
# and its detection is excluded from triangulation or it is not. These are those verbs.


def test_clear_gt_targets_deletes_only_the_pixel(result):
    """Delete GT and Reset are different retractions and must stay different keys."""
    f, p = 0, 5
    state = EditorState.from_result(result)
    state.toggle_invisible(1, p, frame=f)  # an exclusion the operator authored
    state.apply_2d_edit(0, p, (10.0, 20.0), f)  # ... and a pixel they placed

    state.clear_gt_targets([(0, p)], f)
    assert not state.labels.has_gt[0, f, p]  # the pixel is gone ...
    assert state.labels.occluded[1, f, p]  # ... and the exclusion is untouched

    state.reset_targets([(1, p)], f)
    assert not state.labels.occluded[1, f, p]  # Reset is what retracts that


def test_clear_gt_targets_on_cells_with_no_gt_is_a_no_op(result):
    state = EditorState.from_result(result)
    state.clear_gt_targets([(0, 5), (1, 5)], 0)
    assert not state.can_undo  # no undo step for a batch that changed nothing


def test_toggle_exclude_targets_round_trips(result):
    f, p = 0, 5
    state = EditorState.from_result(result)
    assert (
        state.toggle_exclude_targets([(v, p) for v in range(state.n_views)], f) is True
    )
    assert state.labels.occluded[:, f, p].all()
    assert (
        state.toggle_exclude_targets([(v, p) for v in range(state.n_views)], f) is False
    )
    assert not state.labels.occluded[:, f, p].any()


def test_marking_a_labeled_cell_hidden_keeps_the_pixel(result):
    """The case that matters: a joint placed *through* an occluder is both facts.

    The verb used to skip labeled cells, because storing the flag cleared the pixel under it
    and a bulk toggle would have destroyed the operator's work. The two are orthogonal now,
    so the cell records both -- "here it is" from the geometry of the views that can see it,
    and "the pixels here do not show it" for training. Nothing else can produce that pairing.
    """
    f, p = 0, 5
    state = EditorState.from_result(result)
    state.apply_2d_edit(2, p, (30.0, 40.0), f)
    assert (
        state.toggle_exclude_targets([(v, p) for v in range(state.n_views)], f) is True
    )

    assert state.labels.has_gt[2, f, p]  # the pixel stands
    np.testing.assert_allclose(state.labels.gt[2, f, p], [30.0, 40.0])
    assert state.labels.occluded[:, f, p].all()  # ... including the labeled view


def test_toggle_hidden_refuses_only_for_an_absent_point(result):
    f, p = 0, 5
    state = EditorState.from_result(result)
    state.set_absent([p], True, f)
    assert state.toggle_exclude_targets([(0, p)], f) is None


# -- the annotation instance ---------------------------------------------------
#
# The unit of annotation. Before one exists the editor is showing the detector and nothing
# is authored; creating one seeds a position for every (view, point), and from then on the
# skeleton is its own object whose cells are GT or not-GT -- no third value, because there
# is no longer a layer to defer to.


def test_no_instance_until_one_is_created(result):
    state = EditorState.from_result(result)
    assert not state.has_instance(0)
    assert state.display_instance_pts2d(0) is None
    assert state.create_instance(0) is True
    assert state.has_instance(0)
    assert state.create_instance(0) is False  # idempotent; reseed_instance is the redo


def test_a_joint_with_no_evidence_gets_no_seed_but_is_still_drawn(result):
    """Seeds are observations; the drawn position may be invented. Keep those separate.

    A seed feeds the 3D solve, so it may only hold something real -- a detection, or a
    reprojection. A joint the detector predicts in *no* view (an ipsilateral-only model, and
    the contralateral keypoints this project exists to fix) would otherwise be seeded from
    the placeholder chain's last rungs -- its neighbours' mean, the view centroid, the image
    centre -- and triangulating those manufactures a confident 3D out of coordinates the
    editor made up. So the seed stays NaN and the *display* fills the cell instead, which is
    all the operator needs to see it and drag it into place.
    """
    p_gone = 7
    result.pts2d[:, :, p_gone] = np.nan  # the detector never fired for this joint
    result.pts3d[:, p_gone] = np.nan  # ... and triangulation has nothing either
    state = EditorState.from_result(result)
    assert state.create_instance(0) is True
    assert state.has_instance(0)  # the instance exists even with nothing to seed it

    assert np.isnan(state.labels.seeds[:, 0, p_gone]).all()  # no invented evidence
    _, evidence, _ = state._point_obs(0, p_gone)
    assert not np.isfinite(evidence).any(), "invented pixels reached the solve"
    assert np.isfinite(
        state.display_instance_pts2d(0)[:, p_gone]
    ).all()  # still grabbable
    # every other joint does have evidence, so it is seeded
    assert np.isfinite(state.labels.seeds[:, 0, 5]).all()


@pytest.mark.parametrize("mode", ["triangulate", "copy"])
def test_the_two_seeding_modes_differ_where_a_view_disagrees(result, mode):
    """`triangulate` pulls every view onto the consensus; `copy` keeps its own opinion."""
    f, p, odd = 0, 5, 2
    result.pts2d[odd, f, p] += np.array([40.0, 30.0])  # one view out of line
    state = EditorState.from_result(result)
    state.create_instance(f, mode=mode)
    seed = state.labels.seeds[odd, f, p]
    detection = state.detections[odd, f, p]
    if mode == "copy":
        np.testing.assert_allclose(seed, detection)
    else:
        assert np.linalg.norm(seed - detection) > 1.0  # pulled onto the consensus


def test_an_unknown_seeding_mode_is_refused(result):
    state = EditorState.from_result(result)
    with pytest.raises(ValueError, match="mode must be"):
        state.create_instance(0, mode="vibes")


def test_the_first_drag_in_a_frame_creates_the_instance(result):
    """Implicit creation: telling us where a keypoint is presupposes a skeleton."""
    f, p = 0, 5
    state = EditorState.from_result(result)
    assert not state.has_instance(f)
    state.apply_2d_edit(0, p, (11.0, 22.0), f)
    assert state.has_instance(f)
    assert np.isfinite(state.labels.seeds[1, f, p]).all()  # the rest got seeded too


def test_the_implicit_creation_and_its_drag_are_one_undo_step(result):
    """One gesture, one ctrl-Z -- the creation folds into the drag's own entry."""
    f, p = 0, 5
    state = EditorState.from_result(result)
    state.apply_3d_edit(0, p, (110.0, 130.0), f, fix=True)
    assert state.has_instance(f) and state.labels.has_gt[0, f, p]

    state.undo()
    assert not state.labels.has_gt[0, f, p]
    assert not state.has_instance(f), "the undo left the instance behind"
    assert not state.can_undo  # ... and it really was one step


def test_creating_an_instance_is_one_undo_step(result):
    state = EditorState.from_result(result)
    state.create_instance(0)
    assert state.can_undo
    state.undo()
    assert not state.has_instance(0), "the undo left the instance standing"
    assert np.isnan(state.labels.seeds[:, 0]).all()


def test_reseeding_keeps_every_gt_pixel(result):
    """Seeds are frozen at birth; picking up new detections is explicit and non-destructive."""
    f, p = 0, 5
    state = EditorState.from_result(result)
    state.create_instance(f, mode="copy")
    state.apply_2d_edit(0, p, (11.0, 22.0), f)

    result.pts2d[:, f, :] += 5.0  # the detector was re-run and moved
    assert state.reseed_instance(f, mode="copy") is True
    np.testing.assert_allclose(state.labels.gt[0, f, p], [11.0, 22.0])  # GT untouched
    np.testing.assert_allclose(state.labels.seeds[1, f, p], result.pts2d[1, f, p])
    state.undo()  # ... and the reseed is one undoable step
    assert state.labels.has_gt[0, f, p]


def test_the_instance_draws_gt_over_the_derived_position(result):
    f, p = 0, 5
    state = EditorState.from_result(result)
    state.create_instance(f)
    state.apply_2d_edit(0, p, (11.0, 22.0), f)

    inst = state.display_instance_pts2d(f)
    np.testing.assert_allclose(inst[0, p], [11.0, 22.0])  # the operator's pixel
    proj = state.display_pts3d_projected(f)
    np.testing.assert_allclose(inst[1, p], proj[1, p])  # ... the rest follow the 3D


def test_the_seed_display_mode_shows_what_the_instance_started_from(result):
    f, p = 0, 5
    state = EditorState.from_result(result)
    state.create_instance(f, mode="copy")
    state.apply_3d_edit(0, p, state.detections[0, f, p] + 30.0, f, fix=True)

    state.nongt_display = "reprojection"
    moved = state.display_instance_pts2d(f)[1, p].copy()
    state.nongt_display = "seed"
    held = state.display_instance_pts2d(f)[1, p]
    np.testing.assert_allclose(held, state.labels.seeds[1, f, p])
    assert not np.allclose(moved, held), "the two modes should differ after a drag"


def test_an_absent_point_has_no_instance_position(result):
    f, p = 0, 5
    state = EditorState.from_result(result)
    state.create_instance(f)
    state.set_absent([p], True, f)
    assert np.isnan(state.display_instance_pts2d(f)[:, p]).all()


def test_seeds_survive_a_save_load_round_trip(tmp_path, result):
    """Unpersisted seeds would silently re-solve every non-GT point on reopen."""
    from deeperfly.gui.labels import labels_identity, load_labels, save_labels

    state = EditorState.from_result(result)
    state.create_instance(1, mode="copy")
    identity = labels_identity(
        point_names=list(result.skeleton.point_names),
        camera_names=list(result.cameras.names),
        n_frames=result.n_frames,
        image_sizes={n: (256, 256) for n in result.cameras.names},
        footage={n: {"rel": [f"{n}.mp4"]} for n in result.cameras.names},
    )
    path = tmp_path / "labels.h5"
    save_labels(path, state.labels, identity=identity)
    loaded = load_labels(path, identity=identity)
    assert loaded is not None
    np.testing.assert_allclose(
        np.nan_to_num(loaded.seeds), np.nan_to_num(state.labels.seeds)
    )


# -- what triangulation uses ---------------------------------------------------
#
# GT if GT exists (it overrides the detection in its own view), else the detections that
# were not excluded. An exclusion is a claim about a view, and it is honored right up to
# the point where honoring it would leave the joint unsolvable -- see `_point_obs`.


def test_the_solve_reads_the_instance_seeds_once_one_exists(result):
    """The one substitution the instance model makes: seeds take the detections' job."""
    f, p = 0, 5
    state = EditorState.from_result(result)
    _, before, _ = state._point_obs(f, p)
    np.testing.assert_allclose(before, state.detections[:, f, p], equal_nan=True)

    assert state.create_instance(f) is True
    _, after, _ = state._point_obs(f, p)
    np.testing.assert_allclose(after, state.labels.seeds[:, f, p], equal_nan=True)


def test_gt_takes_over_from_the_seeds_one_view_at_a_time(result):
    """First GT overrides its own view; at two GT the evidence is not consulted at all."""
    f, p = 0, 5
    state = EditorState.from_result(result)
    state.create_instance(f)

    xy0 = state.display_pts2d_refine(f)[0, p] + np.array([7.0, -5.0])
    state.apply_3d_edit(0, p, xy0, f, fix=True)
    gt_obs, pred_obs, _ = state._point_obs(f, p)
    assert np.isfinite(gt_obs[0]).all()  # GT here
    assert not np.isfinite(pred_obs[0]).all()  # so the seed is not used in this view
    assert (
        np.isfinite(pred_obs[1:]).all(axis=-1).any()
    )  # the rest still carry the depth

    xy1 = state.display_pts2d_refine(f)[3, p] + np.array([-6.0, 4.0])
    state.apply_3d_edit(3, p, xy1, f, fix=True)
    gt_obs, pred_obs, _ = state._point_obs(f, p)
    assert int(np.isfinite(gt_obs).all(axis=-1).sum()) == 2
    assert np.isfinite(pred_obs).all(axis=-1).sum() == state.n_views - 2


def test_absence_is_never_softened(result):
    """An exclusion is a claim about a view; absence is a claim about the animal."""
    f, p = 0, 5
    state = EditorState.from_result(result)
    state.set_absent([p], True, f)
    gt_obs, pred_obs, _ = state._point_obs(f, p)
    assert not np.isfinite(pred_obs).any()  # no fallback, unlike an exclusion
    assert np.isnan(state.display_pts3d(f)[p]).all()


# -- an edit never moves a point it did not target ----------------------------
#
# The derived 3D is *almost* a pure function of the labels, and the exception is what
# these guard: a drag on a point with fewer than two usable views stores a ray-slide of
# the prior 3D, which no re-derivation can reproduce (one pixel leaves depth free). So a
# gesture that drops more of the derived-3D cache than it touched silently throws away
# hand-placed depth -- and because the *labels* survive, it reads as "I undid one point
# and the others moved". The fixture below is the case that used to break: an unplaced
# joint placed by hand, then an unrelated edit, then history.

UNPLACED = 10  # a joint triangulation dropped: NaN in the cleaned pts2d in every view
OTHER = 3  # an ordinary joint, the one whose edit gets undone


def _hand_placed_unplaced_joint(result, frame=1, xy=(410.0, 260.0)):
    """A state where ``UNPLACED`` was placed by hand from a single view (a ray-slide).

    Returns ``(state, placed_3d)``. ``placed_3d`` is only recoverable from the cache --
    that is the whole point -- so every assertion below compares against it.
    """
    result.pts2d[:, :, UNPLACED] = np.nan
    state = EditorState.from_result(result)
    for i in range(1, 4):  # the client's streamed drag ...
        f = i / 3
        state.apply_3d_edit(
            0, UNPLACED, (xy[0] * f + 200 * (1 - f), xy[1] * f + 200 * (1 - f)), frame
        )
    state.apply_3d_edit(0, UNPLACED, xy, frame, fix=True)  # ... and its settle
    placed = state.display_pts3d(frame)[UNPLACED].copy()
    assert np.all(np.isfinite(placed))
    return state, placed


def _assert_unplaced_held(state, placed, frame=1):
    np.testing.assert_allclose(state.display_pts3d(frame)[UNPLACED], placed)
    assert state.labels.has_gt[0, frame, UNPLACED]  # the label was never in doubt


def test_undo_of_another_points_drag_keeps_a_hand_placed_depth(result):
    state, placed = _hand_placed_unplaced_joint(result)
    before = state.display_pts2d_refine(1).copy()
    state.apply_3d_edit(0, OTHER, result.pts2d[0, 1, OTHER] + 15.0, 1, fix=True)
    state.undo()
    _assert_unplaced_held(state, placed)
    # and nothing else in the frame moved either
    np.testing.assert_allclose(
        np.nan_to_num(state.display_pts2d_refine(1), nan=-1.0),
        np.nan_to_num(before, nan=-1.0),
    )


def test_redo_of_another_points_drag_keeps_a_hand_placed_depth(result):
    state, placed = _hand_placed_unplaced_joint(result)
    state.apply_3d_edit(0, OTHER, result.pts2d[0, 1, OTHER] + 15.0, 1, fix=True)
    state.undo()
    state.redo()
    _assert_unplaced_held(state, placed)


def test_undo_of_another_points_occlusion_keeps_a_hand_placed_depth(result):
    state, placed = _hand_placed_unplaced_joint(result)
    state.toggle_invisible(2, OTHER, frame=1)
    state.undo()
    _assert_unplaced_held(state, placed)


@pytest.mark.parametrize("whole_recording", [False, True])
def test_marking_another_point_absent_keeps_a_hand_placed_depth(
    result, whole_recording
):
    state, placed = _hand_placed_unplaced_joint(result)
    state.set_absent([OTHER], True, 1, whole_recording=whole_recording)
    _assert_unplaced_held(state, placed)
    state.undo()  # ... and undoing the declaration does not move it either
    _assert_unplaced_held(state, placed)


def test_bulk_confirm_keeps_an_untargeted_hand_placed_depth(result):
    state, placed = _hand_placed_unplaced_joint(result)
    state.confirm([(v, OTHER) for v in range(state.n_views)], "all", 1)
    _assert_unplaced_held(state, placed)
    state.undo()
    _assert_unplaced_held(state, placed)


def test_bulk_reset_and_occlude_keep_an_untargeted_hand_placed_depth(result):
    state, placed = _hand_placed_unplaced_joint(result)
    state.occlude_targets([(2, OTHER)], 1)
    _assert_unplaced_held(state, placed)
    state.reset_targets([(2, OTHER)], 1)
    _assert_unplaced_held(state, placed)
    state.undo()
    _assert_unplaced_held(state, placed)


def test_reset_frame_drops_the_hand_placed_depth_but_undo_restores_it(result):
    # reset_frame really does clear every label in the frame, so re-deriving the whole
    # frame is correct there -- but the undo must still be exact.
    state, placed = _hand_placed_unplaced_joint(result)
    state.reset_frame(1)
    assert not np.allclose(state.display_pts3d(1)[UNPLACED], placed, equal_nan=True)
    state.undo()
    _assert_unplaced_held(state, placed)


def test_undo_pressed_mid_drag_moves_nothing_else(result):
    """Ctrl+Z while the point is still held -- only ``fix=False`` edits have landed."""
    state, placed = _hand_placed_unplaced_joint(result)
    before3d = state.display_pts3d(1).copy()
    before2d = state.display_pts2d_refine(1).copy()
    for dx in (5.0, 10.0, 15.0):  # the mid-drag stream, never released
        state.apply_3d_edit(0, OTHER, result.pts2d[0, 1, OTHER] + dx, 1, fix=False)
    assert state.undo() == 1
    _assert_unplaced_held(state, placed)
    keep = np.ones(before3d.shape[0], bool)
    keep[OTHER] = False
    np.testing.assert_allclose(
        np.nan_to_num(state.display_pts3d(1)[keep], nan=-1.0),
        np.nan_to_num(before3d[keep], nan=-1.0),
    )
    np.testing.assert_allclose(
        np.nan_to_num(state.display_pts2d_refine(1)[:, keep], nan=-1.0),
        np.nan_to_num(before2d[:, keep], nan=-1.0),
    )


def test_the_release_after_a_mid_drag_undo_is_its_own_undo_step(result):
    """The pointerup still lands, and must start a fresh step -- not rejoin the undone one.

    The operator ends with the joint under the cursor, so re-authoring the GT there is
    right; what must not happen is the release silently joining the gesture the undo just
    reverted (which also left the undone edit's redo entry alive to clobber it).
    """
    state, _placed = _hand_placed_unplaced_joint(result)
    for dx in (5.0, 10.0, 15.0):
        state.apply_3d_edit(0, OTHER, result.pts2d[0, 1, OTHER] + dx, 1, fix=False)
    state.undo()
    steps = len(state._undo)
    state.apply_3d_edit(0, OTHER, result.pts2d[0, 1, OTHER] + 15.0, 1, fix=True)
    assert state.labels.has_gt[0, 1, OTHER]  # released there, so authored there
    assert not state.can_redo  # the reverted edit's redo branch is gone
    assert len(state._undo) == steps + 1  # a step of its own
    state.undo()  # ... and one ctrl-z takes it away again
    assert not state.labels.has_gt[0, 1, OTHER]


def test_a_drag_settles_onto_the_configured_solve(result):
    """On release the 3D converges to what a plain re-derivation would give.

    That is what keeps the cache a function of the labels for every point the labels
    actually determine, so no later invalidation can move it.
    """
    state = EditorState.from_result(result)
    f, p = 1, 5
    for v in (0, 3):  # two GT views: the labels now determine the 3D
        xy = state.display_pts2d_refine(f)[v, p] + np.array([6.0, -4.0])
        state.apply_3d_edit(v, p, xy, f, fix=True)
    cached = state.display_pts3d(f)[p].copy()
    state._invalidate_frame3d(f)  # force the pure re-derivation
    np.testing.assert_allclose(state.display_pts3d(f)[p], cached, atol=1e-9)


# -- bulk confirm -------------------------------------------------------------


def test_confirm_predictions_promotes_to_gt(result):
    state = EditorState.from_result(result)
    f = 0
    targets = [(v, 5) for v in range(state.n_views)]
    assert state.confirm(targets, "predictions", f) is True
    for v in range(state.n_views):
        assert state.labels.has_gt[v, f, 5]
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


def test_confirm_skips_a_cell_with_nothing_on_screen(result):
    """A cell whose only candidate is off-image (or missing) is skipped, not invented.

    The editor used to clamp such a position into the frame and store it, so the operator
    would have something to grab. That fabricated a coordinate at the image edge -- the
    worst place to put a training target -- and needed its own provenance code to keep it
    out of the export.

    The two skipped cases differ in what is left behind, and both are right:

    * **no position at all** -> the Unplaced seed layer supplies a draggable ghost, so the
      joint is still reachable in this view without anything being stored.
    * **off-image** -> no ghost, because the cell *does* have a position of its own; it
      simply is not inside this camera's image. A joint outside the frame is not placeable
      from this view, which is what the Projected/occluded state is for, and it stays
      placeable from every view that can see it.
    """
    sizes = {name: (480, 640) for name in result.cameras.names}  # (h, w)
    state = EditorState.from_result(result, image_sizes=sizes)
    state.result.pts3d = None  # no 3D -> no reprojection anywhere
    state.result.pts2d[0, 0, 5] = [-40.0, 100.0]  # off-image in x
    state.result.pts2d[1, 0, 6] = [100.0, 999.0]  # off-image in y
    state.result.pts2d[2, 0, 7] = [np.nan, np.nan]  # nothing at all
    n_pts = result.pts2d.shape[2]
    state.confirm(
        [(v, p) for v in range(state.n_views) for p in range(n_pts)], "all", 0
    )

    for v, p in ((0, 5), (1, 6), (2, 7)):
        assert not state.labels.has_gt[v, 0, p]  # skipped: no dot on screen to approve
    # the cell with nothing at all keeps a draggable Unplaced ghost ...
    assert np.isfinite(state.placeholder_pts2d(0)[2, 7]).all()
    # ... while the off-image ones have a position of their own, just not in frame
    assert not np.isfinite(state.placeholder_pts2d(0)[0, 5]).all()
    # every cell that DID have a visible dot was authored, and inside the image
    for v in range(state.n_views):
        for p in range(n_pts):
            if not state.labels.has_gt[v, 0, p]:
                continue
            x, y = state.labels.gt[v, 0, p]
            assert 0 <= x < 640 and 0 <= y < 480, f"view {v} point {p} at {(x, y)}"


# -- absence: "this keypoint is not on this animal" ---------------------------


def test_absent_point_has_no_3d_and_does_not_fall_back_to_the_run_cache(result):
    # The linchpin. `solve_point_3d` already returns NaN once every view is dropped, but
    # `_solve_point`'s run-cache fallback would immediately substitute `result.pts3d` --
    # a perfectly finite phantom -- and the editor would keep drawing the amputated leg.
    state = EditorState.from_result(result)
    p = 4
    assert np.all(np.isfinite(state.result.pts3d[0, p]))  # the phantom is finite
    state.set_absent([p], True)
    assert np.isnan(state.display_pts3d(0)[p]).all()


def test_absent_point_is_nan_in_every_view(result):
    state = EditorState.from_result(result)
    p = 4
    assert np.isfinite(state.display_pts2d(0)[:, p]).any()  # drawn before
    state.set_absent([p], True)
    assert not np.isfinite(state.display_pts2d(0)[:, p]).any()
    assert state.absent_mask(0)[p]


def test_absent_refuses_authoring_and_says_why(result):
    state = EditorState.from_result(result)
    p, v = 4, 0
    state.set_absent([p], True)
    reason = state.absent_refusal(p)
    assert reason and "not on this animal" in reason
    state.apply_2d_edit(v, p, (10.0, 20.0), 0)
    assert not state.labels.gt_authored[v, 0, p]  # nothing authored at all
    assert state.toggle_invisible(v, p, 0) is None
    assert not state.labels.occluded[v, 0, p]
    assert state.absent_refusal(5) is None  # a bystander point is unaffected


def test_bulk_confirm_skips_absent_points(result):
    # Without this, `a` then `1` fabricates GT on the phantom limb on every frame the
    # operator visits -- and `_grabbable` clamps the invented pixel into the image, so it
    # looks like a real observation.
    state = EditorState.from_result(result)
    p = 4
    state.set_absent([p], True)
    targets = [(v, pt) for v in range(state.n_views) for pt in range(state.n_points)]
    state.confirm(targets, "all", 0)
    assert not state.labels.gt_authored[:, 0, p].any()
    assert state.labels.has_gt[:, 0, 5].any()  # other points were confirmed


def test_absent_whole_recording_and_undoable(result):
    state = EditorState.from_result(result)
    p = 4
    state.apply_2d_edit(0, p, (11.0, 22.0), 0)
    assert state.labels.has_gt[0, 0, p]
    changed = state.set_absent([p], True, whole_recording=True)
    assert changed == [p]
    # every frame, not just the current one
    assert all(state.absent_mask(t)[p] for t in (0, 1, state.n_frames - 1))
    assert state.labels.absent_all_frames()[p]
    assert not state.labels.has_gt[0, 0, p]  # vetoed ...
    assert state.labels.gt_authored[0, 0, p]  # ... but not destroyed
    state.undo()
    assert not state.absent_mask(0)[p]
    assert state.labels.has_gt[0, 0, p]  # the pixel is back, untouched
    assert np.allclose(state.labels.gt[0, 0, p], [11.0, 22.0])


def test_absent_defaults_to_the_current_frame_only(result):
    # A keypoint can stop existing part-way through a recording (autotomy), so the base
    # gesture is frame-scoped; "apply to the whole recording" is the separate convenience.
    state = EditorState.from_result(result)
    p = 4
    state.set_absent([p], True, frame=1)
    assert state.absent_mask(1)[p]
    assert not state.absent_mask(0)[p]
    assert not state.labels.absent_all_frames()[p]  # structural consumers see nothing
    assert state.labels.absent_any_frame()[p]
    # ... and the 3D is gone only in that frame
    assert np.isnan(state.display_pts3d(1)[p]).all()
    assert np.all(np.isfinite(state.display_pts3d(0)[p]))


def test_absent_in_one_frame_is_undoable_without_touching_others(result):
    state = EditorState.from_result(result)
    p = 4
    state.set_absent([p], True, whole_recording=True)
    state.set_absent([p], False, frame=1)  # a hole in the declaration
    assert state.absent_mask(0)[p] and not state.absent_mask(1)[p]
    state.undo()
    assert state.absent_mask(0)[p] and state.absent_mask(1)[p]


def test_resetting_a_frame_does_not_undeclare_absence(result):
    # The everyday reset verbs must not silently un-declare an amputation.
    state = EditorState.from_result(result)
    p = 4
    state.set_absent([p], True, whole_recording=True)
    state.reset_frame(0)
    assert state.absent_mask(0)[p]
    state.reset_point(p, frame=0)
    assert state.absent_mask(0)[p]
    state.reset_targets([(0, p)], frame=0)
    assert state.absent_mask(0)[p]


def test_absent_is_not_a_labeled_frame(result):
    # Folding a recording-wide declaration into per-frame progress would mark every frame
    # labeled at a stroke, and the suggestion queue (which skips labeled frames) empties.
    state = EditorState.from_result(result)
    state.set_absent([4], True, whole_recording=True)
    assert state.corrected_frames() == []
    state.apply_2d_edit(0, 5, (1.0, 2.0), 0)
    assert [f["frame"] for f in state.corrected_frames()] == [0]


def test_per_frame_absence_survives_a_labels_roundtrip(tmp_path, result):
    # The whole point of spans: a partial declaration must persist exactly, and a
    # whole-recording one must not blow up to one row per frame.
    from deeperfly.gui.labels import labels_identity, load_labels, save_labels

    state = EditorState.from_result(result)
    state.set_absent([4], True, frame=1)
    state.set_absent([5], True, whole_recording=True)

    identity = labels_identity(
        point_names=list(result.skeleton.point_names),
        camera_names=list(result.cameras.names),
        n_frames=result.n_frames,
    )
    path = tmp_path / "labels.h5"
    save_labels(path, state.labels, identity=identity)
    back = load_labels(path, identity=identity)
    assert back is not None
    np.testing.assert_array_equal(back.absent, state.labels.absent)
    assert back.absent_all_frames()[5] and not back.absent_all_frames()[4]
    assert back.absent_any_frame()[4]
