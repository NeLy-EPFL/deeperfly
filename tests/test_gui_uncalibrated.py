"""The editor with no camera rig -- the from-scratch labeling phase.

A project started from scratch has footage and nothing else: no trained detector, no
solved rig, no 3D. The editor has to open anyway, because labeling 2D by hand is how the
rig eventually gets solved. So the whole point of these tests is that the *absence* of
geometry is a supported state rather than an accident:

- every geometric path returns ``None``/NaN instead of raising;
- the 2D authoring surface -- drag, occlude, absent, reviewed, undo -- is unchanged;
- the payload says ``has_cameras: false`` so the front-end can explain itself;
- nothing silently invents a 3D position the operator cannot check.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly.gui.labels import Labels
from deeperfly.gui.server import _meta_payload, _points_payload
from deeperfly.gui.state import EditorState
from deeperfly.results import PoseResult
from deeperfly.skeleton import Skeleton

VIEWS = ["camera_RH", "camera_F", "camera_LH"]


@pytest.fixture
def blank() -> PoseResult:
    return PoseResult.uncalibrated(
        Skeleton.fly(), n_views=len(VIEWS), n_frames=5, view_names=VIEWS
    )


@pytest.fixture
def state(blank) -> EditorState:
    return EditorState.from_result(blank)


# -- the result ----------------------------------------------------------------


def test_an_uncalibrated_result_has_no_cameras_and_no_observations(blank):
    assert blank.cameras is None
    assert not blank.has_cameras
    assert blank.pts3d is None
    assert blank.conf is None
    assert blank.pts2d.shape == (3, 5, 38, 2)
    assert np.isnan(blank.pts2d).all()


def test_an_uncalibrated_result_refuses_to_be_saved(blank, tmp_path):
    """It is an editing scaffold; a results.h5 without cameras would claim a run happened."""
    with pytest.raises(ValueError, match="cannot be saved"):
        blank.save(tmp_path / "results.h5")


def test_view_names_survive_without_a_rig(state):
    """The operator's own camera names, not view0..viewN -- they label the canvases."""
    assert state.camera_names == VIEWS


def test_view_names_fall_back_to_positional_when_nothing_records_them():
    """A hand-built camera-less result must not crash the editor over display names."""
    result = PoseResult(
        cameras=None, skeleton=Skeleton.fly(), pts2d=np.full((2, 3, 38, 2), np.nan)
    )
    assert EditorState.from_result(result).camera_names == ["view0", "view1"]


# -- the geometric paths are inert, not broken ---------------------------------


def test_every_3d_path_is_none_rather_than_raising(state):
    assert not state.has_cameras
    assert not state.has_3d
    assert not state.has_nmf
    assert state.display_pts3d() is None
    assert state.display_pts3d_projected() is None
    assert state.display_pts2d_refine() is None
    assert state.display_nmf_projected() is None


def test_solving_a_point_without_a_rig_is_nan_not_an_exception(state):
    """`_solve_point` is the one 3D path reachable without an upstream pts3d guard."""
    assert np.isnan(state._solve_point(0, 0)).all()


def test_placeholder_seeds_still_cover_every_cell(state):
    """This is what keeps every joint draggable with nothing detected and no 3D.

    Without a rig there is no reprojection and no detection, so *every* cell needs a
    seed -- and each must be finite, or the operator has nothing to grab and cannot
    author the first label in the project.
    """
    seeds = state.placeholder_pts2d(0)
    assert seeds.shape == (3, 38, 2)
    assert np.isfinite(seeds).all()


def test_a_3d_edit_is_refused_but_a_2d_label_is_not(state):
    """3D dragging needs a rig; 2D authoring is the whole point of this mode."""
    assert state.apply_3d_edit(0, 0, (10.0, 20.0), 0) is None
    state.apply_2d_edit(0, 0, (10.0, 20.0), 0)
    assert state.labels.has_gt[0, 0, 0]
    np.testing.assert_allclose(state.labels.gt[0, 0, 0], [10.0, 20.0])


# -- the 2D authoring surface is unchanged -------------------------------------


def test_the_full_2d_authoring_surface_works(state):
    """Drag, occlude, absent, reviewed and undo must all behave as they do with a rig."""
    state.apply_2d_edit(0, 3, (5.0, 6.0), 1)
    state.toggle_invisible(1, 3, 1)
    state.set_reviewed(True, 1)
    state.set_absent([7], True, 1, whole_recording=False)

    assert state.labels.has_gt[0, 1, 3]
    assert state.labels.occluded[1, 1, 3]
    assert state.labels.reviewed[1]
    assert state.absent_mask(1)[7]

    assert state.can_undo
    state.undo()  # undoes the absent declaration
    assert not state.absent_mask(1)[7]


def test_display_2d_shows_the_authored_pixels(state):
    state.apply_2d_edit(2, 5, (11.0, 12.0), 0)
    shown = state.display_pts2d(0)
    np.testing.assert_allclose(shown[2, 5], [11.0, 12.0])
    # Everything else is unobserved: no detector fired, and there is no 3D ghost.
    assert np.isnan(shown[0]).all()


def test_corrected_frames_tracks_work_without_a_rig(state):
    state.apply_2d_edit(0, 0, (1.0, 2.0), 3)
    assert [f["frame"] for f in state.corrected_frames()] == [3]


# -- the wire payloads ---------------------------------------------------------


def test_the_meta_payload_declares_the_rig_missing(state, tmp_path):
    from deeperfly.gui.readers import FrameSource
    from deeperfly.gui.session import Session

    session = Session.build(
        state,
        FrameSource({}, image_sizes={n: (64, 80) for n in VIEWS}),
        results_path=str(tmp_path / "results.h5"),
        labels_path=tmp_path / "labels.h5",
        image_sizes={n: (64, 80) for n in VIEWS},
    )
    meta = _meta_payload(session, "v")
    assert meta["has_cameras"] is False
    assert meta["has_3d"] is False
    # Empty rather than absent, so the front-end's `?? []` paths and the 3D scene do not
    # have to special-case the key.
    assert meta["cameras_3d"] == []
    assert meta["cameras_proj"] == []
    assert meta["camera_names"] == VIEWS


def test_the_points_payload_carries_no_projection(state, tmp_path):
    from deeperfly.gui.readers import FrameSource
    from deeperfly.gui.session import Session

    session = Session.build(
        state,
        FrameSource({}, image_sizes={n: (64, 80) for n in VIEWS}),
        results_path=str(tmp_path / "results.h5"),
        labels_path=tmp_path / "labels.h5",
    )
    payload = _points_payload(session, 0, "view", verbose=True)
    assert payload["proj"] is None
    assert payload["nmf"] is None
    assert len(payload["points"]) == len(VIEWS)
    # The placeholder layer is what makes the frame labelable at all here.
    assert payload["placeholder"] is not None


# -- a calibrated result is untouched ------------------------------------------


def test_a_calibrated_result_still_reports_cameras(result):
    """The existing path must not change: has_cameras is true wherever a rig exists."""
    state = EditorState.from_result(result)
    assert state.has_cameras
    assert state.has_3d
    assert state.display_pts3d_projected() is not None
    assert state.camera_names == list(result.cameras.names)


# -- the project entry point ---------------------------------------------------


def _fresh_project(tmp_path, n_frames=6):
    """A project holding one never-run recording of real (tiny) videos."""
    import av

    from deeperfly.project import Project

    rec = tmp_path / "rawfly"
    rec.mkdir()
    for i, cam in enumerate(VIEWS):
        with av.open(str(rec / f"{cam}.mp4"), mode="w") as container:
            stream = container.add_stream("libx264", rate=10)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            for t in range(n_frames):
                img = np.full((48, 64, 3), 20 + 10 * i, dtype=np.uint8)
                img[10 + t : 14 + t, 20:30] = 200
                container.mux(
                    stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24"))
                )
            container.mux(stream.encode())
    project = Project.create(tmp_path / "proj", skeleton="fly38")
    entry = project.add_recording(rec)
    return project, entry


def test_open_target_opens_a_never_run_project_recording_uncalibrated(tmp_path):
    """The whole from-scratch entry point: videos in, labelable session out."""
    pytest.importorskip("av")
    from deeperfly.gui import open_target

    project, _ = _fresh_project(tmp_path)
    session = open_target(project.root)

    assert not session.state.has_cameras
    assert session.n_frames == 6
    assert sorted(session.state.camera_names) == sorted(VIEWS)
    # Real footage, really decoded -- not blank fallbacks.
    frame = session.source.frame(session.state.camera_names[0], 2)
    assert frame is not None and frame.shape[:2] == (48, 64)
    # And every joint is grabbable, which is what makes the first label possible.
    assert np.isfinite(session.state.placeholder_pts2d(0)).all()


def test_opening_backfills_the_frame_count_into_the_index(tmp_path):
    """Adoption could not know it; the editor can, so `status` must stop saying "?"."""
    pytest.importorskip("av")
    from deeperfly.gui import open_target
    from deeperfly.project import Project

    project, entry = _fresh_project(tmp_path, n_frames=5)
    assert entry.n_frames is None

    open_target(project.root)
    assert Project.load(project.root).recording(entry.id).n_frames == 5


def test_a_project_with_several_recordings_needs_a_named_one(tmp_path):
    pytest.importorskip("av")
    from deeperfly.gui import open_target

    project, _ = _fresh_project(tmp_path)
    second = tmp_path / "other"
    second.mkdir()
    for cam in VIEWS:
        (second / f"{cam}.mp4").write_bytes(b"\0" * 4321)
    project.add_recording(second)

    with pytest.raises(SystemExit, match="--recording"):
        open_target(project.root)


def test_a_named_recording_opens(tmp_path):
    pytest.importorskip("av")
    from deeperfly.gui import open_target

    project, entry = _fresh_project(tmp_path)
    session = open_target(project.root, recording=entry.slug)
    assert session.n_frames == 6


def test_an_unknown_recording_name_is_a_clear_error(tmp_path):
    pytest.importorskip("av")
    from deeperfly.gui import open_target

    project, _ = _fresh_project(tmp_path)
    with pytest.raises(SystemExit, match="no recording"):
        open_target(project.root, recording="nope")


def test_an_empty_project_says_how_to_add_a_recording(tmp_path):
    from deeperfly.gui import open_target
    from deeperfly.project import Project

    project = Project.create(tmp_path / "proj")
    with pytest.raises(SystemExit, match="project add"):
        open_target(project.root)


def test_a_recording_whose_footage_vanished_is_a_clear_error(tmp_path):
    pytest.importorskip("av")
    from deeperfly.gui import open_target

    project, entry = _fresh_project(tmp_path)
    for video in (tmp_path / "rawfly").glob("*.mp4"):
        video.unlink()
    with pytest.raises(SystemExit, match="could be found|no resolvable footage"):
        open_target(project.root)


def test_open_target_still_opens_a_plain_results_directory(result, tmp_path):
    """The pre-existing behavior must not change."""
    from deeperfly.gui import open_target

    out = tmp_path / "out"
    out.mkdir()
    result.save(out / "results.h5")
    session = open_target(out)
    assert session.state.has_cameras


def test_open_target_on_a_path_with_no_result_names_what_it_looked_for(tmp_path):
    from deeperfly.gui import open_target

    with pytest.raises(SystemExit, match="deeperfly_outputs"):
        open_target(tmp_path)


# -- calibration landmarks in the editor ---------------------------------------


def _landmark_state(n_frames=4, names=("tether_tip", "coverslip_ne")):
    """An uncalibrated state carrying a landmark overlay."""
    from deeperfly.gui.labels import LandmarkLabels

    result = PoseResult.uncalibrated(
        Skeleton.fly(), n_views=len(VIEWS), n_frames=n_frames, view_names=VIEWS
    )
    marks = LandmarkLabels.empty(len(VIEWS), n_frames, names)
    return EditorState.from_result(result, landmarks=marks)


def test_a_landmark_is_placed_into_its_own_namespace():
    """Never the skeleton's: a landmark must not reach the detector or the training export."""
    state = _landmark_state()
    assert state.has_landmarks
    assert state.landmark_names() == ["tether_tip", "coverslip_ne"]

    assert state.set_landmark(0, 0, (12.0, 34.0), 1)
    np.testing.assert_allclose(state.landmarks.xy[0, 1, 0], [12.0, 34.0])
    # The skeleton labels are untouched -- separate overlays, separate namespaces.
    assert not state.labels.gt_authored.any()


def test_placing_a_landmark_makes_the_session_dirty():
    """Saving is one button, so landmark-only work must still prompt before closing."""
    state = _landmark_state()
    assert not state.dirty
    state.set_landmark(1, 0, (5.0, 6.0), 0)
    assert state.dirty


def test_clearing_a_landmark_removes_only_that_observation():
    state = _landmark_state()
    state.set_landmark(0, 0, (1.0, 2.0), 0)
    state.set_landmark(0, 1, (3.0, 4.0), 0)
    state.clear_landmark(0, 0, 0)
    assert not np.isfinite(state.landmarks.xy[0, 0, 0]).all()
    np.testing.assert_allclose(state.landmarks.xy[0, 0, 1], [3.0, 4.0])


def test_landmark_counts_drive_the_readiness_signal():
    state = _landmark_state()
    for view in range(len(VIEWS)):
        state.set_landmark(view, 0, (1.0, 1.0), 0)
    assert state.landmark_counts() == {"tether_tip": 3, "coverslip_ne": 0}


def test_a_project_with_no_landmarks_reports_none():
    state = EditorState.from_result(
        PoseResult.uncalibrated(Skeleton.fly(), n_views=2, n_frames=2)
    )
    assert not state.has_landmarks
    assert state.display_landmarks() is None
    assert not state.set_landmark(0, 0, (1.0, 1.0), 0)  # refused, not crashed


def test_the_payloads_carry_landmarks(tmp_path):
    from deeperfly.gui.readers import FrameSource
    from deeperfly.gui.session import Session

    state = _landmark_state()
    state.set_landmark(0, 0, (7.0, 8.0), 0)
    session = Session.build(
        state,
        FrameSource({}),
        results_path=str(tmp_path / "results.h5"),
        labels_path=tmp_path / "labels.h5",
    )
    meta = _meta_payload(session, "v")
    assert [m["name"] for m in meta["landmarks"]] == ["tether_tip", "coverslip_ne"]
    assert meta["landmarks"][0]["static"] is True
    assert meta["landmarks"][0]["observations"] == 1

    payload = _points_payload(session, 0, "view")
    assert payload["landmarks"][0][0] == [7.0, 8.0]
    assert payload["landmarks"][1][0] is None  # unobserved in view 1


def test_the_edit_socket_places_and_clears_a_landmark(tmp_path):
    from deeperfly.gui.readers import FrameSource
    from deeperfly.gui.server import _handle_edit
    from deeperfly.gui.session import Session

    state = _landmark_state()
    session = Session.build(
        state,
        FrameSource({}),
        results_path=str(tmp_path / "results.h5"),
        labels_path=tmp_path / "labels.h5",
    )
    _handle_edit(
        session,
        {
            "type": "set_landmark",
            "view": 2,
            "landmark": 1,
            "x": 9.0,
            "y": 10.0,
            "frame": 3,
        },
    )
    np.testing.assert_allclose(state.landmarks.xy[2, 3, 1], [9.0, 10.0])

    _handle_edit(
        session, {"type": "clear_landmark", "view": 2, "landmark": 1, "frame": 3}
    )
    assert not np.isfinite(state.landmarks.xy[2, 3, 1]).all()


def test_saving_keeps_the_landmarks(tmp_path):
    """save_labels rewrites the whole file, so a forgotten hand-off would drop them."""
    from deeperfly.gui.labels import load_landmark_labels, save_labels

    state = _landmark_state()
    state.set_landmark(0, 0, (11.0, 12.0), 2)
    identity = {
        "point_names": list(state.result.skeleton.point_names),
        "camera_names": VIEWS,
        "n_frames": state.n_frames,
    }
    path = tmp_path / "labels.h5"
    save_labels(path, state.labels, identity=identity, landmarks=state.landmarks)

    back = load_landmark_labels(path, n_views=len(VIEWS), n_frames=state.n_frames)
    assert back is not None
    assert list(back.names) == ["tether_tip", "coverslip_ne"]
    np.testing.assert_allclose(back.xy[0, 2, 0], [11.0, 12.0])


def test_a_renamed_landmark_set_carries_observations_across_by_name(tmp_path):
    """Reordering landmarks.toml must not move every observation to a different landmark."""
    from deeperfly.gui import _load_landmarks
    from deeperfly.gui.labels import LandmarkLabels, save_labels
    from deeperfly.landmarks import Landmark, LandmarkSet
    from deeperfly.project import Project

    project = Project.create(tmp_path / "proj")
    LandmarkSet([Landmark(name="a"), Landmark(name="b")]).save(project.root)

    marks = LandmarkLabels.empty(2, 3, ["a", "b"])
    marks.set(0, 0, 0, (1.0, 1.0))  # 'a'
    marks.set(0, 0, 1, (2.0, 2.0))  # 'b'
    path = tmp_path / "labels.h5"
    save_labels(
        path,
        Labels.empty(2, 3, 38),
        identity={"point_names": [], "camera_names": ["x", "y"], "n_frames": 3},
        landmarks=marks,
    )

    # Reorder and add one.
    LandmarkSet([Landmark(name="b"), Landmark(name="c"), Landmark(name="a")]).save(
        project.root
    )
    loaded = _load_landmarks(project, path, 2, 3)
    assert list(loaded.names) == ["b", "c", "a"]
    np.testing.assert_allclose(loaded.xy[0, 0, 0], [2.0, 2.0])  # 'b' followed its NAME
    np.testing.assert_allclose(loaded.xy[0, 0, 2], [1.0, 1.0])  # 'a' followed its NAME
    assert not np.isfinite(loaded.xy[0, 0, 1]).all()  # 'c' is new and empty
