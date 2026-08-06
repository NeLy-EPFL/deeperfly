"""Switching the open recording from inside the editor.

The editor shows one recording. ``GET /api/recordings`` lists the project's others with
the counts that decide which is worth opening next, and ``POST /api/recordings/open``
swaps the served session in place and tells every open browser to reload.

Two things here are load-bearing and neither is visible in a happy-path click:

- the **cache token** must move with the recording, or the browser goes on painting the
  previous fly out of its own cache for an hour (frames are served ``immutable``);
- an unsaved label must **survive** the swap. The editor keeps every recording it has
  opened (``opened`` in :func:`~deeperfly.gui.server.create_app`), so unsaved work is
  project-wide: switching is free, ``POST /api/save-all`` is what writes it, and the
  only prompt left is at close time. A switch that quietly dropped the outgoing
  session's labels would be unrecoverable and completely silent.

The front-end half is covered in ``test_gui_browser.py``.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient
from helpers import HEIGHT, WIDTH
from websockets.sync.client import connect as ws_connect

from deeperfly.gui import EditorState, FrameSource, Session, open_target, server
from deeperfly.gui.labels import Labels, labels_identity, save_labels
from deeperfly.gui.server import create_app
from deeperfly.project import Project
from deeperfly.results import StageStore

CAMERAS = ("rh", "rm", "rf", "f", "lf", "lm", "lh")
SIZES = {name: (HEIGHT, WIDTH) for name in CAMERAS}


def _make_recording(
    root, cameras, fly, *, seed, n_frames, gt_cells=0, reviewed=0, names=CAMERAS
):
    """A recording with byte-only footage and a real ``results.h5``.

    The footage bytes differ per recording (via ``seed``) because
    :func:`~deeperfly.project.recording_fingerprint` reads their sizes -- two identical
    recordings would be de-duplicated into one project entry, and there would be nothing
    to switch between. The bytes are not decodable video, so ``/api/frame`` 404s here --
    which is fine: what is under test is which *session* the server holds, not pixels.

    ``names`` is the recording's own camera list, and ``cameras`` must be the matching rig.
    It defaults to the whole set, but a project really does hold recordings filmed on
    different rigs -- a view added, a camera that failed that day -- so anything that
    outlives a switch has to survive the camera list changing under it.
    """
    root.mkdir(parents=True, exist_ok=True)
    names = tuple(names)
    for i, camera in enumerate(names):
        (root / f"camera_{camera}.mp4").write_bytes(b"\0" * (1000 + 7 * i + seed))
    outputs = root / "deeperfly_outputs"
    outputs.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    pts2d = rng.uniform(0, 100, size=(len(names), n_frames, 38, 2))
    StageStore(outputs / "results.h5").write_pose2d(
        cameras=cameras,
        skeleton=fly,
        pts2d=pts2d,
        conf=np.ones(pts2d.shape[:3]),
        image_sizes={name: (HEIGHT, WIDTH) for name in names},
        footage={c: [root / f"camera_{c}.mp4"] for c in names},
    )
    if gt_cells or reviewed:
        labels = Labels.empty(len(names), n_frames, 38)
        for i in range(gt_cells):
            labels.set_gt(i % len(names), i % n_frames, i % 38, (1.0 * i, 2.0 * i))
        for t in range(reviewed):
            labels.set_reviewed(t, True)
        save_labels(
            outputs / "labels.h5",
            labels,
            identity=labels_identity(
                point_names=list(fly.point_names),
                camera_names=list(names),
                n_frames=n_frames,
            ),
        )
    return root


@pytest.fixture
def two_recordings(tmp_path, cameras, fly):
    """A project holding ``flyA`` (labeled) and ``flyB`` (not), with flyA open."""
    project = Project.create(tmp_path / "proj", name="switchproj")
    a = _make_recording(
        tmp_path / "flyA", cameras, fly, seed=0, n_frames=6, gt_cells=5, reviewed=2
    )
    b = _make_recording(tmp_path / "flyB", cameras, fly, seed=500, n_frames=9)
    project.add_recording(a, slug="flyA")
    project.add_recording(b, slug="flyB")
    return project


@pytest.fixture
def client(two_recordings):
    return TestClient(create_app(open_target(two_recordings.root, recording="flyA")))


@pytest.fixture
def bare_client(result, tmp_path):
    """A session on a plain results.h5 -- no project, so nothing to switch to."""
    image_sizes = {name: (HEIGHT, WIDTH) for name in result.cameras.names}
    session = Session.build(
        EditorState.from_result(result),
        FrameSource({}, image_sizes=image_sizes),
        results_path=str(tmp_path / "results.h5"),
        labels_path=tmp_path / "labels.h5",
        image_sizes=image_sizes,
    )
    return TestClient(create_app(session))


# -- listing -------------------------------------------------------------------


def test_a_bare_results_session_says_why_it_cannot_switch(bare_client):
    """``enabled: false`` with a reason, not an unexplained empty menu.

    The same convention as /api/jobs and /api/config: the picker can then say what is
    missing instead of rendering nothing and looking broken.
    """
    body = bare_client.get("/api/recordings").json()
    assert body["enabled"] is False
    assert body["recordings"] == []
    assert "results.h5" in body["reason"]
    assert (
        bare_client.post("/api/recordings/open", json={"recording": "x"}).status_code
        == 409
    )


def test_the_listing_names_every_recording_and_which_one_is_open(client):
    """A 200 here also proves the payload is JSON-safe.

    ``Project.status()`` rows carry a ``RecordingEntry`` dataclass and ``Path``s, which
    FastAPI's encoder refuses -- so the fields are named explicitly by ``_recording_rows``
    rather than dumped.
    """
    body = client.get("/api/recordings").json()
    assert body["enabled"] is True
    assert body["project"] == "switchproj"
    assert [r["slug"] for r in body["recordings"]] == ["flyA", "flyB"]
    active = [r["slug"] for r in body["recordings"] if r["active"]]
    assert active == ["flyA"] == [body["active"]]
    assert client.get("/api/meta").json()["recording"] == "flyA"


def test_the_listing_carries_the_counts_that_decide_what_to_open_next(client):
    rows = {r["slug"]: r for r in client.get("/api/recordings").json()["recordings"]}
    assert rows["flyA"]["gt_points"] == 5
    assert rows["flyA"]["reviewed_frames"] == 2
    assert rows["flyA"]["n_frames"] == 6
    assert rows["flyB"]["gt_points"] == 0
    assert rows["flyB"]["labeled_frames"] == 0
    assert rows["flyB"]["n_frames"] == 9
    assert rows["flyA"]["has_results"] and rows["flyB"]["has_results"]


# -- switching -----------------------------------------------------------------


def test_opening_another_recording_swaps_the_session(client):
    body = client.post("/api/recordings/open", json={"recording": "flyB"}).json()
    assert body["switched"] is True
    meta = client.get("/api/meta").json()
    assert meta["recording"] == "flyB"
    assert meta["n_frames"] == 9  # flyA has 6: this is the new recording, not a relabel
    assert client.get("/api/recordings").json()["active"] == "flyB"


def test_the_cache_token_moves_with_the_recording(client):
    """The regression the whole design turns on.

    Frames are addressed only by camera and frame index, and served ``max-age=3600,
    immutable`` when their URL carries this session's token -- ``immutable`` meaning the
    browser will not even revalidate. If the token survived a switch, every cached
    ``/api/frame/f/3`` would go on being the *previous* recording's picture for an hour.
    """
    before = client.get("/api/meta").json()["cache_v"]
    client.post("/api/recordings/open", json={"recording": "flyB"})
    after = client.get("/api/meta").json()["cache_v"]
    assert before != after


def test_a_stale_cache_stamp_is_never_served_as_immutable(client, monkeypatch):
    """The other half: the token has to actually gate the caching header.

    Asserted on the mesh route rather than the frame route only because this fixture's
    footage is bytes, not decodable video, so ``/api/frame`` 404s before it sets a
    header. Both routes share ``_image_cache_control``.
    """
    monkeypatch.setattr(EditorState, "has_nmf", property(lambda self: True))
    monkeypatch.setattr(server, "_render_mesh_png", lambda s, camera, t: b"png")
    before = client.get("/api/meta").json()["cache_v"]
    client.post("/api/recordings/open", json={"recording": "flyB"})
    after = client.get("/api/meta").json()["cache_v"]

    stale = client.get(f"/api/mesh/{CAMERAS[0]}/0?v={before}")
    fresh = client.get(f"/api/mesh/{CAMERAS[0]}/0?v={after}")
    assert stale.headers["Cache-Control"] == "no-store"
    assert fresh.headers["Cache-Control"] == "max-age=3600, immutable"


def test_the_meta_payload_is_never_cached(client):
    """After a switch, /api/meta is the only thing telling a reloaded page which
    recording it is on -- and which token to stamp into its frame URLs."""
    assert client.get("/api/meta").headers["Cache-Control"] == "no-store"


def test_the_mesh_overlay_is_never_served_from_the_previous_recording(
    client, monkeypatch
):
    """The memo sits *inside* the HTTP cache the token protects.

    ``mesh_cache`` used to be keyed ``(camera, frame)`` with no recording in the key, so
    after a switch the previous recording's rendered PNG came back for the new one -- at
    full ``immutable`` confidence, from process memory, without the token ever getting a
    say.
    """
    monkeypatch.setattr(EditorState, "has_nmf", property(lambda self: True))
    monkeypatch.setattr(
        server, "_render_mesh_png", lambda s, camera, t: s.results_path.encode()
    )
    first = client.get(f"/api/mesh/{CAMERAS[0]}/0")
    assert first.status_code == 200
    client.post("/api/recordings/open", json={"recording": "flyB"})
    second = client.get(f"/api/mesh/{CAMERAS[0]}/0")
    assert second.status_code == 200
    assert second.content != first.content, "served the previous recording's overlay"


# -- unsaved labels across a switch --------------------------------------------


def _edit(client, *, view=0, point=3, x=12.0, y=34.0, frame=1):
    """Author one 2D ground-truth pixel over the socket (the session goes dirty)."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "edit_2d",
                "view": view,
                "point": point,
                "x": x,
                "y": y,
                "frame": frame,
                "mode": "edit_2d",
            }
        )
        return ws.receive_json()


def test_a_switch_keeps_the_unsaved_labels_of_the_recording_it_leaves(client):
    """The whole feature. Switching used to be refused while dirty, because the swap
    dropped the outgoing session; now the session is kept, so the operator can work
    across a project's recordings the way they work across its frames.

    Asserted on the *pixel*, not just on the dirty flag: a retained flag over a rebuilt
    session would be a worse bug than the refusal it replaced.
    """
    reply = _edit(client)
    # `fixed` is the wire name for "this cell carries a GT pixel" (see _points_payload).
    assert reply["fixed"][0][3] is True
    assert reply["points"][0][3] == [12.0, 34.0]
    assert client.get("/api/meta").json()["dirty"] is True

    switched = client.post("/api/recordings/open", json={"recording": "flyB"})
    assert switched.status_code == 200, "refused a switch it no longer has to refuse"
    body = switched.json()
    assert body["restored"] is False, "flyB was never open; it came from disk"
    assert body["dirty_recordings"] == ["flyA"]
    meta = client.get("/api/meta").json()
    assert meta["recording"] == "flyB"
    assert meta["dirty"] is False, "flyA's edit leaked into flyB"
    # The project is unsaved even though the open recording is not -- the distinction the
    # close prompt and the beforeunload guard are built on.
    assert meta["project_dirty"] is True
    assert meta["dirty_recordings"] == ["flyA"]

    back = client.post("/api/recordings/open", json={"recording": "flyA"}).json()
    assert back["restored"] is True, "flyA was reopened from disk, not restored"
    assert client.get("/api/meta").json()["dirty"] is True
    pts = client.get("/api/points/1").json()
    assert pts["fixed"][0][3] is True, "the unsaved label did not survive"
    assert pts["points"][0][3] == [12.0, 34.0], "the label survived but moved"


def test_a_restored_recording_is_not_read_from_disk_again(client, monkeypatch):
    """Reopening a held recording must not re-run ``open_target``.

    Not an optimization detail: opening reads every camera's video header, ``results.h5``
    and ``labels.h5`` (a network share, in this lab) and would rebuild the state from the
    *sidecar* -- which is precisely how the unsaved labels would vanish while every other
    assertion still passed.
    """
    opens = []
    real = server._open_for_switch

    def counted(root, slug):
        opens.append(slug)
        return real(root, slug)

    monkeypatch.setattr(server, "_open_for_switch", counted)
    client.post("/api/recordings/open", json={"recording": "flyB"})
    client.post("/api/recordings/open", json={"recording": "flyA"})
    assert opens == ["flyB"], "flyA was opened again instead of restored"


def test_the_recording_left_keeps_its_labels_but_not_its_pictures(two_recordings):
    """A retained session holds the operator's work, not a pile of decoded frames.

    ``cache_size`` full-resolution frames per held recording is what would make keeping
    them a way to run out of memory; they are also the one part that costs nothing to
    rebuild (the browser refetches, from readers that stayed open). Reaches into the
    private LRU because this fixture's footage is bytes, not decodable video -- there is no
    way to fill the cache through ``/api/frame`` here, and the property under test is a
    memory one.
    """
    session = open_target(two_recordings.root, recording="flyA")
    client = TestClient(create_app(session))
    session.source._cache[("rh", 0)] = np.zeros((4, 4, 3), np.uint8)
    client.post("/api/recordings/open", json={"recording": "flyB"})
    assert session.source._cache == {}, (
        "kept the frames of a recording nobody is showing"
    )
    # The session behind those frames is untouched -- flyA's five saved GT cells included.
    assert int(session.state.labels.has_gt.sum()) == 5, "released more than the cache"


def test_discard_abandons_the_unsaved_labels_of_the_recording_left(client):
    """``discard: true`` is the caller saying "do not keep this one".

    The flag used to mean "swap anyway despite the refusal"; with nothing left to refuse
    it means the same thing it always did to the operator -- throw this recording's
    unsaved work away -- and it is the only way to do so.
    """
    _edit(client)
    client.post("/api/recordings/open", json={"recording": "flyB", "discard": True})
    meta = client.get("/api/meta").json()
    assert meta["project_dirty"] is False
    assert meta["dirty_recordings"] == []

    back = client.post("/api/recordings/open", json={"recording": "flyA"}).json()
    assert back["restored"] is False, "the discarded session was kept after all"
    assert client.get("/api/meta").json()["dirty"] is False
    pts = client.get("/api/points/1").json()
    assert pts["fixed"][0][3] is False, "a discarded label came back"


def test_a_clean_recording_is_pruned_but_an_unsaved_one_never_is(client, monkeypatch):
    """The registry is a cache for clean recordings and a store for dirty ones.

    With the keep-count at zero every clean session is droppable, so the two halves are
    visible in one run: flyA comes back from disk when it was saved, and from memory when
    it was not.
    """
    monkeypatch.setattr(server, "_SESSION_KEEP", 0)
    client.post("/api/recordings/open", json={"recording": "flyB"})
    assert (
        client.post("/api/recordings/open", json={"recording": "flyA"}).json()[
            "restored"
        ]
        is False
    ), "a clean session was kept even with _SESSION_KEEP = 0"

    _edit(client)
    client.post("/api/recordings/open", json={"recording": "flyB"})
    assert (
        client.post("/api/recordings/open", json={"recording": "flyA"}).json()[
            "restored"
        ]
        is True
    ), "pruned a session holding the operator's only copy of their labels"


def test_save_all_writes_every_recording_holding_unsaved_labels(client, two_recordings):
    """The editor's Save. Unsaved work spans the project, so saving only the open
    recording would leave the title starred and the operator hunting for which other one
    still needed a click."""
    _edit(client, frame=1)
    client.post("/api/recordings/open", json={"recording": "flyB"})
    _edit(client, frame=2, x=7.0, y=8.0)
    assert set(client.get("/api/meta").json()["dirty_recordings"]) == {"flyA", "flyB"}

    body = client.post("/api/save-all").json()
    assert set(body["saved"]) == {"flyA", "flyB"}
    assert body["failed"] == []
    assert body["project_dirty"] is False and body["dirty_recordings"] == []
    # On disk, in each recording's own sidecar. The failure this must be incapable of is
    # writing one recording's labels to the other's path, which the counts would show:
    # flyA had 5 saved cells and gained one, flyB had none.
    from deeperfly.project import label_stats

    counts = {
        slug: label_stats(two_recordings.labels_path(two_recordings.recording(slug)))
        for slug in ("flyA", "flyB")
    }
    assert counts["flyA"]["gt_points"] == 6
    assert counts["flyB"]["gt_points"] == 1


def test_the_listing_marks_which_recordings_are_unsaved(client):
    """The cue's other half: the count in the toolbar says how many, the list says which.

    A held recording is also *counted* live -- listing it from its on-disk sidecar would
    say "unlabeled" next to work the operator has just done.
    """
    _edit(client, frame=4, point=9)
    client.post("/api/recordings/open", json={"recording": "flyB"})
    body = client.get("/api/recordings").json()
    rows = {r["slug"]: r for r in body["recordings"]}
    assert rows["flyA"]["dirty"] is True and rows["flyA"]["open"] is True
    assert rows["flyB"]["dirty"] is False and rows["flyB"]["open"] is True
    assert body["dirty_recordings"] == ["flyA"]
    assert body["project_dirty"] is True
    # flyA's sidecar holds 5 GT cells; the sixth is the unsaved one.
    assert rows["flyA"]["gt_points"] == 6, "the row was read from disk, not the session"


# -- bundle adjustment across a switch -----------------------------------------

# The editor runs ONE solve at a time, on one worker, for whatever recording is open --
# but the tab reporting it survives a switch, because the page rebuilds in place. So the
# run has to say which recording it belongs to, or the last solve is announced under
# whichever animal happens to be open next: a calibration written into flyA's directory
# read as flyB's, and a per-camera reprojection table matched against a rig it was never
# solved on.


def _label_every_view(root, cameras, fly, *, n_frames, n_points, names=CAMERAS):
    """Ground truth in EVERY view, at the true projection of a random 3D point.

    A bundle adjustment fits tracks seen from more than one camera. The per-view random
    ``pts2d`` :func:`_make_recording` writes share nothing, so ``preflight`` refuses a
    session built on them before a solve ever starts -- these labels are what make one
    genuinely runnable.
    """
    rng = np.random.default_rng(7)
    pts3d = rng.uniform(-1.5, 1.5, size=(n_frames, 38, 3))
    proj = np.asarray(cameras.project(pts3d))  # (V, T, P, 2)
    labels = Labels.empty(len(names), n_frames, 38)
    for v in range(len(names)):
        for t in range(n_frames):
            for p in range(n_points):
                xy = proj[v, t, p]
                labels.set_gt(v, t, p, (float(xy[0]), float(xy[1])))
    save_labels(
        root / "deeperfly_outputs" / "labels.h5",
        labels,
        identity=labels_identity(
            point_names=list(fly.point_names),
            camera_names=list(names),
            n_frames=n_frames,
        ),
    )


@pytest.fixture
def solvable_client(tmp_path, cameras, fly):
    """Two recordings a bundle adjustment can actually be run on, with flyA open."""
    project = Project.create(tmp_path / "proj", name="solveproj")
    a = _make_recording(tmp_path / "flyA", cameras, fly, seed=0, n_frames=3)
    b = _make_recording(tmp_path / "flyB", cameras, fly, seed=500, n_frames=3)
    for root in (a, b):
        _label_every_view(root, cameras, fly, n_frames=3, n_points=12)
    project.add_recording(a, slug="flyA")
    project.add_recording(b, slug="flyB")
    return TestClient(create_app(open_target(project.root, recording="flyA")))


def _run_bundle_adjustment(client, name):
    """Solve, wait, and return the finished run. Fails loudly rather than timing out."""
    plan = client.get("/api/bundle-adjust").json()
    assert plan["enabled"], plan.get("reason")
    assert plan["readiness"]["ok"], plan["readiness"]["problems"]
    started = client.post(
        "/api/bundle-adjust", json={"name": name, "settings": plan["settings"]}
    )
    assert started.status_code == 200, started.text
    for _ in range(600):
        run = client.get("/api/bundle-adjust/run").json()
        if run["state"] != "running":
            return run
        time.sleep(0.05)
    raise AssertionError("the solve never finished")


def test_a_solve_is_reported_only_under_the_recording_it_was_run_on(solvable_client):
    """flyA's result must vanish on the way to flyB -- and be there again on the way back.

    Both halves matter. Reporting it under flyB attributes another animal's calibration to
    this one; dropping it for good would lose the result of a solve the operator started
    and merely walked away from.
    """
    run = _run_bundle_adjustment(solvable_client, "across-the-switch")
    assert run["state"] == "done", run
    assert run["recording"] == "flyA"

    solvable_client.post("/api/recordings/open", json={"recording": "flyB"})
    assert solvable_client.get("/api/bundle-adjust/run").json() == {"state": "idle"}
    assert solvable_client.get("/api/bundle-adjust").json()["run"] == {"state": "idle"}

    solvable_client.post("/api/recordings/open", json={"recording": "flyA"})
    back = solvable_client.get("/api/bundle-adjust").json()["run"]
    assert back["state"] == "done"
    assert back["calibration_file"].startswith("across-the-switch")


def test_the_plans_camera_list_is_the_open_recordings(solvable_client):
    """The fix/free matrix is keyed by camera name, so this list is what the pane trusts."""
    assert solvable_client.get("/api/bundle-adjust").json()["cameras"] == list(CAMERAS)
    solvable_client.post("/api/recordings/open", json={"recording": "flyB"})
    plan = solvable_client.get("/api/bundle-adjust").json()
    assert plan["recording"] == "flyB", (
        "the plan does not say which recording it is for"
    )
    assert plan["cameras"] == list(CAMERAS)


# -- refusals ------------------------------------------------------------------


def test_an_unknown_recording_is_refused_and_leaves_the_session_alone(client):
    assert (
        client.post("/api/recordings/open", json={"recording": "nope"}).status_code
        == 404
    )
    assert client.post("/api/recordings/open", json={}).status_code == 400
    assert client.get("/api/meta").json()["recording"] == "flyA"


def test_reopening_the_recording_already_open_is_a_no_op(client):
    body = client.post("/api/recordings/open", json={"recording": "flyA"}).json()
    assert body["switched"] is False
    assert body["reason"] == "already open"


def test_a_recording_can_be_named_by_id(client, two_recordings):
    """``Project.recording`` accepts a slug, an id, or an unambiguous id prefix; the
    reply always speaks in slugs so the front-end has one name for the thing."""
    rec_id = two_recordings.recording("flyB").id
    body = client.post("/api/recordings/open", json={"recording": rec_id}).json()
    assert body["switched"] is True
    assert body["recording"] == "flyB"


# -- the broadcast, over a real socket -----------------------------------------


def _serve(app):
    """Run ``app`` on a free port; returns ``(server, port)``."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    threading.Thread(target=srv.run, daemon=True).start()
    deadline = time.monotonic() + 10
    while not srv.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert srv.started, "server did not start"
    return srv, port


def _await_reload(sock, *, deadline=10.0):
    """Drain ``sock`` until the reload push arrives; ``None`` if it never does.

    Messages cannot simply be counted off: the *first* browser to connect is told
    nothing (an unadorned client is editable by default, so its first message is its own
    edit reply), while later ones get a role handshake. So this reads until it sees the
    type it wants, rather than assuming a fixed prelude.
    """
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        try:
            msg = json.loads(sock.recv(timeout=end - time.monotonic()))
        except TimeoutError:
            return None
        if msg.get("type") == "reload":
            return msg
    return None


def test_switching_tells_every_open_browser_to_reload(two_recordings):
    """Every tab, not just the one that clicked.

    The page builds its canvases, key bindings and cache token from /api/meta, fetched
    once per load, so a tab that is not reloaded keeps drawing the old recording's
    geometry against the new recording's session.
    """
    import httpx

    app = create_app(open_target(two_recordings.root, recording="flyA"))
    srv, port = _serve(app)
    try:
        url = f"ws://127.0.0.1:{port}/ws"
        with ws_connect(url) as one, ws_connect(url) as two:
            r = httpx.post(
                f"http://127.0.0.1:{port}/api/recordings/open",
                json={"recording": "flyB"},
                timeout=30,
            )
            assert r.status_code == 200
            for label, sock in (("first", one), ("second", two)):
                msg = _await_reload(sock)
                assert msg is not None, f"the {label} browser was never told to reload"
                assert msg["recording"] == "flyB"
                assert msg["reason"] == "recording"
    finally:
        srv.should_exit = True


def test_a_reload_storm_does_not_stop_the_server_mid_switch(two_recordings):
    """Every browser reloads at once, so every socket drops at once.

    ``exit_on_disconnect`` reads "no sockets left" as "the last tab closed" and stops the
    server after ``disconnect_grace`` -- and the reconnect that would cancel it only
    happens after the reloaded page has fetched /api/meta. Without the switch window,
    switching recordings would routinely kill the editor.
    """
    import httpx

    stopped = []
    app = create_app(
        open_target(two_recordings.root, recording="flyA"),
        on_shutdown=lambda: stopped.append(True),
        exit_on_disconnect=True,
        disconnect_grace=0.05,
    )
    srv, port = _serve(app)
    try:
        # The socket is opened only so that closing it is a "the last tab went away"
        # event; nothing is read from it.
        with ws_connect(f"ws://127.0.0.1:{port}/ws"):
            r = httpx.post(
                f"http://127.0.0.1:{port}/api/recordings/open",
                json={"recording": "flyB"},
                timeout=30,
            )
            assert r.status_code == 200
        # the socket is now closed -- exactly what a reload looks like
        time.sleep(0.5)
        assert not stopped, "the server stopped itself during a recording switch"
    finally:
        srv.should_exit = True


# -- the front-end contract (no browser) ---------------------------------------


def test_the_switcher_ids_agree_across_the_assets(client):
    """Every id the picker's JS binds exists in the HTML it is served with.

    ``el()`` throws on a missing id, which takes the whole editor down at boot -- so a
    renamed id in one file and not the other is a blank page, not a missing button.
    """
    from deeperfly.gui.server import _WEB_DIR

    page = client.get("/").text
    app_js = (_WEB_DIR / "static" / "app.js").read_text()
    api_js = (_WEB_DIR / "static" / "api.js").read_text()
    css = (_WEB_DIR / "static" / "styles.css").read_text()

    for ident in (
        "recording-name",
        "recording-list",
        "recording-empty",
        # The unsaved-changes cue and the close prompt's list of which recordings are
        # unsaved: what replaced the switch prompt, now that a switch cannot lose an edit.
        "unsaved",
        "close-list",
    ):
        assert f'id="{ident}"' in page, f"{ident} missing from the page"
        assert f'el("{ident}")' in app_js, f"{ident} not bound in app.js"
    # The picker lives in the sidebar now, on its own tab -- reached through SIDEBAR_TABS
    # rather than a literal el() call, except for the tab itself (hidden when there is no
    # project to list).
    for ident in ("tab-recordings", "recording-pane"):
        assert f'id="{ident}"' in page, f"{ident} missing from the page"
    assert 'el("tab-recordings")' in app_js
    for cls in (
        "rec-list",
        "rec-row",
        "rec-name",
        "rec-stats",
        "rec-flag",
        "rec-dirty",
        "unsaved-chip",
    ):
        assert f".{cls}" in css, f"{cls} has no style"
    # The cue is a dot with no author `display`, so an unstyled `.rec-dirty` would show on
    # every row: the class has to exist AND the row has to hide it.
    assert 'querySelector(".rec-dirty")' in app_js
    assert "/api/save-all" in api_js and "saveAllCorrections" in app_js
    # The rows are patched in place, not rebuilt, so the click is delegated to the list and
    # each row carries its slug in a data attribute -- that pair IS the wiring. A row built
    # without `dataset.slug` would render fine and do nothing when clicked.
    assert "dataset.slug" in app_js, "the delegated row click has nothing to read"
    assert 'closest?.(".rec-row")' in app_js, "the row click is no longer delegated"
    assert "/api/recordings" in api_js
    assert "openRecording" in api_js and "openRecording" in app_js
    # The socket must branch on the new message type; without it the push falls through
    # to onPoints and is silently eaten by applyPoints' frame guard.
    assert 'msg.type === "reload"' in api_js
