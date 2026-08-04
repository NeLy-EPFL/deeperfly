"""Switching the open recording from inside the editor.

The editor shows one recording. ``GET /api/recordings`` lists the project's others with
the counts that decide which is worth opening next, and ``POST /api/recordings/open``
swaps the served session in place and tells every open browser to reload.

Two things here are load-bearing and neither is visible in a happy-path click:

- the **cache token** must move with the recording, or the browser goes on painting the
  previous fly out of its own cache for an hour (frames are served ``immutable``);
- an unsaved label must **block** the swap, because the swap drops the outgoing session
  -- labels and undo history together -- and no ``beforeunload`` fires for an in-process
  change.

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


def _make_recording(root, cameras, fly, *, seed, n_frames, gt_cells=0, reviewed=0):
    """A recording with byte-only footage and a real ``results.h5``.

    The footage bytes differ per recording (via ``seed``) because
    :func:`~deeperfly.project.recording_fingerprint` reads their sizes -- two identical
    recordings would be de-duplicated into one project entry, and there would be nothing
    to switch between. The bytes are not decodable video, so ``/api/frame`` 404s here --
    which is fine: what is under test is which *session* the server holds, not pixels.
    """
    root.mkdir(parents=True, exist_ok=True)
    for i, camera in enumerate(CAMERAS):
        (root / f"camera_{camera}.mp4").write_bytes(b"\0" * (1000 + 7 * i + seed))
    outputs = root / "deeperfly_outputs"
    outputs.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    pts2d = rng.uniform(0, 100, size=(len(CAMERAS), n_frames, 38, 2))
    StageStore(outputs / "results.h5").write_pose2d(
        cameras=cameras,
        skeleton=fly,
        pts2d=pts2d,
        conf=np.ones(pts2d.shape[:3]),
        image_sizes=SIZES,
        footage={c: [root / f"camera_{c}.mp4"] for c in CAMERAS},
    )
    if gt_cells or reviewed:
        labels = Labels.empty(len(CAMERAS), n_frames, 38)
        for i in range(gt_cells):
            labels.set_gt(i % len(CAMERAS), i % n_frames, i % 38, (1.0 * i, 2.0 * i))
        for t in range(reviewed):
            labels.set_reviewed(t, True)
        save_labels(
            outputs / "labels.h5",
            labels,
            identity=labels_identity(
                point_names=list(fly.point_names),
                camera_names=list(CAMERAS),
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


# -- refusals ------------------------------------------------------------------


def test_a_switch_is_refused_while_labels_are_unsaved(client):
    """The one way this editor could destroy hand work without saying so.

    The swap drops the outgoing session -- labels and undo history together -- and an
    in-process change fires no ``beforeunload``, so the refusal is the whole guard.
    """
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "edit_2d",
                "view": 0,
                "point": 3,
                "x": 12.0,
                "y": 34.0,
                "frame": 1,
                "mode": "edit_2d",
            }
        )
        ws.receive_json()
    assert client.get("/api/meta").json()["dirty"] is True

    refused = client.post("/api/recordings/open", json={"recording": "flyB"})
    assert refused.status_code == 409
    assert "unsaved" in refused.json()["detail"]
    assert client.get("/api/meta").json()["recording"] == "flyA", "swapped anyway"

    forced = client.post(
        "/api/recordings/open", json={"recording": "flyB", "discard": True}
    )
    assert forced.status_code == 200
    assert client.get("/api/meta").json()["recording"] == "flyB"


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


def test_the_picker_ids_agree_across_the_assets(client):
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
        "recording-wrap",
        "recording-toggle",
        "recording-menu",
        "recording-name",
        "recording-list",
        "recording-empty",
        "switch-overlay",
        "switch-cancel",
        "switch-discard",
        "switch-save",
        "switch-from",
        "switch-to",
    ):
        assert f'id="{ident}"' in page, f"{ident} missing from the page"
        assert f'el("{ident}")' in app_js, f"{ident} not bound in app.js"
    for cls in ("rec-list", "rec-row", "rec-name", "rec-stats", "rec-flag"):
        assert f".{cls}" in css, f"{cls} has no style"
    assert "/api/recordings" in api_js
    assert "openRecording" in api_js and "openRecording" in app_js
    # The socket must branch on the new message type; without it the push falls through
    # to onPoints and is silently eaten by applyPoints' frame guard.
    assert 'msg.type === "reload"' in api_js
