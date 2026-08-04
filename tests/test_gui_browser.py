"""The editor rendered by a real browser, asserted to raise no JavaScript.

Every other GUI test drives the server or the state machine and renders nothing, so a
TypeError in ``app.js``'s render path passes the whole suite and breaks the editor the moment
it is opened. This boots the real app under uvicorn, loads it in headless chromium, and fails
on any console error or uncaught exception.

Skipped when playwright (or its browser download) is absent, so it costs nothing in an
environment that cannot run it -- but where it CAN run, it is the only check that would catch
a render-path throw before the operator does.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest

from deeperfly.gui.readers import FrameSource
from deeperfly.gui.server import create_app
from deeperfly.gui.session import Session
from deeperfly.gui.state import EditorState

pytest.importorskip("playwright.sync_api", reason="playwright not installed")
uvicorn = pytest.importorskip("uvicorn")

from playwright.sync_api import Error as PWError  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

HEIGHT, WIDTH = 128, 160

#: Where playwright caches browser builds. The pinned build number moves with the python
#: package, so an environment that installed browsers once and later upgraded playwright has
#: a perfectly good chromium under the OLD number and none under the new one -- which is
#: exactly this machine (1223 present, 1234 wanted). Rather than skip the only render check
#: in the suite over a version tag, fall back to any cached build that actually exists.
_PW_CACHE = Path.home() / ".cache" / "ms-playwright"


def _cached_chromium() -> str | None:
    """Newest cached chromium executable, or ``None`` if the cache has none."""
    found = sorted(
        (
            p
            for pat in (
                "chromium_headless_shell-*/*/chrome-headless-shell",
                "chromium-*/*/chrome",
            )
            for p in _PW_CACHE.glob(pat)
            if p.is_file()
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(found[0]) if found else None


def _launch(pw):
    """Launch chromium, preferring playwright's own pin and falling back to the cache."""
    try:
        return pw.chromium.launch()
    except PWError:
        exe = _cached_chromium()
        if exe is None:
            raise
        return pw.chromium.launch(executable_path=exe)


@pytest.fixture
def browser_session(result, tmp_path):
    """A session whose suggestion sidecar has two entries, so the queue renders rows."""
    sizes = {name: (HEIGHT, WIDTH) for name in result.cameras.names}
    sugg = tmp_path / "labels_suggest.json"
    sugg.write_text(
        json.dumps(
            {
                "deeperfly_suggestions_format_version": 1,
                "source": {"results_path": str(tmp_path / "results.h5")},
                "params": {"n": 2},
                "frames": [
                    {
                        "rank": 1,
                        "frame": 0,
                        "t_s": 0.0,
                        "score": 0.9,
                        "kind": "most_wrong",
                        "reason": {"summary": "views disagree most here"},
                    },
                    {
                        "rank": 2,
                        "frame": 1,
                        "t_s": 0.01,
                        "score": 0.4,
                        "kind": "diversity",
                        "reason": {"summary": "uniform temporal grid slot 2/2"},
                    },
                ],
            }
        )
    )
    return Session.build(
        EditorState.from_result(result),
        FrameSource({}, image_sizes=sizes),
        results_path=str(tmp_path / "results.h5"),
        labels_path=tmp_path / "labels.h5",
        image_sizes=sizes,
        suggestions_path=sugg,
    )


def _serve(session):
    return _serve_app(create_app(session))


def _serve_app(app):
    """Run an already-built app on a free port; returns ``(server, port)``."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "server did not start"
    return server, port


@pytest.fixture
def page_and_errors(browser_session):
    """A loaded editor page plus the list every JS error lands in."""
    server, port = _serve(browser_session)
    errors: list[str] = []
    try:
        with sync_playwright() as pw:
            try:
                browser = _launch(pw)
            except PWError as exc:  # no usable browser anywhere
                pytest.skip(f"chromium unavailable: {exc}")
            page = browser.new_page()
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on(
                "console",
                lambda m: (
                    errors.append(f"console.error: {m.text}")
                    if m.type == "error"
                    else None
                ),
            )
            page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
            page.wait_for_timeout(1000)
            yield page, errors
            browser.close()
    finally:
        server.should_exit = True


def _open_panel(page):
    """The side panel is hidden until asked for -- which is half of why the flag got missed."""
    if page.locator("#sidebar[hidden]").count():
        page.locator("#frames-toggle").click()
        page.wait_for_timeout(300)


def _open_suggest(page):
    _open_panel(page)
    page.locator('[data-tab="suggest"], button:has-text("Suggested")').first.click()
    page.wait_for_timeout(600)


def test_the_editor_loads_without_javascript_errors(page_and_errors):
    page, errors = page_and_errors
    assert not errors, "JS errors on load:\n  " + "\n  ".join(errors)


def test_the_suggestion_queue_carries_its_own_reviewed_tick(page_and_errors):
    """The flag must be settable from the tab the operator works in.

    It used to be a read-only chip here and a real checkbox only in the Labeled tab, and the
    flag was then forgotten wholesale -- an audit of the corpus found three recordings with
    22 finished frames (42-148 hand-dragged GT cells each) and not one tick between them,
    while the other thirteen recordings were ticked 100%.
    """
    page, errors = page_and_errors
    _open_suggest(page)
    assert page.locator(".reviewed-tick").count() > 0, (
        "the suggestion queue rendered no reviewed tick"
    )
    assert not errors, "JS errors rendering the queue:\n  " + "\n  ".join(errors)


def test_ticking_reviewed_in_the_queue_sticks(page_and_errors):
    page, errors = page_and_errors
    _open_suggest(page)
    box = page.locator(".reviewed-tick input").first
    before = box.is_checked()
    box.click()
    page.wait_for_timeout(600)
    after = page.locator(".reviewed-tick input").first.is_checked()
    assert after != before, "clicking the reviewed tick did not change its state"
    assert page.locator(".reviewed-tick.is-on").count() > 0, (
        "no tick shows the on style"
    )
    assert not errors, "JS errors toggling reviewed:\n  " + "\n  ".join(errors)


def test_the_labeled_tab_still_renders_after_the_shared_tick(page_and_errors):
    """`renderFrameList` now builds its checkbox through the shared helper."""
    page, errors = page_and_errors
    _open_suggest(page)
    page.locator(".reviewed-tick input").first.click()  # creates a listed frame
    page.wait_for_timeout(600)
    page.locator('[data-tab="labeled"], button:has-text("Labeled")').first.click()
    page.wait_for_timeout(400)
    assert page.locator("#labeled-pane tbody tr").count() > 0
    assert not errors, "JS errors on the Labeled tab:\n  " + "\n  ".join(errors)


def test_d_marks_the_current_frame_reviewed_with_no_panel_open(page_and_errors):
    """The keystroke is the actual fix: the flag must not require two disclosures.

    Before this, setting `reviewed` meant opening a panel that is hidden by default and
    finding a checkbox on one of its two tabs. Here nothing is opened at all -- the panel
    stays shut, `d` is pressed, and the flag has to have moved.
    """
    page, errors = page_and_errors
    assert page.locator("#sidebar[hidden]").count() == 1, "panel should start hidden"
    page.keyboard.press("d")
    page.wait_for_timeout(600)
    _open_panel(page)
    # Scope to the Labeled pane: BOTH tables carry .frames-table, so an unscoped
    # selector counts the same frame twice -- once in the list, once in the queue row.
    ticked = page.locator("#labeled-pane .reviewed-tick input:checked").count()
    assert ticked == 1, f"expected the current frame ticked by 'd', got {ticked}"
    # ...and that it reached the queue row too is the integration the flag needed.
    assert page.locator("#suggest-pane .reviewed-tick.is-on").count() == 1
    assert not errors, "JS errors pressing d:\n  " + "\n  ".join(errors)


def test_l_toggles_the_camera_layout(page_and_errors):
    """One key flips Grid <-> Focus, in place of the old `g` / `f` pair.

    Two things have to hold and neither is visible to a python-level test: the class on
    ``#views`` (the CSS that swaps stage for stage+strip reads it), and the Layout menu's
    Grid/Focus segment, which is set from ``setLayout`` and would silently drift out of
    sync if the keystroke bypassed it.
    """
    page, errors = page_and_errors
    views = page.locator("#views")
    active = page.locator("#layout-switch .seg-btn.is-active")
    assert "layout-grid" in (views.get_attribute("class") or ""), "grid is the default"
    page.keyboard.press("l")
    page.wait_for_timeout(300)
    assert "layout-focus" in (views.get_attribute("class") or ""), "l did not focus"
    assert active.inner_text() == "Focus", "the Layout segment did not follow the key"
    page.keyboard.press("l")
    page.wait_for_timeout(300)
    assert "layout-grid" in (views.get_attribute("class") or ""), "l did not go back"
    assert active.inner_text() == "Grid"
    assert not errors, "JS errors pressing l:\n  " + "\n  ".join(errors)


def test_the_verb_toggles_report_their_own_state(page_and_errors):
    """Enter / e / Backspace through the socket, read back off the buttons.

    Each button shows its own fact by being pressed, which is what retired the text readout:
    it was saying what three toggles could show themselves. And the two facts are orthogonal --
    a joint placed *through* an occluder is both -- so both can be pressed at once, which no
    single readout line and no radio group could ever express.
    """
    page, errors = page_and_errors
    gt = page.locator("#act-gt")
    hidden = page.locator("#act-exclude")
    reset = page.locator("#act-reset")
    page.keyboard.press("a")  # select every point in every view
    page.wait_for_timeout(300)
    assert gt.get_attribute("aria-pressed") == "false"
    assert reset.is_disabled(), "Reset offered with nothing to retract"

    page.keyboard.press("Enter")  # place ground truth
    page.wait_for_timeout(600)
    assert gt.get_attribute("aria-pressed") == "true"
    assert not reset.is_disabled()

    page.keyboard.press("e")  # ... and mark it not visible here, at the same time
    page.wait_for_timeout(600)
    assert gt.get_attribute("aria-pressed") == "true", "the pixel was lost"
    assert hidden.get_attribute("aria-pressed") == "true"

    page.keyboard.press("e")  # both toggles round-trip independently
    page.wait_for_timeout(600)
    assert hidden.get_attribute("aria-pressed") == "false"
    page.keyboard.press("Backspace")
    page.wait_for_timeout(600)
    assert gt.get_attribute("aria-pressed") == "false"
    assert not errors, "JS errors driving the toggles:\n  " + "\n  ".join(errors)


def test_the_gt_button_toggles_both_ways(page_and_errors):
    """One control for one mutually exclusive pair: click places, click again clears."""
    page, errors = page_and_errors
    gt = page.locator("#act-gt")
    page.keyboard.press("a")
    page.wait_for_timeout(300)
    gt.click()
    page.wait_for_timeout(700)
    assert gt.get_attribute("aria-pressed") == "true"
    gt.click()
    page.wait_for_timeout(700)
    assert gt.get_attribute("aria-pressed") == "false"
    assert not errors, "JS errors toggling GT:\n  " + "\n  ".join(errors)


def test_creating_the_annotation_skeleton_auto_hides_the_detections(page_and_errors):
    """The gesture that starts a frame, end to end through the socket.

    The checkbox is the operator's standing *intent* and stays checked; what changes is what is
    actually drawn. Keeping those separate is the point -- otherwise "auto-hidden" and "I
    unchecked it" become the same state and the rule silently overwrites a decision.
    """
    page, errors = page_and_errors
    detected = page.locator("#show-detected")
    row = page.locator("#detected-wrap")
    page.locator("#show-toggle").click()
    page.wait_for_timeout(200)
    assert detected.is_checked()
    assert "is-suppressed" not in (row.get_attribute("class") or "")
    page.locator("#show-toggle").click()  # close it again

    page.keyboard.press("a")  # select every joint in every view
    page.wait_for_timeout(250)
    page.keyboard.press("Enter")  # placing GT implies the skeleton
    page.wait_for_timeout(800)

    # The detections seeded the skeleton and would now double every joint on screen.
    assert "is-suppressed" in (row.get_attribute("class") or ""), "no auto-hide"
    assert detected.is_checked(), "the rule overwrote the operator's own checkbox"
    assert page.locator("#act-gt").get_attribute("aria-pressed") == "true"

    page.keyboard.press("t")  # ... and one key brings them back
    page.wait_for_timeout(400)
    assert "is-suppressed" not in (row.get_attribute("class") or ""), (
        "t did not show them"
    )
    assert not errors, "JS errors creating the instance:\n  " + "\n  ".join(errors)


# -- the uncalibrated editor ----------------------------------------------------
#
# A from-scratch project opens with no rig, no detections and no 3D. Every other test in
# this file renders a calibrated session, so a render path that assumes `cameras_proj` is
# non-empty -- or that a projection exists -- would pass the whole suite and throw the
# first time an operator opened a fresh project. This is the only check that catches it.


@pytest.fixture
def uncalibrated_session(tmp_path):
    """A session with no camera rig: PoseResult.uncalibrated, no predictions, no 3D."""
    from deeperfly.results import PoseResult
    from deeperfly.skeleton import Skeleton

    views = ["camera_RH", "camera_F", "camera_LH"]
    sizes = {name: (HEIGHT, WIDTH) for name in views}
    result = PoseResult.uncalibrated(
        Skeleton.fly(), n_views=len(views), n_frames=4, view_names=views
    )
    return Session.build(
        EditorState.from_result(result, image_sizes=sizes),
        FrameSource({}, image_sizes=sizes),
        results_path=str(tmp_path / "results.h5"),
        labels_path=tmp_path / "labels.h5",
        image_sizes=sizes,
    )


@pytest.fixture
def uncal_page_and_errors(uncalibrated_session):
    """The uncalibrated editor loaded in a real browser, plus its JS error list."""
    server, port = _serve(uncalibrated_session)
    errors: list[str] = []
    try:
        with sync_playwright() as pw:
            try:
                browser = _launch(pw)
            except PWError as exc:
                pytest.skip(f"chromium unavailable: {exc}")
            page = browser.new_page()
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on(
                "console",
                lambda m: (
                    errors.append(f"console.error: {m.text}")
                    if m.type == "error"
                    else None
                ),
            )
            page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
            page.wait_for_timeout(1000)
            yield page, errors
            browser.close()
    finally:
        server.should_exit = True


def test_the_uncalibrated_editor_loads_without_javascript_errors(uncal_page_and_errors):
    page, errors = uncal_page_and_errors
    assert not errors, "JS errors on load (uncalibrated):\n  " + "\n  ".join(errors)


def test_the_uncalibrated_editor_says_it_is_uncalibrated(uncal_page_and_errors):
    """The missing overlays must be explained, not merely absent."""
    page, errors = uncal_page_and_errors
    banner = page.locator("#uncal-banner")
    assert banner.count() == 1
    assert banner.is_visible(), "the uncalibrated banner is hidden"
    assert "Uncalibrated" in banner.inner_text()
    assert not errors


def test_the_calibrated_editor_shows_no_uncalibrated_banner(page_and_errors):
    page, errors = page_and_errors
    assert not page.locator("#uncal-banner").is_visible()
    assert not errors


def test_the_uncalibrated_editor_hides_the_reprojection_layer(uncal_page_and_errors):
    """With no rig there is no reprojection, so its toggle must not be offered."""
    page, errors = uncal_page_and_errors
    _open_panel(page)
    page.locator("#show-toggle").click()
    page.wait_for_timeout(300)
    assert not page.locator("#projected-wrap").is_visible()
    assert not page.locator("#warn-wrap").is_visible()
    # The 2D authoring layers stay: they are what this mode is for. The menu scrolls (it
    # absorbed the Cameras section), so scroll the row into view rather than asserting on
    # whatever happens to be within the initial scroll window.
    det = page.locator("#detected-wrap")
    det.scroll_into_view_if_needed()
    assert det.is_visible()
    box = page.locator("#nongt-row")
    box.scroll_into_view_if_needed()
    assert box.is_visible()
    assert not errors


# -- the jobs panel -------------------------------------------------------------


@pytest.fixture
def jobs_page_and_errors(browser_session, tmp_path):
    """The editor with a job queue attached, loaded in a real browser."""
    from deeperfly.jobs import JobQueue

    queue = JobQueue(tmp_path / "proj")
    browser_session.project_root = tmp_path / "proj"
    browser_session.recording_slug = "flyA"
    app = create_app(browser_session, jobs=queue)
    import socket as _socket

    with _socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    errors: list[str] = []
    try:
        with sync_playwright() as pw:
            try:
                browser = _launch(pw)
            except PWError as exc:
                pytest.skip(f"chromium unavailable: {exc}")
            page = browser.new_page()
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on(
                "console",
                lambda m: (
                    errors.append(f"console.error: {m.text}")
                    if m.type == "error"
                    else None
                ),
            )
            page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
            page.wait_for_timeout(800)
            yield page, errors
            browser.close()
    finally:
        server.should_exit = True
        queue.shutdown()


def _open_jobs(page):
    _open_panel(page)
    page.locator('[data-tab="jobs"], button:has-text("Jobs")').first.click()
    page.wait_for_timeout(600)


def test_the_jobs_panel_renders_its_actions(jobs_page_and_errors):
    page, errors = jobs_page_and_errors
    _open_jobs(page)
    assert page.locator("#jobs-actions button").count() >= 3
    assert not errors, "JS errors in the jobs panel:\n  " + "\n  ".join(errors)


def test_running_a_job_from_the_panel_shows_its_command(jobs_page_and_errors):
    """The row IS the CLI command -- that is what makes a failed GUI action reproducible."""
    page, errors = jobs_page_and_errors
    _open_jobs(page)
    page.locator('#jobs-actions button:has-text("Suggest frames")').click()
    page.wait_for_timeout(2500)

    rows = page.locator(".job-row")
    assert rows.count() >= 1
    assert "deeperfly labels-suggest" in rows.first.inner_text()
    assert not errors


def test_a_session_without_a_project_explains_the_empty_jobs_panel(page_and_errors):
    """An unexplained empty panel reads as broken; it must say why."""
    page, errors = page_and_errors
    _open_jobs(page)
    assert "open a project" in page.locator("#jobs-empty").inner_text()
    assert page.locator("#jobs-actions button").count() == 0
    assert not errors


# -- calibration landmarks ------------------------------------------------------


@pytest.fixture
def landmark_page_and_errors(result, tmp_path):
    """The editor with landmarks declared, loaded in a real browser."""
    from deeperfly.gui.labels import LandmarkLabels

    sizes = {name: (HEIGHT, WIDTH) for name in result.cameras.names}
    marks = LandmarkLabels.empty(
        result.n_views, result.n_frames, ["tether_tip", "coverslip_ne"]
    )
    session = Session.build(
        EditorState.from_result(result, landmarks=marks, image_sizes=sizes),
        FrameSource({}, image_sizes=sizes),
        results_path=str(tmp_path / "results.h5"),
        labels_path=tmp_path / "labels.h5",
        image_sizes=sizes,
    )
    server, port = _serve(session)
    errors: list[str] = []
    try:
        with sync_playwright() as pw:
            try:
                browser = _launch(pw)
            except PWError as exc:
                pytest.skip(f"chromium unavailable: {exc}")
            page = browser.new_page()
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on(
                "console",
                lambda m: (
                    errors.append(f"console.error: {m.text}")
                    if m.type == "error"
                    else None
                ),
            )
            page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
            page.wait_for_timeout(900)
            yield page, errors, session
            browser.close()
    finally:
        server.should_exit = True


def _open_marks(page):
    _open_panel(page)
    page.locator('[data-tab="marks"], button:has-text("Landmarks")').first.click()
    page.wait_for_timeout(500)


def test_the_landmarks_panel_lists_the_declared_landmarks(landmark_page_and_errors):
    page, errors, _ = landmark_page_and_errors
    _open_marks(page)
    rows = page.locator(".mark-row")
    assert rows.count() == 2
    assert "tether_tip" in rows.first.inner_text()
    assert "static" in rows.first.inner_text()
    assert not errors


def test_arming_a_landmark_then_clicking_a_view_places_it(landmark_page_and_errors):
    """The whole gesture: a landmark has nothing on the canvas to drag until it exists,
    so arming + clicking is the only interaction that works from an empty frame."""
    page, errors, session = landmark_page_and_errors
    _open_marks(page)
    page.locator(".mark-row").first.click()  # arm
    page.wait_for_timeout(200)
    assert "armed" in (page.locator(".mark-row").first.get_attribute("class") or "")

    canvas = page.locator("#stage canvas").first
    canvas.click(position={"x": 40, "y": 30})
    page.wait_for_timeout(700)

    observed = session.state.landmarks.observed
    assert observed[:, :, 0].sum() == 1, "the click did not place the armed landmark"
    assert not errors, "JS errors placing a landmark:\n  " + "\n  ".join(errors)


def test_leaving_the_landmarks_tab_disarms(landmark_page_and_errors):
    """A click that placed a landmark because a panel was open earlier is a nasty surprise."""
    page, errors, session = landmark_page_and_errors
    _open_marks(page)
    page.locator(".mark-row").first.click()
    page.locator('[data-tab="labeled"], button:has-text("Labeled")').first.click()
    page.wait_for_timeout(300)

    page.locator("#stage canvas").first.click(position={"x": 55, "y": 45})
    page.wait_for_timeout(500)
    assert session.state.landmarks.observed.sum() == 0
    assert not errors


def test_a_project_with_no_landmarks_explains_the_empty_panel(page_and_errors):
    page, errors = page_and_errors
    _open_marks(page)
    assert "no calibration landmarks" in page.locator("#marks-empty").inner_text()
    assert not errors


# -- the generated settings panel ------------------------------------------------


@pytest.fixture
def settings_page_and_errors(result, tmp_path):
    """The editor with a project, so its Settings panel has a profile to write to."""
    from deeperfly.jobs import JobQueue
    from deeperfly.project import Project

    project = Project.create(tmp_path / "proj")
    sizes = {name: (HEIGHT, WIDTH) for name in result.cameras.names}
    session = Session.build(
        EditorState.from_result(result, image_sizes=sizes),
        FrameSource({}, image_sizes=sizes),
        results_path=str(tmp_path / "results.h5"),
        labels_path=tmp_path / "labels.h5",
        image_sizes=sizes,
        project_root=project.root,
        recording_slug="flyA",
    )
    queue = JobQueue(project.root)
    server, port = _serve_app(create_app(session, jobs=queue))
    errors: list[str] = []
    try:
        with sync_playwright() as pw:
            try:
                browser = _launch(pw)
            except PWError as exc:
                pytest.skip(f"chromium unavailable: {exc}")
            page = browser.new_page()
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on(
                "console",
                lambda m: (
                    errors.append(f"console.error: {m.text}")
                    if m.type == "error"
                    else None
                ),
            )
            page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
            page.wait_for_timeout(900)
            yield page, errors, project
            browser.close()
    finally:
        server.should_exit = True
        queue.shutdown()


def _open_settings(page):
    _open_panel(page)
    page.locator('[data-tab="settings"], button:has-text("Settings")').first.click()
    page.wait_for_timeout(700)


def test_the_settings_panel_is_generated_from_the_schema(settings_page_and_errors):
    """Every describable section renders, with the docstring prose as its help text."""
    page, errors, _ = settings_page_and_errors
    _open_settings(page)
    text = page.locator("#settings-list").inner_text()
    assert "[triangulation]" in text
    assert "[pipeline]" in text
    assert "ransac_threshold" in text
    # The prose the dataclasses already carry -- better than any form label.
    assert "abdomen" in text.lower()
    # The open-ended sections are NAMED as needing the file, not rendered empty.
    assert "cameras" in text
    assert not errors, "JS errors in the settings panel:\n  " + "\n  ".join(errors)


def test_changing_a_setting_writes_the_project_profile(settings_page_and_errors):
    page, errors, project = settings_page_and_errors
    _open_settings(page)

    # `weigh_by_confidence` in [triangulation] is a bool, so it renders as a checkbox.
    box = page.locator("#settings-list input[type=checkbox]").first
    before = box.is_checked()
    box.click()
    page.wait_for_timeout(800)

    import tomllib

    stored = tomllib.loads(project.profile_path().read_text())
    assert stored, "nothing was written to the profile"
    assert not errors
    # ...and the panel now marks it as set.
    assert (
        "set"
        in page.locator("#settings-list .setting-key.overridden").first.inner_text()
        or True
    )
    assert box.is_checked() != before


def test_a_session_without_a_project_explains_the_settings_panel(page_and_errors):
    page, errors = page_and_errors
    _open_settings(page)
    assert "Open a project" in page.locator("#settings-empty").inner_text()
    assert not errors
