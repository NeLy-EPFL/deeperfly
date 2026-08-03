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
    app = create_app(session)
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
    # The 2D authoring layers stay: they are what this mode is for.
    assert page.locator("#gt-wrap").is_visible()
    assert page.locator("#placeholder-wrap").is_visible()
    assert not errors
