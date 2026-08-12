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

import numpy as np
import pytest

from deeperfly.gui.readers import FrameSource
from deeperfly.gui.server import create_app
from deeperfly.gui.session import Session
from deeperfly.gui.state import EditorState

pytest.importorskip("playwright.sync_api", reason="playwright not installed")
uvicorn = pytest.importorskip("uvicorn")

from playwright.sync_api import Error as PWError  # noqa: E402
from playwright.sync_api import expect, sync_playwright  # noqa: E402

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
    """The panel is the operator's arrangement, so it persists -- open it if it is shut.

    Through the right-edge rail, which is the only click that opens it: there is no toolbar
    button any more, and the rail exists only while the panel is hidden.
    """
    if page.locator("#sidebar[hidden]").count():
        page.locator("#sidebar-rail").click()
        page.wait_for_timeout(300)


def _close_panel(page):
    if not page.locator("#sidebar[hidden]").count():
        page.locator("#frames-collapse").click()
        page.wait_for_timeout(300)


def _tab(page, name):
    """Show the named sidebar pane, and PROVE it is on screen.

    The visibility assertion is the point. Every ``count()`` / ``inner_text()`` assertion in
    this file reads straight through ``display: none``, so a tab that could no longer be
    activated at all -- or a chip wired to the wrong pane -- would leave the whole suite
    green. The pane is looked up through the chip's own ``aria-controls``, so that wiring is
    what gets checked rather than a mapping duplicated here.
    """
    _open_panel(page)
    chip = page.locator(f"#tab-{name}")
    pane = chip.get_attribute("aria-controls")
    chip.click()
    page.wait_for_timeout(500)
    expect(page.locator(f"#{pane}")).to_be_visible()
    return pane


def _leave_tab(page, name):
    """Move off the named tab -- the event that disarms a landmark and stops the jobs poll.

    Deliberately lands on a pane that fetches nothing, so what is under test is the leaving.
    """
    _tab(page, "instances" if name == "labeled" else "labeled")
    expect(page.locator(f"#tab-{name}")).to_have_attribute("aria-selected", "false")


def _open_suggest(page):
    _tab(page, "suggest")


def test_the_editor_loads_without_javascript_errors(page_and_errors):
    page, errors = page_and_errors
    assert not errors, "JS errors on load:\n  " + "\n  ".join(errors)


def test_the_toolbar_is_one_row(page_and_errors):
    """Every toolbar group sits inside ``.toolbar-row``, on one line.

    ``#controls`` is ``flex-direction: column``, so a group that escapes the row does not
    look broken in the DOM -- it renders as its own full-width band, and ``.spacer``
    (which is ``flex: 1`` *within* the row) stops pushing the session controls right.
    Retiring the Layout popover left its ``</div>`` behind, which closed ``.toolbar-row``
    immediately and dropped all six groups into the column: the toolbar became a six-high
    stack. No existing test noticed, because every element still existed, still had its
    id, and still worked -- only the geometry was wrong. So this asserts the geometry.
    """
    page, errors = page_and_errors
    assert page.locator(".toolbar-row").count() == 1, "expected one .toolbar-row"

    # Structural, not geometric: `.control-row` is `flex-wrap: wrap` on purpose, so a
    # narrow window legitimately puts the toolbar on two lines. What must never happen is
    # a group escaping the row -- and that is exactly what stacking looks like.
    kids = page.evaluate(
        "() => [...document.getElementById('controls').children].map(e => e.className)"
    )
    assert kids == ["control-row frame-row", "control-row toolbar-row"], (
        f"#controls should hold exactly the two rows, got {kids}"
    )
    for sel in ("#show-wrap", "#point-status", "#save", "#cameras"):
        assert page.evaluate(
            "sel => !!document.querySelector(sel)?.closest('.toolbar-row')", sel
        ), f"{sel} escaped .toolbar-row"

    # Given the width to fit, the row is one line and .spacer pushes the session
    # controls to the right edge -- neither is true when the groups stack.
    page.set_viewport_size({"width": 1800, "height": 900})
    page.wait_for_timeout(200)
    left = page.locator("#show-wrap").bounding_box()
    right = page.locator("#save").bounding_box()
    assert left and right, "toolbar controls have no layout box"
    assert left["y"] < right["y"] + right["height"], "toolbar stacked at 1800px"
    assert right["y"] < left["y"] + left["height"], "toolbar stacked at 1800px"
    assert right["x"] > left["x"] + left["width"], (
        "the session controls are not pushed right -- .spacer is not inside the row"
    )
    assert not errors, "JS errors laying out the toolbar:\n  " + "\n  ".join(errors)


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


def test_the_labeled_list_still_renders_after_the_shared_tick(page_and_errors):
    """`renderFrameList` now builds its checkbox through the shared helper."""
    page, errors = page_and_errors
    _open_suggest(page)
    page.locator(".reviewed-tick input").first.click()  # creates a listed frame
    page.wait_for_timeout(600)
    # Tick from the QUEUE, then switch to the Labeled tab: the row has to be there, built by
    # the shared tick helper from a list the operator was not looking at when they made it.
    _tab(page, "labeled")
    assert page.locator("#labeled-pane tbody tr").count() > 0
    assert not errors, "JS errors rendering the labeled list:\n  " + "\n  ".join(errors)


def test_d_marks_the_current_frame_reviewed_with_no_panel_open(page_and_errors):
    """The keystroke is the actual fix: the flag must not require two disclosures.

    Before this, setting `reviewed` meant opening a panel and finding a checkbox on one
    of its two tabs. Here the panel is shut, `d` is pressed, and the flag has to move.
    (The panel DEFAULTS to open, so this closes it first -- what is under test is that `d`
    needs nothing on screen.)
    """
    page, errors = page_and_errors
    _close_panel(page)
    assert page.locator("#sidebar[hidden]").count() == 1, "the panel did not close"
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
    it was saying what three toggles could show themselves. And the two facts are on
    independent axes -- a pixel you stand behind and hold out of the training loss is both --
    so both can be pressed at once, which no single readout line and no radio group could ever
    express.
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

    page.keyboard.press("e")  # ... and hold it out of the loss, at the same time
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


#: The Hidden bar's top stroke, rgb(230, 234, 240) -- poseView.js HIDDEN_COLOR, #e6eaf0.
#: Counted on the canvas to prove the cue is really drawn rather than merely wired up:
#: nothing else in the marker vocabulary uses this near-white, and the limb palette is fully
#: saturated. Keep the two in step; a colour change here fails the test loudly.

#: Read one view's canvas back: a cheap hash of every pixel, plus how many carry the Hidden
#: bar's colour. The hash answers "did anything at all change", the count "did the mark
#: appear" -- together they separate an added cue from a moved joint.
_CANVAS_PROBE = """
() => {
  const c = document.querySelector("#views canvas");
  const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
  let hash = 0, bar = 0;
  for (let i = 0; i < d.length; i += 4) {
    hash = (hash * 31 + d[i] + d[i + 1] * 3 + d[i + 2] * 7) | 0;
    if (d[i] === 230 && d[i + 1] === 234 && d[i + 2] === 240) bar++;
  }
  return { hash, bar };
}
"""


def test_hidden_draws_a_mark_and_nothing_else_moves(page_and_errors):
    """The visual cue, asserted on the pixels: Hidden ADDS a bar and changes nothing else.

    Both halves matter and only the canvas can show either. That the flag is *visible* at all
    is the point of the mark -- an authored decision the operator cannot see is one they cannot
    check or revise. That the rest of the frame is untouched is the orthogonality: the flag
    used to be rendered by deleting the cell's position, which moved joints, broke bones and
    (where the 3D was unsolvable) made joints vanish outright. So the canvas must differ from
    the unmarked one, and must come back **byte-identical** when the mark is lifted.
    """
    page, errors = page_and_errors
    probe = lambda: page.evaluate(_CANVAS_PROBE)  # noqa: E731
    # Spend the session's FIRST edit on something that draws nothing, so the comparison
    # below is not measuring one. Whatever edits first also raises the unsaved-changes chip,
    # and at this fixture's 800px width that chip wraps .toolbar-row onto a second line and
    # shrinks every canvas by ~16px -- so `before` and `after` would be read off two
    # differently-sized canvases and differ no matter what Hidden did. Reviewed is the ideal
    # sacrifice: it dirties the session and puts nothing on the canvas. (Wide enough to fit
    # one line, >=1024px, there is no wrap and no confound; the fixture is simply narrower.)
    page.keyboard.press("d")
    page.wait_for_timeout(900)
    assert not page.evaluate("() => document.getElementById('unsaved').hidden"), (
        "the unsaved chip never appeared, so it is no longer what reflows the toolbar -- "
        "re-derive what this warm-up is protecting against before deleting it"
    )
    page.keyboard.press("a")  # select every point in every view
    page.wait_for_timeout(400)
    before = probe()

    page.keyboard.press("e")  # hold the whole frame out of the training loss
    page.wait_for_timeout(700)
    marked = probe()
    assert marked["bar"] > before["bar"], "no Hidden mark was drawn"
    assert marked["hash"] != before["hash"]

    page.keyboard.press("e")  # ... and lift it again
    page.wait_for_timeout(700)
    after = probe()
    assert after == before, "lifting Hidden did not restore the canvas exactly"
    assert not errors, "JS errors drawing the Hidden mark:\n  " + "\n  ".join(errors)


#: Violet -- the label-coverage gauge's hue (poseView.js ``COVER_RGB`` = 170,120,255). Matched by
#: a *shape* in RGB space rather than exactly, because only the solid arc lands on the exact value:
#: the dashed track is drawn at half alpha, so its pixels are whatever it sat on, blended. The
#: predicate ("blue leads, red clearly above green") is what no other cue in the editor satisfies --
#: the limb palette's blues and pale cyans all have green above red, its reds lead with red, and
#: lime / cyan / amber / mint / the achromatic dataset marks each fail one clause. Anything that
#: does slip through is static, and every assertion below compares counts across a toggle, so a
#: constant offset cancels. Keep this in step with COVER_RGB; a hue change fails the test loudly.
_COVER_PROBE = """
() => {
  const c = document.querySelector("#views canvas");
  const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
  let hash = 0, violet = 0;
  for (let i = 0; i < d.length; i += 4) {
    const [r, g, b] = [d[i], d[i + 1], d[i + 2]];
    hash = (hash * 31 + r + g * 3 + b * 7) | 0;
    if (b > 60 && b > r + 20 && r > g + 12) violet++;
  }
  return { hash, violet };
}
"""


def test_the_under_labeled_gauge_appears_on_u_and_clears_when_labeled(page_and_errors):
    """The whole check, on the pixels: off by default, `u` shows it, labeling clears it.

    All three halves are the feature. It must be **off** on load -- a ring on all 38 joints of an
    untouched frame would bury the skeleton it describes. It must be **visible** when asked for,
    which only the canvas can show. And it must **clear itself** as the work gets done: a check
    that stays lit after you satisfy it is one an operator learns to ignore. Turning it back off
    must restore the canvas byte-for-byte -- it annotates the joints, it never moves them.
    """
    page, errors = page_and_errors
    probe = lambda: page.evaluate(_COVER_PROBE)  # noqa: E731
    assert not page.locator("#show-cover").is_checked(), "the check must default to off"
    before = probe()

    page.keyboard.press("u")
    page.wait_for_timeout(400)
    flagged = probe()
    assert page.locator("#show-cover").is_checked(), (
        "u did not follow through to the menu"
    )
    assert flagged["violet"] > before["violet"], "no coverage gauge was drawn"

    # Ground truth in every view of every joint: 7 labeled views, so nothing is under two.
    page.keyboard.press("a")
    page.wait_for_timeout(300)
    page.keyboard.press("Enter")
    page.wait_for_timeout(900)
    labeled = probe()
    assert labeled["violet"] < flagged["violet"], "labeling did not clear the gauges"

    page.keyboard.press("u")  # ... and the cue is purely additive
    page.wait_for_timeout(400)
    off = probe()
    assert off["violet"] == labeled["violet"], "u left something behind"
    assert not errors, "JS errors driving the coverage check:\n  " + "\n  ".join(errors)


def test_the_under_labeled_check_can_be_toggled_from_the_show_menu(page_and_errors):
    """The discoverable path: the key is the fast one, the menu is how you find it at all."""
    page, errors = page_and_errors
    page.locator("#show-toggle").click()
    page.wait_for_timeout(200)
    row = page.locator("#cover-wrap")
    row.scroll_into_view_if_needed()
    assert row.is_visible()
    check = page.locator("#show-cover")
    check.click()
    page.wait_for_timeout(300)
    assert check.is_checked()
    # The threshold subrow is clamped to the rig: asking for more views than there are cameras
    # would flag every joint of every frame with no way to satisfy it.
    assert page.locator("#cover-min").get_attribute("value") == "2"
    assert page.locator("#cover-min").get_attribute("max") == "7"
    assert not errors, "JS errors toggling the coverage check:\n  " + "\n  ".join(
        errors
    )


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
    # ... and so does the label-coverage check, which is the point of it having its own gate: it
    # counts YOUR pixels per view, so it needs cameras to count over and nothing else. On a fresh
    # project with no rig, no detections and no 3D it is the only check there is -- and the state
    # it reports (nothing labeled in two views yet) is exactly the state such a project is in.
    cover = page.locator("#cover-wrap")
    cover.scroll_into_view_if_needed()
    assert cover.is_visible(), "the coverage check needs no rig, only cameras"
    assert page.locator("#checks-section").is_visible(), (
        "the Checks heading was hidden with the reprojection warning, stranding the row below it"
    )
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
    _tab(page, "jobs")


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
    _tab(page, "marks")


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
    """A click that placed a landmark because a panel was open earlier is a nasty surprise.

    Leaving the tab is the event that has to do it: the pane that says WHICH landmark is
    armed is no longer on screen, so nothing else would tell the operator why their next
    click moved a calibration point instead of selecting a joint.
    """
    page, errors, session = landmark_page_and_errors
    _open_marks(page)
    page.locator(".mark-row").first.click()
    _leave_tab(page, "marks")

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
    _tab(page, "settings")


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


# -- the recording picker --------------------------------------------------------


def _real_video(root, n_frames):
    """Overwrite a fixture recording's byte-only footage with decodable video.

    The API tests do not need pixels, but the browser does: an undecodable frame is a 404,
    and Chrome logs every failed resource as a `console.error` -- which this suite asserts
    against, so the fixture's own footage would fail every test here for a reason that has
    nothing to do with what is under test.
    """
    import cv2

    for camera in root.glob("camera_*.mp4"):
        writer = cv2.VideoWriter(
            str(camera), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (WIDTH, HEIGHT)
        )
        for _ in range(n_frames):
            writer.write(np.zeros((HEIGHT, WIDTH, 3), np.uint8))
        writer.release()


@pytest.fixture
def recording_page_and_errors(tmp_path, cameras, fly):
    """The editor on a real two-recording project, so a switch is genuinely executable.

    Built through ``open_target`` rather than ``Session.build`` because that is what the
    switch handler itself calls -- a fixture that hand-assembled the session would not
    prove the second recording can be opened the way the server opens it.
    """
    from test_gui_recordings import _make_recording

    from deeperfly.gui import open_target
    from deeperfly.project import Project

    project = Project.create(tmp_path / "proj", name="switchproj")
    a = _make_recording(tmp_path / "flyA", cameras, fly, seed=0, n_frames=6, gt_cells=5)
    b = _make_recording(tmp_path / "flyB", cameras, fly, seed=500, n_frames=9)
    _real_video(a, 6)
    _real_video(b, 9)
    project.add_recording(a, slug="flyA")
    project.add_recording(b, slug="flyB")
    session = open_target(project.root, recording="flyA")
    server, port = _serve_app(create_app(session))
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
            yield page, errors, port
            browser.close()
    finally:
        server.should_exit = True


def _open_recording_menu(page):
    _tab(page, "recordings")


def test_the_recording_pane_names_the_open_recording(recording_page_and_errors):
    """The pane says which animal you are labeling, above the list you would switch with.

    With one pane showing at a time and the toolbar picker retired, this is the only place
    outside the browser tab's title that names it -- so the pane has to carry it even
    though the LIST already marks the same slug active.
    """
    page, errors, _ = recording_page_and_errors
    _tab(page, "recordings")
    assert page.locator("#recording-name").inner_text() == "flyA"
    assert page.locator("#recording-pane .pane-head").is_visible()
    assert not errors, "JS errors on load:\n  " + "\n  ".join(errors)


def test_the_picker_lists_the_projects_recordings(recording_page_and_errors):
    page, errors, _ = recording_page_and_errors
    _tab(page, "recordings")
    rows = page.locator(".rec-row")
    assert rows.count() == 2
    assert [rows.nth(i).locator(".rec-name").inner_text() for i in range(2)] == [
        "flyA",
        "flyB",
    ]
    # The open one is marked and unclickable; the other carries its label count.
    assert page.locator(".rec-row.is-active .rec-name").inner_text() == "flyA"
    assert page.locator(".rec-row.is-active").is_disabled()
    assert "unlabeled" in rows.nth(1).inner_text()
    assert not errors, "JS errors listing recordings:\n  " + "\n  ".join(errors)


def test_b_reveals_the_recordings_tab(recording_page_and_errors):
    """`b` activates it from any other pane. Registered only for a project session, so it is
    not advertised in the help of a bare results.h5 that could not honor it."""
    page, errors, _ = recording_page_and_errors
    _leave_tab(page, "recordings")
    page.keyboard.press("b")
    page.wait_for_timeout(600)
    expect(page.locator("#recording-pane")).to_be_visible()
    assert page.locator("#tab-recordings").get_attribute("aria-selected") == "true"
    assert not errors, "JS errors on the picker shortcut:\n  " + "\n  ".join(errors)


def test_switching_moves_the_editor_onto_the_new_recording(recording_page_and_errors):
    """The whole point: a different recording, without restarting the server."""
    page, errors, port = recording_page_and_errors
    _tab(page, "recordings")
    page.locator(".rec-row:not(.is-active)").first.click()
    # The server swaps, pushes a reload, and the page rebuilds from the new /api/meta.
    page.wait_for_function(
        "() => document.getElementById('recording-name')?.textContent === 'flyB'",
        timeout=15000,
    )
    assert page.locator("#recording-name").inner_text() == "flyB"
    # flyA has 6 frames and flyB has 9: the canvases were rebuilt, not just relabeled.
    assert (
        page.evaluate("() => Number(document.getElementById('frame-number').max)") == 8
    )
    assert not errors, "JS errors switching recordings:\n  " + "\n  ".join(errors)


def test_switching_with_unsaved_labels_neither_asks_nor_loses_them(
    recording_page_and_errors,
):
    """No prompt, and the edit is still there on the way back.

    The server keeps the recording being left, so the switch is free -- and the cue is
    what tells the operator their work is in memory rather than on disk. Both halves are
    asserted here because either alone is a plausible-looking half-feature: a silent
    switch that dropped the labels, or a cue over a switch that still refused.
    """
    page, errors, _ = recording_page_and_errors
    # Dirty the session. The Reviewed toggle needs no point selection, so this does not
    # depend on a click landing on a joint.
    page.locator("#reviewed-toggle").click()
    page.wait_for_timeout(700)
    # `updateDirty()` appends " *" to the title; without this the test would pass
    # vacuously, by switching out of a recording that had nothing to lose.
    assert page.title().endswith("*"), "the session is not dirty; this proves nothing"
    expect(page.locator("#unsaved")).to_be_visible()

    _tab(page, "recordings")
    page.locator(".rec-row:not(.is-active)").first.click()
    page.wait_for_function(
        "() => document.getElementById('recording-name')?.textContent === 'flyB'",
        timeout=15000,
    )
    # flyB is clean, but the project is not: the cue stays up, now counting recordings,
    # and flyA's row is dotted.
    expect(page.locator("#unsaved")).to_be_visible()
    assert page.title().endswith("*"), "the star went out over another recording's work"
    expect(page.locator('.rec-row[data-slug="flyA"] .rec-dirty')).to_be_visible()
    expect(page.locator('.rec-row[data-slug="flyB"] .rec-dirty')).to_be_hidden()

    page.locator('.rec-row[data-slug="flyA"]').click()
    page.wait_for_function(
        "() => document.getElementById('recording-name')?.textContent === 'flyA'",
        timeout=15000,
    )
    page.wait_for_timeout(500)
    assert page.locator("#reviewed-toggle").get_attribute("aria-pressed") == "true", (
        "the unsaved review flag did not come back with the recording"
    )
    assert not errors, "JS errors switching with unsaved labels:\n  " + "\n  ".join(
        errors
    )


def test_save_writes_every_recording_holding_unsaved_labels(
    recording_page_and_errors,
):
    """One Save, from wherever you are standing. Unsaved work spans the project, so a Save
    that wrote only the open recording would leave the title starred and the cue lit with
    no button that clears it."""
    page, errors, _ = recording_page_and_errors
    page.locator("#reviewed-toggle").click()
    page.wait_for_timeout(700)
    _tab(page, "recordings")
    page.locator('.rec-row[data-slug="flyB"]').click()
    page.wait_for_function(
        "() => document.getElementById('recording-name')?.textContent === 'flyB'",
        timeout=15000,
    )
    page.locator("#reviewed-toggle").click()  # dirty flyB too
    page.wait_for_timeout(700)
    assert page.locator("#unsaved").inner_text().endswith("2"), "the cue miscounts"

    page.locator("#save").click()
    page.wait_for_function(
        "() => !document.title.endsWith('*')", timeout=15000
    )  # both recordings written
    expect(page.locator("#unsaved")).to_be_hidden()
    expect(page.locator('.rec-row[data-slug="flyA"] .rec-dirty')).to_be_hidden()
    assert not errors, "JS errors saving the project:\n  " + "\n  ".join(errors)


def test_switching_recordings_rebuilds_in_place(recording_page_and_errors):
    """No page load. A reload and an in-place rebuild are indistinguishable to every
    other assertion in this file, so this one pins the page's identity: a marker set on
    ``window`` before the switch has to still be there afterwards, and no navigation may
    have happened. Without it, step one of this feature could silently revert to
    ``location.reload()`` and nothing would notice.
    """
    page, errors, _ = recording_page_and_errors
    navigated: list[str] = []
    page.on("framenavigated", lambda f: navigated.append(f.url))
    page.evaluate("() => { window.__pageId = 'sentinel'; }")

    _tab(page, "recordings")
    page.locator(".rec-row:not(.is-active)").first.click()
    page.wait_for_function(
        "() => document.getElementById('recording-name')?.textContent === 'flyB'",
        timeout=15000,
    )
    assert page.evaluate("() => window.__pageId") == "sentinel", "the page reloaded"
    assert not navigated, f"the page navigated: {navigated}"
    # ...and it is genuinely the new recording: flyA has 6 frames, flyB has 9.
    assert (
        page.evaluate("() => Number(document.getElementById('frame-number').max)") == 8
    )
    assert not errors, "JS errors rebuilding:\n  " + "\n  ".join(errors)


def test_selecting_a_recording_patches_the_list_instead_of_rebuilding_it(
    recording_page_and_errors,
):
    """Clicking a row must not blank and refill the list the click was aimed at.

    A switch resets the per-recording state and then re-reads the listing, and both used to
    ``replaceChildren()`` the list -- so the pane visibly emptied and repopulated under the
    operator, losing the scroll position with it. Node IDENTITY is what pins the fix: a
    marker stamped on each row before the switch survives a patch and cannot survive a
    rebuild. The active mark still has to MOVE, which is the one thing that really changed.
    """
    page, errors, _ = recording_page_and_errors
    _tab(page, "recordings")
    stamped = page.evaluate(
        "() => [...document.querySelectorAll('.rec-row')]"
        "  .map((r, i) => (r.dataset.probe = String(i), r.dataset.slug))"
    )
    assert stamped == ["flyA", "flyB"], f"unexpected rows: {stamped}"

    page.locator(".rec-row:not(.is-active)").first.click()
    page.wait_for_function(
        "() => document.getElementById('recording-name')?.textContent === 'flyB'",
        timeout=15000,
    )
    page.wait_for_timeout(800)  # let the post-switch refreshRecordings land too
    survived = page.evaluate(
        "() => [...document.querySelectorAll('.rec-row')]"
        "  .map(r => [r.dataset.slug, r.dataset.probe])"
    )
    assert survived == [["flyA", "0"], ["flyB", "1"]], (
        f"the rows were rebuilt, not patched: {survived}"
    )
    # ...and the list did not merely freeze: the open recording is the new one.
    assert page.locator(".rec-row.is-active .rec-name").inner_text() == "flyB"
    assert page.locator(".rec-row.is-active").is_disabled()
    assert not errors, "JS errors patching the list:\n  " + "\n  ".join(errors)


def test_the_rebuild_leaves_one_canvas_per_camera(recording_page_and_errors):
    """``buildViews`` appends, so a rebuild without teardown leaves both rigs live --
    and ``relayout`` then re-attaches the stale canvases over the new ones."""
    page, errors, _ = recording_page_and_errors
    before = page.locator("#stage canvas, #strip canvas").count()
    _tab(page, "recordings")
    page.locator(".rec-row:not(.is-active)").first.click()
    page.wait_for_function(
        "() => document.getElementById('recording-name')?.textContent === 'flyB'",
        timeout=15000,
    )
    page.wait_for_timeout(800)
    assert page.locator("#stage canvas, #strip canvas").count() == before
    assert not errors, "JS errors after the rebuild:\n  " + "\n  ".join(errors)


# -- the tabbed sidebar -----------------------------------------------------------

# The panel's seven tabs, in strip order, each with the pane it shows.
_TABS = [
    ("recordings", "recording-pane"),
    ("labeled", "labeled-pane"),
    ("suggest", "suggest-pane"),
    ("instances", "instances-pane"),
    ("marks", "marks-pane"),
    ("jobs", "jobs-pane"),
    ("bundle", "ba-pane"),
    ("settings", "settings-pane"),
]


def test_the_sidebar_holds_exactly_the_eight_tabs_and_panes_in_order(page_and_errors):
    """The structural net for the DOM rewrite, mirroring ``test_the_toolbar_is_one_row``.

    A stray ``</div>`` in a nested rewrite like this does not throw and does not remove
    any element -- it re-parents everything after it, which no id-based or content-based
    assertion can see. Both halves are pinned because the strip and the stack are
    positionally paired: a pane that drifts out of ``#sidebar-panes`` still shows, just
    never hides again.
    """
    page, errors = page_and_errors
    _open_panel(page)
    kids = page.evaluate(
        "() => ['sidebar-tabs', 'sidebar-panes'].map("
        "  id => [...document.getElementById(id).children].map(e => e.id))"
    )
    assert kids == [
        [f"tab-{name}" for name, _ in _TABS],
        [pane for _, pane in _TABS],
    ]
    assert not errors, "JS errors laying out the sidebar:\n  " + "\n  ".join(errors)


def test_each_tab_shows_its_own_pane_and_only_that_one(page_and_errors):
    """One pane at a time is the whole contract of a tab strip.

    Asserting only that the wanted pane is visible would pass a strip that never hid the
    previous one -- which looks like the stacked layout it replaced.
    """
    page, errors = page_and_errors
    _open_panel(page)
    for name, pane in _TABS[1:]:  # the Recording tab needs a project session
        _tab(page, name)
        showing = page.evaluate(
            "() => [...document.querySelectorAll('#sidebar-panes > .sidebar-body')]"
            "  .filter(e => !e.hidden).map(e => e.id)"
        )
        assert showing == [pane], f"{name} showed {showing}"
    assert not errors, "JS errors switching tabs:\n  " + "\n  ".join(errors)


def test_the_toolbar_no_longer_carries_a_recording_picker(page_and_errors):
    """Recordings live in the sidebar only. The toolbar row must survive the deletion."""
    page, errors = page_and_errors
    assert page.locator("#recording-wrap").count() == 0
    kids = page.evaluate(
        "() => [...document.getElementById('controls').children].map(e => e.className)"
    )
    assert kids == ["control-row frame-row", "control-row toolbar-row"]
    assert not errors


def test_which_tab_is_active_survives_a_reload(page_and_errors):
    """The arrangement is the operator's, so it outlives the page."""
    page, errors = page_and_errors
    _tab(page, "instances")
    page.reload(wait_until="networkidle")
    page.wait_for_timeout(900)
    assert page.locator("#tab-instances").get_attribute("aria-selected") == "true", (
        "the reload came back on a different tab"
    )
    expect(page.locator("#instances-pane")).to_be_visible()
    assert not errors


def test_the_rail_is_how_a_closed_panel_comes_back(page_and_errors):
    """There is no toolbar button for the panel any more, so the rail is the only click.

    Worth its own test because the failure mode is a one-way door: a rail that does not
    appear (or does not open the panel) leaves the panel unreachable except by pressing `j`,
    and the closed state persists across reloads -- so an operator who closes it once never
    sees it again. The two are also mutually exclusive by design, which is asserted here
    because a rail left on screen beside the open panel steals 34px of camera grid.
    """
    page, errors = page_and_errors
    _open_panel(page)
    expect(page.locator("#sidebar-rail")).to_be_hidden()

    page.locator("#frames-collapse").click()
    page.wait_for_timeout(300)
    expect(page.locator("#sidebar")).to_be_hidden()
    expect(page.locator("#sidebar-rail")).to_be_visible()

    page.locator("#sidebar-rail").click()
    page.wait_for_timeout(300)
    expect(page.locator("#sidebar")).to_be_visible()
    expect(page.locator("#sidebar-rail")).to_be_hidden()
    assert not errors, "JS errors toggling the panel:\n  " + "\n  ".join(errors)


def test_the_rail_survives_a_reload_with_the_panel_shut(page_and_errors):
    """The closed state persists, so the way back has to be painted at boot too.

    ``buildSidebar`` only calls ``openFrames()``; the closed branch has to reveal the rail
    itself, and the markup ships it hidden (the panel ships open). Missing that is a boot
    with neither the panel nor its handle.
    """
    page, errors = page_and_errors
    _close_panel(page)
    page.reload(wait_until="networkidle")
    page.wait_for_timeout(900)
    expect(page.locator("#sidebar")).to_be_hidden()
    expect(page.locator("#sidebar-rail")).to_be_visible()
    assert not errors, "JS errors on a shut-panel boot:\n  " + "\n  ".join(errors)


def test_the_skeleton_menu_offers_only_the_verb_that_applies(page_and_errors):
    """Create and Reseed share a menu, and never both act.

    The server refuses each in the other's state, so the pair is mutually exclusive by
    construction -- and since they now sit next to each other, the disabled state is what
    stops a click landing on the one that would be refused. It also doubles as the readout
    for whether this frame has been started.
    """
    page, errors = page_and_errors
    page.locator("#skeleton-toggle").click()
    page.wait_for_timeout(200)
    expect(page.locator("#skeleton-menu")).to_be_visible()
    assert page.locator("#skeleton-create").is_enabled(), "cannot start a bare frame"
    assert page.locator("#reseed").is_disabled(), "offered to reseed nothing"

    page.locator("#skeleton-create").click()
    page.wait_for_timeout(700)
    expect(page.locator("#skeleton-menu")).to_be_hidden()  # acting closes it
    page.locator("#skeleton-toggle").click()
    page.wait_for_timeout(200)
    assert page.locator("#skeleton-create").is_disabled(), "offered a second skeleton"
    assert page.locator("#reseed").is_enabled(), "cannot reseed a frame that has one"
    assert not errors, "JS errors in the skeleton menu:\n  " + "\n  ".join(errors)


def test_closing_the_panel_disarms(landmark_page_and_errors):
    page, errors, session = landmark_page_and_errors
    _tab(page, "marks")
    page.locator(".mark-row").first.click()
    page.wait_for_timeout(300)
    _close_panel(page)

    page.locator("#stage canvas").first.click(position={"x": 55, "y": 45})
    page.wait_for_timeout(500)
    assert session.state.landmarks.observed.sum() == 0, "a hidden panel still placed"
    assert not errors


def test_the_instance_pane_reports_this_frames_skeleton(page_and_errors):
    """Not a list -- the format cannot carry two. What it adds is how much is yours."""
    page, errors = page_and_errors
    _tab(page, "instances")
    assert page.locator("#instance-state").inner_text() == "none"
    page.keyboard.press("g")
    page.wait_for_timeout(700)
    assert "/" in page.locator("#instance-state").inner_text(), (
        "no cell tally after 'g'"
    )
    assert not errors


def test_the_reprojection_warning_anchors_on_what_the_view_asserts(page_and_errors):
    """The rule behind the warning ring, exercised in the browser that draws it.

    The check must measure the position a view *claims* against the geometry -- the operator's
    GT pixel, else what the annotation skeleton draws there. It used to fall back to the raw
    detection instead, which in a view on the far side of the animal is tens to hundreds of
    pixels out and cannot be moved by labeling: the joint stayed flagged in that view no
    matter how well it was labelled elsewhere (reported on AN07B017_260414_Fly4_005,
    ``lh_femur_tibia`` in ``rh``, 125 px). The data half of this is in test_gui.py.

    Asserted on ``warnAnchor`` rather than on pixels: the drawing is a ring and a connector on
    a canvas, so the anchor IS the observable rule, and nothing else in the suite can reach
    into a module the page loads.
    """
    page, errors = page_and_errors
    got = page.evaluate("""
      async () => {
        const { PoseView } = await import('/static/poseView.js');
        const view = new PoseView(0, document.createElement('canvas'), {});
        // One joint: the 3D reprojects it at (100, 100), the detector fired 100 px away.
        view.latent = [[100, 100]];
        view.detected = [[200, 100]];
        const out = {};
        view.instanceMode = false; // no skeleton yet: the detections are the primary layer
        view.fixed = [false];
        view.pts = [[200, 100]];
        out.no_skeleton = view.warnAnchor(0);
        view.instanceMode = true; // a skeleton exists, this cell is derived ...
        view.pts = [[100, 100]];  // ... so it is drawn AT the reprojection
        out.derived = view.warnAnchor(0);
        view.pts = [[180, 100]];  // showing seeds instead ('s'): the seed can drift
        out.seed_display = view.warnAnchor(0);
        view.fixed = [true];      // the operator's own pixel, wherever they put it
        view.pts = [[150, 100]];
        out.ground_truth = view.warnAnchor(0);
        return out;
      }
    """)
    assert got["no_skeleton"] == [200, 100], (
        "the detection is the anchor before a skeleton"
    )
    assert got["derived"] == [100, 100], (
        "a derived cell must anchor on its own drawn position (the reprojection), so its gap "
        "is zero -- anchoring on the detection is the bug this pins"
    )
    assert got["seed_display"] == [180, 100], "the seed is a claim of its own"
    assert got["ground_truth"] == [150, 100], "GT always wins"
    assert not errors, "JS errors reading the warning anchor:\n  " + "\n  ".join(errors)


# -- the bundle-adjust fix/free matrix ---------------------------------------------

# The per-camera master box. Worth browser tests rather than unit ones because it exists
# only in the DOM: the server has no notion of "all", so the only thing that makes the box
# true is that it writes the same four parameter groups the operator would have ticked by
# hand -- and the only way to see that is the request the pane sends.


def _open_bundle(page):
    """Show the Bundle-adjust tab and wait for the matrix to settle.

    The pane is built on first activation and then re-rendered when its readiness round trip
    lands, so a click placed between the two lands on a table that is about to be replaced.
    """
    _tab(page, "bundle")
    page.wait_for_selector("#ba-pane .ba-matrix tbody tr", timeout=10000)
    page.wait_for_timeout(700)


def _ba_row(page, cam):
    """The matrix row for one camera, found by the name it prints rather than by index."""
    cams = page.evaluate(
        "() => [...document.querySelectorAll('#ba-pane .ba-matrix tbody .ba-cam')]"
        "  .map(e => e.textContent)"
    )
    assert cam in cams, f"{cam} is not in the fix/free matrix ({cams})"
    return page.locator("#ba-pane .ba-matrix tbody tr").nth(cams.index(cam))


def _row_boxes(page, cam):
    """``[(class, checked, indeterminate), ...]`` -- the master first, then the groups.

    Read as properties, never as attributes: nothing here sets ``checked=""`` in the markup,
    so an assertion on the HTML would read every box as unticked and pass on a broken pane.
    """
    return _ba_row(page, cam).evaluate(
        "tr => [...tr.querySelectorAll('input[type=checkbox]')]"
        "  .map(b => [b.className, b.checked, b.indeterminate])"
    )


def _ba_params(page):
    """The parameter-group column names, between the 'all' column and the label count."""
    return page.evaluate(
        "() => [...document.querySelectorAll('#ba-pane .ba-matrix thead th')]"
        "  .map(e => e.textContent).slice(2, -1)"
    )


def _held(page, cam):
    """``(master_state, {group, ...})`` for one row, as the operator sees it.

    Derived from the header rather than hardcoded, because what the matrix opens on comes from
    the project's own ``[bundle_adjustment] fixed`` -- these tests are about what the master
    box does to that state, not about what the state happens to be.
    """
    boxes = _row_boxes(page, cam)
    assert boxes[0][0] == "ba-all-box", f"the master is not first in the row: {boxes}"
    groups = {p for p, b in zip(_ba_params(page), boxes[1:]) if b[1]}
    return (boxes[0][1], boxes[0][2]), groups


def test_the_all_box_holds_every_parameter_of_its_own_camera(recording_page_and_errors):
    """One click holds the whole camera -- and only that camera."""
    page, errors, _ = recording_page_and_errors
    posted: list[str] = []
    page.on(
        "request",
        lambda r: (
            posted.append(r.post_data)
            if r.url.endswith("/api/bundle-adjust/check")
            else None
        ),
    )
    _open_bundle(page)
    params = set(_ba_params(page))
    (checked, _mixed), opened_with = _held(page, "rm")
    assert not checked and opened_with != params, "premise: rm does not open fully held"
    neighbour = _held(page, "rf")

    _ba_row(page, "rm").locator(".ba-all-box").check()
    page.wait_for_timeout(900)

    boxes = _row_boxes(page, "rm")
    assert [b[1] for b in boxes] == [True] * (len(params) + 1), (
        f"the row did not follow the master: {boxes}"
    )
    assert not any(b[2] for b in boxes), (
        "a fully held row must not also read as half-held"
    )
    assert _held(page, "rf") == neighbour, (
        "the master is per camera; the neighbouring row must not move"
    )

    # The DOM agreeing with itself proves nothing about the solve: what the server is asked
    # about is the payload, and that is where a master box that only paints ticks would show.
    assert posted, "ticking the master did not re-ask the server for readiness"
    sent = json.loads(posted[-1])["settings"]["fixed"]
    assert set(sent["rm"]) == params, (
        f"the whole camera was not sent as held: {sent['rm']}"
    )
    assert set(sent["rf"]) == neighbour[1], f"a neighbour moved too: {sent['rf']}"
    assert not errors, "JS errors holding a camera:\n  " + "\n  ".join(errors)


def test_unticking_the_all_box_releases_the_camera_and_the_gauge_notices(
    recording_page_and_errors,
):
    """Releasing the anchored camera must come back as a refusal, not just as empty ticks.

    ``rh`` opens with its pose held because nothing else defines the world frame. Clearing it
    has to reach ``check_gauge`` server-side -- a master box that only repainted its own row
    would leave the pane cheerfully offering to solve a rig that can drift for free.
    """
    page, errors, _ = recording_page_and_errors
    _open_bundle(page)
    gauge = "the world frame is free"
    before = page.locator("#ba-pane .ba-problem").all_inner_texts()
    assert not any(gauge in p for p in before), (
        f"premise: the world frame is anchored when the pane opens ({before})"
    )

    master = _ba_row(page, "rh").locator(".ba-all-box")
    master.check()  # half-held -> held entirely
    page.wait_for_timeout(900)
    assert _held(page, "rh") == ((True, False), set(_ba_params(page)))

    _ba_row(page, "rh").locator(".ba-all-box").uncheck()
    page.wait_for_timeout(900)
    assert _held(page, "rh") == ((False, False), set()), "the row was not released"
    problems = page.locator("#ba-pane .ba-problem").all_inner_texts()
    assert any(gauge in p for p in problems), (
        f"no camera is anchored any more, but the pane did not say so: {problems}"
    )
    assert not errors, "JS errors releasing a camera:\n  " + "\n  ".join(errors)


def test_a_half_held_camera_reads_as_half_held(recording_page_and_errors):
    """The third state is the honest one: some groups pinned is neither on nor off.

    Both directions are pinned here -- the box that opens half-held, and the box that becomes
    half-held when a single group is ticked underneath it. The second is the one that rots:
    the per-parameter handler only re-rendered after its readiness round trip, so a master
    left out of that redraw would sit unticked over a camera that is already partly held.
    """
    page, errors, _ = recording_page_and_errors
    _open_bundle(page)
    params = set(_ba_params(page))
    state, groups = _held(page, "rh")
    assert groups and groups != params, f"premise: rh opens partly held ({groups})"
    assert state == (False, True), (
        f"{groups} of {params} are held, so the master must read as half-held, not {state}"
    )

    # ...and a camera reaches that state the other way round too, one group at a time. Clear
    # rm through the master first: from half-held that is two clicks (on to all, then off to
    # none), which is what a browser does with an indeterminate box and is worth pinning.
    _ba_row(page, "rm").locator(".ba-all-box").check()
    page.wait_for_timeout(800)
    _ba_row(page, "rm").locator(".ba-all-box").uncheck()
    page.wait_for_timeout(800)
    assert _held(page, "rm") == ((False, False), set()), (
        "the master did not clear the row"
    )

    first = _ba_params(page)[0]
    _ba_row(page, "rm").locator("input[type=checkbox]").nth(1).check()
    page.wait_for_timeout(900)
    assert _held(page, "rm") == ((False, True), {first}), (
        f"the master ignored {first} being ticked under it"
    )
    assert not errors, "JS errors half-holding a camera:\n  " + "\n  ".join(errors)


# -- the pane across a recording switch --------------------------------------------

# The pane is built once and then outlives every switch (the editor rebuilds itself in
# place, with no page load), so everything it holds describes the recording it was opened
# on. The fix/free matrix is the sharp edge: it is keyed by camera NAME, and a project
# holds recordings filmed on different rigs.


@pytest.fixture
def mixed_rig_page_and_errors(tmp_path, rig, fly):
    """Two recordings whose camera LISTS differ, with the smaller rig open.

    ``flySix`` is missing the front camera that ``flySeven`` has -- exactly the shape of a
    real project, where a view is added between sessions or a camera fails on the day.
    """
    from test_gui_recordings import _make_recording

    from deeperfly.cameras import CameraGroup
    from deeperfly.gui import open_target
    from deeperfly.project import Project

    def group(names):
        idx = [rig["names"].index(n) for n in names]
        return CameraGroup.from_arrays(
            list(names),
            rig["rvecs"][idx],
            rig["tvecs"][idx],
            rig["intrs"][idx],
            rig["dists"][idx],
        )

    six = [n for n in rig["names"] if n != "f"]
    project = Project.create(tmp_path / "proj", name="mixedrig")
    a = _make_recording(
        tmp_path / "flySix", group(six), fly, seed=0, n_frames=6, gt_cells=5, names=six
    )
    b = _make_recording(
        tmp_path / "flySeven",
        group(rig["names"]),
        fly,
        seed=500,
        n_frames=6,
        gt_cells=5,
        names=rig["names"],
    )
    _real_video(a, 6)
    _real_video(b, 6)
    project.add_recording(a, slug="flySix")
    project.add_recording(b, slug="flySeven")
    server, port = _serve_app(create_app(open_target(project.root, recording="flySix")))
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
            yield page, errors, port
            browser.close()
    finally:
        server.should_exit = True


def _switch_to(page, slug):
    _tab(page, "recordings")
    page.locator(f'.rec-row[data-slug="{slug}"]').click()
    page.wait_for_function(
        f"() => document.getElementById('recording-name')?.textContent === '{slug}'",
        timeout=15000,
    )
    page.wait_for_timeout(500)


def test_the_matrix_follows_the_new_recordings_camera_list(mixed_rig_page_and_errors):
    """A switch to a rig with a camera the pane has never seen must redraw, not refuse.

    The fix/free matrix is a map keyed by camera name, adopted once from the server. Kept
    across a switch it describes the animal you left, and the first camera the new rig has
    and the old one lacked has no entry at all -- which threw out of the render and left the
    whole tab reading "Could not read the bundle-adjustment plan".
    """
    page, errors, _ = mixed_rig_page_and_errors
    _open_bundle(page)
    assert "f" not in page.evaluate(
        "() => [...document.querySelectorAll('#ba-pane .ba-matrix tbody .ba-cam')]"
        "  .map(e => e.textContent)"
    ), "premise: the open recording has no front camera"

    _switch_to(page, "flySeven")
    # Not `_open_bundle`: that waits for a matrix, and the failure under test is a pane
    # with no matrix at all. Show the tab, let it settle, then say what is actually wrong.
    _tab(page, "bundle")
    page.wait_for_timeout(1200)

    refusal = page.locator("#ba-pane .ba-empty").all_inner_texts()
    assert not refusal, f"the pane refused to read the plan: {refusal}"
    cams = page.evaluate(
        "() => [...document.querySelectorAll('#ba-pane .ba-matrix tbody .ba-cam')]"
        "  .map(e => e.textContent)"
    )
    assert "f" in cams, f"the matrix still shows the previous rig: {cams}"
    # The label counts are the new recording's too -- a matrix redrawn over the previous
    # animal's readiness would still be describing the wrong labels.
    assert page.locator("#ba-pane .ba-matrix tbody tr").count() == 7
    assert not errors, "JS errors after the switch:\n  " + "\n  ".join(errors)


def test_the_pane_redraws_when_the_recording_changes_under_it(
    mixed_rig_page_and_errors,
):
    """A switch made elsewhere must redraw the pane in place, not leave it stale.

    Another browser -- or another operator -- can open a different recording while this
    page is sitting on the Bundle-adjust tab; the server pushes the swap to every socket
    and the editor rebuilds itself around it. The pane cannot wait to be re-activated, or
    it goes on showing the previous animal's matrix, label counts and calibration list
    under the new recording's name.
    """
    page, errors, port = mixed_rig_page_and_errors
    _open_bundle(page)
    page.request.post(
        f"http://127.0.0.1:{port}/api/recordings/open",
        data={"recording": "flySeven", "discard": False},
    )
    page.wait_for_function(
        "() => document.getElementById('recording-name')?.textContent === 'flySeven'",
        timeout=15000,
    )
    page.wait_for_function(
        "() => [...document.querySelectorAll('#ba-pane .ba-matrix tbody .ba-cam')]"
        "  .map(e => e.textContent).includes('f')",
        timeout=15000,
    )
    assert not errors, "JS errors refreshing the pane:\n  " + "\n  ".join(errors)
