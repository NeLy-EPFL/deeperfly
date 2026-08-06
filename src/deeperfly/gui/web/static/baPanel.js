// @ts-check
// The Bundle adjustment pane: solve the rig from the ground truth the operator placed, save
// it as a NEW calibration, and choose which calibration the editor derives non-GT positions
// from. Server side is deeperfly/gui/ba.py + the /api/bundle-adjust routes in server.py.
//
// Deliberately self-contained: one class that owns one DOM node and talks to the server
// itself. app.js only has to construct it, hand it the pane element, and call `refresh()`
// when the tab is activated -- so the sidebar registry can be rearranged (it is being
// rewritten in parallel) without touching anything in here.
//
// The design rule the pane follows throughout: never show a number without saying what rig
// it was measured against, and never let a solve start that the labels cannot determine.
// The readiness block is the whole point -- bundle adjustment reports a small residual for
// an unidentifiable rig just as happily as for a good one.

const PARAM_DOC = {
  rvec: "orientation",
  tvec: "position",
  intr: "focal length + principal point",
  dist: "lens distortion",
};

/** @param {string} tag @param {string} [cls] @param {string} [text] */
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

async function getJson(url) {
  const r = await fetch(url);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `GET ${url} -> ${r.status}`);
  return body;
}

async function postJson(url, payload) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `POST ${url} -> ${r.status}`);
  return body;
}

export class BundleAdjustPanel {
  /**
   * @param {HTMLElement} root the pane element
   * @param {{ onRigChanged?: () => void, isReadOnly?: () => boolean, isDirty?: () => boolean }} hooks
   */
  constructor(root, hooks = {}) {
    this.root = root;
    this.hooks = hooks;
    /** @type {any} */ this.plan = null;
    /** @type {any} */ this.run = null;
    /** @type {any[]} */ this.calibrations = [];
    /** @type {Record<string, Set<string>>} */ this.fixed = {};
    /** @type {string | null} which recording everything above describes */
    this.recording = null;
    this.name = "from-labels";
    this.pollTimer = null;
    this.busy = false;
  }

  /** Fetch the plan and the calibration list, then draw. Safe to call repeatedly. */
  async refresh() {
    try {
      const [plan, cals] = await Promise.all([
        getJson("/api/bundle-adjust"),
        getJson("/api/calibrations").catch(() => ({ items: [], active: null })),
      ]);
      if (plan.enabled === false) {
        // Same convention as Settings and Jobs: a bare results.h5 is not a fault, so it is
        // explained in place rather than logged as an error.
        this.forget();
        this.root.replaceChildren(el("p", "ba-empty", plan.reason || "Unavailable here."));
        return;
      }
      this.adoptPlan(plan);
      this.calibrations = cals.items || [];
      this.activeCalibration = cals.active || null;
      this.run = plan.run && plan.run.state !== "idle" ? plan.run : this.run;
      this.render();
      if (this.run && this.run.state === "running") this.startPolling();
    } catch (err) {
      this.root.replaceChildren(
        el("p", "ba-empty", `Could not read the bundle-adjustment plan: ${err.message}`),
      );
    }
  }

  /**
   * Take a freshly fetched plan, starting over if it describes a DIFFERENT recording.
   *
   * Both routes into the pane come through here, because the pane outlives its subject: a
   * recording switch rebuilds the editor in place with no page load, so `this` survives it
   * holding the animal the operator just left. Every one of those fields is per recording
   * -- the fix/free matrix most sharply, since it is keyed by camera NAME and a project
   * holds recordings filmed on different rigs. A camera the new rig has and the old one
   * lacked had no entry at all, which threw out of the render and left the whole tab
   * reading "Could not read the bundle-adjustment plan".
   * @param {any} plan
   */
  adoptPlan(plan) {
    const recording = plan.recording ?? null;
    if (recording !== this.recording) this.forget();
    this.recording = recording;
    this.plan = plan;
    // Adopt the server's fix/free matrix once per recording, never on every poll, or the
    // operator's ticks would be reverted underneath them.
    if (!Object.keys(this.fixed).length) this.adoptFixed(plan.settings.fixed);
  }

  /**
   * Drop everything that belongs to one recording, and blank the pane.
   *
   * Called on a switch -- by `adoptPlan` when the server names another recording, and by
   * the editor as it rebuilds, so the stale matrix is not left on screen in the meantime.
   * The settings overrides go too: they were entered against another recording's config,
   * and `has_rig` alone changes what the defaults mean. So does the result block, which
   * names a calibration written into another recording's directory.
   */
  forget() {
    this.stopPolling();
    this.plan = null;
    this.recording = null;
    this.fixed = {};
    this.edits = {};
    this.run = null;
    this.busy = false;
    this.name = "from-labels";
    this.calibrations = [];
    this.activeCalibration = null;
    this.root.replaceChildren();
  }

  /** The rig's camera names. Read from the plan, never cached -- see `adoptPlan`. */
  get cameras() {
    return (this.plan && this.plan.cameras) || [];
  }

  /**
   * The held-parameter set for one camera, created empty if this pane has never seen it.
   *
   * Never index `this.fixed` directly. It is the operator's map and `cameras` is the
   * server's list; anything that lets the two disagree -- a switch, a view added under the
   * session -- must render as "nothing held" rather than throw out of the render.
   * @param {string} cam
   */
  heldBy(cam) {
    let held = this.fixed[cam];
    if (!held) this.fixed[cam] = held = new Set();
    return held;
  }

  adoptFixed(matrix) {
    this.fixed = {};
    for (const cam of this.cameras) this.fixed[cam] = new Set((matrix || {})[cam] || []);
  }

  /**
   * Hold -- or release -- every parameter group of one camera at once.
   *
   * Convenience over the per-parameter boxes, never a separate mode: it writes the same
   * `fixed` sets, so the payload the solver sees is indistinguishable from ticking the row
   * by hand. That matters because holding a camera *completely* is the common case rather
   * than the exotic one -- a refine trusts whole cameras and frees whole cameras.
   * @param {string} cam @param {boolean} fix
   */
  setAllFixed(cam, fix) {
    const held = this.heldBy(cam);
    for (const param of this.plan.params) {
      if (fix) held.add(param);
      else held.delete(param);
    }
    this.applyFixedChange();
  }

  /** Redraw now, then re-ask the server what this fix/free split can determine.
   *
   * The redraw is not cosmetic. The master box and the four parameter boxes describe the
   * same state from two directions, so whichever one was clicked leaves the others showing
   * the previous split until the row is rebuilt -- and `refreshReadiness` only rebuilds once
   * its round trip returns.
   */
  applyFixedChange() {
    this.render();
    this.refreshReadiness();
  }

  settingsPayload() {
    const s = { ...this.plan.settings };
    s.fixed = {};
    for (const cam of this.cameras) s.fixed[cam] = [...this.heldBy(cam)];
    for (const [k, v] of Object.entries(this.edits || {})) s[k] = v;
    return s;
  }

  // -- rendering -------------------------------------------------------------

  render() {
    const p = this.plan;
    this.root.replaceChildren();
    if (!p) return;

    if (p.cold_start) {
      this.root.append(
        el(
          "p",
          "ba-note",
          "This recording has no rig yet, so the solve starts from scratch: the cameras are " +
            "placed from the config's orbit prior (or by incremental SfM if there is none) " +
            "and then refined. Without a known distance the result is correct up to scale — " +
            "angles are meaningful, lengths are not.",
        ),
      );
    }
    if (p.defaults_note) this.root.append(el("p", "ba-note", p.defaults_note));
    if ((p.provisional || []).length) {
      this.root.append(
        el(
          "p",
          "ba-note",
          `${p.provisional.join(", ")} ${p.provisional.length > 1 ? "are" : "is"} shown but ` +
            "not solved from: the pose is a guess, so labels placed there do not move the " +
            "other views. Label it, run the solve below with it free, then switch to the " +
            "result — it starts informing the other views as soon as a calibration has " +
            "actually solved it.",
        ),
      );
    }

    this.root.append(this.fixedTable());
    this.root.append(this.settingsBox());
    this.root.append(this.readinessBox());
    this.root.append(this.runBox());
    this.root.append(this.calibrationsBox());
  }

  /** The fix/free matrix: rows are cameras, columns are parameter groups. Ticked = held. */
  fixedTable() {
    const box = el("div", "ba-section");
    box.append(el("h4", undefined, "What to hold fixed"));
    box.append(
      el(
        "p",
        "sec-doc",
        "Ticked means the solver may not change it. Anything unticked is solved for. One " +
          "camera must keep both its orientation and position ticked — that is what defines " +
          "the world frame; with nothing held still the whole rig can drift for free. The " +
          "'all' box holds a whole camera at once, which is the usual shape of a refine: " +
          "every camera you trust held, the one you are solving for free.",
      ),
    );
    const table = el("table", "ba-matrix");
    const head = el("tr");
    head.append(el("th", undefined, "camera"));
    const allTh = el("th", "ba-all", "all");
    allTh.title = "hold this camera entirely — ticks every parameter group in the row";
    head.append(allTh);
    for (const param of this.plan.params) {
      const th = el("th", undefined, param);
      th.title = PARAM_DOC[param] || param;
      head.append(th);
    }
    head.append(el("th", undefined, "labels"));
    table.append(el("thead").appendChild(head).parentElement);

    const body = el("tbody");
    for (const cam of this.cameras) {
      const fixed = this.heldBy(cam);
      const tr = el("tr");
      const free = !(fixed.has("rvec") && fixed.has("tvec"));
      if (free) tr.classList.add("is-free");
      tr.append(el("td", "ba-cam", cam));

      // The per-camera master: one click holds (or releases) the whole camera. Half-held is
      // shown as indeterminate rather than as either extreme -- a box reading "unticked" for
      // a camera whose orientation is in fact pinned would misdescribe the solve.
      const held = this.plan.params.filter((p) => fixed.has(p)).length;
      const allTd = el("td", "ba-all");
      const all = document.createElement("input");
      all.type = "checkbox";
      all.className = "ba-all-box";
      all.checked = held === this.plan.params.length;
      all.indeterminate = held > 0 && held < this.plan.params.length;
      all.disabled = this.busy || (this.hooks.isReadOnly?.() ?? false);
      // ...and what "released" really means depends on free_focal: with it off the solver
      // re-fixes every camera's intrinsics whatever this row says, so promising the untick
      // frees the whole camera would be the one claim the box cannot keep.
      const freeFocal = Boolean({ ...this.plan.settings, ...(this.edits || {}) }.free_focal);
      all.title = all.checked
        ? `${cam} is held entirely — untick to solve for it` +
          (freeFocal ? "" : ", except intrinsics (see 'free focal length' below)")
        : `hold all of ${cam} (${this.plan.params.join(", ")})`;
      all.setAttribute("aria-label", `hold all of ${cam}`);
      all.addEventListener("change", () => this.setAllFixed(cam, all.checked));
      allTd.append(all);
      tr.append(allTd);

      for (const param of this.plan.params) {
        const td = el("td");
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = fixed.has(param);
        cb.disabled = this.busy || (this.hooks.isReadOnly?.() ?? false);
        cb.title = `${cam}.${param} — ${PARAM_DOC[param] || param}`;
        cb.addEventListener("change", () => {
          if (cb.checked) this.heldBy(cam).add(param);
          else this.heldBy(cam).delete(param);
          this.applyFixedChange();
        });
        td.append(cb);
        tr.append(td);
      }
      const n = (this.plan.readiness.per_view || {})[cam] ?? 0;
      const td = el("td", n ? "ba-count" : "ba-count is-zero", String(n));
      td.title = n
        ? `${n} labeled cell(s) in this view are usable`
        : "no usable ground truth in this view — a camera cannot be solved for without any";
      tr.append(td);
      body.append(tr);
    }
    table.append(body);
    box.append(table);
    return box;
  }

  settingsBox() {
    const box = el("div", "ba-section");
    box.append(el("h4", undefined, "Settings"));
    box.append(
      el("p", "sec-doc", "Defaulted from the project config's [bundle_adjustment]."),
    );
    this.edits = this.edits || {};
    const s = { ...this.plan.settings, ...this.edits };
    const grid = el("div", "ba-grid");

    const row = (label, node, doc) => {
      const wrap = el("label", "ba-field");
      wrap.append(el("span", "ba-label", label));
      wrap.append(node);
      if (doc) wrap.title = doc;
      grid.append(wrap);
    };
    const select = (key, options) => {
      const sel = document.createElement("select");
      for (const o of options) {
        const opt = document.createElement("option");
        opt.value = o;
        opt.textContent = o;
        if (String(s[key]) === o) opt.selected = true;
        sel.append(opt);
      }
      sel.disabled = this.busy;
      sel.addEventListener("change", () => {
        this.edits[key] = sel.value;
        this.refreshReadiness();
      });
      return sel;
    };
    const number = (key, step) => {
      const inp = document.createElement("input");
      inp.type = "number";
      inp.step = String(step);
      inp.value = s[key] === null || s[key] === undefined ? "" : String(s[key]);
      inp.disabled = this.busy;
      inp.addEventListener("change", () => {
        this.edits[key] = inp.value === "" ? null : Number(inp.value);
        this.refreshReadiness();
      });
      return inp;
    };
    const check = (key) => {
      const inp = document.createElement("input");
      inp.type = "checkbox";
      inp.checked = Boolean(s[key]);
      inp.disabled = this.busy;
      inp.addEventListener("change", () => {
        this.edits[key] = inp.checked;
        this.refreshReadiness();
      });
      return inp;
    };

    row("loss", select("loss", this.plan.losses),
      "Robust losses down-weight large residuals. cauchy at a few px is what this lab measured as helping a hard rig rather than flattering it.");
    row("f_scale (px)", number("f_scale", 0.5),
      "The pixel scale of the robust loss: residuals beyond about this get down-weighted.");
    row("max_nfev", number("max_nfev", 100), "Iteration cap handed to least_squares.");
    row("max_frames", number("max_frames", 10),
      "Cap on labeled frames used. Blank uses them all; more frames means more 3D unknowns, not just more constraints.");
    row("frame sampling", select("frame_sampling", this.plan.samplings),
      "Which labeled frames to keep when capped. 'coverage' prefers frames with the most multi-view cells, which are the best conditioned.");
    row("weigh by confidence", check("weigh_by_confidence"),
      "Scale each residual by the detector's confidence. Irrelevant here unless landmarks or predictions are mixed in: these observations are hand-placed.");
    row("free focal length", check("free_focal"),
      "Off for a reason: focal error trades against depth, so freeing it can lower the residual while making the rig worse.");
    box.append(grid);
    return box;
  }

  readinessBox() {
    const rd = this.plan.readiness || {};
    const box = el("div", "ba-section");
    box.append(el("h4", undefined, "Readiness"));
    const line = el(
      "p",
      "sec-doc",
      `${rd.n_tracks ?? 0} usable point(s) across ${rd.n_frames ?? 0} labeled frame(s). ` +
        `Solving for: ${(rd.free || []).join(", ") || "nothing"}.`,
    );
    box.append(line);
    for (const problem of rd.problems || []) {
      box.append(el("p", "ba-problem", problem));
    }
    for (const w of rd.warnings || []) {
      box.append(el("p", "ba-warn", w));
    }
    // The co-visibility of each free camera is what actually ties it to the rig.
    for (const cam of rd.free || []) {
      const co = (rd.covisibility || {})[cam] || {};
      const shared = Object.entries(co)
        .filter(([, n]) => n > 0)
        .sort((a, b) => b[1] - a[1])
        .map(([k, n]) => `${k} ${n}`)
        .join(", ");
      box.append(
        el("p", "sec-doc", `${cam} shares points with: ${shared || "nothing — it cannot be tied to the rig"}`),
      );
    }
    return box;
  }

  runBox() {
    const box = el("div", "ba-section");
    box.append(el("h4", undefined, "Solve"));
    const nameWrap = el("label", "ba-field");
    nameWrap.append(el("span", "ba-label", "save as"));
    const nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.value = this.name;
    nameInput.placeholder = "from-labels";
    nameInput.disabled = this.busy;
    nameInput.addEventListener("change", () => {
      this.name = nameInput.value.trim() || "from-labels";
    });
    nameWrap.append(nameInput);
    box.append(nameWrap);
    box.append(
      el(
        "p",
        "sec-doc",
        "Written as a new calibration in the project's calibrations/ directory. An existing " +
          "one is never overwritten — a name that is taken gets a numbered suffix.",
      ),
    );

    const button = document.createElement("button");
    button.className = "ba-run";
    button.textContent = this.busy ? "Solving…" : "Run bundle adjustment";
    const blocked = !(this.plan.readiness || {}).ok;
    button.disabled = this.busy || blocked || (this.hooks.isReadOnly?.() ?? false);
    if (blocked) button.title = "Fix the problems above first";
    button.addEventListener("click", () => this.start());
    box.append(button);

    if (this.run) box.append(this.resultBox());
    return box;
  }

  resultBox() {
    const r = this.run;
    const box = el("div", "ba-result");
    if (r.state === "running") {
      box.append(el("p", "sec-doc", `Solving on ${r.n_tracks} point(s)…`));
      return box;
    }
    if (r.state === "failed") {
      box.append(el("p", "ba-problem", r.error || "the solve failed"));
      return box;
    }
    if (r.state !== "done") return box;
    box.append(
      el(
        "p",
        "ba-ok",
        `Wrote ${r.calibration_file} — reprojection ${fmt(r.before.median)} → ${fmt(r.after.median)} px ` +
          `on ${r.after.n} observations (${r.nfev} iterations).`,
      ),
    );
    const table = el("table", "ba-matrix");
    const head = el("tr");
    for (const h of ["camera", "before", "after", "moved"]) head.append(el("th", undefined, h));
    table.append(el("thead").appendChild(head).parentElement);
    const body = el("tbody");
    for (const cam of this.cameras) {
      const b = (r.before.per_view || {})[cam] || {};
      const a = (r.after.per_view || {})[cam] || {};
      const m = (r.moved || {})[cam] || { rvec: 0, tvec: 0 };
      const tr = el("tr");
      tr.append(el("td", "ba-cam", cam));
      tr.append(el("td", undefined, fmt(b.median)));
      tr.append(el("td", undefined, fmt(a.median)));
      const movedText = m.rvec === 0 && m.tvec === 0 ? "held" : `${m.rvec.toFixed(4)} / ${m.tvec.toFixed(3)}`;
      const td = el("td", m.rvec === 0 && m.tvec === 0 ? "ba-held" : undefined, movedText);
      td.title = "rotation (rad) / position, relative to where it started";
      tr.append(td);
      body.append(tr);
    }
    table.append(body);
    box.append(table);
    box.append(
      el(
        "p",
        "sec-doc",
        "Judge this on the reprojection column, not on how far a camera moved: for a distant " +
          "long-focal rig, lateral position trades against rotation almost exactly, so a " +
          "well-fitted camera can land some way from where it started.",
      ),
    );
    return box;
  }

  calibrationsBox() {
    const box = el("div", "ba-section");
    box.append(el("h4", undefined, "Calibration in use"));
    box.append(
      el(
        "p",
        "sec-doc",
        "Which rig the editor derives non-GT positions from. Switching re-derives every 3D " +
          "point from the labels, so save first — a depth you placed by hand in a single view " +
          "was measured along the old camera's ray and cannot be carried over.",
      ),
    );
    const list = el("div", "ba-cal-list");
    const rows = [
      {
        file: null,
        name: "the rig in results.h5",
        note: "what a pipeline run would use",
        active: !this.activeCalibration,
      },
      ...this.calibrations,
    ];
    for (const row of rows) {
      const item = el("div", row.active ? "ba-cal is-active" : "ba-cal");
      const label = el("div", "ba-cal-name", row.name || row.file);
      item.append(label);
      const bits = [];
      if (row.median_px !== undefined && row.median_px !== null) bits.push(`${fmt(row.median_px)} px`);
      if (row.n_observations) bits.push(`${row.n_observations} obs`);
      if (row.units && row.units !== "arbitrary") bits.push(row.units);
      else if (row.units === "arbitrary") bits.push("scale arbitrary");
      if (row.created_utc) bits.push(String(row.created_utc).slice(0, 16).replace("T", " "));
      if (row.note) bits.push(row.note);
      if (row.error) bits.push(`unreadable: ${row.error}`);
      if (row.file && row.covers_session === false) bits.push("does not cover this session's views");
      item.append(el("div", "ba-cal-meta", bits.join(" · ")));
      if (!row.active) {
        const actions = el("div", "ba-cal-actions");
        const use = document.createElement("button");
        use.className = "ba-use";
        use.textContent = "Use";
        use.disabled = Boolean(row.error) || row.covers_session === false || (this.hooks.isReadOnly?.() ?? false);
        use.addEventListener("click", () => this.select(row.file));
        actions.append(use);
        if (row.file && row.deletable) {
          const del = document.createElement("button");
          del.className = "ba-del";
          del.textContent = "Delete";
          del.title = "Remove this calibration (only ones solved in this tab, never the one in use)";
          del.disabled = this.hooks.isReadOnly?.() ?? false;
          del.addEventListener("click", () => this.remove(row));
          actions.append(del);
        }
        item.append(actions);
      } else {
        item.append(el("span", "ba-active-tag", "in use"));
      }
      list.append(item);
    }
    box.append(list);
    return box;
  }

  // -- actions ---------------------------------------------------------------

  /** Re-ask the server for readiness with the operator's current ticks, without solving. */
  async refreshReadiness() {
    try {
      const plan = await getJson("/api/bundle-adjust");
      if (plan.enabled === false) return;
      // Through `adoptPlan` like every other read: a switch can land between the click and
      // this reply, and then the ticks being checked belong to another rig.
      this.adoptPlan(plan);
      // Keep the operator's edits; take only the derived block.
      this.plan = { ...this.plan, settings: { ...plan.settings, ...(this.edits || {}) } };
      const payload = this.settingsPayload();
      this.plan.readiness = await postJson("/api/bundle-adjust/check", { settings: payload });
      this.render();
    } catch {
      this.render();
    }
  }

  async start() {
    if (this.busy) return;
    this.busy = true;
    this.run = { state: "running", n_tracks: this.plan.readiness.n_tracks };
    this.render();
    try {
      await postJson("/api/bundle-adjust", {
        name: this.name,
        settings: this.settingsPayload(),
      });
      this.startPolling();
    } catch (err) {
      this.busy = false;
      this.run = { state: "failed", error: err.message };
      this.render();
    }
  }

  startPolling() {
    if (this.pollTimer !== null) return;
    this.pollTimer = window.setInterval(async () => {
      try {
        const st = await getJson("/api/bundle-adjust/run");
        // "idle" is what the server answers for a solve belonging to another recording, so
        // it must clear the result block rather than be rendered as an empty one.
        this.run = st && st.state !== "idle" ? st : null;
        if (st.state !== "running") {
          this.stopPolling();
          this.busy = false;
          await this.refresh();
        } else {
          this.render();
        }
      } catch {
        this.stopPolling();
        this.busy = false;
      }
    }, 1000);
  }

  stopPolling() {
    if (this.pollTimer !== null) {
      window.clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }

  /** @param {any} row the calibration to remove */
  async remove(row) {
    if (!window.confirm(`Delete ${row.file}? This cannot be undone.`)) return;
    try {
      await postJson("/api/calibrations/delete", { calibration: row.file });
      await this.refresh();
    } catch (err) {
      window.alert(`Could not delete: ${err.message}`);
    }
  }

  /** @param {string | null} file null = go back to the rig in results.h5 */
  async select(file) {
    // An empty file means "back to the rig in results.h5" -- the server handles it.
    const dirty = this.hooks.isDirty?.() ?? false;
    if (dirty) {
      const ok = window.confirm(
        "You have unsaved labels. Switching the calibration re-derives every 3D point and " +
          "clears the undo history. Save them first?",
      );
      if (ok) return; // let the operator save, then click again
    }
    try {
      await postJson("/api/calibrations/select", { calibration: file || "", discard: true });
      await this.refresh();
      this.hooks.onRigChanged?.();
    } catch (err) {
      window.alert(`Could not switch calibration: ${err.message}`);
    }
  }
}

function fmt(v) {
  return v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(2);
}
