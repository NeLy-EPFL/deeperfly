// @ts-check
// REST + WebSocket client for the deeperfly gui server (deeperfly/gui/server.py).
// This .js is the source -- there is no build step. VS Code type-checks it via
// `// @ts-check` and the JSDoc payload types in types.js.

/** @typedef {import("./types.js").Meta} Meta */
/** @typedef {import("./types.js").PointsPayload} PointsPayload */
/** @typedef {import("./types.js").ScenePayload} ScenePayload */
/** @typedef {import("./types.js").EditMode} EditMode */
/** @typedef {import("./types.js").EditMessage} EditMessage */

// Identifies the recording this page is editing; stamped into the frame and mesh
// URLs by `frameUrl`/`meshUrl` so the browser cannot serve another recording's
// picture for the same camera+frame. See `_session_version` in server.py. Set from
// /api/meta rather than by the caller so it cannot drift out of sync with the
// session -- until it arrives the URLs go unstamped, which the server answers
// `no-store` (correct, merely uncached).
let cacheVersion = "";

/** @returns {Promise<Meta>} */
export async function fetchMeta() {
  const r = await fetch("/api/meta");
  if (!r.ok) throw new Error(`GET /api/meta -> ${r.status}`);
  const meta = await r.json();
  cacheVersion = meta.cache_v ?? "";
  return meta;
}

/**
 * @param {number} frame
 * @param {EditMode} mode
 * @param {boolean} [verbose]  also return `pred` (the raw detections) for the Detected layer
 * @returns {Promise<PointsPayload>}
 */
export async function fetchPoints(frame, mode, verbose = false) {
  const q = verbose ? `?mode=${mode}&verbose=true` : `?mode=${mode}`;
  const r = await fetch(`/api/points/${frame}${q}`);
  if (!r.ok) throw new Error(`GET /api/points/${frame} -> ${r.status}`);
  return r.json();
}

/**
 * @param {number} frame
 * @returns {Promise<ScenePayload>}
 */
export async function fetchScene(frame) {
  const r = await fetch(`/api/scene/${frame}`);
  if (!r.ok) throw new Error(`GET /api/scene/${frame} -> ${r.status}`);
  return r.json();
}

/**
 * The frames the operator has touched, sorted, each with its reviewed flag -- the
 * editor's frame list.
 * @returns {Promise<{ frames: import("./types.js").CorrectedFrame[] }>}
 */
export async function fetchCorrected() {
  const r = await fetch("/api/corrected");
  if (!r.ok) throw new Error(`GET /api/corrected -> ${r.status}`);
  return r.json();
}

/**
 * The ranked "label these frames next" queue: the `labels_suggest.json` sidecar written
 * by `deeperfly labels-suggest`, joined server-side with the live labeled/reviewed
 * state. Always resolves -- `present: false` means no queue has been computed yet, which
 * is a normal state the panel renders as an invitation to run the command.
 * @returns {Promise<import("./types.js").SuggestionsPayload>}
 */
export async function fetchSuggestions() {
  const r = await fetch("/api/suggestions");
  if (!r.ok) throw new Error(`GET /api/suggestions -> ${r.status}`);
  return r.json();
}

/**
 * This project's recordings with their label counts, and which one is open.
 * `enabled: false` (with a `reason`) for a bare results.h5 session.
 * @returns {Promise<any>}
 */
export async function fetchRecordings() {
  const r = await fetch("/api/recordings");
  if (!r.ok) throw new Error(`GET /api/recordings -> ${r.status}`);
  return r.json();
}

/**
 * Switch the whole editor to another recording of this project. The server swaps the
 * session in place and pushes a reload to every open browser. `discard` abandons unsaved
 * labels, which the server refuses to do without it -- so a 409 here is the server
 * protecting hand work, not a failure.
 * @param {string} recording
 * @param {boolean} discard
 * @returns {Promise<any>}
 */
export async function openRecording(recording, discard) {
  const res = await fetch("/api/recordings/open", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ recording, discard }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `POST /api/recordings/open -> ${res.status}`);
  }
  return res.json();
}

/**
 * Write every recording holding unsaved labels -- what the editor's Save does.
 *
 * Unsaved work spans the project: the server keeps every recording the operator has
 * opened, so a switch loses nothing and "save" means all of it. There is deliberately no
 * wrapper for `POST /api/save` (the open recording alone) -- the editor has one Save,
 * because a per-recording one would leave the title starred with no button that clears it.
 *
 * `failed` is per recording and non-fatal to the others, so a caller that is about to
 * close must check it rather than trusting the 200.
 * @returns {Promise<{ saved: string[], failed: {recording: string, error: string}[],
 *   dirty: boolean, project_dirty: boolean, dirty_recordings: string[] }>}
 */
export async function saveAllCorrections() {
  const r = await fetch("/api/save-all", { method: "POST" });
  if (!r.ok) throw new Error(`POST /api/save-all -> ${r.status}`);
  return r.json();
}

/** Stop the server. Resolves even if the reply is cut short by the shutdown. */
export async function shutdownServer() {
  await fetch("/api/shutdown", { method: "POST" }).catch(() => {});
}

/**
 * @param {string} camera
 * @param {number} frame
 * @returns {string}
 */
export function frameUrl(camera, frame) {
  return `/api/frame/${encodeURIComponent(camera)}/${frame}${cacheStamp()}`;
}

/** `?v=<recording token>`, or "" before /api/meta has answered. */
function cacheStamp() {
  return cacheVersion ? `?v=${encodeURIComponent(cacheVersion)}` : "";
}

/**
 * URL of the posed NeuroMechFly mesh overlay (RGBA PNG) for a camera + frame.
 * @param {string} camera
 * @param {number} frame
 * @returns {string}
 */
export function meshUrl(camera, frame) {
  return `/api/mesh/${encodeURIComponent(camera)}/${frame}${cacheStamp()}`;
}

/**
 * The static model mesh topology + per-vertex colors (binary; fetched once). See
 * `_model_asset_bytes` in server.py for the layout.
 * @returns {Promise<ArrayBuffer>}
 */
export async function fetchModelAsset() {
  const r = await fetch("/api/model/asset");
  if (!r.ok) throw new Error(`GET /api/model/asset -> ${r.status}`);
  return r.arrayBuffer();
}

/**
 * The posed model vertices + smooth normals + valid-face mask for a frame (re-fit from
 * the edits). The head/abdomen size is the IK data estimate (no knob). See
 * `_model_verts_bytes` in server.py.
 * @param {number} frame
 * @returns {Promise<ArrayBuffer>}
 */
export async function fetchModelVerts(frame) {
  const r = await fetch(`/api/model/verts/${frame}`);
  if (!r.ok) throw new Error(`GET /api/model/verts/${frame} -> ${r.status}`);
  return r.arrayBuffer();
}

// A tiny request->reply WebSocket client: send an edit, get the refreshed points
// payload back through the `onPoints` callback. The server also pushes a role
// handshake ({type:"role"}) telling this browser whether it may edit or is
// read-only (only one connected browser edits at a time); that goes to `onRole`.
export class EditSocket {
  /**
   * @param {(p: PointsPayload) => void} onPoints
   * @param {(r: import("./types.js").RoleMessage) => void} [onRole]  the role
   *   handshake: whether this browser is the writer (editable) or read-only
   */
  constructor(onPoints, onRole, onReload) {
    this.onPoints = onPoints;
    this.onRole = onRole;
    this.onReload = onReload;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${proto}://${location.host}/ws`);
    this.ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      // Three message kinds share this socket, discriminated by `type`: the role
      // handshake, the "the open recording changed underneath you" push, and an edit
      // reply (a points payload), which never carries one. A new type WITHOUT a branch
      // here falls through to onPoints and is then silently eaten by applyPoints' frame
      // guard -- it looks exactly like a message that never arrived.
      if (msg && msg.type === "role") {
        this.onRole?.(msg);
        return;
      }
      if (msg && msg.type === "reload") {
        this.onReload?.(msg);
        return;
      }
      this.onPoints(msg);
    };
  }

  /** @param {EditMessage} msg */
  send(msg) {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  /** Ask the server to hand editing to this browser (the read-only "Take over" action). */
  claim() {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "claim" }));
    }
  }
}

// -- pipeline jobs -------------------------------------------------------------
//
// Polled by the Jobs panel while it is open. Separate from the /ws edit stream on purpose:
// that socket is single-writer, and a read-only tab must still be able to watch the queue.

/** @returns {Promise<any>} the queue, or `{enabled: false, reason}` when there is none */
export async function jobs(tail = 3) {
  const res = await fetch(`/api/jobs?tail=${tail}`);
  if (!res.ok) throw new Error(`GET /api/jobs -> ${res.status}`);
  return res.json();
}

/** Queue a job. `kind` must be server-allow-listed; arbitrary argv is refused. */
export async function submitJob(body) {
  const res = await fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST /api/jobs -> ${res.status}`);
  return res.json();
}

export async function cancelJob(id) {
  const res = await fetch(`/api/jobs/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`DELETE /api/jobs/${id} -> ${res.status}`);
  return res.json();
}

// -- project settings ----------------------------------------------------------
//
// The schema is DERIVED from the config dataclasses, so the forms built on it cannot drift
// from the code: a new option appears with nothing to keep in sync, and its help text is
// the prose already written for it.

/** @returns {Promise<any>} every describable section's fields, defaults and documentation */
export async function configSchema() {
  const res = await fetch("/api/schema");
  if (!res.ok) throw new Error(`GET /api/schema -> ${res.status}`);
  return res.json();
}

/** @returns {Promise<any>} current values, and which of them were actually set */
export async function configValues() {
  const res = await fetch("/api/config");
  if (!res.ok) throw new Error(`GET /api/config -> ${res.status}`);
  return res.json();
}

/** Set one key in the project's profile; `value: null` clears the override. */
export async function setConfig(section, key, value) {
  const res = await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ section, key, value }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `POST /api/config -> ${res.status}`);
  }
  return res.json();
}
