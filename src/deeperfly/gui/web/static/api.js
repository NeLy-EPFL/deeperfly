// @ts-check
// REST + WebSocket client for the deeperfly gui server (deeperfly/gui/server.py).
// This .js is the source -- there is no build step. VS Code type-checks it via
// `// @ts-check` and the JSDoc payload types in types.js.

/** @typedef {import("./types.js").Meta} Meta */
/** @typedef {import("./types.js").PointsPayload} PointsPayload */
/** @typedef {import("./types.js").ScenePayload} ScenePayload */
/** @typedef {import("./types.js").EditMode} EditMode */
/** @typedef {import("./types.js").EditMessage} EditMessage */

/** @returns {Promise<Meta>} */
export async function fetchMeta() {
  const r = await fetch("/api/meta");
  if (!r.ok) throw new Error(`GET /api/meta -> ${r.status}`);
  return r.json();
}

/**
 * @param {number} frame
 * @param {EditMode} mode
 * @returns {Promise<PointsPayload>}
 */
export async function fetchPoints(frame, mode) {
  const r = await fetch(`/api/points/${frame}?mode=${mode}`);
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

/** @returns {Promise<{ dirty: boolean }>} */
export async function saveCorrections() {
  const r = await fetch("/api/save", { method: "POST" });
  if (!r.ok) throw new Error(`POST /api/save -> ${r.status}`);
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
  return `/api/frame/${encodeURIComponent(camera)}/${frame}`;
}

// A tiny request->reply WebSocket client: send an edit, get the refreshed points
// payload back through the `onPoints` callback.
export class EditSocket {
  /** @param {(p: PointsPayload) => void} onPoints */
  constructor(onPoints) {
    this.onPoints = onPoints;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${proto}://${location.host}/ws`);
    this.ws.onmessage = (ev) => this.onPoints(JSON.parse(ev.data));
  }

  /** @param {EditMessage} msg */
  send(msg) {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }
}
