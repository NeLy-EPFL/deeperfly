// @ts-check
// Shared payload shapes exchanged with the deeperfly gui server (see
// deeperfly/gui/server.py). This file holds only JSDoc @typedef declarations --
// there is no runtime code, and it is never imported at run time (the other
// modules reference these types via `import("./types.js")` in JSDoc, which the
// type system erases). VS Code's built-in TypeScript service uses them to
// type-check the rest of the GUI; no build step or npm is involved.

/**
 * The wire value selecting the per-view overlay the server returns. The editor has a
 * single unified editing model (a drag authors GT and, when the result has 3D,
 * re-solves it) -- this is no longer a user-facing switch; the client always requests
 * the plain per-view overlay ("edit_2d"). "edit_3d" (the 3D reprojected into every
 * view) is retained on the server for tests/back-compat.
 * @typedef {"edit_2d" | "edit_3d"} EditMode
 */

/**
 * A drawn 2D point `[x, y]`, or null when the keypoint is not visible in a view.
 * @typedef {[number, number] | null} Point
 */

/**
 * A world-frame 3D point `[x, y, z]`, or null when not triangulated.
 * @typedef {[number, number, number] | null} Point3
 */

/**
 * One camera's world-frame pose, for the 3D rig plot. `right`/`up`/`forward` are
 * unit axes; `forward` is the optical axis (the direction it looks).
 * @typedef {object} Camera3D
 * @property {string} name
 * @property {[number, number, number]} position  camera centre in world coords
 * @property {[number, number, number]} right
 * @property {[number, number, number]} up
 * @property {[number, number, number]} forward
 */

/**
 * One camera's pinhole projection, for the client's WebGL mesh overlay.
 * @typedef {object} CameraProj
 * @property {string} name
 * @property {[number, number, number, number]} intr  [fx, fy, cx, cy]
 * @property {number[]} rmat  3x3 world->camera rotation, row major
 * @property {[number, number, number]} tvec
 * @property {[number, number]} size  footage [width, height]
 */

/**
 * One-time metadata the front-end needs to lay out and draw the editor.
 * @typedef {object} Meta
 * @property {string} results_path
 * @property {number} n_views
 * @property {number} n_frames
 * @property {number} n_points
 * @property {boolean} has_3d
 * @property {boolean} has_nmf  whether a fitted NMF model overlay is available
 * @property {string[]} camera_names
 * @property {Record<string, [number, number]>} image_sizes  camera -> [height, width]
 * @property {string[]} point_names
 * @property {[number, number][]} bones
 * @property {[number, number, number][]} point_colors  0-255 RGB, one per point
 * @property {{name: string, color: [number, number, number]}[]} limbs  per-limb name + 0-255 RGB swatch, derived from the skeleton palette, for the legend
 * @property {Camera3D[]} cameras_3d  per-camera world poses for the rig plot
 * @property {CameraProj[]} cameras_proj  per-camera pinhole projection for the mesh overlay
 * @property {boolean} dirty
 */

/**
 * The per-view 2D overlay (with the fixed/invisible masks) to draw for one frame.
 * @typedef {object} PointsPayload
 * @property {number} frame
 * @property {EditMode} mode
 * @property {Point[][]} points  [view][point]
 * @property {boolean[][]} fixed  [view][point]
 * @property {boolean[][]} invisible  [view][point]  occluded: dropped from triangulation
 * @property {Point[][] | null} proj  [view][point] the 3D reprojection, or null. Ghosted by the "3D estimate" overlay, and the canvas's fallback position + "projection" source for a joint with no observed pixel (occluded / undetected in that view).
 * @property {Point[][] | null} [nmf]  [view][point] fitted NMF model reprojection (display only), or null. Omitted on mid-drag replies (the server skips the per-frame re-fit) -- treat "absent" as "unchanged".
 * @property {(number | null)[][]} [conf]  [view][point] detector confidence, or null. Rides the settle/plain reply only (not the mid-drag stream).
 * @property {Point[][] | null} [pred]  [view][point] the raw detector prediction (before GT override), for the verbose overlay. Present only when verbose was requested.
 * @property {boolean} dirty
 * @property {boolean} [can_undo]  whether an undo step is available
 * @property {boolean} [can_redo]  whether a redo step is available
 * @property {number | null} [seq]  the seq of the edit this reply answers, echoed so a superseded reply can be dropped; absent on plain frame fetches
 * @property {number | null} [goto]  for undo/redo: the frame the reverted edit was on, so the client navigates there; null for in-place edits
 */

/**
 * The current frame's 3D pose, for the 3D scene view.
 * @typedef {object} ScenePayload
 * @property {number} frame
 * @property {Point3[] | null} points3d  triangulated keypoints, or null when 2D-only
 * @property {Point3[] | null} nmf3d  fitted NMF model joints, or null when no IK model
 */

/**
 * One frame in the corrected-frames list: the frame index and how many of its
 * keypoints carry a manual correction.
 * @typedef {object} CorrectedFrame
 * @property {number} frame
 * @property {number} count  number of corrected keypoints in the frame
 */

/**
 * An edit sent over the WebSocket; the server dispatches on `type` and replies
 * with a refreshed {@link PointsPayload}.
 * @typedef {object} EditMessage
 * @property {"edit_2d" | "edit_3d" | "set_gt" | "clear_gt" | "toggle_fixed" | "toggle_invisible" | "toggle_occluded" | "confirm" | "reset" | "occlude" | "undo" | "redo" | "reset_point" | "reset_point_view" | "reset_frame"} type
 * @property {number} [view]
 * @property {number} [point]
 * @property {number} [x]
 * @property {number} [y]
 * @property {[number, number][]} [targets]  the (view, point) pairs a batched op acts on: "confirm" promotes them to GT, "reset" clears them to unset, "occlude" flags them occluded
 * @property {"all" | "predictions" | "projections"} [sources]  for "confirm": which suggestions to snapshot
 * @property {number} frame
 * @property {boolean} [fix]
 * @property {boolean} [verbose]  request the raw-prediction overlay array in the reply
 * @property {EditMode} mode
 * @property {number} [seq]  monotonic id stamped by App.sendEdit; echoed in the reply
 */

export {};
