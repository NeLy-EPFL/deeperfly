// @ts-check
// Shared payload shapes exchanged with the deeperfly gui server (see
// deeperfly/gui/server.py). This file holds only JSDoc @typedef declarations --
// there is no runtime code, and it is never imported at run time (the other
// modules reference these types via `import("./types.js")` in JSDoc, which the
// type system erases). VS Code's built-in TypeScript service uses them to
// type-check the rest of the GUI; no build step or npm is involved.

/** @typedef {"edit_2d" | "edit_3d"} EditMode */

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
 * One-time metadata the front-end needs to lay out and draw the editor.
 * @typedef {object} Meta
 * @property {string} results_path
 * @property {number} n_views
 * @property {number} n_frames
 * @property {number} n_points
 * @property {boolean} has_3d
 * @property {string[]} camera_names
 * @property {Record<string, [number, number]>} image_sizes  camera -> [height, width]
 * @property {string[]} point_names
 * @property {[number, number][]} bones
 * @property {[number, number, number][]} point_colors  0-255 RGB, one per point
 * @property {Camera3D[]} cameras_3d  per-camera world poses for the rig plot
 * @property {boolean} dirty
 */

/**
 * The per-view 2D overlay (and fixed mask) to draw for one frame.
 * @typedef {object} PointsPayload
 * @property {number} frame
 * @property {EditMode} mode
 * @property {Point[][]} points  [view][point]
 * @property {boolean[][]} fixed  [view][point]
 * @property {Point[][] | null} proj  [view][point] latent 3D reprojection (display only), or null
 * @property {boolean} dirty
 */

/**
 * The current frame's triangulated 3D keypoints, for the rig plot.
 * @typedef {object} ScenePayload
 * @property {number} frame
 * @property {Point3[] | null} points3d  one per point, or null when 2D-only
 */

/**
 * An edit sent over the WebSocket; the server dispatches on `type` and replies
 * with a refreshed {@link PointsPayload}.
 * @typedef {object} EditMessage
 * @property {"edit_2d" | "edit_3d" | "toggle_fixed" | "reset_point" | "reset_point_view"} type
 * @property {number} [view]
 * @property {number} [point]
 * @property {number} [x]
 * @property {number} [y]
 * @property {number} frame
 * @property {boolean} [fix]
 * @property {EditMode} mode
 */

export {};
