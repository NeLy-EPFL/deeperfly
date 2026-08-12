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
 * What a navigation expects to be asked for next, so `App.schedulePrefetch` can warm
 * those frames in every view while the operator looks at the current one. Purely a
 * performance hint -- an absent or wrong one costs nothing but the miss.
 *
 * `step` is a signed frame delta to repeat (the arrow keys: the operator stepping +1 will
 * very likely step +1 again). `then` is an explicit frame index (the suggestion queue
 * knows exactly which frame its next entry is).
 * @typedef {object} NavHint
 * @property {number} [step]
 * @property {number} [then]
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
 * @property {string} cache_v  token identifying this recording, stamped into frame/mesh URLs
 * @property {number} n_views
 * @property {number} n_frames
 * @property {number} n_points
 * @property {boolean} has_3d
 * @property {boolean} has_nmf  whether a fitted NMF model overlay is available
 * @property {boolean} [has_cameras]  whether a camera rig is solved for this recording.
 *   False => uncalibrated: every view is an independent 2D canvas (no 3D, no
 *   reprojection, no cross-view help). Absent on an older server, which means calibrated.
 * @property {string[]} camera_names
 * @property {Record<string, [number, number]>} image_sizes  camera -> [height, width]
 * @property {string[]} point_names
 * @property {[number, number][]} bones
 * @property {[number, number, number][]} point_colors  0-255 RGB, one per point
 * @property {{name: string, color: [number, number, number]}[]} limbs  per-limb name + 0-255 RGB swatch, derived from the skeleton palette, for the legend
 * @property {Camera3D[]} cameras_3d  per-camera world poses for the rig plot
 * @property {CameraProj[]} cameras_proj  per-camera pinhole projection for the mesh overlay
 * @property {boolean} dirty
 * @property {boolean} has_jobs  whether this session can run pipeline commands
 * @property {string | null} project_root  the project this recording belongs to, if any
 * @property {string | null} recording  the project slug of the open recording
 * @property {any[]} landmarks  the calibration landmarks placed in this recording
 */

/**
 * One row of `GET /api/recordings`: a recording of the open project, with the counts
 * that decide which is worth opening next.
 * @typedef {object} RecordingRow
 * @property {string} slug  the name to pass back to `POST /api/recordings/open`
 * @property {string} id  the content-derived recording id
 * @property {string | null} subject  animal identifier, when the result records one
 * @property {number | null} n_frames
 * @property {number | null} fps
 * @property {boolean} active  whether this is the recording currently open
 * @property {boolean} has_results  false for a recording that has never been run (2D only)
 * @property {boolean} has_labels
 * @property {boolean} outputs_missing  the adopted outputs directory has gone away
 * @property {number} gt_points
 * @property {number} occluded
 * @property {number} labeled_frames
 * @property {number} reviewed_frames
 */

/**
 * The per-view 2D overlay (with the fixed/invisible masks) to draw for one frame.
 * @typedef {object} PointsPayload
 * @property {number} frame
 * @property {EditMode} mode
 * @property {Point[][]} points  [view][point]
 * @property {boolean[][]} fixed  [view][point]
 * @property {boolean[][]} invisible  [view][point]  occluded: dropped from triangulation
 * @property {boolean[][]} absent  [view][point]  "not on this animal" (amputated / ablated). Per-*point* truth broadcast over views, so it indexes like `fixed`/`invisible`. Rides EVERY reply including the lean mid-drag stream, because it gates whether the joint is drawn at all. Drawn as a dim grey tombstone with no bones; never draggable.
 * @property {Point[][] | null} proj  [view][point] the 3D reprojection, or null. Ghosted by the "3D estimate" overlay, and the canvas's fallback position + "projection" source for a joint with no observed pixel (occluded / undetected in that view).
 * @property {Point[][] | null} [nmf]  [view][point] fitted NMF model reprojection (display only), or null. Omitted on mid-drag replies (the server skips the per-frame re-fit) -- treat "absent" as "unchanged".
 * @property {(number | null)[][]} [conf]  [view][point] detector confidence, or null. Rides the settle/plain reply only (not the mid-drag stream).
 * @property {Point[][] | null} [pred]  [view][point] the raw detector prediction (before GT override), for the verbose overlay. Present only when verbose was requested.
 * @property {Point[][] | null} [placeholder]  [view][point] seed positions for joints ABSENT from a view (no detection / reprojection), so a GT can still be dragged into being; NaN->null elsewhere. Present only when verbose was requested.
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
 * One frame in the corrected-frames list: the frame index and whether the operator
 * has ticked it as reviewed.
 * @typedef {object} CorrectedFrame
 * @property {number} frame
 * @property {boolean} reviewed  whether the operator has marked this frame reviewed
 */

/**
 * One driver joint behind a suggested frame's score: the joint whose views disagree, how
 * badly, and in which view -- the "why am I being sent here" detail.
 * @typedef {object} SuggestionDriver
 * @property {number} [point]
 * @property {string} [point_name]
 * @property {number} [disagreement]  the joint's mean-over-views normalized residual
 * @property {string} [worst_camera]
 * @property {number} [worst_px]  the largest single-view reprojection residual, in pixels
 * @property {number} [views_over_threshold]
 * @property {string} [relation]  "far" / "near": geometrically, is the joint beyond the body centroid from that camera
 */

/**
 * Why a frame was suggested. `summary` is the one-line human reading the panel shows;
 * the rest is the detail behind it (absent for a diversity pick, which reports its grid
 * slot instead -- it is in the queue precisely because it is a *typical* pose).
 * @typedef {object} SuggestionReason
 * @property {string} [summary]
 * @property {number} [n_joints_over_threshold]
 * @property {SuggestionDriver[]} [drivers]
 * @property {[number, number]} [grid_slot]  [slot, n_slots] for a diversity pick
 */

/**
 * One frame in the suggestion queue: where it ranks, when it is, how badly its views
 * disagree, why it was picked, and whether the operator has since labeled it.
 * @typedef {object} Suggestion
 * @property {number | null} rank  1-based position in the queue
 * @property {number} frame
 * @property {number | null} t_s  time into the recording, in seconds
 * @property {number | null} score  multi-view disagreement; ranks within THIS recording only
 * @property {number | null} percentile  the score's percentile among this recording's frames
 * @property {"most-wrong" | "diversity" | string} kind  why it is in the list at all: the worst-scoring frames, or the uniform-temporal-grid reserve that keeps typical poses in the round
 * @property {SuggestionReason} reason
 * @property {boolean} labeled  the frame now carries ground truth (or is marked reviewed) -- i.e. done
 * @property {boolean} reviewed
 */

/**
 * The suggestion queue with its provenance and staleness, from `/api/suggestions`.
 *
 * `stale.level` escalates: "none", "progress" (the operator is working through it --
 * expected), "predictions" (results.h5 changed since it was computed, so it describes
 * predictions that no longer exist), "hard" (a different recording entirely -- the panel
 * shows no rows). `notes` are caveats about the queue itself (an under-delivered count,
 * a reseeded result, uncalibrated cameras) that would otherwise only reach the CLI log.
 * @typedef {object} SuggestionsPayload
 * @property {boolean} present  false when no sidecar has been computed yet
 * @property {string | null} [path]  the sidecar's path
 * @property {string} [command]  the exact command that (re)computes the queue
 * @property {string | null} [computed_utc]
 * @property {Record<string, any>} [params]  the ranking parameters (count, min_gap_s, threshold_px, ...)
 * @property {Record<string, any>} [source]  provenance: which array was scored, whether the result was reseeded, which cameras
 * @property {Record<string, any>} [coverage]
 * @property {Record<string, any>} [shortfall]
 * @property {{level: "none" | "progress" | "predictions" | "hard", reasons: string[]}} [stale]
 * @property {string[]} [notes]
 * @property {number} [n_done]  how many queue frames are already labeled
 * @property {Suggestion[]} [frames]  in rank order
 */

/**
 * An edit sent over the WebSocket; the server dispatches on `type` and replies
 * with a refreshed {@link PointsPayload}.
 * @typedef {object} EditMessage
 * @property {"edit_2d" | "edit_3d" | "set_gt" | "clear_gt" | "toggle_fixed" | "toggle_invisible" | "toggle_occluded" | "confirm" | "reset" | "occlude" | "clear_gt_targets" | "toggle_exclude" | "undo" | "redo" | "reset_point" | "reset_point_view" | "reset_frame" | "set_reviewed" | "set_absent"} type
 * @property {number} [view]
 * @property {number} [point]
 * @property {boolean} [reviewed]  for "set_reviewed": the frame's new reviewed state
 * @property {number} [x]
 * @property {number} [y]
 * @property {[number, number][]} [targets]  the (view, point) pairs a batched op acts on: "confirm" creates GT at the position already shown, "clear_gt_targets" deletes just the GT pixel, "toggle_exclude" toggles "exclude this detection from triangulation" (skipping cells that carry GT), "reset" retracts both, "occlude" is the set-only form "toggle_exclude" is built on, "set_absent" collapses them to a point SET (absence is recording-wide, so the view half is discarded)
 * @property {boolean} [absent]  for "set_absent": the value to set. Sent explicitly rather than toggled per point, so a mixed selection resolves one way instead of splitting.
 * @property {"all" | "predictions" | "projections"} [sources]  for "confirm": which suggestions to snapshot
 * @property {number} frame
 * @property {boolean} [fix]
 * @property {boolean} [verbose]  request the raw-prediction overlay array in the reply
 * @property {EditMode} mode
 * @property {number} [seq]  monotonic id stamped by App.sendEdit; echoed in the reply
 */

/**
 * The role handshake the server pushes over the edit socket: whether this browser
 * may edit the shared session (writer) or is read-only (reader) because another
 * browser already holds the writer slot. Only one connected browser edits at a time.
 * @typedef {object} RoleMessage
 * @property {"role"} type
 * @property {"writer" | "reader"} role
 * @property {number} clients  how many browsers are currently connected
 */

/**
 * Pushed to every open browser when the server swaps the recording underneath them
 * (`POST /api/recordings/open`). The page must do a full `location.reload()`: its
 * canvases, key bindings and frame-URL cache token were all built from `/api/meta`,
 * which it fetches exactly once per load.
 * @typedef {object} ReloadMessage
 * @property {"reload"} type
 * @property {string} reason  why -- currently always "recording"
 * @property {string} recording  the slug now open
 */

export {};
