# deeperfly web GUI assets

The browser front-end for `deeperfly gui`, served by
[`server.py`](../server.py). No bundler, no framework, **no build step** — plain
ES modules served straight from `static/`.

- `index.html` — the page shell, served at `/`.
- `static/` — the source, served at `/static/`:
  - `app.js` — controller: layout (grid / focus + thumbnails), frame scrubbing,
    the display toggles (skeleton, keypoint labels, 3D-estimate overlay, NMF
    skeleton + mesh), a two-step **select → act** annotation model — build a
    multi-point selection (click / Shift+click / double-click a keypoint across views /
    Shift+drag marquee / `a` all / `v` this view) then apply one of the Confirm / Reset /
    Occlude actions to all of it — undo / redo, the labelled-frames list, the 3D view,
    the keypoint-locations reference link (opens the docs viewer in a new tab) +
    keyboard-shortcut help, and edit routing. There is one unified editing
    model: a drag authors a ground-truth 2D pixel and, when the result carries 3D,
    re-solves the 3D live (no 2D/3D mode switch). The editor authors ground-truth 2D
    labels (saved to `labels.h5`); the 3D pose is re-derived from them server-side.
  - `poseView.js` — one `<canvas>` per camera: frame + draggable skeleton overlay,
    with wheel-zoom + drag-to-pan on the large view(s). Each joint's marker style
    encodes its source — ground truth (solid lime ring over a filled disc), detector
    prediction (thin dark ring over a disc that fades with confidence), or a point
    derived by reprojecting the 3D (a hollow circle in the point's limb palette
    colour; an occluded view shows the same way, since occluding just deletes the
    observation) — and it also draws the ghosted 3D-estimate reprojection overlay and
    optional per-joint name labels.
  - `scene3d.js` — the on-demand 3D view (a hand-rolled orbit camera on a canvas;
    drag to orbit, Shift/right-drag to pan, wheel to zoom). Shown in a non-modal
    floating panel (app.js owns the move/resize) that overlays the editor without
    blocking it, so the main frame scrubber still steps the 3D pose. Draws the camera rig
    (each camera an RGB axis triad x/right=red, y/down=green, z/optical=blue, as in
    the bundle-adjustment notebook), the triangulated pose, the fitted NMF skeleton,
    and the posed NMF mesh (composited from `meshGL.js` through a matching camera).
  - `meshGL.js` — WebGL2 renderer for the posed NMF mesh (per-camera overlay + the
    3D view).
  - `api.js` — REST + WebSocket client.
  - `types.js` — JSDoc `@typedef`s for the server payloads (comment-only; never
    fetched at runtime).
  - `styles.css`.

Press `?` in the page for the full list of keyboard shortcuts.

## Editing

Edit the `.js` files directly and reload the page — there is nothing to compile.

The files start with `// @ts-check` and use JSDoc type annotations, so VS Code's
built-in TypeScript service (no npm install required) type-checks them live and
gives autocomplete on the server payload shapes in `types.js`. The annotations
are purely advisory: they never affect what runs in the browser.
