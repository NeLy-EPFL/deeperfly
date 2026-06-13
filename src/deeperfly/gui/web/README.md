# deeperfly web GUI assets

The browser front-end for `deeperfly gui`, served by
[`server.py`](../server.py). No bundler, no framework, **no build step** — plain
ES modules served straight from `static/`.

- `index.html` — the page shell, served at `/`.
- `static/` — the source, served at `/static/`:
  - `app.js` — controller: layout (grid / focus + thumbnails), frame scrubbing,
    Edit 2D / Edit 3D mode switching, the display toggles (skeleton, keypoint
    labels, 3D-estimate overlay), hover/selection, the per-view / all-views Reset
    buttons, the camera-rig plot + keyboard-shortcut help, and edit routing.
  - `poseView.js` — one `<canvas>` per camera: frame + draggable skeleton overlay,
    with wheel-zoom + drag-to-pan on the large view(s); also draws the ghosted
    3D-estimate reprojection and optional per-joint name labels.
  - `scene3d.js` — the on-demand 3D camera-rig plot (a hand-rolled orbit camera on
    a canvas; no 3D engine). Each camera is an RGB axis triad (x/right=red,
    y/down=green, z/optical=blue), as in the bundle-adjustment notebook.
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
