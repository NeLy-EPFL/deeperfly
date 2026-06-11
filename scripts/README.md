# scripts/

Maintenance scripts that are **not** part of the installed package or the docs
build. Run them by hand when the inputs they depend on change.

## `build_keypoint_viewer_assets.py`

Regenerates the static assets for the interactive keypoint-locations docs page
(`docs/explanation/keypoints.md`, viewer at `docs/keypoints/viewer.html`). It
composes the NeuroMechFly model with [flygym], exports a flattened, self-contained
MJCF + simplified meshes, and writes the pose/keypoint metadata the browser viewer
loads. Outputs land under `docs/keypoints/assets/` and are committed, so the docs
CI never needs flygym or mujoco.

Run it whenever the NeuroMechFly model (flygym) or the deeperfly skeleton
(`src/deeperfly/data/default_config.toml`) changes:

```sh
uv run --with flygym --with mujoco --python 3.12 \
    python scripts/build_keypoint_viewer_assets.py
```

`flygym` and `mujoco` are heavy and intentionally kept out of the project's
dependency groups; the `uv run --with` invocation installs them into a throwaway
environment. Review the regenerated `docs/keypoints/assets/keypoints.json` —
especially points listed under `approximate` (antennae, abdomen markers), which
have no exact NeuroMechFly counterpart — before committing.

The vendored runtime libraries under `docs/keypoints/vendor/` (the official
`@mujoco/mujoco` WebAssembly build and Three.js) are committed separately and do
not need regenerating unless you bump their versions.

[flygym]: https://github.com/NeLy-EPFL/flygym
