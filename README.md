# deeperfly

JAX-based multi-view geometry and bundle adjustment for multi-camera rigs.

The computer-vision primitives follow OpenCV's conventions (Rodrigues
rotations, `projectPoints` distortion model, DLT triangulation) and are
cross-checked against OpenCV in the test suite. Everything is written in JAX so
projection and its Jacobian are JIT- and autodiff-friendly.

## Layout

| Path | Contents |
| --- | --- |
| `src/deeperfly/geometry.py` | Projection, distortion, triangulation, Rodrigues (JAX). |
| `src/deeperfly/cameras.py` | `Camera` / `CameraGroup` and the TOML config loader. |
| `src/deeperfly/bundle_adjustment/` | Packed state, fixed/shared grammar, and the SciPy (TRF + LSMR) solver. |
| `examples/` | Worked example notebook and a sample rig config. |
| `tests/` | Pytest suite, including OpenCV cross-checks. |
| `dev/` | Benchmarks. |

## Usage

```python
from deeperfly import CameraGroup, bundle_adjust

group = CameraGroup.from_config("examples/cameras.toml")
pts2d = group.project(pts3d)                      # (V, N, 2) observations
result, optimized, points = bundle_adjust(group, pts2d, fixed=["*.intr"])
```

See [`examples/bundle_adjustment.ipynb`](examples/bundle_adjustment.ipynb) for an
end-to-end walkthrough (load rig → synthesize observations → perturb →
bundle-adjust → align and compare).

## Development

```bash
uv sync --group test    # install with test dependencies
uv run pytest           # run the suite
```
