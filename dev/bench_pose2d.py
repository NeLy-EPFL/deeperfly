"""Benchmark the JAX (Equinox) hourglass detector vs the PyTorch original.

This is the decision gate from the implementation plan: keep the JAX detector if
it is at least as fast as PyTorch on the target GPU; otherwise wrap PyTorch
behind the same ``predict_heatmaps`` / ``heatmap_to_points`` interface. Run on
the GPU you intend to deploy on:

    uv run python dev/bench_pose2d.py --batch 7 --frames 8

Reports images/second for both backends at the given batch shape (the JAX side
is timed after JIT warmup; the PyTorch side under inference_mode).
"""

from __future__ import annotations

import argparse
import time

import jax
import numpy as np


def time_it(fn, *, warmup=2, repeat=10):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def bench_jax(n_images, h=256, w=512):
    import equinox as eqx
    from deeperfly.pose2d.backends.jax import HourglassNet, predict_heatmaps

    model = HourglassNet.deepfly2d(key=jax.random.PRNGKey(0))
    x = jax.random.normal(
        jax.random.PRNGKey(1), (n_images, 3, h, w), dtype=jax.numpy.float32
    )

    @eqx.filter_jit
    def run(inputs):
        return predict_heatmaps(model, inputs)

    def call():
        jax.block_until_ready(run(x))

    dt = time_it(call)
    print(
        f"JAX    : {dt * 1e3:8.2f} ms / batch   {n_images / dt:8.1f} img/s   ({jax.devices()[0]})"
    )
    return dt


def bench_torch(n_images, h=256, w=512):
    try:
        import torch
    except ImportError:
        print(
            "torch not installed; skipping PyTorch benchmark (install the 'torch' extra)"
        )
        return None
    from deeperfly.pose2d.backends import torch as torch_backend

    device = torch_backend.device()
    model = torch_backend.HourglassNet().eval().to(device)
    x = torch.randn(n_images, 3, h, w, device=device)

    @torch.inference_mode()
    def call():
        out = model(x)[-1]
        if device == "cuda":
            torch.cuda.synchronize()
        return out

    dt = time_it(call)
    print(
        f"PyTorch: {dt * 1e3:8.2f} ms / batch   {n_images / dt:8.1f} img/s   ({device})"
    )
    return dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=7, help="cameras per frame")
    parser.add_argument("--frames", type=int, default=1, help="frames per batch")
    args = parser.parse_args()
    n = args.batch * args.frames
    print(f"batch = {n} images (256x512)\n")
    dt_jax = bench_jax(n)
    dt_torch = bench_torch(n)
    if dt_torch is not None:
        ratio = dt_torch / dt_jax
        verdict = "KEEP JAX" if ratio >= 0.9 else "consider PyTorch backend"
        print(f"\nJAX is {ratio:.2f}x PyTorch's speed  ->  {verdict}")


if __name__ == "__main__":
    main()
