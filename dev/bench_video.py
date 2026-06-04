"""Benchmark video decode vs detector inference, to size the streaming window.

Finding (RTX 4090, 7-cam 480x960): **inference is the bottleneck** -- the 8-stack
detector runs ~28 multi-camera frames/s and is compute-bound (batching frames does
not help), while every (CPU) decoder is far faster. So the decode backend and CPU
parallelism barely move total throughput, and ``[detector] chunk_frames`` is a
*memory* knob, not a speed one -- keep it small to bound VRAM.

This is also why deeperfly decodes on the CPU only: an earlier sweep over GPU/NVDEC
decode (torchcodec-CUDA + nvidia-npp) reached ~23 mcam-fps end to end vs ~22 for
plain CPU decode -- a ~6% edge that did not justify the CUDA-video dependency stack,
so it was dropped. ``bench_pipeline`` times the real streaming detect (serial vs a
prefetch thread overlapping CPU decode with the GPU forward).

Run::

    uv run --no-sync python dev/bench_video.py data/videos/camera_0.mp4
"""

from __future__ import annotations

import contextlib
import io
import sys
import time


def _timeit(fn, reps=2):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def bench_decode(path: str, n: int) -> None:
    """Decode-in-windows throughput (frames/s) per (CPU) backend x chunk."""
    from deeperfly import video

    def run_windows(backend, chunk):
        for s in range(0, n, chunk):
            video.read_video(path, backend=backend, start=s, stop=min(s + chunk, n))

    print(f"\n{'backend':16s} {'chunk':>5s} {'frames/s':>9s}")
    with contextlib.redirect_stderr(io.StringIO()):
        for backend in video.available_read_backends():
            for chunk in (64, 256):
                try:
                    fps = n / _timeit(lambda: run_windows(backend, chunk))
                    print(f"{backend:16s} {chunk:5d} {fps:9.0f}")
                except Exception as e:  # noqa: BLE001
                    print(f"{backend:16s} {chunk:5d}  err {str(e)[:40]}")


def bench_inference(path: str, t: int) -> None:
    """Detector throughput (multi-cam frames/s) and forward-pass scaling with batch."""
    import numpy as np

    from deeperfly import video
    from deeperfly.pose2d import backends, inference
    from deeperfly.pose2d.download import download_torch_weights

    model = backends.load_detector(download_torch_weights())

    print(f"\n{'fwd batch':>9s} {'img/s':>7s}")
    for b in (8, 32, 64):
        x = np.zeros((b, 3, 256, 512), np.float32)
        np.asarray(backends.predict_heatmaps(model, x))  # compile
        fps = (
            b
            * 3
            / _timeit(lambda: np.asarray(backends.predict_heatmaps(model, x)), reps=1)
            / 3
        )
        print(f"{b:9d} {fps:7.0f}")

    sides, flips = inference.fly_camera_layout(
        ["rh", "rm", "rf", "f", "lf", "lm", "lh"]
    )
    paths = [path.replace("_0", f"_{i}") for i in range(7)]
    frames = [video.read_video(p, backend="auto", start=0, stop=t) for p in paths]
    inference.detect_sequence(model, [f[:4] for f in frames], sides, flips)  # warmup
    dt = _timeit(
        lambda: np.asarray(inference.detect_sequence(model, frames, sides, flips)[0]),
        reps=1,
    )
    print(
        f"\ndetect_sequence: {t / dt:.0f} multi-cam frames/s  (the pipeline's real ceiling)"
    )


def bench_pipeline(path: str, t: int, chunk: int = 64) -> None:
    """End-to-end streaming detect throughput: serial vs prefetch overlap.

    Times the serial loop (decode a window, then forward it) against the prefetched
    loop (decode the next window on the CPU in a background thread while the GPU
    forwards the current one). The forward is batched to the GPU via
    :func:`auto_batch_size`, exactly as the CLI runs it.
    """
    import time

    from deeperfly import video
    from deeperfly.cli import _prefetch_windows
    from deeperfly.pose2d import backends, inference
    from deeperfly.pose2d.download import download_torch_weights

    model = backends.load_detector(download_torch_weights())
    sides, flips = inference.fly_camera_layout(
        ["rh", "rm", "rf", "f", "lf", "lm", "lh"]
    )
    paths = [path.replace("_0", f"_{i}") for i in range(7)]
    bs = backends.auto_batch_size(inference.IMG_SIZE)

    def serial():
        done = 0
        while done < t:
            window = [
                video.read_frames(p, start=done, stop=min(done + chunk, t))
                for p in paths
            ]
            n = len(window[0])
            if n == 0:
                break
            inference.detect_sequence(model, window, sides, flips, batch_size=bs)
            done += n
        return done

    def prefetch():
        done = 0
        for window, n in _prefetch_windows(paths, backend="auto", chunk=chunk):
            inference.detect_sequence(model, window, sides, flips, batch_size=bs)
            done += n
            if done >= t:
                break
        return done

    print(f"\n{'mode':9s} {'mcam fps':>9s}  (chunk={chunk}, fwd batch={bs})")
    for name, fn in (("serial", serial), ("prefetch", prefetch)):
        try:
            fn()  # warmup: JIT compile + decoder init
            t0 = time.perf_counter()
            done = fn()
            dt = time.perf_counter() - t0
            print(f"{name:9s} {done / dt:9.1f}")
        except Exception as e:  # noqa: BLE001
            print(f"{name:9s}  err {str(e)[:48]}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/videos/camera_0.mp4"
    bench_decode(src, n=900)
    bench_inference(src, t=128)
    bench_pipeline(src, t=128)
