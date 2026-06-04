"""Benchmark video decode vs detector inference, to size the streaming window.

Finding (RTX 4090, 7-cam 480x960): **inference is the bottleneck** -- the 8-stack
detector runs ~28 multi-camera frames/s and is compute-bound (batching frames does
not help), while every decoder is far faster (torchcodec-CUDA ~2400 fps). So decode
backend and CPU parallelism barely move total throughput, and
``[detector] chunk_frames`` is a *memory* knob, not a speed one -- keep it small to
bound VRAM.

``bench_pipeline`` settles the follow-up question -- *is GPU decode worth it?* -- by
timing the real streaming detect end to end for {cpu, cuda} x {serial, prefetch}.
Finding (RTX 4090, 7-cam 480x960): with frames uploaded once per window rather than
once per pass (see ``inference._window_to_device`` -- the per-pass version decoded
~30x slower than the forward) and a prefetch thread overlapping decode with the
forward, **CPU decode reaches ~22 mcam-fps vs ~23 for CUDA/NVDEC** -- a ~6% edge
that does not justify the torchcodec-CUDA + nvidia-npp dependency stack. Prefetch
adds ~11% to either decoder. So plain CPU decode is the sensible default.

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
    """Decode-in-windows throughput (frames/s) per backend x chunk x threads."""
    import torch

    def run_windows(read, chunk):
        for s in range(0, n, chunk):
            read(s, min(s + chunk, n))
            torch.cuda.synchronize()

    def torchcodec(chunk, threads):
        from torchcodec.decoders import VideoDecoder

        from deeperfly.video.backends.torchcodec_io import _preload_npp

        _preload_npp()

        def read(s, e):
            return VideoDecoder(path, device="cuda", num_ffmpeg_threads=threads)[s:e]

        return n / _timeit(lambda: run_windows(read, chunk))

    print(f"\n{'backend':12s} {'chunk':>5s} {'thr':>3s} {'frames/s':>9s}")
    with contextlib.redirect_stderr(io.StringIO()):
        for name, fnb, threads in (("torchcodec", torchcodec, (1,)),):
            for chunk in (64, 256):
                for thr in threads:
                    try:
                        print(f"{name:12s} {chunk:5d} {thr:3d} {fnb(chunk, thr):9.0f}")
                    except Exception as e:  # noqa: BLE001
                        print(f"{name:12s} {chunk:5d} {thr:3d}  err {str(e)[:40]}")


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
    frames = [
        video.read_video(p, backend="auto", device="cuda", start=0, stop=t)
        for p in paths
    ]
    inference.detect_sequence(model, [f[:4] for f in frames], sides, flips)  # warmup
    dt = _timeit(
        lambda: np.asarray(inference.detect_sequence(model, frames, sides, flips)[0]),
        reps=1,
    )
    print(
        f"\ndetect_sequence: {t / dt:.0f} multi-cam frames/s  (the pipeline's real ceiling)"
    )


def bench_pipeline(path: str, t: int, chunk: int = 64) -> None:
    """End-to-end streaming detect throughput: decode device x prefetch overlap.

    For each ``{cpu, cuda}`` decode device, times the serial loop (decode a window,
    then forward it) against the prefetched loop (decode the next window in a
    background thread while the GPU forwards the current one). The forward is
    batched to the GPU via :func:`auto_batch_size`, exactly as the CLI runs it.
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

    def serial(device):
        done = 0
        while done < t:
            window = [
                video.read_frames(
                    p, device=device, start=done, stop=min(done + chunk, t)
                )
                for p in paths
            ]
            n = len(window[0])
            if n == 0:
                break
            inference.detect_sequence(model, window, sides, flips, batch_size=bs)
            done += n
        return done

    def prefetch(device):
        done = 0
        for window, n in _prefetch_windows(
            paths, backend="auto", device=device, chunk=chunk
        ):
            inference.detect_sequence(model, window, sides, flips, batch_size=bs)
            done += n
            if done >= t:
                break
        return done

    print(
        f"\n{'decode':6s} {'mode':9s} {'mcam fps':>9s}  (chunk={chunk}, fwd batch={bs})"
    )
    for device in ("cuda", "cpu"):
        for name, fn in (("serial", serial), ("prefetch", prefetch)):
            try:
                fn(device)  # warmup: JIT compile + decoder init
                t0 = time.perf_counter()
                done = fn(device)
                dt = time.perf_counter() - t0
                print(f"{device:6s} {name:9s} {done / dt:9.1f}")
            except Exception as e:  # noqa: BLE001
                print(f"{device:6s} {name:9s}  err {str(e)[:48]}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/videos/camera_0.mp4"
    bench_decode(src, n=900)
    bench_inference(src, t=128)
    bench_pipeline(src, t=128)
