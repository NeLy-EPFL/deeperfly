"""Several video files read as one stream.

A camera's ``video`` pattern may match more than one file -- the acquisition splits a
long recording at a byte threshold -- and everything one pattern matches is ONE stream.
:class:`ConcatReader` is that stream: the parts laid end to end, with a global frame index
that runs straight through the boundaries.

This is the one part of the v2 schema change that is not a surface change, and its failure
mode is silent: get a boundary wrong and the run succeeds while every keypoint after it
belongs to a different frame. Two things follow from that.

**Exact lengths are the hard requirement.** A global index is undefined unless every part
but the last has an EXACT frame count. ``VideoReader.count()`` returns ``None`` when the
container header carries no ``nb_frames``, so such a part is counted once by a full decode
-- logged with what it cost -- and never estimated from ``duration * fps``: an off-by-one
there shifts every frame index after the boundary, and nothing downstream can detect it.

**The parts must agree.** A concatenated stream with two frame sizes has no single
intrinsics and a mixed frame rate has no single timebase, so either is an error naming the
two files rather than a stream that reads plausibly.
"""

from __future__ import annotations

import bisect
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
from jaxtyping import Float

from .base import FrameCursor, FrameReader
from .video import VideoReader

log = logging.getLogger("deeperfly")


class ConcatReader(FrameReader):
    """The parts of one camera's footage, read as a single stream.

    Attributes
    ----------
    paths
        The parts, in decode order.
    """

    def __init__(self, paths, *, counts=None) -> None:
        self.paths = [Path(p) for p in paths]
        if len(self.paths) < 2:
            raise ValueError(
                "ConcatReader is for two or more parts; open_reader returns a plain "
                "VideoReader for one"
            )
        self._readers = [VideoReader(p) for p in self.paths]
        self._counts = (
            [int(c) for c in counts] if counts is not None else self._measure()
        )
        # Exclusive prefix sums: `_starts[i]` is the global index of part i's frame 0.
        self._starts: list[int] = []
        total = 0
        for n in self._counts:
            self._starts.append(total)
            total += n
        self._total = total
        self._check_agreement()

    # -- construction helpers ------------------------------------------------

    def _measure(self) -> list[int]:
        """Each part's exact frame count, decoding a part whose header will not say.

        The last part's count may come from the header too -- it is the one part whose
        length no other index depends on -- but it is measured the same way, because the
        total is what ``count()`` reports and a wrong total truncates a run.
        """
        out: list[int] = []
        for reader in self._readers:
            n = reader.count()
            if n is None:
                import time

                started = time.perf_counter()
                n = sum(1 for _ in reader.stream_frames(gray_ok=True))
                log.warning(
                    "%s carries no frame count in its container header, so it was "
                    "counted by a full decode (%d frames in %.1f s). A concatenated "
                    "stream needs exact lengths -- an estimate would shift every frame "
                    "index after this part.",
                    reader.path.name,
                    n,
                    time.perf_counter() - started,
                )
            out.append(int(n))
        return out

    def _check_agreement(self) -> None:
        """Refuse parts that disagree on frame size or frame rate."""
        sizes: dict[tuple[int, int], Path] = {}
        rates: dict[float, Path] = {}
        for reader, path in zip(self._readers, self.paths):
            frame = reader[0]
            size = (int(frame.shape[0]), int(frame.shape[1]))
            if size not in sizes and sizes:
                other = next(iter(sizes.items()))
                raise ValueError(
                    f"{path.name} is {size[1]}x{size[0]} but {other[1].name} is "
                    f"{other[0][1]}x{other[0][0]}; the parts of one camera's stream have "
                    "to agree on frame size -- a concatenated stream with two "
                    "resolutions has no single intrinsics"
                )
            sizes[size] = path
            rate = reader.fps()
            if rate is None:
                continue
            rate = round(float(rate), 3)
            if rate not in rates and rates:
                other_rate, other_path = next(iter(rates.items()))
                raise ValueError(
                    f"{path.name} is {rate} fps but {other_path.name} is {other_rate} "
                    "fps; the parts of one camera's stream have to agree on frame rate"
                )
            rates[rate] = path

    # -- index mapping -------------------------------------------------------

    def _locate(self, idx: int) -> tuple[int, int]:
        """A global frame index -> ``(part, local index)``."""
        if idx < 0:
            idx += self._total
        if not 0 <= idx < self._total:
            raise IndexError(
                f"frame {idx} is outside this {self._total}-frame stream "
                f"({len(self.paths)} parts)"
            )
        part = bisect.bisect_right(self._starts, idx) - 1
        return part, idx - self._starts[part]

    def _indices(self, key) -> list[int]:
        if isinstance(key, slice):
            return list(range(*key.indices(self._total)))
        return [int(i) for i in key]

    # -- FrameReader ---------------------------------------------------------

    def __getitem__(self, key) -> Float[np.ndarray, "..."]:
        if isinstance(key, (int, np.integer)):
            part, local = self._locate(int(key))
            return self._readers[part][local]
        wanted = self._indices(key)
        if not wanted:
            return np.empty((0,), dtype=np.uint8)
        # Grouped per part and decoded in one pass each, then put back in the caller's
        # order -- so a slice spanning a boundary costs two forward decodes, not one
        # re-open per frame.
        per_part: dict[int, list[int]] = {}
        for i in wanted:
            part, local = self._locate(i)
            per_part.setdefault(part, []).append(local)
        decoded: dict[int, np.ndarray] = {}
        for part, locals_ in per_part.items():
            block = self._readers[part][locals_]
            for local, frame in zip(locals_, block):
                decoded[self._starts[part] + local] = frame
        return np.stack([decoded[i] for i in wanted])

    def stream_frames(
        self,
        *,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
        gray_ok: bool = False,
        thread_count: int | None = None,
    ) -> Iterator[Float[np.ndarray, "H W 3"]]:
        """Walk the parts in order, forward, never seeking across a boundary."""
        stop = self._total if stop is None else min(stop, self._total)
        wanted = range(start, stop, step)
        for part, reader in enumerate(self._readers):
            base, n = self._starts[part], self._counts[part]
            first = next((i for i in wanted if base <= i < base + n), None)
            if first is None:
                continue
            local_stop = min(stop, base + n) - base
            yield from reader.stream_frames(
                start=first - base,
                stop=local_stop,
                step=step,
                gray_ok=gray_ok,
                thread_count=thread_count,
            )

    def stream_blocks(
        self,
        *,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
        block_size: int = 64,
        gray_ok: bool = False,
        thread_count: int | None = None,
    ) -> Iterator[Float[np.ndarray, "T H W 3"]]:
        """Blocks of up to ``block_size`` frames, re-grouped ACROSS part boundaries.

        Re-grouped rather than yielded per part, so a boundary never emits a short block:
        a consumer batching a detector forward pass would otherwise see one odd-sized
        batch per split, which changes nothing about correctness and everything about
        whether a batch-size assertion holds.
        """
        buffer: list[np.ndarray] = []
        for frame in self.stream_frames(
            start=start,
            stop=stop,
            step=step,
            gray_ok=gray_ok,
            thread_count=thread_count,
        ):
            buffer.append(frame)
            if len(buffer) == block_size:
                yield np.stack(buffer)
                buffer = []
        if buffer:
            yield np.stack(buffer)

    def count(self) -> int | None:
        """The exact total, the sum over parts (never ``None`` -- see :meth:`_measure`)."""
        return self._total

    def fps(self) -> float | None:
        """The parts' shared frame rate (they are checked to agree)."""
        return self._readers[0].fps()

    def cursor(
        self, *, gray_ok: bool = False, same_as_rgb: bool = False
    ) -> FrameCursor:
        """A cursor holding one underlying cursor per part, opened lazily.

        Which is what keeps the seek-not-walk and held-open-decoder work from being
        undone by a boundary: crossing one swaps which cursor answers, and neither is
        re-opened.
        """
        return ConcatCursor(self, gray_ok=gray_ok, same_as_rgb=same_as_rgb)

    def close(self) -> None:
        for reader in self._readers:
            reader.close()

    def __repr__(self) -> str:
        parts = ", ".join(p.name for p in self.paths)
        return f"ConcatReader({parts}; {self._total} frames)"


class ConcatCursor(FrameCursor):
    """One cursor per part of a :class:`ConcatReader`, opened on first use."""

    def __init__(self, reader: ConcatReader, *, gray_ok=False, same_as_rgb=False):
        super().__init__(reader)
        self._concat = reader
        self._kwargs = {"gray_ok": gray_ok, "same_as_rgb": same_as_rgb}
        self._cursors: dict[int, FrameCursor] = {}

    def frame(self, idx: int) -> np.ndarray:
        part, local = self._concat._locate(int(idx))
        cursor = self._cursors.get(part)
        if cursor is None:
            cursor = self._concat._readers[part].cursor(**self._kwargs)
            self._cursors[part] = cursor
        return cursor.frame(local)

    def close(self) -> None:
        for cursor in self._cursors.values():
            cursor.close()
        self._cursors.clear()
