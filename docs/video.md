# Frame I/O

`deeperfly.io` reads and writes video files with **PyAV** — in-process libx264,
with FFmpeg bundled in the wheel (no system FFmpeg needed). All decoding and
encoding runs on the CPU and yields `(T, H, W, 3)` uint8 RGB NumPy.

Footage is read through a small reader hierarchy: `open_reader(source)` resolves a
source to a `VideoReader` (a video file) or an `ImageSequenceReader` (a directory,
glob, or explicit file list), both subclasses of `FrameReader`. You then index
(`reader[:]`, `reader[i]`, `reader[[0, 3, 5]]`), stream (`stream_frames` /
`stream_blocks`), or probe metadata (`count` / `fps`) against the returned reader.

Image *sequences* (a directory or glob of PNG/JPG/…) are decoded by OpenCV, in
parallel across threads (JPEG/PNG decoders release the GIL).

```python
from deeperfly import io

reader = io.open_reader(path)                 # video file or image dir/glob/list
frames = reader[:]                            # (T, H, W, 3) uint8 NumPy (host)
clip = io.VideoReader("clip.mp4")[[0, 50]]    # random access (seeks per frame)
for block in io.open_reader(path).stream_blocks(block_size=64):  # forward
    ...

# VideoWriter encodes a frame, a batch, or any iterable -- so a long clip can be
# written as it is produced, without ever holding every frame in memory.
with io.VideoWriter("out.mp4", fps=30) as writer:
    writer.write_frames(frames)
```

`deeperfly run` decodes on the CPU and uploads each window to the detector device
in one shot — decode is not the bottleneck, the detector forward is. The only
frame-I/O configuration is the image-decode thread count (`[io.image] workers`),
applied across every stage. See the config comments and `deeperfly.io` docstrings
for details.
