# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the video decode fixtures.

Most clips are ten 256x256 frames at 10 fps (large enough for NVDEC's minimum
sizes); frame i is one solid colour, `COLORS[i]`, so a decoder can be checked
frame by frame (order, sampling and YUV->RGB conversion). Colour conversion is
FFmpeg's default for untagged video (BT.601, limited range).

Clips:
* h264.mp4, hevc.mp4, vp9.webm, vp9.mp4 and av1.mp4 decode through NVDEC;
  vp8.webm exists to show that VP8 is refused;
* vp9_sar.mp4 is vp9.mp4 with 2:1 pixels: its track header says 512x256 (the
  display size) while the sample entry and bitstream are 256x256;
* vp9_texture.webm is ten frames of a texture that moves one pixel per frame,
  so each frame is predicted from the previous one; vp9_invisible.webm is the
  same clip with its third block flagged invisible (the decoder must still
  decode it, or the frames after it are predicted from the wrong reference);
* vp9_huge.webm is vp9.webm whose first keyframe header claims 16384x16384
  inside a 256x256 container; av1_small_container.mp4 is av1.mp4 whose sample
  entry claims 128x128 (both must be refused before a decoder allocates);
* vp9_offset.webm is vp9.webm with its first frame at 5 s (the decoder
  reports time from the first frame, as for MP4);
* vp9_709.webm and h264_709.mp4 are two frames of `COLOR_709` encoded with the
  BT.709 matrix (limited range) and tagged BT.709, which the decoder must
  honour (read as BT.601 the colour shifts visibly);
* vp9_odd.webm is one 130x129 frame of the quadrants (NVDEC must decode an
  odd height one row taller and crop it); vp9_odd.rgb is its exact RGB,
  computed like h264_quad.rgb;
* vp9_altref.webm is twenty 128x128 frames of a moving texture, encoded in
  two passes with alternate reference frames: hidden frames packed into
  superframes with a shown one. Frame i has its left 32 columns at grey level
  `altref_grey(i)`, so each output frame can be matched to its timestamp;
* av1_odd.mp4 is one 130x129 AV1 frame, which NVDEC resamples rather than
  crops (it must be refused);
* h264_quad.mp4 is one 96x64 frame of four coloured quadrants (`QUADRANTS`);
  h264_quad.rgb is its exact RGB: a software decoder's YUV planes converted
  with BT.601 limited range and nearest chroma, the conversion the crate
  implements (H.264 decoding is bit-exact, so NVDEC must match it).

Requires PyAV (tested with av 19.0.1, whose wheels bundle the encoders),
numpy, and for vp9_altref.webm the ffmpeg CLI with libvpx (tested with 6.1.1):

    python multimodal/tests/fixtures/video_decode/generate.py
"""

import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

SIZE = 256
FPS = 10
COLORS = [(40 + 16 * i, 200 - 12 * i, 90 + (i % 3) * 40) for i in range(10)]
# Top-left, top-right, bottom-left, bottom-right.
QUADRANTS = [(220, 40, 40), (40, 200, 60), (50, 60, 220), (230, 220, 40)]
QUAD_W, QUAD_H = 96, 64
COLOR_709 = (220, 40, 200)

CLIPS = [
    ("vp8.webm", "webm", "libvpx", {"crf": "4", "b": "1M"}),
    ("vp9.webm", "webm", "libvpx-vp9", {"crf": "4", "b": "0"}),
    ("vp9.mp4", "mp4", "libvpx-vp9", {"crf": "4", "b": "0"}),
    ("av1.mp4", "mp4", "libsvtav1", {"crf": "10"}),
    ("h264.mp4", "mp4", "libx264", {"crf": "10"}),
    ("hevc.mp4", "mp4", "libx265", {"crf": "10", "x265-params": "log-level=error"}),
]


def solid_frames():
    return [np.full((SIZE, SIZE, 3), color, dtype=np.uint8) for color in COLORS]


def texture_frames():
    # A smooth, deterministic texture; frame i is the window shifted by i px.
    y, x = np.mgrid[0 : SIZE + 16, 0 : SIZE + 16]
    base = np.stack(
        [
            128 + 100 * np.sin(x / 3.0) * np.cos(y / 5.0),
            128 + 100 * np.sin((x + y) / 4.0),
            128 + 100 * np.cos(x / 6.0 - y / 2.0),
        ],
        axis=-1,
    ).astype(np.uint8)
    return [base[i : i + SIZE, i : i + SIZE].copy() for i in range(10)]


def quadrant_frame(width=QUAD_W, height=QUAD_H):
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    top, left = slice(0, height // 2), slice(0, width // 2)
    bottom, right = slice(height // 2, height), slice(width // 2, width)
    for (rows, cols), color in zip(
        [(top, left), (top, right), (bottom, left), (bottom, right)], QUADRANTS
    ):
        rgb[rows, cols] = color
    return [rgb]


def write(
    path: Path, fmt: str, codec: str, options: dict, frames=None, sar=None, start=0
) -> None:
    frames = solid_frames() if frames is None else frames
    height, width = frames[0].shape[:2]
    with av.open(str(path), "w", format=fmt) as out:
        stream = out.add_stream(codec, rate=FPS, options=options)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, FPS)
        if sar is not None:
            stream.codec_context.sample_aspect_ratio = sar
        for i, rgb in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = start + i
            frame.time_base = Fraction(1, FPS)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def altref_grey(i: int) -> int:
    return 20 + 10 * i


def write_altref(path: Path, size=128, count=20) -> None:
    # libvpx only places alternate reference frames in two-pass mode, which
    # PyAV does not drive, so this clip comes from the ffmpeg CLI.
    y, x = np.mgrid[0 : size + 2 * count, 0 : size + 3 * count]
    base = np.stack(
        [
            128 + 100 * np.sin(x / 3.0) * np.cos(y / 5.0),
            128 + 100 * np.sin((x + y) / 4.0),
            128 + 100 * np.cos(x / 6.0 - y / 2.0),
        ],
        axis=-1,
    ).astype(np.uint8)
    raw = b""
    for i in range(count):
        rgb = base[2 * i : 2 * i + size, 3 * i : 3 * i + size].copy()
        rgb[:, :32] = altref_grey(i)
        raw += rgb.tobytes()
    with tempfile.TemporaryDirectory() as tmp:
        for npass, target in ((1, "/dev/null"), (2, str(path))):
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{size}x{size}", "-r", str(FPS), "-i", "-",
                 "-c:v", "libvpx-vp9", "-b:v", "30k", "-auto-alt-ref", "1",
                 "-lag-in-frames", "25", "-pass", str(npass),
                 "-passlogfile", str(Path(tmp) / "pass"),
                 "-pix_fmt", "yuv420p", "-f", "webm", target],
                input=raw,
                check=True,
            )
    with av.open(str(path)) as container:
        packets = [bytes(p) for p in container.demux(video=0) if p.size]
    assert any(p[-1] & 0xE0 == 0xC0 for p in packets), "no superframes"


def read_vint(data: bytes, pos: int, keep_marker: bool) -> tuple[int, int]:
    """An EBML variable-length integer at `pos`: (value, length)."""
    first = data[pos]
    length = 8 - first.bit_length() + 1
    value = first if keep_marker else first & (0xFF >> length)
    for b in data[pos + 1 : pos + length]:
        value = (value << 8) | b
    return value, length


def simple_blocks(data: bytes, start: int, end: int):
    """Offsets of SimpleBlock payloads under Segment > Cluster, in order."""
    pos = start
    while pos < end:
        element, n = read_vint(data, pos, True)
        size, m = read_vint(data, pos + n, False)
        body = pos + n + m
        if element in (0x18538067, 0x1F43B675):  # Segment, Cluster: descend
            yield from simple_blocks(data, body, min(body + size, end))
        elif element == 0xA3:  # SimpleBlock
            yield body
        pos = body + size


def mark_invisible(src: Path, dst: Path, index: int) -> None:
    data = bytearray(src.read_bytes())
    block = list(simple_blocks(bytes(data), 0, len(data)))[index]
    # Payload: track number (1-byte vint for track 1), int16 timecode, flags.
    assert data[block] == 0x81, "expected track 1"
    data[block + 3] |= 0x08  # the invisible flag
    dst.write_bytes(bytes(data))


def patch_tkhd_size(path: Path, width: int, height: int) -> None:
    data = bytearray(path.read_bytes())
    at = data.index(b"tkhd")
    assert data[at + 4] == 0, "expected a version 0 tkhd"
    # After the fourcc: version/flags, creation, modification, track id,
    # reserved, duration, 8 reserved, layer, group, volume, reserved, matrix,
    # then 16.16 fixed-point width and height.
    data[at + 80 : at + 84] = (width << 16).to_bytes(4, "big")
    data[at + 84 : at + 88] = (height << 16).to_bytes(4, "big")
    path.write_bytes(bytes(data))


def patch_av01_size(src: Path, dst: Path, width: int, height: int) -> None:
    data = bytearray(src.read_bytes())
    at = data.index(b"av01", data.index(b"stsd"))
    # VisualSampleEntry after the fourcc: 6 reserved, data reference index,
    # pre_defined, reserved, 12 pre_defined, then 16-bit width and height.
    data[at + 28 : at + 30] = width.to_bytes(2, "big")
    data[at + 30 : at + 32] = height.to_bytes(2, "big")
    dst.write_bytes(bytes(data))


def patch_vp9_keyframe_size(src: Path, dst: Path, width: int, height: int) -> None:
    data = bytearray(src.read_bytes())
    frame = list(simple_blocks(bytes(data), 0, len(data)))[0] + 4
    # VP9 profile 0 keyframe: frame marker, profile, show_existing_frame,
    # frame type, show frame, error resilience (8 bits), then the sync code.
    assert data[frame] & 0xF4 == 0x80 and data[frame] & 0x02, "not a profile 0 keyframe"
    assert data[frame + 1 : frame + 4] == b"\x49\x83\x42", "no VP9 sync code"
    # Then colour space (3 bits) and range (1 bit), then width - 1 and
    # height - 1 as 16-bit fields: bits 36..52 and 52..68 of the frame.
    bits = int.from_bytes(data[frame : frame + 9], "big")
    total = 9 * 8
    for offset, value in ((36, width - 1), (52, height - 1)):
        shift = total - offset - 16
        bits = (bits & ~(0xFFFF << shift)) | (value << shift)
    data[frame : frame + 9] = bits.to_bytes(9, "big")
    dst.write_bytes(bytes(data))


def reference_rgb(path: Path) -> bytes:
    """BT.601 limited-range RGB of the first frame, nearest chroma."""
    with av.open(str(path)) as container:
        frame = next(container.decode(video=0))
    planes = []
    for plane in frame.planes:
        rows = np.frombuffer(bytes(plane), np.uint8).reshape(-1, plane.line_size)
        planes.append(rows[: plane.height, : plane.width].astype(np.float64))
    y, u, v = planes
    h, w = y.shape
    u = u[np.arange(h) // 2][:, np.arange(w) // 2]
    v = v[np.arange(h) // 2][:, np.arange(w) // 2]
    kr, kb = 0.299, 0.114
    kg = 1 - kr - kb
    yy = (y - 16) * 255 / 219
    cb = (u - 128) * 255 / 224
    cr = (v - 128) * 255 / 224
    r = yy + 2 * (1 - kr) * cr
    b = yy + 2 * (1 - kb) * cb
    g = yy - (2 * kb * (1 - kb) * cb + 2 * kr * (1 - kr) * cr) / kg
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8).tobytes()


def ycbcr(rgb, kr: float, kb: float):
    """Limited-range Y'CbCr of an 8-bit RGB triple for a matrix (Kr, Kb)."""
    r, g, b = (c / 255 for c in rgb)
    y = kr * r + (1 - kr - kb) * g + kb * b
    return (
        16 + 219 * y,
        128 + 224 * (b - y) / (2 * (1 - kb)),
        128 + 224 * (r - y) / (2 * (1 - kr)),
    )


def write_bt709(path: Path, fmt: str, codec: str, options: dict) -> None:
    y, cb, cr = (round(v) for v in ycbcr(COLOR_709, 0.2126, 0.0722))
    # The same samples read with BT.601 must land well away from the colour.
    kr, kb = 0.299, 0.114
    yy, u, v = (y - 16) * 255 / 219, (cb - 128) * 255 / 224, (cr - 128) * 255 / 224
    as_601 = (yy + 2 * (1 - kr) * v, yy - (2 * kb * (1 - kb) * u + 2 * kr * (1 - kr) * v) / (1 - kr - kb), yy + 2 * (1 - kb) * u)
    assert max(abs(a - b) for a, b in zip(as_601, COLOR_709)) > 15, as_601
    with av.open(str(path), "w", format=fmt) as out:
        stream = out.add_stream(codec, rate=FPS, options=options)
        stream.width = stream.height = SIZE
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, FPS)
        for attr in ("colorspace", "color_primaries", "color_trc"):
            setattr(stream.codec_context, attr, 1)  # BT.709
        stream.codec_context.color_range = 1  # limited ("MPEG")
        for i in range(2):
            frame = av.VideoFrame(SIZE, SIZE, "yuv420p")
            for plane, value in zip(frame.planes, (y, cb, cr)):
                plane.update(bytes([value]) * (plane.line_size * plane.height))
            frame.colorspace, frame.color_range = 1, 1
            frame.pts = i
            frame.time_base = Fraction(1, FPS)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def main() -> None:
    here = Path(__file__).parent
    for name, fmt, codec, options in CLIPS:
        write(here / name, fmt, codec, options)
    write(here / "vp9_sar.mp4", "mp4", "libvpx-vp9", CLIPS[2][3], sar=Fraction(2, 1))
    patch_tkhd_size(here / "vp9_sar.mp4", 2 * SIZE, SIZE)
    write(here / "h264_quad.mp4", "mp4", "libx264", CLIPS[4][3], frames=quadrant_frame())
    write(here / "vp9_texture.webm", "webm", "libvpx-vp9", {"crf": "30", "b": "0"}, frames=texture_frames())
    mark_invisible(here / "vp9_texture.webm", here / "vp9_invisible.webm", 2)
    patch_vp9_keyframe_size(here / "vp9.webm", here / "vp9_huge.webm", 16384, 16384)
    patch_av01_size(here / "av1.mp4", here / "av1_small_container.mp4", 128, 128)
    write(here / "vp9_offset.webm", "webm", "libvpx-vp9", CLIPS[1][3], start=5 * FPS)
    write_bt709(here / "vp9_709.webm", "webm", "libvpx-vp9", CLIPS[1][3])
    write_bt709(here / "h264_709.mp4", "mp4", "libx264", CLIPS[4][3])
    (here / "h264_quad.rgb").write_bytes(reference_rgb(here / "h264_quad.mp4"))
    write(here / "vp9_odd.webm", "webm", "libvpx-vp9", CLIPS[1][3], frames=quadrant_frame(130, 129))
    (here / "vp9_odd.rgb").write_bytes(reference_rgb(here / "vp9_odd.webm"))
    write_altref(here / "vp9_altref.webm")
    write(here / "av1_odd.mp4", "mp4", "libsvtav1", CLIPS[3][3], frames=quadrant_frame(130, 129))
    for path in sorted(here.glob("*.*")):
        if path.suffix in (".mp4", ".webm"):
            print(f"{path.name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
