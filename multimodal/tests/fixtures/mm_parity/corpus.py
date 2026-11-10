# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# `gradient`, `encode` and the images that keep SGLang's corpus names are adapted
# from sgl-project/sglang `rust/sglang-processor/tests/scripts/generate_mm_parity.py`
# (Apache-2.0).
"""Build the shared image corpus `images/` and the decode stage table `decode.json`.

Usage:
    python corpus.py [--check]

`CORPUS` builds every image except the files in `COMMITTED`, which no Pillow
option writes (see README.md). `decode.json` holds, for each image and each
decode entry point of the crate, the output of Pillow in `COLUMNS`, and the
cases where the crate differs: `REJECTED` (both refuse), `KNOWN_GAPS` (the
crate refuses what Pillow decodes) and `DIVERGES` (the crate decodes other
pixels). `resize_rgba` rows hold Pillow's RGBA resize. The script never runs
the crate.

`--check` writes nothing and fails when a committed file differs.

Needs Pillow 12.3, numpy, and SGLang for `smart_to_rgb`.
"""

import argparse
import hashlib
import io
import json
import re
import struct
import zlib
from pathlib import Path

import numpy as np
import PIL
import sglang
from PIL import Image, ImageOps, PngImagePlugin
from sglang.srt.multimodal.media_io import smart_to_rgb

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"


def gradient(width, height, seed=0, channels=3, noise=24):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = np.stack(
        [
            x * 255 / max(width - 1, 1),
            y * 255 / max(height - 1, 1),
            (x + y) % 256,
            x % 200,
        ],
        axis=-1,
    )[..., :channels]
    return np.clip(base + rng.integers(0, noise + 1, base.shape), 0, 255).astype(
        np.uint8
    )


def draw(kind, width, height, seed):
    """Integer noise, with fully transparent and fully opaque cells in the alpha band."""
    bands = {"rgb": 3, "rgba": 4, "la": 2}[kind]
    i = np.arange(height * width * bands, dtype=np.uint64)
    v = (
        ((i + np.uint64(seed) * np.uint64(0x9E3779B9)) * np.uint64(2654435761))
        & np.uint64(0xFFFFFFFF)
    ) >> np.uint64(24)
    px = v.astype(np.uint8).reshape(height, width, bands)
    if bands in (2, 4):
        y, x = np.mgrid[0:height, 0:width]
        cell = (x // 7 + y // 5) % 3
        px[..., -1] = np.where(cell == 0, 0, np.where(cell == 1, 255, px[..., -1]))
    return Image.fromarray(px)


def encode(image, fmt, **options):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **options)
    return buffer.getvalue()


def exif_rotated_jpeg():
    exif = Image.Exif()
    exif[0x0112] = 6
    return encode(Image.fromarray(gradient(64, 48, 7)), "JPEG", quality=90, exif=exif)


def animated_gif():
    frames = [Image.fromarray(gradient(48, 40, seed)) for seed in (8, 9)]
    return encode(
        frames[0], "GIF", save_all=True, append_images=frames[1:], duration=100
    )


def transparent_gif():
    image = Image.fromarray(gradient(56, 56, 18)).convert("P")
    image.paste(0, (12, 8, 44, 40))
    return encode(image, "GIF", transparency=0)


def raw_profile_exif_rgba_png():
    """Orientation in a text chunk: PIL reads it, eXIf-only readers miss it."""
    exif = Image.Exif()
    exif[0x0112] = 6
    data = exif.tobytes()
    info = PngImagePlugin.PngInfo()
    info.add_text("Raw profile type exif", f"\nexif\n{len(data):8d}\n{data.hex()}\n")
    return encode(
        Image.fromarray(gradient(56, 48, 19, channels=4)), "PNG", pnginfo=info
    )


def bitfields_alpha_bmp():
    """Pillow saves RGBA as 32-bit BI_RGB and reads that back as RGB; only an
    alpha bitmask opens as RGBA, so write the BITMAPV4HEADER by hand."""
    pixels = gradient(56, 56, 20, channels=4)
    pixels[..., 3] = 255
    pixels[8:40, 12:44, 3] = np.arange(0, 256, 8, dtype=np.uint8)
    height, width = pixels.shape[:2]
    data = pixels[::-1, :, [2, 1, 0, 3]].tobytes()
    fields = (108, width, height, 1, 32, 3, len(data), 2835, 2835, 0, 0)
    masks = (0xFF0000, 0xFF00, 0xFF, 0xFF000000)
    header = struct.pack("<IiiHHIIiiII4I", *fields, *masks).ljust(108, b"\0")
    offset = 14 + len(header)
    file_header = b"BM" + struct.pack("<IHHI", offset + len(data), 0, 0, offset)
    return file_header + header + data


def huge_palette_png():
    """182 M pixels: above PIL's decompression-bomb error threshold."""
    image = Image.new("P", (14000, 13000), 0)
    image.putpalette([200, 30, 60, 10, 220, 90])
    return encode(image, "PNG", bits=1)


def gray_png(width, height, depth, seed, trns=None):
    """A grayscale PNG below 8 bits, which Pillow cannot write with a tRNS chunk."""

    def chunk(tag, data):
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    samples = np.random.default_rng(seed).integers(0, 1 << depth, (height, width))
    bits = np.unpackbits(samples.astype(np.uint8)[..., None], axis=-1)[..., -depth:]
    rows = np.packbits(bits.reshape(height, -1), axis=-1)
    raw = b"".join(b"\x00" + row.tobytes() for row in rows)
    header = struct.pack(">IIBBBBB", width, height, depth, 0, 0, 0, 0)
    extra = b"" if trns is None else chunk(b"tRNS", struct.pack(">H", trns))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + extra
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def most_used_transparent_gif(width, height, seed):
    """Pillow drops a transparent index no pixel uses, so take the most used one."""
    image = Image.fromarray(gradient(width, height, seed)).convert("P")
    index = int(np.bincount(np.asarray(image).ravel()).argmax())
    return encode(image, "GIF", transparency=index)


def first_image(data):
    """The offset of a GIF's first image descriptor."""
    at = 13 + (3 << (data[10] & 7) + 1 if data[10] & 0x80 else 0)
    while data[at] == 0x21:
        at += 2
        while data[at]:
            at += 1 + data[at]
        at += 1
    assert data[at] == 0x2C
    return at


def local_ramp_gif(width, height, seed):
    """A first frame whose own color table is the gray ramp, beside a global
    table of colors: Pillow paints the frame with the global table."""
    gray = Image.fromarray(gradient(width, height, seed)[..., 0])
    data = encode(gray, "GIF", optimize=False)
    ramp = bytes(value for value in range(256) for _ in range(3))
    assert data[10] & 0x87 == 0x87 and data[13:781] == ramp
    at = first_image(data)
    assert not data[at + 9] & 0x80
    colors = gradient(256, 1, seed).tobytes()
    return (
        data[:13]
        + colors
        + data[781 : at + 9]
        + bytes([data[at + 9] | 0x87])
        + ramp
        + data[at + 10 :]
    )


def two_control_gif(width, height, seed):
    """A transparent index, a Plain Text Extension, then a Graphic Control
    Extension with no transparent index: Pillow keeps the first index."""
    data = most_used_transparent_gif(width, height, seed)
    at = first_image(data)
    assert data[at - 8 : at - 5] == b"!\xf9\x04" and data[at - 5] & 1
    text = b"!\x01\x0c" + struct.pack("<4H4B", 0, 0, 8, 8, 8, 8, 1, 0) + b"\x01A\x00"
    return data[:at] + text + b"!\xf9\x04" + bytes(5) + data[at:]


def newline_trns_png(width, height, seed):
    """Palette alpha `FF 00 0A`: Pillow's pattern for one transparent entry
    also matches before a final newline byte, so entry 2 stays opaque."""
    image = Image.fromarray(gradient(width, height, seed)).quantize(8)
    assert 2 in np.asarray(image)
    data = encode(image, "PNG", transparency=b"\xff\x00\x0a")
    assert b"tRNS\xff\x00\x0a" in data
    return data


def interlace_text_png(width, height, seed):
    """A text chunk named `interlace`: Pillow keeps it in `info` and reads the
    rows of this plain file in Adam7 order. At 1 pixel wide the rows keep their
    length, so Pillow decodes."""
    text = PngImagePlugin.PngInfo()
    text.add_text("interlace", "1")
    image = Image.fromarray(gradient(width, height, seed))
    return encode(image, "PNG", pnginfo=text)


def bmp(width, height, bits, rows, compression=0, masks=(), palette=b"", colors=0):
    """A BMP form Pillow does not write: a 40-byte header, or BITMAPV4HEADER
    for 4 masks."""
    header = struct.pack(
        "<IiiHHIIiiII",
        108 if len(masks) == 4 else 40,
        width,
        height,
        1,
        bits,
        compression,
        len(rows),
        2835,
        2835,
        colors,
        0,
    )
    header += struct.pack(f"<{len(masks)}I", *masks)
    if len(masks) == 4:
        header += b"BGRs" + bytes(48)
    offset = 14 + len(header) + len(palette)
    file_header = b"BM" + struct.pack("<IHHI", offset + len(rows), 0, 0, offset)
    return file_header + header + palette + rows


def bmp16(width, height, seed, masks=()):
    """16 bits: x555 as BI_RGB, or BI_BITFIELDS with 3 masks or an alpha mask."""
    a = gradient(width, height, seed).astype(np.uint16)
    green = 2 if masks and masks[1] == 0x07E0 else 3
    px = ((a[..., 0] >> 3) << (5 + 8 - green)) | ((a[..., 1] >> green) << 5) | (a[..., 2] >> 3)
    rows = b"".join(
        row.astype("<u2").tobytes() + bytes((-2 * width) % 4) for row in px[::-1]
    )
    return bmp(width, height, 16, rows, compression=3 if masks else 0, masks=masks)


def palette_bmp(width, height, seed, colors, rle=False):
    """8 bits per pixel. `colors` 2 is a black and white palette, which Pillow
    reads as mode 1. `rle` skips 3 rows and ends one row early."""
    rng = np.random.default_rng(seed)
    index = rng.integers(0, colors, (height, width), dtype=np.uint8)
    palette = bytes([0, 0, 0, 0, 255, 255, 255, 0])
    if colors > 2:
        palette = rng.integers(1, 256, (colors, 4), dtype=np.uint8).tobytes()
    if not rle:
        rows = b"".join(row.tobytes() + bytes(-width % 4) for row in index[::-1])
        return bmp(width, height, 8, rows, palette=palette, colors=colors)
    stream = b""
    for y, row in enumerate(index[::-1]):
        if y == 5:
            stream += bytes([0, 2, 0, 3])
        elif y == 10:
            stream += bytes([15, row[0], 0, 0])
        elif y not in (6, 7):
            stream += b"".join(bytes([1, value]) for value in row) + bytes(2)
    stream += bytes([0, 1])
    return bmp(width, height, 8, stream, compression=1, palette=palette, colors=colors)


def animated_webp(alpha, blend_first=False):
    channels = 4 if alpha else 3
    frames = [Image.fromarray(gradient(48, 40, seed, channels)) for seed in (45, 46)]
    options = {"lossless": True} if alpha else {"quality": 80}
    data = bytearray(
        encode(frames[0], "WEBP", save_all=True, append_images=frames[1:], **options)
    )
    if blend_first:
        # Pillow sets "do not blend" (bit 1 of the ANMF flags) on the first frame.
        position = 12
        while data[position : position + 4] != b"ANMF":
            (size,) = struct.unpack_from("<I", data, position + 4)
            position += 8 + size + (size & 1)
        data[position + 8 + 15] &= ~0b10
    return bytes(data)


def muxed_webp():
    """An animation with no alpha and one lossless frame that asks for a
    blend: a legal form no Pillow option writes."""

    def chunk(tag, data):
        return tag + struct.pack("<I", len(data)) + data + bytes(len(data) & 1)

    def u24(value):
        return struct.pack("<I", value)[:3]

    still = encode(Image.fromarray(gradient(48, 40, 49)), "WEBP", lossless=True)
    assert still[12:16] == b"VP8L"
    size = (47, 39)
    frame = u24(0) + u24(0) + u24(size[0]) + u24(size[1]) + u24(100) + b"\0" + still[12:]
    body = (
        b"WEBP"
        + chunk(b"VP8X", bytes([0x02, 0, 0, 0]) + u24(size[0]) + u24(size[1]))
        + chunk(b"ANIM", bytes(6))
        + chunk(b"ANMF", frame)
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


def quantized(width, height, seed):
    """RGB in 4 levels per channel, so one exact color covers many pixels."""
    return Image.fromarray(gradient(width, height, seed) // 64 * 64)


def xor_pattern(width, height, seed):
    """RGB that holds every value of each channel at 512x512."""
    y, x = np.mgrid[0:height, 0:width].astype(np.int64)
    red = (x * 255 // max(width - 1, 1)) ^ ((y * 37 + seed) & 31)
    green = (y * 255 // max(height - 1, 1)) ^ ((x * 53 + seed) & 31)
    blue = (x * x + y * 7 + seed * 13) & 255
    return np.stack([red, green, blue], axis=-1).astype(np.uint8)


def la_png(width, height):
    y, x = np.mgrid[0:height, 0:width]
    return encode(Image.fromarray(np.stack([x * 6 + y, 255 - y * 8], -1).astype(np.uint8)), "PNG")


# name -> encoded bytes, one file per decode form. The first block keeps the
# names and bytes of SGLang's corpus.
CORPUS = {
    "rgb_96x80.png": lambda: encode(Image.fromarray(gradient(96, 80)), "PNG"),
    "rgb_333x251.jpg": lambda: encode(
        Image.fromarray(gradient(333, 251, 1)), "JPEG", quality=90
    ),
    "rgb_444_120x90.jpg": lambda: encode(
        Image.fromarray(gradient(120, 90, 2)), "JPEG", quality=95, subsampling=0
    ),
    "progressive_200x150.jpg": lambda: encode(
        Image.fromarray(gradient(200, 150, 3)), "JPEG", quality=85, progressive=True
    ),
    "cmyk_80x64.jpg": lambda: encode(
        Image.fromarray(gradient(80, 64, 4)).convert("CMYK"), "JPEG"
    ),
    "exif_orientation6_64x48.jpg": exif_rotated_jpeg,
    "gray_64x64.png": lambda: encode(
        Image.fromarray(gradient(64, 64, 5)[..., 0]), "PNG"
    ),
    "rgba_80x60.png": lambda: encode(
        Image.fromarray(gradient(80, 60, 6, channels=4)), "PNG"
    ),
    "palette_transparency_70x50.png": lambda: encode(
        Image.fromarray(gradient(70, 50, 10)).convert("P"), "PNG", transparency=0
    ),
    "gray16_40x30.png": lambda: encode(
        Image.fromarray((gradient(40, 30, 11)[..., 0].astype(np.uint16) * 257)), "PNG"
    ),
    "animated_48x40.gif": animated_gif,
    "lossy_90x70.webp": lambda: encode(
        Image.fromarray(gradient(90, 70, 12)), "WEBP", quality=80
    ),
    "lossless_alpha_60x40.webp": lambda: encode(
        Image.fromarray(gradient(60, 40, 13, channels=4)), "WEBP", lossless=True
    ),
    "transparent_56x56.gif": transparent_gif,
    "rgba_raw_exif6_56x48.png": raw_profile_exif_rgba_png,
    "rgb_50x40.bmp": lambda: encode(Image.fromarray(gradient(50, 40, 14)), "BMP"),
    "rgba_bitfields_56x56.bmp": bitfields_alpha_bmp,
    "tiny_10x10.png": lambda: encode(Image.fromarray(gradient(10, 10, 15)), "PNG"),
    "large_800x600.jpg": lambda: encode(
        Image.fromarray(gradient(800, 600, 16, noise=0)), "JPEG", quality=80
    ),
    "aspect_201_4020x20.png": lambda: encode(
        Image.fromarray(gradient(4020, 20, 17, noise=0)), "PNG"
    ),
    # Refused by Pillow and the crate.
    "truncated_96x80.png": lambda: CORPUS["rgb_96x80.png"]()[:400],
    "huge_palette_14000x13000.png": huge_palette_png,
    # Alpha, transparency and native modes.
    "alpha_noise_84x60.png": lambda: encode(draw("rgba", 84, 60, 20), "PNG"),
    "gray_alpha_noise_84x60.png": lambda: encode(draw("la", 84, 60, 21), "PNG"),
    "rgb_trns_64x48.png": lambda: encode(
        quantized(64, 48, 22), "PNG", transparency=(64, 64, 64)
    ),
    "gray_trns_64x48.png": lambda: encode(
        Image.fromarray(gradient(64, 48, 23)[..., 0] // 16 * 16), "PNG", transparency=96
    ),
    "palette_64x48.png": lambda: encode(
        Image.fromarray(gradient(64, 48, 24)).convert("P"), "PNG"
    ),
    "bilevel_64x48.png": lambda: encode(
        Image.fromarray(gradient(64, 48, 25)[..., 0]).convert("1"), "PNG"
    ),
    "bilevel_trns_64x48.png": lambda: gray_png(64, 48, 1, 26, trns=1),
    "gray4_64x48.png": lambda: gray_png(64, 48, 4, 27),
    "gray4_trns_64x48.png": lambda: gray_png(64, 48, 4, 28, trns=5),
    "lossy_alpha_60x40.webp": lambda: encode(
        Image.fromarray(gradient(60, 40, 31, channels=4)), "WEBP", quality=80
    ),
    "palette_64x48.bmp": lambda: encode(
        Image.fromarray(gradient(64, 48, 32)).convert("P"), "BMP"
    ),
    # Pillow writes RGBA as 32-bit BI_RGB and reads it back as RGB.
    "rgbx_56x40.bmp": lambda: encode(
        Image.fromarray(gradient(56, 40, 33, channels=4)), "BMP"
    ),
    "gray_90x70.jpg": lambda: encode(
        Image.fromarray(gradient(90, 70, 34)[..., 0]), "JPEG", quality=90
    ),
    # The forms whose decode is known to differ from Pillow.
    "bilevel_trns2_64x48.png": lambda: gray_png(64, 48, 1, 43, trns=2),
    "bmp16_555_50x40.bmp": lambda: bmp16(50, 40, 44),
    "bmp16_565_50x40.bmp": lambda: bmp16(50, 40, 47, (0xF800, 0x07E0, 0x001F)),
    "bmp16_alpha_mask_50x40.bmp": lambda: bmp16(
        50, 40, 48, (0x7C00, 0x03E0, 0x001F, 0x8000)
    ),
    "animated_48x40.webp": lambda: animated_webp(alpha=False),
    "animated_alpha_blend_48x40.webp": lambda: animated_webp(alpha=True, blend_first=True),
    "animated_lossless_blend_48x40.webp": muxed_webp,
    "rle8_gap_40x30.bmp": lambda: palette_bmp(40, 30, 53, 256, rle=True),
    "two_color_8bit_40x30.bmp": lambda: palette_bmp(40, 30, 54, 2),
    "local_ramp_table_40x30.gif": lambda: local_ramp_gif(40, 30, 56),
    "two_controls_40x30.gif": lambda: two_control_gif(40, 30, 57),
    "palette_trns_newline_40x30.png": lambda: newline_trns_png(40, 30, 58),
    "interlace_text_1x32.png": lambda: interlace_text_png(1, 32, 59),
    "rgb_32x24.tiff": lambda: encode(Image.fromarray(gradient(32, 24, 55)), "TIFF"),
    "garbage.bin": lambda: b"not an image",
    # Sizes below one patch and at extreme aspect ratios, with and without alpha (Kimi-K3).
    "wide_300x24.png": lambda: encode(draw("rgb", 300, 24, 35), "PNG"),
    "wide_alpha_300x24.png": lambda: encode(draw("rgba", 300, 24, 36), "PNG"),
    "tall_2x1000.png": lambda: encode(draw("rgb", 2, 1000, 37), "PNG"),
    "tall_alpha_3x700.png": lambda: encode(draw("rgba", 3, 700, 38), "PNG"),
    "one_1x1.png": lambda: encode(draw("rgb", 1, 1, 39), "PNG"),
    "rgb_3x5.png": lambda: encode(draw("rgb", 3, 5, 40), "PNG"),
    # Pixtral sizes: a fit into the 1540-pixel longest edge, a short side that
    # fits to 0 pixels, and a 3-pixel-high image.
    "rgb_2048x1536.png": lambda: encode(
        Image.fromarray(gradient(2048, 1536, 23, noise=0)), "PNG"
    ),
    "thin_3080x1.png": lambda: encode(
        Image.fromarray(gradient(3080, 1, 24, noise=0)), "PNG"
    ),
    "thin_640x3.png": lambda: encode(Image.fromarray(gradient(640, 3, 60)), "PNG"),
    "la_40x30.png": lambda: la_png(40, 30),
    # 512x512 is a tile of the Nemotron processor: no resize.
    "on_tile_512x512.png": lambda: encode(Image.fromarray(xor_pattern(512, 512, 0)), "PNG"),
    "corrupt.png": lambda: encode(Image.fromarray(xor_pattern(64, 64, 0)), "PNG")[:200],
}

# Files no Pillow option writes, kept as committed bytes: README.md has their recipe.
COMMITTED = {
    "libvpx_filter_below0_96x80.webp": "ee8fb942c46060ca28e04c0d87ce06e94624fb2a3574aeb9952a75cfc14f2491",
    "libvpx_filter_above63_96x80.webp": "904c4b8d7a111980f59fb2fccf152b597bbe1ea4238eb41d2831dd63cab66d5f",
    "libvpx_filter_in_range_96x80.webp": "4d9f3de9686e1a2718567589cf71ae7358fc63a3c4e6862082155be07ccdf5be",
}


def native(image):
    """The Kimi-K3 checkpoint's rule: RGBA when the mode is not RGB and has an
    alpha band or a `transparency` entry, else RGB. No EXIF transpose."""
    alpha = image.mode != "RGB" and (
        "A" in image.getbands() or "transparency" in image.info
    )
    return image.convert("RGBA" if alpha else "RGB")


# Crate decode entry point -> Pillow's output for `Image.open(bytes)`.
COLUMNS = {
    "decode_rgb": lambda image: image.convert("RGB"),
    "decode_rgb_exif_transposed": lambda image: ImageOps.exif_transpose(image).convert(
        "RGB"
    ),
    "decode_rgb_smart": smart_to_rgb,
    "decode_native": native,
}
RGB_COLUMNS = ["decode_rgb", "decode_rgb_exif_transposed", "decode_rgb_smart"]
ALL_COLUMNS = list(COLUMNS)

# Pillow and the crate both refuse these; the crate's error kind.
REJECTED = {
    "garbage.bin": "InvalidInput",
    "truncated_96x80.png": "InvalidInput",
    "huge_palette_14000x13000.png": "InvalidInput",
    "corrupt.png": "InvalidInput",
}

# Pillow decodes and the crate refuses: image -> [(columns, error kind, reason)].
# The `decode_native` reasons are the decode gate's refusals.
KNOWN_GAPS = {
    "gray16_40x30.png": [
        (
            ALL_COLUMNS,
            "InvalidInput",
            "Dynamo rejects 16-bit samples, which PIL's convert('RGB') clips",
        ),
    ],
    "rgb_32x24.tiff": [
        (
            RGB_COLUMNS,
            "InvalidInput",
            "Dynamo decodes JPEG, PNG, GIF, WebP and BMP only, and TIFF when another crate "
            "in the build turns on the image crate's tiff feature",
        ),
        (["decode_native"], "Unsupported", "Dynamo decodes JPEG, PNG, GIF, WebP and BMP only"),
    ],
    "gray4_trns_64x48.png": [
        (
            ["decode_native"],
            "Unsupported",
            "Dynamo refuses tRNS on 2-bit or 4-bit grayscale; Pillow compares the "
            "unscaled tRNS sample with the scaled pixels",
        ),
    ],
    "bilevel_trns2_64x48.png": [
        (
            ["decode_native"],
            "Unsupported",
            "Dynamo refuses a tRNS sample over the bit depth; Pillow reads 2 as white",
        ),
    ],
    **{
        name: [
            (
                ["decode_native"],
                "Unsupported",
                "Dynamo refuses 16-bit BMP; the image crate rounds 5-bit and 6-bit "
                "fields where Pillow truncates, and reads an alpha mask Pillow ignores",
            )
        ]
        for name in ["bmp16_555_50x40.bmp", "bmp16_565_50x40.bmp", "bmp16_alpha_mask_50x40.bmp"]
    },
    **{
        name: [
            (
                ["decode_native"],
                "Unsupported",
                "Dynamo refuses palette BMP as a category; Pillow fills RLE gaps with "
                "palette entry 0 and reads a black and white palette as mode 1",
            )
        ]
        for name in ["palette_64x48.bmp", "rle8_gap_40x30.bmp", "two_color_8bit_40x30.bmp"]
    },
    **{
        name: [
            (
                ["decode_native"],
                "Unsupported",
                "Dynamo refuses animated WebP as a category; image-webp blends a "
                "lossless first frame that libwebp does not",
            )
        ]
        for name in [
            "animated_48x40.webp",
            "animated_alpha_blend_48x40.webp",
            "animated_lossless_blend_48x40.webp",
        ]
    },
    "local_ramp_table_40x30.gif": [
        (
            ["decode_native"],
            "Unsupported",
            "Dynamo refuses a gray-ramp local color table beside a global table; "
            "Pillow paints the frame with the global table",
        ),
    ],
    "two_controls_40x30.gif": [
        (
            ["decode_native"],
            "Unsupported",
            "Dynamo refuses a second Graphic Control Extension before the image; "
            "Pillow keeps a transparent index that the second one clears",
        ),
    ],
    "palette_trns_newline_40x30.png": [
        (
            ["decode_native"],
            "Unsupported",
            "Dynamo refuses palette alpha FF* 00 FF* 0A; Pillow reads it as one "
            "transparent entry and drops the last alpha",
        ),
    ],
    "interlace_text_1x32.png": [
        (
            ["decode_native"],
            "Unsupported",
            "Dynamo refuses a text chunk named interlace before IDAT; Pillow reads "
            "the keyword back as its deinterlace switch",
        ),
    ],
    "libvpx_filter_below0_96x80.webp": [
        (
            ["decode_native"],
            "Unsupported",
            "Dynamo refuses a segment filter level outside 0..63 that a loop-filter "
            "delta moves; libwebp clamps after the deltas and image-webp before them",
        ),
    ],
}

# Known gaps that close in a build with the image crate's decoder for the format, such
# as a workspace build: (image, column). The cell also holds Pillow's pixels.
DECODER_GAPS = {("rgb_32x24.tiff", column) for column in RGB_COLUMNS}

# The crate decodes other pixels than Pillow, with no error: image -> [(columns, reason)].
DIVERGES = {
    **{
        name: [
            (
                RGB_COLUMNS,
                "image-webp blends a lossless frame that asks for it; libwebp does not "
                "blend a first frame",
            )
        ]
        for name in ["animated_alpha_blend_48x40.webp", "animated_lossless_blend_48x40.webp"]
    },
    "bmp16_555_50x40.bmp": [
        (RGB_COLUMNS, "the image crate rounds 5-bit fields where Pillow truncates"),
    ],
    "bmp16_565_50x40.bmp": [
        (RGB_COLUMNS, "the image crate rounds 5-bit and 6-bit fields where Pillow truncates"),
    ],
    "bmp16_alpha_mask_50x40.bmp": [
        (
            ["decode_rgb", "decode_rgb_exif_transposed"],
            "the image crate rounds 5-bit fields where Pillow truncates",
        ),
        (
            ["decode_rgb_smart"],
            "the image crate reads an alpha mask Pillow ignores; the mask is 0 in "
            "every pixel of this file",
        ),
    ],
    "rle8_gap_40x30.bmp": [
        (RGB_COLUMNS, "Pillow fills RLE gaps with palette entry 0"),
    ],
    "two_color_8bit_40x30.bmp": [
        (RGB_COLUMNS, "Pillow reads an 8-bit BMP with a black and white palette as mode 1"),
    ],
    "interlace_text_1x32.png": [
        (
            RGB_COLUMNS,
            "Pillow reads a text chunk named interlace back as its deinterlace switch, "
            "and reads the rows of this plain file in Adam7 order",
        ),
    ],
    "libvpx_filter_below0_96x80.webp": [
        (
            RGB_COLUMNS,
            "libwebp adds the loop-filter deltas to a segment's filter level before it "
            "clamps; image-webp clamps first",
        ),
    ],
    "local_ramp_table_40x30.gif": [
        (
            RGB_COLUMNS,
            "Pillow paints a first frame whose own table is a gray ramp with the global table",
        ),
    ],
    "bilevel_trns2_64x48.png": [
        (["decode_rgb_smart"], "Pillow reads a 1-bit tRNS sample above 1 as white"),
    ],
    "gray4_trns_64x48.png": [
        (
            ["decode_rgb_smart"],
            "Pillow compares the unscaled tRNS sample with the scaled pixels at 2 and 4 bits",
        ),
    ],
    "palette_trns_newline_40x30.png": [
        (
            ["decode_rgb_smart"],
            "Pillow reads palette alpha FF* 00 FF* 0A as one transparent entry",
        ),
    ],
    "two_controls_40x30.gif": [
        (
            ["decode_rgb_smart"],
            "Pillow keeps a transparent index that a second Graphic Control Extension clears",
        ),
    ],
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def pillow_cell(data, column):
    """Pillow's output for one column, or its error message. A `decode_native`
    cell also holds the mode of `Image.open`, which decides `exact_resize`."""
    try:
        source = Image.open(io.BytesIO(data))
        mode = source.mode
        image = COLUMNS[column](source)
    except Exception as error:  # noqa: BLE001 -- the refusal is the recorded result
        # Pillow prints the address of the stream it could not read.
        return None, re.sub(r" at 0x[0-9a-f]+", "", f"{type(error).__name__}: {error}")
    key = "rgba_sha256" if image.mode == "RGBA" else "rgb_sha256"
    assert image.mode in ("RGB", "RGBA"), image.mode
    pixels = np.asarray(image).tobytes()
    cell = {key: sha256(pixels), "width": image.width, "height": image.height}
    if column == "decode_native":
        cell["mode"] = mode
    return cell, None


def declared(table, name, column):
    matches = [entry for entry in table.get(name, []) if column in entry[0]]
    assert len(matches) <= 1, (name, column)
    return matches[0][1:] if matches else None


def decode_row(name, data):
    row = {}
    for column in COLUMNS:
        cell, error = pillow_cell(data, column)
        gap, diverges = declared(KNOWN_GAPS, name, column), declared(DIVERGES, name, column)
        if error is not None:
            assert name in REJECTED and not gap and not diverges, f"{name} {column}: {error}"
            row[column] = {"rejected": REJECTED[name], "pillow_error": error}
        elif gap:
            assert not diverges, (name, column)
            row[column] = {"rejected": gap[0], "known_gap": gap[1]}
            if (name, column) in DECODER_GAPS:
                row[column].update(cell)
        elif diverges:
            row[column] = {"diverges": diverges[0], **cell}
        else:
            assert name not in REJECTED, f"{name} {column}: Pillow decodes"
            row[column] = cell
    return row


# (image, (width, height)): Pillow's BICUBIC resize of the RGBA or LA image,
# which it runs premultiplied. The sources cover alpha 0, 255 and values between.
RESIZE_RGBA = [
    ("alpha_noise_84x60.png", (50, 36)),
    ("alpha_noise_84x60.png", (130, 97)),
    ("alpha_noise_84x60.png", (84, 31)),
    ("alpha_noise_84x60.png", (37, 60)),
    ("alpha_noise_84x60.png", (84, 60)),
    ("rgba_80x60.png", (40, 30)),
    ("gray_alpha_noise_84x60.png", (50, 36)),
    ("la_40x30.png", (23, 61)),
]


def resize_row(name, size):
    image = Image.open(IMAGES / name)
    assert native(image).mode == "RGBA", name
    resized = image.resize(size, Image.BICUBIC).convert("RGBA")
    return {
        "image": name,
        "size": list(size),
        "rgba_sha256": sha256(np.asarray(resized).tobytes()),
    }


def dump(table):
    """The table's fields one per line, and each image row and resize row on one line."""
    rows = {"images", "resize_rgba"}
    lines = [
        f"  {json.dumps(key)}: {json.dumps(value)}"
        for key, value in table.items()
        if key not in rows
    ]
    images = ",\n".join(
        f"    {json.dumps(name)}: {json.dumps(row)}" for name, row in table["images"].items()
    )
    resizes = ",\n".join(f"    {json.dumps(row)}" for row in table["resize_rgba"])
    return (
        "{\n"
        + ",\n".join(lines)
        + ',\n  "images": {\n'
        + images
        + '\n  },\n  "resize_rgba": [\n'
        + resizes
        + "\n  ]\n}\n"
    )


def image_bytes(name, check):
    path = IMAGES / name
    if name in COMMITTED:
        data = path.read_bytes()
        assert sha256(data) == COMMITTED[name], f"{name} differs"
        return data
    data = CORPUS[name]()
    if check:
        assert path.read_bytes() == data, f"{name} differs"
    else:
        path.write_bytes(data)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    assert PIL.__version__.startswith("12.3."), "the crate mirrors Pillow 12.3"
    names = sorted({*CORPUS, *COMMITTED})
    for table in (REJECTED, KNOWN_GAPS, DIVERGES):
        assert set(table) <= set(names), set(table) - set(names)
    IMAGES.mkdir(exist_ok=True)
    extra = {path.name for path in IMAGES.iterdir()} - set(names)
    assert not extra, f"images not in the corpus: {sorted(extra)}"
    table = {
        "pillow": PIL.__version__,
        "versions": {"numpy": np.__version__, "sglang": sglang.__version__},
        "columns": {
            "decode_rgb": 'Image.open(f).convert("RGB")',
            "decode_rgb_exif_transposed": 'ImageOps.exif_transpose(Image.open(f)).convert("RGB")',
            "decode_rgb_smart": "sglang.srt.multimodal.media_io.smart_to_rgb(Image.open(f))",
            "decode_native": 'Image.open(f).convert("RGBA" if alpha else "RGB"), alpha: '
            'mode is not RGB and has an alpha band or a transparency entry',
        },
        "images": {name: decode_row(name, image_bytes(name, args.check)) for name in names},
        "resize_rgba": [resize_row(name, size) for name, size in RESIZE_RGBA],
    }
    text = dump(table)
    if args.check:
        assert (HERE / "decode.json").read_text() == text, "decode.json differs"
        print(f"checked {len(names)} images and decode.json")
        return
    (HERE / "decode.json").write_text(text)
    print(f"wrote {len(names)} images and decode.json")


if __name__ == "__main__":
    main()
