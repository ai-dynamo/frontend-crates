# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate EXIF RGB goldens and legacy PNG profiles with Pillow 12.0.0.

Run: python multimodal/tests/fixtures/decode/generate_exif.py
Each .rgb concatenates eight ImageOps.exif_transpose results in tag order.
The JPEG source is the existing Pillow/libjpeg-turbo parity fixture; the
lossless source is the same asymmetric 6x4 pattern the Rust tests encode.
Legacy PNG profiles use orientation 6 in tEXt/zTXt before or after IDAT;
their transposed RGB is the sixth existing lossless golden.
"""

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps, PngImagePlugin, __version__

ROOT = Path(__file__).parent
assert __version__ == "12.0.0", __version__

lossless = Image.new("RGB", (6, 4))
lossless.putdata([(x * 40, y * 60, x + y * 10) for y in range(4) for x in range(6)])

for name, source in [
    ("pillow_exif_jpeg.rgb", Image.open(ROOT / "pillow_noise_13x9.jpg")),
    ("pillow_exif_lossless.rgb", lossless),
]:
    results = []
    for orientation in range(1, 9):
        image = source.copy()
        image.getexif()[274] = orientation
        results.append(ImageOps.exif_transpose(image).convert("RGB").tobytes())
    (ROOT / name).write_bytes(b"".join(results))

exif: Image.Exif = Image.Exif()
exif[274] = 6
raw_exif: bytes = exif.tobytes()
profile: str = f"\nexif\n{len(raw_exif)}\n{raw_exif.hex()}\n"
golden: bytes = (ROOT / "pillow_exif_lossless.rgb").read_bytes()[5 * 72 : 6 * 72]

for compressed in (False, True):
    chunk_type: bytes = b"zTXt" if compressed else b"tEXt"
    info: PngImagePlugin.PngInfo = PngImagePlugin.PngInfo()
    info.add_text("Raw profile type exif", profile, zip=compressed)
    encoded: BytesIO = BytesIO()
    lossless.save(encoded, format="PNG", pnginfo=info)
    png: bytes = encoded.getvalue()
    for trailing in (False, True):
        output: bytes = png
        if trailing:
            # Move the intact text chunk (including CRC) before IEND. Pillow's
            # writer only honors after_idat for private chunks, not tEXt/zTXt.
            offset: int = 8
            while png[offset + 4 : offset + 8] != chunk_type:
                offset += int.from_bytes(png[offset : offset + 4], "big") + 12
            end: int = offset + int.from_bytes(png[offset : offset + 4], "big") + 12
            output = png[:offset] + png[end:-12] + png[offset:end] + png[-12:]
        location: str = "after_idat" if trailing else "before_idat"
        path: Path = ROOT / f"pillow_raw_exif_{chunk_type.decode()}_{location}.png"
        path.write_bytes(output)
        with Image.open(path) as image:
            transposed: Image.Image = ImageOps.exif_transpose(image).convert("RGB")
            assert image.size == (6, 4)
            assert transposed.size == (4, 6)
            assert transposed.tobytes() == golden
        print(f"{path.name}: {len(output)} bytes; Pillow 6x4 -> 4x6; RGB golden matches")
