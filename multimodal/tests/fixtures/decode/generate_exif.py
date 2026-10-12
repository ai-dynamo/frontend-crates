# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate EXIF orientation 1..8 RGB goldens with Pillow 12.0.0.

Run: python multimodal/tests/fixtures/decode/generate_exif.py
Each .rgb concatenates eight ImageOps.exif_transpose results in tag order.
The JPEG source is the existing Pillow/libjpeg-turbo parity fixture; the
lossless source is the same asymmetric 6x4 pattern the Rust tests encode.
"""

from pathlib import Path

from PIL import Image, ImageOps, __version__

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
