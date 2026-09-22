# Pixel parity fixtures

The CMYK/YCCK JPEG, offset GIF, and normalization fixtures were captured with
Pillow 12.3.0 and NumPy 2.3.5. Expected `.rgb` files contain the complete output
of `Image.open(path).convert("RGB").tobytes()`, with no alpha compositing.

- `pillow_cmyk_13x9.jpg`: 13×9 CMYK noise from
  `np.random.default_rng(225).integers(0, 256, 13*9*4, dtype=np.uint8)`, saved
  by Pillow at quality 87.
- `turbojpeg_ycck_13x9.jpg`: encoded with Rust turbojpeg 1.5.1, YCCK colorspace,
  quality 87, and 2×2 chroma subsampling. The 13×9 CMYK input byte at index `i`
  is `(((i + 225) * 2654435761) >> 20) & 255`. Pillow supplies the expected RGB.
- `pillow_*offset_8x8.gif`: a 4×4 indexed frame at offset (2, 2) in an 8×8
  canvas. Each row contains indices `[0, 1, 2, 3]`. The global palette begins
  with `(200,100,50), (10,220,30), (40,60,240), (180,90,20)`; remaining entries
  are black. The declared background index is 1, deliberately different from
  Pillow's initial fill index 0. The transparent variant uses transparent
  index 2. The local-palette variant overrides the first four colors with
  `(70,80,90), (200,10,150), (5,160,130), (240,180,40)`.
- `pillow_expanded_6x6.gif`: the opaque offset fixture with the logical screen
  changed to 1×1. Pillow expands it to the first frame's 6×6 extent.

To regenerate expected decode outputs from the committed inputs:

```python
from pathlib import Path
from PIL import Image

for path in Path("mm-preprocessor/tests/fixtures/decode").iterdir():
    if path.suffix in {".jpg", ".gif"}:
        path.with_suffix(".rgb").write_bytes(Image.open(path).convert("RGB").tobytes())
```

`transforms/hf_normalized_u8.f32le` contains 256 RGB pixels as little-endian
float32 values, following HF's separate rescale and normalize operations:

```python
import numpy as np
from pathlib import Path

rgb = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
scaled = (rgb.astype(np.float64) * (1 / 255)).astype(np.float32)
mean = np.array([0.5, 0.4, 0.3], dtype=np.float32)
std = np.array([0.5, 0.25, 0.75], dtype=np.float32)
Path("mm-preprocessor/tests/fixtures/transforms/hf_normalized_u8.f32le").write_bytes(
    ((scaled - mean) / std).astype("<f4").tobytes()
)
```
