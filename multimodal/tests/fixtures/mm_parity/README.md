# Multimodal parity fixtures

## The image corpus

`images/` is the one image corpus of the crate's parity tests. Each file
tests one idea: a format, a mode, a transparency form, an EXIF form, or a
form whose decode is known to differ from Pillow. A file that SGLang's corpus
(`rust/sglang-processor/tests/fixtures/mm_parity/images/`) also has keeps
SGLang's name and bytes. Other names are `<idea>_<w>x<h>.<ext>`.

`corpus.py` builds every image except the three `libvpx_filter_*_96x80.webp`
files, which no Pillow option writes. These are VP8 key frames from libvpx
1.14.0 (`vpx_codec_encode` at quantizer 60 on a 96x80 picture, with a
region-of-interest map of 4 segments), in the simple RIFF container. The name
gives the filter levels of the segments before the loop-filter deltas: the
frame level plus the map's deltas (55 + [0, 12, -12, -63] for `below0`,
44 + [0, 5, 20, 63] for `above63`, 63 + [0, -10, -30, -63] for `in_range`).
`corpus.py` checks their SHA-256.

## The decode table

`decode.json` has one row per corpus image and one column per decode entry
point of the crate. `corpus.py` records each column from Pillow 12.3.0:

| column | Pillow's output |
| --- | --- |
| `decode_rgb` | `Image.open(f).convert("RGB")` |
| `decode_rgb_exif_transposed` | `ImageOps.exif_transpose(Image.open(f)).convert("RGB")` |
| `decode_rgb_smart` | SGLang's `smart_to_rgb(Image.open(f))` (GLM) |
| `decode_native` | `Image.open(f).convert("RGBA")` when the mode is not RGB and has an alpha band or a `transparency` entry, else `convert("RGB")`; no EXIF transpose (Kimi-K3) |

A cell is one of:

- `{"rgb_sha256" | "rgba_sha256", "width", "height"}`: Pillow's output. The
  crate must give the same pixels, channels and size. A `decode_native` cell
  also holds `mode`, the mode of `Image.open`. The crate's `exact_resize` must
  be false for `P`, `1`, `CMYK` and an `L` with a `transparency` entry, else true.
- `{"rejected": "<MmError kind>", "pillow_error"}`: both sides refuse. The
  crate must refuse with this kind.
- `{"rejected": "<MmError kind>", "known_gap"}`: Pillow decodes and the crate
  refuses with this kind, for the reason in `known_gap`.
- `{"rejected": "<MmError kind>", "known_gap", "rgb_sha256", "width", "height"}`:
  the crate decodes this format only when the build has the `image` crate's
  decoder for it. A workspace build has every `image` decoder, because another
  crate turns on the `image` default features. Without the decoder, the crate
  must refuse with this kind. With the decoder, the crate must give Pillow's
  pixels. `corpus.py` lists these cells in `DECODER_GAPS`.
- `{"diverges", "rgb_sha256", "width", "height"}`: the crate decodes other
  pixels than Pillow at Pillow's size, with no error, for the reason in
  `diverges`. The test fails when the crate starts to match Pillow: then remove
  the entry from `DIVERGES`.

`resize_rgba` rows hold Pillow's `resize(size, BICUBIC).convert("RGBA")` of an
RGBA or LA corpus image. Pillow resizes RGBA and LA premultiplied. The crate's
`image::resize::resize_rgba` must give the same bytes from the
`decode_native` pixels.

`corpus.py` never runs the crate. The crate's error kinds and the reasons are
in its `REJECTED`, `KNOWN_GAPS` and `DIVERGES` tables. The test
`image::decode_table` in `src/image/decode_table.rs` replays the table, and
fails when an image in `images/` has no row.

## Regenerate

Use Pillow 12.3.0, numpy 2.3.5 and SGLang at
`a2ffd214696ea7144d3c237133a15df3315d900b` (for `smart_to_rgb`):

```sh
python multimodal/tests/fixtures/mm_parity/corpus.py [--check]
cargo test -p dynamo-multimodal --lib image::decode_table
```

`--check` writes nothing and fails when an image or `decode.json` differs.
To add an image, add its recipe to `CORPUS` and run the script.
