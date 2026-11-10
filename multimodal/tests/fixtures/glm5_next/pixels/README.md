# GLM pixel parity oracle

Copied byte-for-byte from SGLang PR #46563 at commit
`916577f5450a0475807c3a07b0b5b8377708935c`:
`rust/sglang-processor/tests/fixtures/mm_parity/glm-5.3-flash.json` and its
20 referenced images. `cases.json` records the model revision and Pillow
version (12.3.0). The 26 cases cover formats, EXIF, alpha, resize backends,
multiple images, and token overrides. The 16-bit PNG case retains its known
gap: Python clips while this crate refuses deeper-than-8-bit samples.

Regenerate using SGLang's Python processor, independently of this Rust code:

```bash
cd rust/sglang-processor
python tests/scripts/generate_mm_parity.py zai-org/GLM-5.3-Flash --revision 46a050bec13002ad29ee795c16e61cf4fa3c4a9c --formats
```

Copy the resulting GLM JSON to `cases.json` and the referenced image files to
`images/`, then run `cargo test -p dynamo-multimodal --test glm5_next_pixels`.
The original generator and image-pipeline authors are Kan Wu and Zhihan Guo;
SGLang is licensed under Apache-2.0.
