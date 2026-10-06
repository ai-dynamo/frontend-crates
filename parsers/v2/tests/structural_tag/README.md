# Run executable structural-tag contracts

Use Python 3.12 on Linux x86_64. From the repository root:

```bash
python3.12 -m venv /tmp/xgrammar
/tmp/xgrammar/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r parsers/v2/tests/structural_tag/requirements.txt
/tmp/xgrammar/bin/pip check
XGRAMMAR_PYTHON=/tmp/xgrammar/bin/python cargo test --locked -p dynamo-parsers-v2 --test structural_tag_xgrammar -- --include-ignored --nocapture
```

The dedicated `structural-tag` CI job executes the ignored XGrammar test explicitly.
Ordinary Rust tests run the authored Unified samples and the extraction-policy
contrast without Python. Missing XGrammar, a version mismatch, compilation errors,
empty cases, and incomplete worker results fail the integration test.

The Rust test calls each public v2 builder and sends its JSON unchanged to the
Python worker. The worker compiles it with XGrammar 0.2.8. Positive cases require
both string acceptance and grammar completion. Truncated named calls explicitly
prove that prefix acceptance alone is insufficient.

The matrix covers DeepSeek V4, Qwen3 Coder, and GLM47. It checks tool choice,
parallel calls, schemas, Unicode, reasoning initialization, external reasoning
boundaries, response alternatives, marker exclusions, and property ordering.
Every accepted sample goes through the corresponding native Unified parser with
the same tools. Chunkings include the entire string, individual Unicode characters,
and every two-chunk boundary, including empty chunks. Expected events are authored
independently, and assembly removes differences in delta segmentation.

Keep these contracts distinct when adding cases:

- Grammar-negative cases do not require parser rejection. The enum contrast
  explicitly shows that Unified extracts an argument outside its schema enum.
- Native extraction preserves some separator text. DeepSeek emits the leading
  block newlines; Qwen emits the separator between calls and after reasoning.
- Without schemas, Qwen and GLM extract numeric XML values as strings. DeepSeek
  has explicit string flags in its markup.
- `any_order` permits reordered properties and duplicate keys. XGrammar still
  requires an entry count at least as large as the number of required keys.
  Duplicate city entries can meet that bound while the required count is absent.
- Qwen must dispatch at `<tool_call>` before its free-text exclusion of
  `<function=` can fire. The full tag still requires `\n<function=...>\n`.

The local byte vocabulary avoids model downloads. These tests qualify complete
string behavior, not production tokenizer masks or model output quality. Python
packages are frozen in `requirements.txt`, including the CPU Torch build. When
updating XGrammar, refresh that environment and rerun the entire matrix.
