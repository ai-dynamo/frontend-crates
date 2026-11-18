# mock-tekken

A small v15 Mistral `tekken.json` for `TekkenTokenizer` tests.

- `vocab`: the first 1,000 entries of mistral-common's public `src/mistral_common/data/tekken_240911.json`, with its split pattern.
- `special_tokens`: the standard mistral-common names at ranks 0–37 (`<s>`, `[INST]`, `[TOOL_CALLS]`, `[ARGS]`, `[THINK]`, `[MODEL_SETTINGS]`, …); ranks 38–39 are left to the `<SPECIAL_i>` fillers.
- `model_settings_builder`: `reasoning_effort` (`none`/`high`, default `none`) and a `json_schema` field.

`expected_ids.json` holds `Tekkenizer.from_file("tekken.json").encode(text, bos=False, eos=False)` from mistral-common 1.13.0 for each text.
