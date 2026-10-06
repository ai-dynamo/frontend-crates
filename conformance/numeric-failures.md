# Numeric conformance follow-ups

These reproduced failures remain red in the conformance report. Parser fixes belong in follow-up work; PR #339 adds cases and captures only. The report groups numeric variants under `UNIFIED.7-15` and `TOOLCALLING.streamv1.7-15`; each popup names the full scenario ID listed below.

## TODO NUM-1: preserve decimal values through argument parsing

The current candidate rounds exact decimal tokens when it builds tool arguments. The parser returns a valid call with a changed number, so the report must keep these cases red. This TODO groups the same number-conversion defect across current Unified and tool-stream paths and historical Dynamo v1 jail captures.

- Affected Unified families: `deepseek_v4`, `deepseek_v41`, `glm47`, `kimi_k2`, and `muse_glimmer` each round all six values below; `gemma4` rounds five and drops the exponent argument, which is tracked separately under NUM-2. This is 35 failing Unified family/case results: five families round all six values, and Gemma rounds five.
- Affected tool-stream families: `deepseek_v4`, `glm47`, `harmony`, `harmony_text`, `kimi_k2`, `minimax_m3`, and `muse_glimmer` each round all six values below; `gemma4` rounds five and drops the exponent argument, which is tracked separately under NUM-2. This is 47 failing stream family/case results: seven families round all six values, and Gemma rounds five.
- Case IDs: suffixes `round_down`, `round_up`, `quarter`, `exponent`, `small`, and `negative` under both `UNIFIED.7-15.*` and `TOOLCALLING.streamv1.7-15.*`.
- Historical Dynamo v1 jail captures also round all six values for `deepseek_v4`, `glm47`, `harmony_text`, `kimi_k2`, `kimi_k3`, `minimax_m2`, `minimax_m3`, and `qwen3_coder`, and round five values for `gemma4`, in each of `dynamo_v1-9.1.0`, `dynamo_v1-9.2.4`, and `dynamo_v1-9.2.8`: 159 additional rounding failures. Gemma's exponent drop is tracked separately under NUM-2.
- Input: one native tool call with arguments `{"value":<token>}` and schema `{"type":"object","properties":{"value":{"type":"number"}}}`; the request does not declare `required`.
- Expected and actual argument values from candidate source `9e4fe9e4c383e3ce54024c8c571add4bff8e3bd9`, crate `dynamo-parsers-v2 0.7.14`, source SHA-256 `9886568549f43124587e6a508130dd270f875800efe8875236d8f513255cf2af`:

| Case suffix | Input token | Expected JSON argument | Actual JSON argument in rounding failures |
|---|---|---|---|
| `round_down` | `9007199254740992.5` | `{"value":9007199254740992.5}` | `{"value":9007199254740992.0}` |
| `round_up` | `9007199254740993.1` | `{"value":9007199254740993.1}` | `{"value":9007199254740994.0}` |
| `quarter` | `9007199254740993.25` | `{"value":9007199254740993.25}` | `{"value":9007199254740994.0}` |
| `exponent` | `9.0071992547409925e15` | `{"value":9.0071992547409925e15}` | `{"value":9007199254740992.0}` |
| `small` | `0.10000000000000000001` | `{"value":0.10000000000000000001}` | `{"value":0.1}` |
| `negative` | `-9007199254740992.5` | `{"value":-9007199254740992.5}` | `{"value":-9007199254740992.0}` |

- Reproduce tool-stream results by building `record_dynamo_stream` from the candidate and passing `conformance/toolcalling/fixtures-stream-v1/inputs/deepseek_v4/TOOLCALLING.streamv1.7-numeric.yaml` to it; add `--text` for `harmony_text`. The captured per-family results are in the current `dynamo_v2-0.7.14` archive and source-identified by the metadata above.
- Reproduce Unified output with `CARGO_TARGET_DIR=/tmp/pr339-numeric/target cargo test --locked -p dynamo-conformance-fixtures-v2 --test unified_render -- --exact render_unified_conformance_html --nocapture`; the Rust gate recognizes the exact Gemma exponent failure through `conformance/unified-known-divergences.yaml`, while the decimal comparison in the canonical report keeps the numeric failures red. Inspect chunk arguments as raw strings; the assembled `serde_json::Value` also rounds numbers and cannot establish decimal preservation.
- Reproduce the historical jail captures from their exact source commits with `CARGO_TARGET_DIR=/tmp/pr339-numeric/target-<version> python3 conformance/utils/src/capture_dynamo_jail_stream.py --root <source-checkout>`. Use `/tmp/pr339-numeric/history-v1/source-9.1.0`, `/tmp/pr339-numeric/history-v1/source-9.2.4`, and `/tmp/pr339-numeric/d1-9.2.8` for the three versions. The result is in each checkout's `conformance/toolcalling/fixtures-stream-v1/dynamo_v1-<version>/` directory.
- Numeric case links in the report: [Unified numeric cases](CONFORMANCE_v2.html) and [tool-stream numeric cases](CONFORMANCE_v2.html).

## TODO NUM-2: Gemma drops exponent-form arguments

For `gemma4`, exponent input produces an empty object instead of a call argument containing the number. This is separate from the rounding cases above and reproduces in both current interfaces.

- Affected family and IDs: `gemma4`, `UNIFIED.7-15.exponent`, and `TOOLCALLING.streamv1.7-15.exponent`.
- Unified input: `<|tool_call>call:get_weather{value:9.0071992547409925e15}<tool_call|>`. Stream input uses the same Gemma native call syntax, with schema `{"type":"object","properties":{"value":{"type":"number"}}}` and no `required` field.
- Expected: `{"value":9.0071992547409925e15}`. Actual: `{}` in both current captures.
- Source identity: candidate commit `9e4fe9e4c383e3ce54024c8c571add4bff8e3bd9`, crate `dynamo-parsers-v2 0.7.14`, source SHA-256 `9886568549f43124587e6a508130dd270f875800efe8875236d8f513255cf2af`.
- Reproduction commands: use the Unified and stream commands under NUM-1 with the Gemma input files; the current stream capture is `conformance/fixtures/toolcalling/fixtures-stream-v1/dynamo_v2-0.7.14.tar.gz`.
- The historical Dynamo v1 jail also drops this argument for `TOOLCALLING.streamv1.7-15.exponent` in `dynamo_v1-9.1.0`, `dynamo_v1-9.2.4`, and `dynamo_v1-9.2.8`; expected `{"value":9.0071992547409925e15}`, actual `{}`. These captures reproduce with the NUM-1 historical command.

## TODO NUM-3: convert integral decimal and exponent tokens in Dynamo v1 jail

The Dynamo v1 jail leaves integral decimal and exponent tokens as JSON strings for the Qwen3-Coder and MiniMax-M2 schemas. The shared contract expects mathematically integral values as JSON numbers, including literal constraints and integer-or-string schemas; the three fractional fallback cases pass.

- Affected families: `qwen3_coder` and `minimax_m2` in each of `dynamo_v1-9.1.0`, `dynamo_v1-9.2.4`, and `dynamo_v1-9.2.8`, for 54 failing family/case/version results (18 per source version).
- Case IDs: `TOOLCALLING.streamv1.7-14.const_decimal`, `TOOLCALLING.streamv1.7-14.const_exponent`, `TOOLCALLING.streamv1.7-14.enum_decimal`, `TOOLCALLING.streamv1.7-14.enum_exponent`, `TOOLCALLING.streamv1.7-14.large_decimal`, `TOOLCALLING.streamv1.7-14.large_exponent`, `TOOLCALLING.streamv1.7-14.negative`, `TOOLCALLING.streamv1.7-14.negative_exponent`, and `TOOLCALLING.streamv1.7-14.zero_underflow`.
- Concrete case: Qwen3-Coder input token `42.0` with schema `{"type":["integer","null"],"const":42}`; expected `{"value":42}`, actual `{"value":"42.0"}`. The exponent form `4.2e1` has the same expected value and is returned as `{"value":"4.2e1"}`. Integer-or-string cases use schema `{"type":["integer","string"]}`.
- Source identities: Dynamo v1 crate `9.1.0`, commit `23bdc23bca16bb14974303ae73fd97d19b0be84d`; crate `9.2.4`, commit `8ff9a8127941a04ba5cd4a0b33678fb2f1045649`; crate `9.2.8`, commit `47ef952237deefe0643f94025272a558f249687d`.
- Reproduction command for each source checkout: `CARGO_TARGET_DIR=/tmp/pr339-numeric/target-<version> python3 conformance/utils/src/capture_dynamo_jail_stream.py --root /tmp/pr339-numeric/history-v1/source-<version>`. The `9.2.8` checkout is `/tmp/pr339-numeric/d1-9.2.8`.

## TODO NUM-4: preserve Muse Glimmer numeric calls in Dynamo v1 jail

The Dynamo v1 jail emits no tool call for any fractional numeric case in Muse Glimmer, so no argument value reaches the caller. This is separate from the number-rounding failures in NUM-1.

- Affected family: `muse_glimmer` in each of `dynamo_v1-9.1.0`, `dynamo_v1-9.2.4`, and `dynamo_v1-9.2.8`, for 21 failing family/case/version results (seven per source version).
- Case IDs: `TOOLCALLING.streamv1.7-15.ordinary`, `TOOLCALLING.streamv1.7-15.round_down`, `TOOLCALLING.streamv1.7-15.round_up`, `TOOLCALLING.streamv1.7-15.quarter`, `TOOLCALLING.streamv1.7-15.exponent`, `TOOLCALLING.streamv1.7-15.small`, and `TOOLCALLING.streamv1.7-15.negative`.
- Concrete case: Muse Glimmer's native numeric call for `{"value":42.5}` with schema `{"type":"object","properties":{"value":{"type":"number"}}}`; expected `{"value":42.5}`, actual assembled calls `[]`.
- Source identities and reproduction commands: the three exact commits and the capture command are listed under NUM-3.

## Historical captures that could not be measured

These are missing source captures, not parser failures. No result from a different parser build was substituted.

- Unified `dynamo_v2-0.7.7` for `glm47`: the checked-out producer source did not match the archive's recorded fingerprint, so no capture was assigned.
- Unified `dynamo_v2-0.7.10` for `qwen3`: the recorded producer commit was unavailable locally.
- Stream `dynamo_v2-0.1.11.patch1` for `deepseek_v4`, `harmony`, `harmony_text`, and `qwen3_coder`: the archive retained no producer source identity; the new archive records these cases as unavailable.
- Stream v1 `dynamo_v1-3.0.0` has applicable numeric families but no numeric cases. The archived release result identifies source commit `5ad6b709` and records no `JailedStream`/recorder; the archive also retains no producer identity. The historical capture audit therefore could not run a numeric back-capture, and this release is missing evidence rather than a measured pass or failure.
- Older parser versions with no Unified interface remain outside Unified applicability. Their stream archive family and mode coverage is unchanged.
