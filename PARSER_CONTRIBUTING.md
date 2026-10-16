<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contributing Parser Changes

Follow the [repository contribution workflow](CONTRIBUTING.md#how-to-contribute), including the Contribution Request requirement for large external changes. Use the [parser-specific PR template](.github/PULL_REQUEST_TEMPLATE/parser.md) for changes that affect parser behavior, including changes outside `parsers/`. See [template selection](CONTRIBUTING.md#required-pr-description-format) for how to load it on GitHub.

## Required Bug-Fix Evidence

Every parser bug-fix PR must complete all five numbered evidence subsections before requesting review. Start with the [three common bug-fix evidence sections](CONTRIBUTING.md#required-pr-description-format): minimal sample input, actual buggy output, and expected/fixed output. Include raw model text, tools/schema, request initialization or mode, and chunk boundaries when they affect the failure. Use the same concrete reproduction for the before and after results.

### 4. Unified Regression Cases

For fixes that change parser behavior, add the reproduction and expected GOLDEN events as `UNIFIED.*` regression cases (test columns). List the exact case IDs, what each verifies, applicable model families, and intentional grammar/mode exclusions. Follow [case applicability](conformance/README.md#case-applicability) and the [parser-fix capture workflow](conformance/README.md#2-fix-a-dynamo-parser-and-refresh-its-expected-outputs). Capture the before and after versions against the same input, tools, initialization, chunk schedule, and GOLDEN so the regression column shows red before the fix and green after it.

### 5. Visual Conformance Verification

For fixes that change parser behavior, regenerate `conformance/CONFORMANCE_v2.html` from the final candidate worktree using `conformance/utils/render_table_v2.sh`, open the report in a browser, and inspect the affected `UNIFIED.*` columns and their input/output popups. State the versions, model families, case IDs, and observed before/after results in the PR description. Confirm that the affected current Dynamo family has zero red and zero empty cells under the [Unified parser hard gate](conformance/README.md#unified-parser-hard-gate). A passing unit test or JSON check alone does not replace this visual verification. Keep generated HTML and JSON out of the commit.

For changes with no bug fix, mark Bug Fix Evidence as not applicable and state why; still list the tests and validation performed for the change.

## Minimal Bug-Fix Examples

Keep the input and output excerpts as small as possible while preserving the failure. These examples come from the linked fixes and illustrate the first three required subsections; your PR must also supply its own reproduction command/test, Unified case IDs, and visual verification where applicable. The output excerpts show only the relevant fields, with generated IDs and other unchanged fields omitted.

**Dotted tool name truncated ([#416](https://github.com/ai-dynamo/frontend-crates/pull/416)).** V1 Harmony complete parsing with default configuration and no tool definitions:

```text
input:  <|channel|>commentary to=functions.a.b<|message|>{}<|call|>
bad:    {"name":"a","arguments":"{}"}
good:   {"name":"a.b","arguments":"{}"}
```

The tool name must remain whole. The PR's minimal example uses `a.b`; its [committed regression test](https://github.com/ai-dynamo/frontend-crates/blob/4cc56ca5e06434bd720080013d4090eef3f676df/parsers/v1/src/tool_calling/harmony/harmony_parser.rs#L553) uses `my-tool.v2`.

**DeepSeek separator leaked as content ([#405](https://github.com/ai-dynamo/frontend-crates/pull/405)).** DeepSeek V4 Unified parsing in the Response starting state, with no tool definitions. The input below is a JSON string: `\n\n` represents two actual leading newline characters.

```json
"\n\n<｜DSML｜tool_calls><｜DSML｜invoke name=\"run\"><｜DSML｜parameter name=\"value\" string=\"true\">Paris</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>"
```

```text
bad:    content = "\n\n"
good:   content = ""
both:   call = {"name":"run","arguments":{"value":"Paris"}}
```

The two newlines are tool-call framing, not visible content. The [regression test](https://github.com/ai-dynamo/frontend-crates/blob/75f04d5c03c1701340d94ae3d450da19603b1c06/parsers/v2/tests/dsml_separator.rs#L127) exercises the separator across chunk splits. Only the content changes; the call stays intact.

**Gemma marker leaked between streamed calls ([#408](https://github.com/ai-dynamo/frontend-crates/pull/408)).** V1 `JailedStream` with the `gemma4` tool-call parser. Feed these exact chunks in order, then finish the stream:

```json
[
  "<|tool_call>call",
  ":f{",
  "a:<|\"|>1<|\"|>",
  "}<tool_call|><|tool_call>ca",
  "ll:g{",
  "b:",
  "<|\"|>2<|\"|>}<tool_call|>"
]
```

```text
bad:    content = "<|tool_call>"
good:   content = ""
both:   tool_call_names = ["f", "g"]
```

Keep the chunk boundaries: the fourth chunk closes one call and begins the next, which triggers the bug. Joining the chunks into one input loses that reproduction. The [committed regression test](https://github.com/ai-dynamo/frontend-crates/blob/f9f8e8788b2efaa41407248d9549a28512efa47d/parsers/v1/tests/jail.rs#L4253) preserves this sequence.
