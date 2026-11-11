---
name: tool-parser-generator
description: Create or extend a v2 Unified tool-calling and reasoning parser from a model's grammar, independently authored GOLDEN cases, and shared parser components. Use when adding support for a model's output format.
license: "Apache-2.0"
---

# Tool Parser Generator Skill

Create or extend a **v2 Unified parser** in `parsers/v2`. V1 is being deprecated; do not create a v1 parser, add v1 captures, or make v1 parity a requirement for this workflow. Preserve existing historical evidence without regenerating it.

The current API, parser goals, family registry, and ordered contributor workflow are in [parsers/v2/README.md](../../../parsers/v2/README.md). Follow its [Adding A Day-0 Tool-Calling Parser](../../../parsers/v2/README.md#adding-a-day-0-tool-calling-parser) section. This skill supplies the agent's review checkpoints; the README owns implementation details and validation commands.

## 1. Research the model and prepare GOLDEN

Read the model's tool-calling specification, chat template, and special-token definitions. Fetch the published `tokenizer_config.json` when needed; handle either a string chat template or a named template list, preferring `tool_use` when supplied. Record the source URL and revision used. Identify reasoning initialization, native call framing, argument encoding, and end-of-stream behavior. Do not infer a model's grammar from its name or another engine's parser.

Use the README's [Parser families](../../../parsers/v2/README.md#parser-families) mapping to find an existing v2 scanner, emitter, or factory. Prefer reusing or extending a shared grammar owner. Write a new state machine only when the grammar requires one. The [pattern recognition guide](references/parser-patterns.md) helps identify syntax; current v2 APIs and the model's specification govern implementation. Peer vLLM and SGLang implementations provide behavioral evidence, not the correctness oracle.

Read [UNIFIED_CASES.md](../../../conformance/utils/lib/parsers/UNIFIED_CASES.md) and inspect **all `UNIFIED.*` cases across every registered family**. Make a best-effort mapping to the target grammar and request modes. Adapt the behavior with family-native bytes; do not copy foreign markers. Record applicable cases, coverage aliases, and each grammar or request-mode exclusion. A current parser failure is not an exclusion.

For example, GPT-OSS uses Harmony framing and JSON arguments, not XML parameter tags. A missing `</parameter>` stimulus cannot transfer literally; retain applicable missing-message-boundary and truncated-JSON tests instead. Conversely, adapt a generic recovery behavior to XML when the target grammar can express it. [UNIFIED_CASES.md](../../../conformance/utils/lib/parsers/UNIFIED_CASES.md#single-family-case-review) owns applicability policy.

Declare the conformance family, its grammar markers, and scenario applicability before generating cases; this is authoring metadata, not parser implementation. For a new family, also create its canonical authored family document as described in the [GOLDEN and latest-capture workflow](../../../conformance/README.md#prepare-golden-and-refresh-latest-dynamo-v2-observations).

The GOLDEN test set is the most important artifact: inputs, request settings, chunk schedules, and independently reasoned expected events. Author it through the current generator and taxonomy described in [Fixture Files To Add](../../../parsers/v2/README.md#fixture-files-to-add); the reviewable family output is `conformance/fixtures-unified-v2/families/<name>/inputs_and_golden.yaml`. Include specification sources and recovery rationale. Never derive GOLDEN from recorded parser results or change it merely to make a comparison pass. Document the reason for any deliberate expectation change.

**Pause for humans to review GOLDEN before implementing.** Present the family file, applicability mapping, exclusions, and unresolved recovery choices. AI-generated expectations can be wrong. When vendors disagree, use the model's specification, reference implementations, request settings, existing issues, and caller needs to explain the proposed decision. Resume implementation after the human review approves the expectations.

## 2. Implement and validate

After review, implement against the approved GOLDEN using existing v2 components and current Unified APIs. Register the factory as described in the README. Add or reuse a structural-tag builder for the new grammar under `parsers/v2/src/structural_tag/builders/` and register its lookup. This builder constrains generation; the output parser alone does not provide guided-decoding support. If the grammar or a request mode cannot be represented safely, document that limitation and the supported fallback rather than claiming guidance is available. Required tool-only compatibility should project Unified events rather than duplicate grammar state. No v1 implementation or comparison is required.

Test the structural-tag builder as well as the parser: use the registered builder to construct the actual grammar, compile it with the pinned XGrammar verifier, check accepted and rejected streams, and parse accepted output across chunk boundaries. Cover the supported tool-choice modes, argument schemas, and reasoning initialization; test unsupported modes explicitly. See [Add and validate structural tags](../../../parsers/v2/README.md#add-and-validate-structural-tags).

Run the focused parser tests and applicable shared v2 tests. Follow [Unified parser validation](../../../parsers/v2/README.md#unified-parser-validation), including its streaming and performance checks. Inspect intermediate emissions as well as assembled results; whole-input agreement alone does not prove incremental behavior. Use fixed fixtures so validation does not require model generation. Live model output is optional for the library contribution. Before claiming deployment support for a model/backend combination, run the real-serving qualification in the structural-tag section; before claiming guidance improves output, run its controlled off/on comparison.

Fix implementation failures against the reviewed contract. If a case exposes an incorrect or ambiguous expectation, document the evidence and return that GOLDEN change for human review. Keep intentional differences from peer parsers explicit; matching a reference implementation alone does not establish correctness.

## 3. Capture and hand off the report

Recapture affected latest Dynamo v2 observations using the commands in [Prepare GOLDEN and refresh latest Dynamo v2 observations](../../../conformance/README.md#prepare-golden-and-refresh-latest-dynamo-v2-observations). Follow the [conformance README](../../../conformance/README.md#generate-the-latest-or-history-report) to generate the latest report locally and refresh `conformance/CONFORMANCE_v2.html` when requested. Historical backfills and v1 captures are outside this workflow. Do not commit generated reports unless specifically requested.

Serve the report through the workspace's approved local HTTP route and provide its URL. Inspect every affected case in the GUI, including input, initialization, chunks, GOLDEN, and actual output. Check event order, argument fidelity, markup leaks, and missing observations. **Pause for humans to visually review the report.** Report the candidate revision, changed files, executed tests, microbenchmark setup and results, exclusions, and unresolved decisions. Distinguish completed checks from checks that were not run.
