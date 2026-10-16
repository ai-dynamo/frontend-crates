## Overview

<!-- Describe the change, affected components/model families, and why it is needed. -->

## Details

<!-- Explain the implementation and link the related issue when required. -->

## Bug Fix Evidence

Bug-fix PRs in every crate must complete subsections 1-3 with concrete input and before/after outputs. See the [minimal input/output examples](https://github.com/ai-dynamo/frontend-crates/blob/main/CONTRIBUTING.md#minimal-bug-fix-examples).

Subsections 4-5 apply only to fixes that change parser behavior, including fixes outside `parsers/`; those fixes must complete all five subsections before requesting review. For other bug fixes (for example, multimodal, tokenizers, or protocols changes that do not affect parser behavior), mark subsections 4-5 `Not applicable` with a brief reason and list the affected crate's regression tests under Validation.

<!-- REQUIRED for bug fixes. Follow CONTRIBUTING.md#required-pr-description-format. For changes with no bug fix, replace this section's contents with "Not applicable" and the reason. -->

### 1. Minimal Sample Input

<!-- Paste a minimal concrete input, affected crate/version, and a runnable reproduction. For parser bugs, include raw model text, tools/schema, initialization/mode, and relevant chunk boundaries. -->

### 2. Buggy Output (Before)

<!-- Paste the actual incorrect output from the previous version for the input above. -->

### 3. Expected/Fixed Output (After)

<!-- Paste the expected output and actual output from the fixed version for the same input. Explain any remaining difference. -->

### 4. Unified Regression Cases (Parser Bug Fixes Only)

<!-- Parser bug fixes: add UNIFIED.* regression cases/test columns and list exact IDs, meanings, applicable families, and exclusions. Capture before/after with identical input, tools, initialization, chunks, and GOLDEN; verify red before and green after. Other bug fixes: mark this subsection Not applicable and explain why parser behavior is unaffected. -->

### 5. Visual Conformance Verification (Parser Bug Fixes Only)

<!-- Parser bug fixes: regenerate conformance/CONFORMANCE_v2.html from the final candidate, open it in a browser, and inspect the affected columns and input/output popups. Record versions, families, case IDs, and before/after results; confirm zero red and zero empty current Dynamo cells for the affected family. Do not commit generated HTML/JSON. Other bug fixes: mark this subsection Not applicable and explain why parser behavior is unaffected. -->

## Validation

<!-- List commands and results, including conformance capture/status checks when applicable. Do not claim verification that was not performed. -->

## Where should the reviewer start?

<!-- List the main files to review. -->

See the [required PR description format](https://github.com/ai-dynamo/frontend-crates/blob/main/CONTRIBUTING.md#required-pr-description-format).
