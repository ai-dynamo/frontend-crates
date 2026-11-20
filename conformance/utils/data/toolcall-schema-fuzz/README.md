# OpenRouter toolcall-schema-fuzz dataset

`wordings.json` is the unmodified version 6 realistic wording dataset from [OpenRouter's benchmark-harness repository](https://github.com/OpenRouterTeam/benchmark-harness/blob/fba7364d1996ceb4a1383e358fc36d32a1a428b6/src/benchmarks/toolcall-schema-fuzz/realistic/wordings.json). It contains 1,358 schema entries and 2,716 expected argument objects, with two argument objects per entry.

The dataset was created by OpenRouter and is synthetic. It is distributed under [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/); see the [source dataset attribution and license](https://github.com/OpenRouterTeam/benchmark-harness/blob/fba7364d1996ceb4a1383e358fc36d32a1a428b6/src/benchmarks/toolcall-schema-fuzz/README.md#source--license). Credit OpenRouter and retain the source and license links when redistributing this data.

No changes were made to `wordings.json`: its bytes match the pinned source revision exactly. `manifest.json` records that revision, the source path, the SHA-256 digest, and the expected dataset counts. The manifest and this attribution README are local additions. Preserve each entry's stable `key` when creating test case IDs.
