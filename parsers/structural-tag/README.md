# Shared Kimi structural tags

`dynamo-structural-tag` owns the native Kimi K2 and K3 call grammars and channel envelopes used by both Dynamo parser generations. It uses `serde` and `serde_json` for grammar serialization, `regex` for schema patterns, and `anyhow` for validation errors.

Each parser adapter selects tools, resolves schema enforcement, and decides call cardinality before invoking the shared grammar. V1 re-exports the shared format types at its existing public paths. V2 translates the grammar nodes into its private wire types to apply argument-order options and compose reasoning or structured-response branches.

V1 preserves its existing Kimi policy: an omitted `strict` flag uses the declared argument schema. V2 uses its common policy: only `strict: true` or global strict mode selects the declared schema. V2 named choice permits exactly one call, regardless of `parallel_tool_calls`.

K2 supports Instruct and prompt-opened K2.5/K2.6 reasoning. Original K2-Thinking, which generates its own opening reasoning marker, remains unsupported by the structural-tag builder.

K3 shares the upstream grammar for typed arguments, required-property order, schema references, unions, and positive call indices. Unsupported schema shapes retain the upstream fallback behavior. Native calls allow omitted response markers and EOS directly after the tools channel. K3 structured final responses retain their native response-channel envelope.

The workspace release configuration includes this publishable crate through its defaults. Both consumers declare a path and a registry version, so the shared crate must be published before either consumer release.

## Validate changes

Run the parser and shared-format tests from the repository root:

```sh
cargo test --locked -p dynamo-parsers -p dynamo-parsers-v2 -p dynamo-structural-tag
```

Export the native samples and validate them with XGrammar:

```sh
KIMI_GRAMMAR_CASES=/tmp/kimi.json cargo test --locked -p dynamo-parsers-v2 --test kimi_structural_tag
uv run --with xgrammar==0.2.8 python parsers/v2/tests/check_kimi_xgrammar.py /tmp/kimi.json
```

The Rust test checks native parser output. The Python guard compiles the generated structural tags and checks accepted and rejected samples.

Verify the publishable archives and their dependency resolution:

```sh
cargo package --locked --allow-dirty -p dynamo-structural-tag -p dynamo-parsers -p dynamo-parsers-v2
```
