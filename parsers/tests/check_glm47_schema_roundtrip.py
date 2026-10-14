# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
import json
import sys

import xgrammar as xg
from jsonschema import Draft202012Validator


def accepts(compiled: object, wire: str) -> bool:
    matcher = xg.GrammarMatcher(compiled)
    return matcher.accept_string(wire) and matcher.is_completed()


def main() -> None:
    version = importlib.metadata.version("xgrammar")
    version_tuple = tuple(int(part) for part in version.split(".")[:3])
    compiler = xg.GrammarCompiler(xg.TokenizerInfo([bytes([i]) for i in range(256)]))
    checked = 0
    rejected = 0
    skipped = []
    failures = []
    for probe in json.load(sys.stdin):
        minimum = probe.get("xgrammar_min")
        if minimum and version_tuple < tuple(int(part) for part in minimum.split(".")):
            skipped.append({"name": probe["name"], "minimum": minimum})
            continue
        try:
            Draft202012Validator.check_schema(probe["schema"])
            Draft202012Validator(probe["schema"]).validate(probe["arguments"])
            compiled = compiler.compile_grammar(xg.Grammar.from_structural_tag(probe["tag"]))
            assert accepts(compiled, probe["wire"]), "valid wire rejected"
            for wire in probe.get("accept", []):
                assert accepts(compiled, wire), f"permissive wire rejected: {wire}"
            for wire in probe["reject"]:
                assert not accepts(compiled, wire), f"invalid wire accepted: {wire}"
                rejected += 1
            checked += 1
        except Exception as error:
            failures.append({"name": probe["name"], "error": str(error)})
    print(json.dumps({"xgrammar": version, "checked": checked, "rejected": rejected,
                      "skipped": skipped, "failures": failures}, ensure_ascii=False))
    assert checked > 0 and not failures, "schema roundtrip failed"


if __name__ == "__main__":
    main()
