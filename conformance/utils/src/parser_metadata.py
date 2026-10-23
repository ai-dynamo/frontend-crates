# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Index parser source references and inheritance independently of captures."""
import re
from pathlib import Path

def family_inheritance(refs: dict[str, tuple[str, int]], source_directory: Path) -> dict[str, dict]:
    """Derive each family's parser-inheritance map from config.rs + parsers.rs.

    Detects:
      • `ParserConfig::<Variant>(...)` — top-level backend variant
      • `JsonParserType::<Sub>`         — Json sub-dispatch (Basic / DeepseekV3 / DeepseekV31)
      • `Self::<factory>(...)`          — private factories (e.g. `deepseek_dsml`)
      • `map.insert("alias", ToolCallConfig::<family>())` — aliases (parsers.rs)

    Backend file is derived from the resolved (variant, sub_variant) tuple.
    Returns `{family: {variant, sub_variant, factory, backend_file,
    base_label, shared_with, aliases, filed_under_xml_misleading}}`.
    """
    cfg = (source_directory / "config.rs").read_text()
    pars_path = source_directory / "parsers.rs"
    pars = pars_path.read_text() if pars_path.exists() else ""

    # Extract all ctor bodies (pub fn + fn) — captures private factories too.
    ctor_pat = re.compile(
        r"^\s*(?:pub )?fn (\w+)\([^)]*\)\s*->\s*Self\s*\{", re.MULTILINE
    )
    bodies: dict[str, str] = {}
    for m in ctor_pat.finditer(cfg):
        start = m.end()
        depth, i = 1, start
        while i < len(cfg) and depth > 0:
            c = cfg[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        bodies[m.group(1)] = cfg[start : i - 1]

    def _classify(body: str) -> tuple[str | None, str | None, str | None]:
        vm = re.search(r"ParserConfig::(\w+)\b", body)
        variant = vm.group(1) if vm else None
        sub = None
        if variant == "Json":
            sm = re.search(r"JsonParserType::(\w+)\b", body)
            sub = sm.group(1) if sm else "Basic"
        fm = re.search(r"Self::(\w+)\(([^)]*)\)", body)
        factory = f"{fm.group(1)}({fm.group(2).strip()})" if fm else None
        return variant, sub, factory

    backend_file = {
        ("Json", "Basic"): "json/base_json_parser.rs",
        ("Json", "DeepseekV3"): "json/deepseek_v3_parser.rs",
        ("Json", "DeepseekV31"): "json/deepseek_v3_1_parser.rs",
        ("Xml", None): "xml/parser.rs",
        ("Pythonic", None): "pythonic/pythonic_parser.rs",
        ("Harmony", None): "harmony/harmony_parser.rs",
        ("Dsml", None): "dsml/parser.rs",
        ("Glm47", None): "xml/glm47_parser.rs",
        ("KimiK2", None): "xml/kimi_k2_parser.rs",
        ("MiniMaxM3", None): "xml/minimax_m3_parser.rs",
        ("Gemma4", None): "gemma4/parser.rs",
    }
    base_label = {
        ("Json", "Basic"): "base_json_parser (JsonParserType::Basic)",
        ("Json", "DeepseekV3"): "deepseek_v3_parser (JsonParserType::DeepseekV3)",
        ("Json", "DeepseekV31"): "deepseek_v3_1_parser (JsonParserType::DeepseekV31)",
        ("Xml", None): "xml::parser (shared XML base)",
        ("Pythonic", None): "pythonic::parser (standalone)",
        (
            "Harmony",
            None,
        ): "harmony::parser (standalone; partial reuse of base_json's try_repair_truncated_json)",
        ("Dsml", None): "dsml::parser (shared via deepseek_dsml() factory)",
        ("Glm47", None): "glm47_parser (standalone; filed under xml/)",
        ("KimiK2", None): "kimi_k2_parser (standalone; filed under xml/)",
        ("MiniMaxM3", None): "minimax_m3_parser (standalone; filed under xml/)",
        ("Gemma4", None): "gemma4::parser (standalone)",
    }

    out: dict[str, dict] = {}
    for family in refs:
        body = bodies.get(family)
        if body is None:
            continue
        variant, sub, factory = _classify(body)
        if variant is None and factory:
            # Resolve through factory (e.g. deepseek_dsml)
            fbody = bodies.get(factory.split("(")[0])
            if fbody:
                variant, sub, _ = _classify(fbody)
        key = (variant, sub) if variant == "Json" else (variant, None)
        out[family] = {
            "variant": variant,
            "sub_variant": sub,
            "factory": factory,
            "backend_file": backend_file.get(key, "unknown"),
            "base_label": base_label.get(key, f"{variant}"),
            "key": key,
            "aliases": [],
            "shared_with": [],
            "filed_under_xml_misleading": False,
        }

    # Aliases from parsers.rs (only ones where alias name != family name).
    alias_pat = re.compile(r'map\.insert\("([^"]+)",\s*ToolCallConfig::(\w+)\(\)\)')
    alias_to_target: dict[str, str] = {}
    for m in alias_pat.finditer(pars):
        alias, fam = m.group(1), m.group(2)
        if alias != fam and fam in out:
            out[fam]["aliases"].append(alias)
            alias_to_target[alias] = fam

    # shared_with — other families with the same (variant, sub_variant).
    by_key: dict[tuple, list[str]] = {}
    for fam, info in out.items():
        by_key.setdefault(info["key"], []).append(fam)
    for fam, info in out.items():
        info["shared_with"] = [s for s in by_key[info["key"]] if s != fam]
        info["filed_under_xml_misleading"] = (
            info["backend_file"].startswith("xml/") and info["variant"] != "Xml"
        )

    # Synthesize entries for alias-only families (e.g. nemotron_nano, qwen25).
    # These are in `refs` (registered in parsers.rs) but have no ctor of their
    # own — the alias `map.insert("nemotron_nano", ToolCallConfig::qwen3_coder())`
    # routes to the target's config. The alias gets the target's full
    # inheritance tree, plus `alias_of` so the tooltip can mark itself as a
    # leaf under the target rather than as the target itself.
    for alias, target in alias_to_target.items():
        if alias in out or target not in out:
            continue
        tgt = out[target]
        out[alias] = {
            **tgt,
            "alias_of": target,
        }

    return out


def family_references(source_directory: Path) -> dict[str, tuple[str, int]]:
    """Scan the Rust source for each family's anchor point.

    Two patterns:
      `config.rs` :  `pub fn <family>() -> Self`               (parser config ctor)
      `parsers.rs`:  `map.insert("<family>", ToolCallConfig::...);`  (aliases)

    Config-ctor wins when the same family appears in both (the ctor is
    the canonical definition; the registration is just plumbing). Aliases
    (e.g. `nemotron_nano`, `qwen25`) only appear in `parsers.rs`.
    Returns `{family: (filename, line)}`; line is 1-indexed.
    """
    out: dict[str, tuple[str, int]] = {}

    config_rs = source_directory / "config.rs"
    if config_rs.exists():
        pat = re.compile(r"^\s*pub fn (\w+)\(\)\s*->\s*Self\b")
        for lineno, line in enumerate(config_rs.read_text().splitlines(), 1):
            m = pat.match(line)
            if m:
                out[m.group(1)] = ("config.rs", lineno)

    parsers_rs = source_directory / "parsers.rs"
    if parsers_rs.exists():
        pat = re.compile(r'^\s*map\.insert\("([^"]+)",\s*ToolCallConfig::')
        for lineno, line in enumerate(parsers_rs.read_text().splitlines(), 1):
            m = pat.match(line)
            if m and m.group(1) not in out:
                out[m.group(1)] = ("parsers.rs", lineno)

    return out




_SHARED_BACKEND_SHORT = {
    ("Json", "Basic"): "base_json",
    ("Xml", None): "xml",
    ("Dsml", None): "dsml",
}

_V2_STREAM_PARSER_CELLS: dict[str, tuple[str, str, str]] = {
    "gemma4": ("Gemma4ToolStreamParser text path", "gemma4.rs", "Gemma"),
    "glm47": ("Glm47ToolStreamParser text path", "glm47.rs", "GLM XML"),
    "kimi_k2": ("KimiK2ToolStreamParser text path", "kimi_k2.rs", "Kimi XML"),
    "minimax_m2": ("MiniMaxM2ToolStreamParser text path", "minimax_m2.rs", "MiniMax XML"),
    "minimax_m3": ("MiniMaxM3ToolStreamParser text path", "minimax_m3.rs", "MiniMax-M3 XML"),
    "qwen3_coder": ("Qwen3CoderToolStreamParser text path", "qwen3_coder.rs", "Qwen XML"),
}
