# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixture loading and sub-case taxonomy for the conformance table (audit B5).

Owns reading the per-family YAML fixtures, the batch/stream sub-case group tables,
split-parent normalization, peer-availability derivation, and the parser-family
inheritance/rust-ref maps. Comparison/marker semantics live in markers.py; this
module depends only on markers + impls + the on-disk fixtures, never on the HTML
rendering code.
"""
import copy
import json
import re
from pathlib import Path

import yaml

from impls import IMPL_DISPLAY, IMPL_KEYS, PEER_IMPL_KEYS
from markers import (
    VLLM_RUST_UNAVAILABLE,
    _canonical_impl_key,
    _impl_get,
    _normalize_impl_mapping,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests/parity/toolcalling/fixtures"
RUST_TOOL_CALLING_DIR = REPO_ROOT / "lib/parsers/src/tool_calling"


def _build_family_inheritance(
    refs: dict[str, tuple[str, int]],
) -> dict[str, dict]:
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
    cfg = (RUST_TOOL_CALLING_DIR / "config.rs").read_text()
    pars_path = RUST_TOOL_CALLING_DIR / "parsers.rs"
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


def _build_family_to_rust_ref() -> dict[str, tuple[str, int]]:
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

    config_rs = RUST_TOOL_CALLING_DIR / "config.rs"
    if config_rs.exists():
        pat = re.compile(r"^\s*pub fn (\w+)\(\)\s*->\s*Self\b")
        for lineno, line in enumerate(config_rs.read_text().splitlines(), 1):
            m = pat.match(line)
            if m:
                out[m.group(1)] = ("config.rs", lineno)

    parsers_rs = RUST_TOOL_CALLING_DIR / "parsers.rs"
    if parsers_rs.exists():
        pat = re.compile(r'^\s*map\.insert\("([^"]+)",\s*ToolCallConfig::')
        for lineno, line in enumerate(parsers_rs.read_text().splitlines(), 1):
            m = pat.match(line)
            if m and m.group(1) not in out:
                out[m.group(1)] = ("parsers.rs", lineno)

    return out


BATCH_SUB_CASE_GROUPS = [
    ("Single-call", ("1.a", "1.b", "1.c", "1.d")),
    ("Core", ("1", "3", "9", "9.a", "9.b")),
    ("Multi-call", ("2.a", "2.b", "2.c", "2.d", "2.e", "10")),
    (
        "Malformed / recovery",
        (
            "4.a",
            "4.b",
            "4.c",
            "4.d",
            "4.e",
            "4.f",
            "5.a",
            "5.b",
            "5.c",
            "5.d",
            "5.e",
            "5.f",
            "5.g",
        ),
    ),
    (
        "Args",
        (
            "6.a",
            "6.b",
            "6.c",
            "7.a",
            "7.b",
            "7.c",
            "7.d",
            "7.e",
            "7.f",
        ),
    ),
    ("Text interleaving", ("8.a", "8.b", "8.c", "8.d")),
    ("Unknown tools", ("13", "13.a", "13.c")),
    (
        "String contents",
        ("30", "30.a", "30.b", "30.c", "31", "31.a", "31.b"),
    ),
]

SPLIT_PARENT_SUBCASES = {
    # Once a taxonomy bucket has leaf cases, the matrix should render only the
    # leaves. Existing parent fixtures still carry useful expectations/reasons
    # for parser families that have not been rewritten to leaf IDs yet.
    "1": ("1.a",),
    "9": ("9.a",),
    "30": ("30.a", "30.b", "30.c"),
    "31": ("31.a", "31.b"),
    "13": ("13.a",),
}

# Streaming taxonomy: TOOLCALLING.streamv2.<batch#> mirrors the batch taxonomy
# 1:1 (streamv2.1 == batch.1, etc.), so the columns group exactly like the batch
# tab. Only sub-cases that actually exist in fixtures become columns; the rest
# fill in over time. Streaming-only cases with no batch analog use the >=50 band.
STREAM_SUB_CASE_GROUPS = BATCH_SUB_CASE_GROUPS + [
    ("Partial-token", ("50",)),
]

SUB_CASE_GROUPS_BY_MODE = {
    "batch": BATCH_SUB_CASE_GROUPS,
    "streamv2": STREAM_SUB_CASE_GROUPS,
}

_SUB_CASE_GROUP_KEY_BY_LABEL_BY_MODE = {
    mode: {
        label: re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        for label, _subs in groups
    }
    for mode, groups in SUB_CASE_GROUPS_BY_MODE.items()
}

_SUB_CASE_GROUP_KEY_BY_SUB_BY_MODE = {
    mode: {
        sub: _SUB_CASE_GROUP_KEY_BY_LABEL_BY_MODE[mode][label]
        for label, subs in groups
        for sub in subs
    }
    for mode, groups in SUB_CASE_GROUPS_BY_MODE.items()
}


def _display_order(mode: str) -> dict[str, tuple[int, int]]:
    return {
        sub: (group_idx, sub_idx)
        for group_idx, (_label, subs) in enumerate(SUB_CASE_GROUPS_BY_MODE[mode])
        for sub_idx, sub in enumerate(subs)
    }


def _group_index_by_sub(mode: str) -> dict[str, int]:
    return {
        sub: group_idx
        for group_idx, (_label, subs) in enumerate(SUB_CASE_GROUPS_BY_MODE[mode])
        for sub in subs
    }


def _group_by_sub(mode: str) -> dict[str, str]:
    return {sub: label for label, subs in SUB_CASE_GROUPS_BY_MODE[mode] for sub in subs}


def _natural_sub_sort_key(sub: str) -> tuple[int, str]:
    """`8.a` → (8, 'a'); `9` → (9, '')."""
    parts = sub.split(".")
    return (int(parts[0]), parts[1] if len(parts) > 1 else "")


def _sub_sort_key(mode: str, sub: str) -> tuple[int, int, int, str]:
    """Sort known cases by semantic display group, future cases naturally last."""
    display_order = _display_order(mode).get(sub)
    if display_order is not None:
        group_idx, sub_idx = display_order
        return (0, group_idx, sub_idx, "")
    num, suffix = _natural_sub_sort_key(sub)
    return (1, num, 0, suffix)


def _subcase_band_class(mode: str, sub: str) -> str:
    group_idx = _group_index_by_sub(mode).get(sub, len(SUB_CASE_GROUPS_BY_MODE[mode]))
    return f"case-band-{group_idx % 2}"


def _subcase_group_key(mode: str, sub: str) -> str:
    return _SUB_CASE_GROUP_KEY_BY_SUB_BY_MODE[mode].get(sub, "other")


def _discover_sub_cases(mode: str, cases: dict) -> list[str]:
    """Union of sub-case IDs across all loaded fixtures, in stable order."""
    return sorted(
        {sub for _fam, sub in cases.keys()}, key=lambda s: _sub_sort_key(mode, s)
    )


def _normalize_split_parent_cases(cases: dict) -> dict:
    """Render split taxonomy buckets as leaf cases only.

    Some older fixture files still define parent buckets such as
    `TOOLCALLING.batch.30`, while newer files define leaf buckets such as
    `TOOLCALLING.batch.30.a`. For display, parent+leaf duplication is confusing:
    once any leaf exists for a parent bucket, the table should show only the
    leaf columns. Parent entries are copied into missing leaf cells so their
    existing expectations or n/a reasons remain visible until the YAML itself
    is migrated.
    """
    all_subs = {sub for _fam, sub in cases.keys()}
    active_split_parents = {
        parent
        for parent, children in SPLIT_PARENT_SUBCASES.items()
        if any(child in all_subs for child in children)
    }
    if not active_split_parents:
        return cases

    normalized = dict(cases)
    families = {fam for fam, _sub in cases.keys()}
    for family in families:
        for parent in active_split_parents:
            parent_key = (family, parent)
            parent_case = normalized.get(parent_key)
            if parent_case is None:
                continue
            for child in SPLIT_PARENT_SUBCASES[parent]:
                child_key = (family, child)
                if child_key in normalized:
                    continue
                child_case = copy.deepcopy(parent_case)
                child_case["__case_id"] = f"TOOLCALLING.batch.{child}"
                child_case["__synthetic_from_case_id"] = parent_case.get("__case_id")
                normalized[child_key] = child_case
            del normalized[parent_key]
    return normalized


def _derive_no_peer_sets(cases: dict) -> tuple[set[str], set[str]]:
    """Families where every case marks the engine `unavailable`.

    Used to render the † (no vLLM Python peer) and § (no SGLang peer) footnote
    markers next to a family's name. A family qualifies when every case
    in every fixture file under that family has
    `expected.<impl>.unavailable: <reason>` recorded — i.e. the wrapper
    rejected the family for that parser in `capture_toolcalling_outputs.py`.
    """
    by_family: dict[str, list[dict]] = {}
    for (fam, _sub), case in cases.items():
        by_family.setdefault(fam, []).append(case)

    def all_unavail(fam_cases: list[dict], impl: str) -> bool:
        expected_cases = [c for c in fam_cases if isinstance(c.get("expected"), dict)]
        if not expected_cases:
            return False
        for c in expected_cases:
            block = _impl_get(c.get("expected") or {}, impl)
            if not isinstance(block, dict) or "unavailable" not in block:
                return False
        return True

    no_vllm = {fam for fam, cs in by_family.items() if all_unavail(cs, "vllm_python")}
    no_sglang = {fam for fam, cs in by_family.items() if all_unavail(cs, "sglang_python")}
    return no_vllm, no_sglang


def family_suffix(fam: str, no_vllm: set[str], no_sglang: set[str]) -> str:
    suff = ""
    if fam in no_vllm:
        suff += "†"
    if fam in no_sglang:
        suff += "§"
    return suff


# Engine versions the per-chunk stream data was captured against, keyed by mode.
# Populated by load_all_cases; read when rendering the panel footer.
_CAPTURED_WITH_BY_MODE: dict[str, dict[str, str]] = {}


def load_all_cases(mode: str) -> tuple[dict[tuple[str, str], dict], dict[str, str]]:
    """Load every fixture YAML for one parser mode.

    Returns `(cases, labels)`:
      cases  — `{(family, sub_case_id): case_data}`; each case dict gets
               `__fixture_path` (relative to this script) and `__case_id`
               annotations for the HTML renderer.
      labels — `{family: model_label}` collected from the fixtures' doc-level
               `model_label:` field. Falls back to the family ID if a fixture
               doesn't declare one.
    """
    cases: dict[tuple[str, str], dict] = {}
    labels: dict[str, str] = {}
    captured_with: dict[str, str] = {}
    script_dir = Path(__file__).resolve().parent
    for fp in sorted(FIXTURES.glob(f"*/TOOLCALLING.{mode}*.yaml")):
        doc = yaml.safe_load(fp.read_text())
        if doc.get("mode") != mode:
            continue
        family = doc["family"]
        rel = str(fp.relative_to(script_dir))
        if "model_label" in doc:
            labels.setdefault(family, doc["model_label"])
        for impl, ver in (doc.get("captured_with") or {}).items():
            captured_with.setdefault(_canonical_impl_key(str(impl)), str(ver))
        for cid, case in doc["cases"].items():
            case["__family"] = family
            sub = cid.replace(f"TOOLCALLING.{mode}.", "")
            case["__fixture_path"] = rel
            case["__case_id"] = cid
            # Stream fixtures use the per-chunk format: each chunk carries
            # `expected.<impl>` deltas, plus a case-level `unavailable` block.
            # Derive a case-level `expected` (assembled calls + normal_text per
            # impl, or {unavailable}) so the rest of the generator — which expects
            # the batch-style {calls, normal_text} shape — works unchanged.
            if mode == "streamv2":
                case["expected"] = _derive_stream_expected(case)
            elif isinstance(case.get("expected"), dict):
                case["expected"] = _normalize_impl_mapping(case["expected"])
            cases[(family, sub)] = case
    _CAPTURED_WITH_BY_MODE[mode] = captured_with
    if mode == "streamv2":
        _attach_streamv2_batch_expected(cases)
    return _normalize_split_parent_cases(cases), labels


def _attach_streamv2_batch_expected(cases: dict) -> None:
    """For each streamv2 case, attach `batch_expected[impl]` from the matching batch
    case (same family + sub), so the stream tab can color each engine's stream
    against its own batch (streamv2.<sub> mirrors batch.<sub>)."""
    batch_exp: dict[tuple[str, str], dict] = {}
    for fp in sorted(FIXTURES.glob("*/TOOLCALLING.batch*.yaml")):
        doc = yaml.safe_load(fp.read_text()) or {}
        if doc.get("mode") != "batch":
            continue
        family = doc["family"]
        for cid, c in (doc.get("cases") or {}).items():
            sub = cid.replace("TOOLCALLING.batch.", "")
            exp = c.get("expected")
            if isinstance(exp, dict):
                batch_exp[(family, sub)] = _normalize_impl_mapping(exp)
    for (family, sub), case in cases.items():
        if isinstance(case, dict):
            case["batch_expected"] = batch_exp.get((family, sub), {})


def _derive_stream_expected(case: dict) -> dict:
    """Assemble case-level {impl: {calls, normal_text}} (or {unavailable}) from
    the per-chunk `expected.<impl>` deltas and `normal_text.<impl>` fragments.

    Tool calls are reconstructed per index: the first delta carrying a `name`
    sets the call name; `arguments` fragments are concatenated and parsed as JSON
    (kept as a raw string if not valid JSON — e.g. a truncated body)."""
    unavailable = case.get("unavailable", {}) or {}
    chunks = case.get("chunks", []) or []
    derived: dict = {}
    unavailable = _normalize_impl_mapping(unavailable)
    for impl in IMPL_KEYS:
        if impl in unavailable:
            derived[impl] = {"unavailable": unavailable[impl]}
            continue
        has_chunk_data = any(
            impl in _normalize_impl_mapping((chunk.get("expected") or {}))
            or impl in _normalize_impl_mapping((chunk.get("normal_text") or {}))
            for chunk in chunks
            if isinstance(chunk, dict)
        )
        if impl == "vllm_rust" and not has_chunk_data:
            derived[impl] = {"unavailable": VLLM_RUST_UNAVAILABLE}
            continue
        # impl is not in `unavailable` → it was run for this case. Always emit a
        # {calls, normal_text} block, even if empty (emitting zero calls is a real
        # result that may diverge from another impl, not a "not applicable").
        names: dict[int, str] = {}
        args: dict[int, str] = {}
        order: list[int] = []
        normal = ""
        for chunk in chunks:
            chunk_expected = _normalize_impl_mapping(chunk.get("expected") or {})
            for d in _impl_get(chunk_expected, impl, []) or []:
                idx = d["index"]
                if idx not in order:
                    order.append(idx)
                if d.get("name") is not None:
                    names[idx] = names.get(idx, "") + d["name"]
                if d.get("arguments") is not None:
                    args[idx] = args.get(idx, "") + d["arguments"]
            nt = _impl_get(chunk.get("normal_text") or {}, impl)
            if nt:
                normal += nt
        calls = []
        for idx in order:
            raw = args.get(idx, "")
            try:
                parsed = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                parsed = raw
            calls.append({"name": names.get(idx, ""), "arguments": parsed})
        block = {"calls": calls, "normal_text": normal}
        # Peer divergence from Dynamo parser v2 streaming is captured
        # ground truth, not an un-triaged gap — text vs token streaming differ by
        # design. Tag peer blocks with a reason so the cell shows `S`/`V` (known
        # divergence), never `S_rs?`/`V_ps?` (research-needed). The per-chunk `expected`
        # in the fixture is the detailed evidence.
        if impl in PEER_IMPL_KEYS:
            block["explanation"] = (
                f"Captured from the {IMPL_DISPLAY[impl]} streaming parser. Streaming output differs "
                "from Dynamo parser v2 token-incremental behavior by design (text vs token "
                "streaming); see per-chunk `expected` in the fixture."
            )
        derived[impl] = block
    return derived
