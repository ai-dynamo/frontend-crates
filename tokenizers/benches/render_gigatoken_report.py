#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract Criterion results and render the Gigatoken cross-model report."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGETS = [128, 512, 2048, 8192, 32768]
BASELINE = "cross-model-20260723"

MODELS = {
    "qwen3-0.6b": {
        "display": "Qwen3-0.6B",
        "repo": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "artifact": "tokenizer.json",
        "sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "reference": "huggingface",
        "scheme": "Qwen2/Qwen3 byte-level BPE",
        "color": "#66e3ff",
    },
    "kimi-k2.6": {
        "display": "Kimi-K2.6",
        "repo": "moonshotai/Kimi-K2.6",
        "revision": "7eb5002f6aadc958aed6a9177b7ed26bb94011bb",
        "artifact": "tiktoken.model",
        "sha256": "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
        "reference": "tiktoken",
        "scheme": "Kimi rank-file BPE",
        "color": "#b58cff",
    },
    "glm-5.2": {
        "display": "GLM-5.2",
        "repo": "zai-org/GLM-5.2",
        "revision": "b4734de4facf877f85769a911abafc5283eab3d9",
        "artifact": "tokenizer.json",
        "sha256": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
        "reference": "huggingface",
        "scheme": "3-digit byte-level BPE",
        "color": "#7ee787",
    },
    "deepseek-v4": {
        "display": "DeepSeek-V4 Pro",
        "repo": "deepseek-ai/DeepSeek-V4-Pro",
        "revision": "b5968e9190ef611bbf34a7229255be88a0e937c1",
        "artifact": "tokenizer.json",
        "sha256": "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf",
        "reference": "huggingface",
        "scheme": "DeepSeek-V3/V4 split BPE",
        "color": "#ffb86b",
    },
    "minimax-m3": {
        "display": "MiniMax-M3",
        "repo": "MiniMaxAI/MiniMax-M3",
        "revision": "f0e1c1e04d40177e4673a22097036854f536e9c0",
        "artifact": "tokenizer.json",
        "sha256": "bb1f1626cf01448f1e3b6036d0a061ffc66c91d9046aada14ea23a5441b5ad6e",
        "reference": "huggingface",
        "scheme": "MiniMax o200k-family BPE",
        "color": "#ff7ca8",
    },
}

FASTOKEN_GAPS = {
    ("kimi-k2.6", 128): "No official tokenizer.json",
    ("kimi-k2.6", 512): "No official tokenizer.json",
    ("kimi-k2.6", 2048): "No official tokenizer.json",
    ("kimi-k2.6", 8192): "No official tokenizer.json",
    ("kimi-k2.6", 32768): "No official tokenizer.json",
    ("glm-5.2", 8192): "Fastokens PCRE2 panic",
    ("deepseek-v4", 8192): "Fastokens PCRE2 panic",
    ("deepseek-v4", 32768): "Fastokens PCRE2 panic",
}


def extract_results(criterion_root: Path, baseline: str) -> list[dict]:
    results: list[dict] = []
    pattern = f"tokenizer_steady_state_*/*/*/{baseline}/estimates.json"
    for path in criterion_root.glob(pattern):
        relative = path.relative_to(criterion_root)
        group, backend, target, _, _ = relative.parts
        model = group.removeprefix("tokenizer_steady_state_")
        if model not in MODELS:
            continue
        estimate = json.loads(path.read_text())["median"]
        median_ns = estimate["point_estimate"]
        interval = estimate["confidence_interval"]
        tokens = int(target)
        results.append(
            {
                "model": model,
                "backend": backend,
                "tokens": tokens,
                "median_ns": median_ns,
                "ci95_lower_ns": interval["lower_bound"],
                "ci95_upper_ns": interval["upper_bound"],
                "mtok_per_s": tokens * 1000.0 / median_ns,
            }
        )
    return sorted(
        results,
        key=lambda row: (
            list(MODELS).index(row["model"]),
            row["tokens"],
            row["backend"],
        ),
    )


LOAD_RE = re.compile(
    r"model=(?P<model>\S+).*?reference=(?P<reference>\S+) "
    r"reference_load_ms=(?P<reference_ms>[\d.]+) "
    r"fastokens_load_ms=(?P<fastokens_ms>[\d.]+|unavailable) "
    r"gigatoken_load_ms=(?P<gigatoken_ms>[\d.]+)"
)


def extract_loads(log_path: Path) -> dict[str, dict[str, float | None]]:
    samples: dict[str, dict[str, list[float]]] = {
        model: {"reference": [], "fastokens": [], "gigatoken": []} for model in MODELS
    }
    for match in LOAD_RE.finditer(log_path.read_text()):
        model = match["model"]
        if model not in samples:
            continue
        samples[model]["reference"].append(float(match["reference_ms"]))
        if match["fastokens_ms"] != "unavailable":
            samples[model]["fastokens"].append(float(match["fastokens_ms"]))
        samples[model]["gigatoken"].append(float(match["gigatoken_ms"]))
    return {
        model: {
            backend: statistics.median(values) if values else None
            for backend, values in backends.items()
        }
        for model, backends in samples.items()
    }


def index_results(results: list[dict]) -> dict[tuple[str, int, str], dict]:
    return {
        (row["model"], row["tokens"], row["backend"]): row for row in results
    }


def build_matrix(results: list[dict]) -> list[dict]:
    indexed = index_results(results)
    matrix = []
    for model, metadata in MODELS.items():
        reference_name = metadata["reference"]
        for tokens in TARGETS:
            reference = indexed[(model, tokens, reference_name)]
            gigatoken = indexed[(model, tokens, "gigatoken")]
            fastokens = indexed.get((model, tokens, "fastokens"))
            matrix.append(
                {
                    "model": model,
                    "tokens": tokens,
                    "reference_backend": reference_name,
                    "reference": reference,
                    "fastokens": fastokens,
                    "gigatoken": gigatoken,
                    "gigatoken_vs_reference": (
                        reference["median_ns"] / gigatoken["median_ns"]
                    ),
                    "gigatoken_vs_fastokens": (
                        fastokens["median_ns"] / gigatoken["median_ns"]
                        if fastokens
                        else None
                    ),
                    "fastokens_gap": FASTOKEN_GAPS.get((model, tokens)),
                }
            )
    return matrix


def summary(matrix: list[dict]) -> dict:
    gigatoken_rates = [row["gigatoken"]["mtok_per_s"] for row in matrix]
    reference_speedups = [row["gigatoken_vs_reference"] for row in matrix]
    fastokens_speedups = [
        row["gigatoken_vs_fastokens"]
        for row in matrix
        if row["gigatoken_vs_fastokens"] is not None
    ]
    return {
        "models": len(MODELS),
        "sequence_lengths": len(TARGETS),
        "measured_backend_cells": sum(
            2 + (row["fastokens"] is not None) for row in matrix
        ),
        "gigatoken_mtok_per_s_min": min(gigatoken_rates),
        "gigatoken_mtok_per_s_max": max(gigatoken_rates),
        "gigatoken_vs_reference_min": min(reference_speedups),
        "gigatoken_vs_reference_max": max(reference_speedups),
        "gigatoken_vs_fastokens_min": min(fastokens_speedups),
        "gigatoken_vs_fastokens_max": max(fastokens_speedups),
    }


def fmt_rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def fmt_ratio(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}×"


def fmt_tokens(tokens: int) -> str:
    return f"{tokens:,}"


def line_chart(matrix: list[dict]) -> str:
    width, height = 920, 350
    left, right, top, bottom = 72, 26, 28, 54
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_rate = max(row["gigatoken"]["mtok_per_s"] for row in matrix)
    y_max = math.ceil(max_rate / 10.0) * 10
    x_positions = {
        target: left + index * plot_width / (len(TARGETS) - 1)
        for index, target in enumerate(TARGETS)
    }

    elements = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Gigatoken throughput by model and sequence length">'
    ]
    for value in range(0, y_max + 1, 20):
        y = top + plot_height - value / y_max * plot_height
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" '
            'class="gridline"/>'
        )
        elements.append(
            f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" '
            f'class="axis">{value}</text>'
        )
    for target in TARGETS:
        x = x_positions[target]
        elements.append(
            f'<text x="{x:.1f}" y="{height-22}" text-anchor="middle" '
            f'class="axis">{fmt_tokens(target)}</text>'
        )

    by_key = index_results([row["gigatoken"] for row in matrix])
    for model, metadata in MODELS.items():
        points = []
        for target in TARGETS:
            rate = by_key[(model, target, "gigatoken")]["mtok_per_s"]
            x = x_positions[target]
            y = top + plot_height - rate / y_max * plot_height
            points.append(f"{x:.1f},{y:.1f}")
        color = metadata["color"]
        elements.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
            'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        for point in points:
            x, y = point.split(",")
            elements.append(
                f'<circle cx="{x}" cy="{y}" r="4.5" fill="{color}" '
                'stroke="#091019" stroke-width="2"/>'
            )
    elements.append(
        '<text x="18" y="185" transform="rotate(-90 18 185)" '
        'text-anchor="middle" class="axis-title">Million tokens / second</text>'
    )
    elements.append("</svg>")
    return "".join(elements)


def report_html(payload: dict) -> str:
    matrix = payload["matrix"]
    summary_data = payload["summary"]
    loads = payload["load_medians_ms"]

    legend = "".join(
        f'<span><i style="background:{metadata["color"]}"></i>'
        f'{html.escape(metadata["display"])}</span>'
        for metadata in MODELS.values()
    )

    model_rows = []
    for model, metadata in MODELS.items():
        rows = [row for row in matrix if row["model"] == model]
        giga_rates = [row["gigatoken"]["mtok_per_s"] for row in rows]
        ref_speedups = [row["gigatoken_vs_reference"] for row in rows]
        fast_speedups = [
            row["gigatoken_vs_fastokens"]
            for row in rows
            if row["gigatoken_vs_fastokens"] is not None
        ]
        fast_text = (
            f"{min(fast_speedups):.1f}–{max(fast_speedups):.1f}×"
            if fast_speedups
            else "not available"
        )
        model_rows.append(
            f"""
            <article class="model-card" style="--accent:{metadata['color']}">
              <div class="model-head">
                <h3>{html.escape(metadata['display'])}</h3>
                <span>{html.escape(metadata['scheme'])}</span>
              </div>
              <div class="model-stats">
                <div><strong>{min(giga_rates):.1f}–{max(giga_rates):.1f}</strong>
                  <small>Gigatoken Mtok/s</small></div>
                <div><strong>{min(ref_speedups):.1f}–{max(ref_speedups):.1f}×</strong>
                  <small>vs {metadata['reference']}</small></div>
                <div><strong>{fast_text}</strong><small>vs Fastokens</small></div>
              </div>
            </article>
            """
        )

    detail_rows = []
    for row in matrix:
        metadata = MODELS[row["model"]]
        fast = row["fastokens"]
        if fast:
            fast_cell = f'{fast["mtok_per_s"]:.2f}'
            fast_ratio = fmt_ratio(row["gigatoken_vs_fastokens"])
        else:
            reason = html.escape(row["fastokens_gap"] or "Unavailable")
            fast_cell = f'<span class="missing" title="{reason}">n/a</span>'
            fast_ratio = "—"
        detail_rows.append(
            f"""
            <tr>
              <td><span class="model-dot" style="background:{metadata['color']}"></span>
                {html.escape(metadata['display'])}</td>
              <td class="num">{fmt_tokens(row['tokens'])}</td>
              <td class="num">{row['reference']['mtok_per_s']:.2f}</td>
              <td class="num">{fast_cell}</td>
              <td class="num giga">{row['gigatoken']['mtok_per_s']:.2f}</td>
              <td class="num ratio">{fmt_ratio(row['gigatoken_vs_reference'])}</td>
              <td class="num ratio">{fast_ratio}</td>
            </tr>
            """
        )

    availability_rows = []
    for model, metadata in MODELS.items():
        cells = []
        for target in TARGETS:
            gap = FASTOKEN_GAPS.get((model, target))
            cells.append(
                '<td class="bad">fail</td>'
                if gap and "panic" in gap.lower()
                else '<td class="na">n/a</td>'
                if gap
                else '<td class="ok">pass</td>'
            )
        availability_rows.append(
            f"<tr><td>{html.escape(metadata['display'])}</td>{''.join(cells)}</tr>"
        )

    load_rows = []
    for model, metadata in MODELS.items():
        values = loads[model]
        load_rows.append(
            f"""
            <tr><td>{html.escape(metadata['display'])}</td>
              <td class="num">{values['reference']:.1f}</td>
              <td class="num">{fmt_rate(values['fastokens'])}</td>
              <td class="num">{values['gigatoken']:.1f}</td></tr>
            """
        )

    speedup_rows = []
    max_speedup = max(
        row["gigatoken_vs_reference"] for row in matrix if row["tokens"] == 32768
    )
    for row in (row for row in matrix if row["tokens"] == 32768):
        metadata = MODELS[row["model"]]
        speedup = row["gigatoken_vs_reference"]
        width = speedup / max_speedup * 100
        speedup_rows.append(
            f"""
            <div class="bar-row">
              <span>{html.escape(metadata['display'])}</span>
              <div class="bar-track"><i style="width:{width:.1f}%;background:{metadata['color']}"></i></div>
              <strong>{speedup:.1f}×</strong>
            </div>
            """
        )

    artifact_rows = []
    for metadata in MODELS.values():
        artifact_rows.append(
            f"""
            <tr>
              <td><a href="https://huggingface.co/{metadata['repo']}">{html.escape(metadata['repo'])}</a></td>
              <td class="mono">{metadata['revision'][:12]}</td>
              <td>{metadata['artifact']}</td>
              <td class="mono">{metadata['sha256'][:16]}…</td>
            </tr>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gigatoken cross-model tokenizer benchmark</title>
  <style>
    :root {{
      --bg:#080d14; --panel:#101823; --panel2:#141f2d; --line:#273648;
      --text:#eef5ff; --muted:#9cafc5; --cyan:#66e3ff; --green:#7ee787;
      --amber:#ffb86b; --red:#ff7188; --purple:#b58cff;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; color:var(--text); background:
      radial-gradient(circle at 80% -10%,#18334b 0,transparent 30%),
      radial-gradient(circle at -10% 25%,#25183c 0,transparent 25%),var(--bg);
      font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    }}
    main {{ max-width:1180px; margin:auto; padding:54px 28px 80px; }}
    a {{ color:var(--cyan); text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
    h1 {{ font-size:clamp(38px,6vw,72px); line-height:.98; margin:16px 0 22px; letter-spacing:-.045em; }}
    h2 {{ font-size:27px; letter-spacing:-.025em; margin:0 0 8px; }}
    h3 {{ margin:0; }}
    p {{ color:var(--muted); }}
    .eyebrow {{ color:var(--cyan); text-transform:uppercase; letter-spacing:.18em; font-size:12px; font-weight:800; }}
    .lede {{ max-width:790px; font-size:18px; }}
    .verdict {{
      margin:34px 0; padding:18px 22px; border:1px solid #2e5770; border-left:4px solid var(--cyan);
      background:linear-gradient(90deg,#102738,#101823); border-radius:10px;
    }}
    .verdict strong {{ color:var(--cyan); text-transform:uppercase; letter-spacing:.1em; font-size:12px; }}
    .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:28px 0 36px; }}
    .kpi,.panel,.model-card {{
      background:linear-gradient(145deg,rgba(20,31,45,.96),rgba(13,21,31,.96));
      border:1px solid var(--line); border-radius:14px; box-shadow:0 18px 45px rgba(0,0,0,.18);
    }}
    .kpi {{ padding:20px; }}
    .kpi strong {{ display:block; font-size:27px; line-height:1.1; color:var(--cyan); }}
    .kpi span {{ display:block; color:var(--muted); margin-top:7px; font-size:12px; }}
    .panel {{ padding:24px; margin:18px 0; overflow:hidden; }}
    .grid.two {{ display:grid; grid-template-columns:1.25fr .75fr; gap:18px; }}
    .chart {{ margin-top:16px; }}
    .gridline {{ stroke:#243346; stroke-width:1; }}
    .axis {{ fill:#91a4ba; font-size:12px; }}
    .axis-title {{ fill:#91a4ba; font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
    .legend {{ display:flex; gap:18px; flex-wrap:wrap; color:var(--muted); font-size:12px; }}
    .legend i {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px; }}
    .models {{ display:grid; grid-template-columns:repeat(2,1fr); gap:14px; margin:18px 0; }}
    .model-card {{ padding:19px; border-top:3px solid var(--accent); }}
    .model-head {{ display:flex; justify-content:space-between; gap:12px; align-items:baseline; }}
    .model-head span {{ color:var(--muted); font-size:12px; }}
    .model-stats {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:18px; }}
    .model-stats strong {{ display:block; font-size:19px; }}
    .model-stats small {{ color:var(--muted); }}
    table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
    th {{ color:#aabbd0; font-size:11px; text-transform:uppercase; letter-spacing:.07em; text-align:left; }}
    th,td {{ padding:11px 10px; border-bottom:1px solid #223043; }}
    tbody tr:hover {{ background:#172332; }}
    .num {{ text-align:right; }}
    .giga,.ratio {{ color:var(--cyan); font-weight:700; }}
    .model-dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:8px; }}
    .missing,.bad {{ color:var(--red); }}
    .ok {{ color:var(--green); }}
    .na {{ color:var(--muted); }}
    .availability td:not(:first-child) {{ text-align:center; font-weight:700; }}
    .bar-row {{ display:grid; grid-template-columns:125px 1fr 64px; gap:12px; align-items:center; margin:15px 0; }}
    .bar-row > span {{ color:var(--muted); }}
    .bar-track {{ height:12px; background:#223043; border-radius:8px; overflow:hidden; }}
    .bar-track i {{ display:block; height:100%; border-radius:8px; }}
    .bar-row strong {{ text-align:right; color:var(--cyan); }}
    .finding {{ padding:15px 0; border-bottom:1px solid #223043; }}
    .finding:last-child {{ border:0; }}
    .finding strong {{ display:block; margin-bottom:4px; }}
    .finding p {{ margin:0; }}
    code,.mono {{ font-family:"SFMono-Regular",Consolas,monospace; font-size:12px; }}
    code {{ color:#bfeeff; background:#0a111a; padding:2px 5px; border-radius:4px; }}
    .callout {{ border-left:3px solid var(--amber); padding:4px 0 4px 16px; margin:18px 0; }}
    .callout p {{ margin:3px 0; }}
    ul {{ color:var(--muted); padding-left:19px; }}
    footer {{ margin-top:36px; color:#70849b; font-size:12px; }}
    @media(max-width:850px) {{
      .kpis,.models,.grid.two {{ grid-template-columns:1fr 1fr; }}
      .grid.two {{ grid-template-columns:1fr; }}
      .kpis {{ grid-template-columns:1fr 1fr; }}
      .table-wrap {{ overflow-x:auto; }}
    }}
    @media(max-width:560px) {{
      main {{ padding:34px 16px 60px; }}
      .kpis,.models {{ grid-template-columns:1fr; }}
      .model-stats {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">Dynamo frontend-crates · CPU tokenizer study · 23 July 2026</div>
    <h1>Gigatoken across five<br>production tokenizer families</h1>
    <p class="lede">A pinned, single-logical-CPU microbenchmark of Hugging Face,
      Dynamo TikToken, Fastokens 0.2.0, and the experimental Gigatoken adapter at
      128 through 32,768 exact model tokens.</p>
  </header>

  <div class="verdict"><strong>Verdict · pursue a narrow upstream integration</strong>
    <div>Gigatoken wins every valid cell, but direct adoption remains blocked by
      nightly Cargo, dependency weight, packaging, and unqualified cold/concurrent behavior.</div>
  </div>

  <section class="kpis">
    <div class="kpi"><strong>{summary_data['gigatoken_mtok_per_s_min']:.1f}–{summary_data['gigatoken_mtok_per_s_max']:.1f}</strong>
      <span>Gigatoken Mtok/s across 25 model-length cells</span></div>
    <div class="kpi"><strong>{summary_data['gigatoken_vs_reference_min']:.1f}–{summary_data['gigatoken_vs_reference_max']:.1f}×</strong>
      <span>Speedup versus each official Dynamo reference backend</span></div>
    <div class="kpi"><strong>{summary_data['gigatoken_vs_fastokens_min']:.1f}–{summary_data['gigatoken_vs_fastokens_max']:.1f}×</strong>
      <span>Speedup versus valid Fastokens cells</span></div>
    <div class="kpi"><strong>{summary_data['measured_backend_cells']}</strong>
      <span>Measured backend cells; exact IDs checked before timing</span></div>
  </section>

  <section class="panel">
    <h2>Gigatoken throughput separates by tokenizer scheme</h2>
    <p>Hot steady-state median. Each length ran in a fresh process; lines connect
      independent measurements, not a single accumulating tokenizer cache.</p>
    <div class="legend">{legend}</div>
    <div class="chart">{line_chart(matrix)}</div>
  </section>

  <section class="models">{''.join(model_rows)}</section>

  <section class="grid two">
    <div class="panel">
      <h2>32K speedup over official reference</h2>
      <p>Hugging Face for JSON tokenizers; Dynamo TikToken for Kimi.</p>
      {''.join(speedup_rows)}
    </div>
    <div class="panel">
      <h2>What changes across models</h2>
      <div class="finding"><strong>Qwen, Kimi, and MiniMax reach the top tier.</strong>
        <p>Their Gigatoken hot paths stabilize around 76–85 Mtok/s at long lengths.</p></div>
      <div class="finding"><strong>GLM settles near 58–65 Mtok/s.</strong>
        <p>Its 3-digit pretokenizer costs more, but scaling stays linear and predictable.</p></div>
      <div class="finding"><strong>DeepSeek-V4 is the hardest scheme.</strong>
        <p>It plateaus near 42–45 Mtok/s, still 7–11× faster than valid Fastokens cells.</p></div>
    </div>
  </section>

  <section class="panel">
    <h2>Complete measured matrix</h2>
    <p>Million tokens per second from Criterion median point estimates. Confidence
      intervals are retained in the JSON artifact.</p>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Model</th><th class="num">Tokens</th><th class="num">Reference</th>
          <th class="num">Fastokens</th><th class="num">Gigatoken</th>
          <th class="num">Giga / ref</th><th class="num">Giga / fast</th></tr></thead>
        <tbody>{''.join(detail_rows)}</tbody>
      </table>
    </div>
  </section>

  <section class="grid two">
    <div class="panel">
      <h2>Fastokens compatibility matrix</h2>
      <p>Pre-timing exact-ID validation in a fresh model/length process.</p>
      <table class="availability">
        <thead><tr><th>Model</th>{''.join(f'<th>{fmt_tokens(t)}</th>' for t in TARGETS)}</tr></thead>
        <tbody>{''.join(availability_rows)}</tbody>
      </table>
      <div class="callout"><strong>One bug, several shapes.</strong>
        <p>GLM 8K and DeepSeek 8K/32K panic at Fastokens
          <code>split.rs:419</code>. GLM 32K passes, proving the fault is sensitive
          to tokenizer/input/cache state rather than monotonic length.</p></div>
    </div>
    <div class="panel">
      <h2>Fresh-process construction</h2>
      <p>Median milliseconds across the five per-length processes. This is object
        construction only—not first-encode latency.</p>
      <table><thead><tr><th>Model</th><th class="num">Reference</th>
        <th class="num">Fastokens</th><th class="num">Gigatoken</th></tr></thead>
        <tbody>{''.join(load_rows)}</tbody></table>
      <p>Kimi’s official rank-file path avoids Hugging Face and Fastokens entirely.
        Gigatoken construction remains 2–3× slower than its Dynamo TikToken reference.</p>
    </div>
  </section>

  <section class="panel">
    <h2>Prototype integration</h2>
    <div class="grid two">
      <div>
        <p><strong>JSON path</strong></p>
        <p><code>tokenizer.json → Gigatoken BPE encoder → Dynamo Encoding</code><br>
          <code>token IDs → Hugging Face decoder</code></p>
      </div>
      <div>
        <p><strong>Kimi path</strong></p>
        <p><code>tiktoken.model + tokenizer_config → Gigatoken Kimi encoder</code><br>
          <code>token IDs → Dynamo TikToken decoder</code></p>
      </div>
    </div>
    <ul>
      <li>Gigatoken matched official reference IDs for all 175 timed text inputs:
        five models × five lengths × seven rotations.</li>
      <li>The adapter now supports both Hugging Face byte-level BPE JSON and native
        rank-per-line TikToken models with an explicit pretokenizer scheme.</li>
      <li>Decode remains on the established Dynamo backend; these results measure encode only.</li>
      <li>Config-only special tokens are not imported into Gigatoken yet; timed inputs
        avoid that unresolved parity surface.</li>
      <li>Prefix-cache wrapping is still conservatively rejected pending boundary qualification.</li>
    </ul>
  </section>

  <section class="grid two">
    <div class="panel">
      <h2>Findings</h2>
      <div class="finding"><strong>Gigatoken’s advantage is robust, not Qwen-specific.</strong>
        <p>Every model and length is materially faster than its official reference.</p></div>
      <div class="finding"><strong>Model choice changes absolute throughput by ~2×.</strong>
        <p>At 32K, DeepSeek runs at {next(r['gigatoken']['mtok_per_s'] for r in matrix if r['model']=='deepseek-v4' and r['tokens']==32768):.1f}
          Mtok/s versus MiniMax at {next(r['gigatoken']['mtok_per_s'] for r in matrix if r['model']=='minimax-m3' and r['tokens']==32768):.1f}.</p></div>
      <div class="finding"><strong>Fastokens is much closer to Gigatoken than HF, when it works.</strong>
        <p>The valid-cell gap is {summary_data['gigatoken_vs_fastokens_min']:.1f}–{summary_data['gigatoken_vs_fastokens_max']:.1f}×,
          compared with {summary_data['gigatoken_vs_reference_min']:.1f}–{summary_data['gigatoken_vs_reference_max']:.1f}× over references.</p></div>
      <div class="finding"><strong>Kimi is the strongest relative result.</strong>
        <p>Native Gigatoken rank-file encoding is roughly 48–73× faster than Dynamo TikToken.</p></div>
    </div>
    <div class="panel">
      <h2>Gaps and limitations</h2>
      <ul>
        <li>Hot cache only: parity checks warm pretoken/BPE caches before timing.</li>
        <li>Single ASCII-heavy source corpus; multilingual checks are probes, not a performance matrix.</li>
        <li>One logical CPU, powersave governor, shared host; no architecture or NUMA sweep.</li>
        <li>Fixed backend order is reference → Fastokens → Gigatoken inside each process.</li>
        <li>No batch-size, concurrency, prefix-cache, cold-unique, or end-to-end frontend measurements.</li>
        <li>No per-backend RSS isolation; construction time does not include first encode.</li>
        <li>Gigatoken still requires nightly <code>portable_simd</code>, unstable Cargo
          <code>profile-rustflags</code>, and a large Python/data dependency closure.</li>
        <li>The git dependency is not crates.io-packageable as currently specified.</li>
      </ul>
    </div>
  </section>

  <section class="panel">
    <h2>Methodology</h2>
    <ul>
      <li>Intel Core i7-7800X; process and all worker threads pinned to logical CPU 5;
        <code>RAYON_NUM_THREADS=1</code>.</li>
      <li>Criterion 0.8.2: 3 s warmup, 7 s target measurement, 60 samples, 95% bootstrap CI.</li>
      <li>Fresh process per model/length; release binary built once before timing.</li>
      <li>Seven rotating inputs from ten checked-in frontend-crates source files, ASCII-filtered.</li>
      <li>Inputs are produced by reference encode → exact token prefix → decode → exact re-encode.</li>
      <li>Every present candidate backend must return the same IDs for all seven inputs before timing.</li>
      <li>Reported throughput derives from exact target tokens divided by Criterion median nanoseconds.</li>
    </ul>
  </section>

  <section class="panel">
    <h2>Pinned tokenizer artifacts</h2>
    <p>DeepSeek-V4 Pro is the representative V4 artifact; Pro and Flash resolved to
      the same <code>tokenizer.json</code> SHA-256 during preflight.</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Repository</th><th>Revision</th><th>Artifact</th><th>SHA-256</th></tr></thead>
      <tbody>{''.join(artifact_rows)}</tbody>
    </table></div>
  </section>

  <section class="panel">
    <h2>Reproduce</h2>
    <p>Build once, then run the resulting Criterion binary in a fresh process for
      each model/length combination:</p>
    <p><code>CARGO_UNSTABLE_PROFILE_RUSTFLAGS=true cargo +nightly build --release
      -p dynamo-tokenizers --features gigatoken --bench gigatoken_models</code></p>
    <p><code>taskset -c 5 env RAYON_NUM_THREADS=1
      TOKENIZER_BENCH_MODEL=&lt;model&gt; TOKENIZER_BENCH_TARGET=&lt;tokens&gt;
      TOKENIZER_PATH=&lt;artifact&gt; &lt;bench-binary&gt; --bench --noplot
      --save-baseline {BASELINE}</code></p>
    <p><code>python3 tokenizers/benches/render_gigatoken_report.py</code></p>
  </section>

  <section class="panel">
    <h2>Recommendation</h2>
    <p>Do not merge the upstream crate directly. The cross-model result is strong enough
      to justify an upstreamable tokenization-only core: stable Rust, optional Python/data
      features, explicit HF/TikToken loaders, and a packageable release. Then qualify cold
      unique prompts, mixed-language/chat/tool corpora, batch/concurrency scaling, memory,
      prefix-cache boundaries, and full Dynamo frontend CPU/request behavior. Fix the
      Fastokens PCRE2 chunk-count bug independently before using its long-input results.</p>
  </section>

  <footer>
    Source branch <code>rmccormick/gigatoken</code> · frontend-crates base
    <code>186268a1b127</code> · Gigatoken <code>ecf968da2b73</code> ·
    structured results: <a href="gigatoken-tokenizer-results.json">gigatoken-tokenizer-results.json</a>
  </footer>
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--criterion-root", type=Path, default=ROOT / "target" / "criterion"
    )
    parser.add_argument(
        "--log", type=Path, default=Path("/tmp/gigatoken-cross-model-20260723.log")
    )
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument(
        "--json-output", type=Path, default=ROOT / "gigatoken-tokenizer-results.json"
    )
    parser.add_argument(
        "--html-output", type=Path, default=ROOT / "gigatoken-tokenizer-report.html"
    )
    args = parser.parse_args()

    results = extract_results(args.criterion_root, args.baseline)
    if len(results) != 67:
        raise SystemExit(f"expected 67 measured backend cells, found {len(results)}")
    matrix = build_matrix(results)
    payload = {
        "schema_version": 2,
        "benchmark_date": "2026-07-23",
        "branch": "rmccormick/gigatoken",
        "frontend_crates_base": "186268a1b127c930c3480f767ffeedd7e0450ddb",
        "gigatoken_revision": "ecf968da2b7300e33f90e8bd9c96a11a335a01ae",
        "environment": {
            "cpu": "Intel Core i7-7800X @ 3.50GHz",
            "logical_cpu": 5,
            "rayon_threads": 1,
            "kernel": "6.8.0-106-generic",
            "governor": "powersave",
            "host": "shared",
        },
        "methodology": {
            "warmup_seconds": 3,
            "measurement_target_seconds": 7,
            "samples": 60,
            "working_set_rotations": 7,
            "sequence_lengths": TARGETS,
            "statistic": "Criterion median point estimate with 95% bootstrap CI",
            "cache_state": "hot after exact-ID parity validation",
            "process_isolation": "fresh process per model and sequence length",
        },
        "models": MODELS,
        "load_medians_ms": extract_loads(args.log),
        "results": results,
        "matrix": matrix,
        "summary": summary(matrix),
        "known_failures": [
            {
                "backend": "fastokens",
                "model": model,
                "tokens": tokens,
                "reason": reason,
            }
            for (model, tokens), reason in FASTOKEN_GAPS.items()
        ],
    }
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
    html = "\n".join(line.rstrip() for line in report_html(payload).splitlines()) + "\n"
    args.html_output.write_text(html)

    print(f"wrote {args.json_output}")
    print(f"wrote {args.html_output}")
    print(
        "sha256 "
        f"{hashlib.sha256(args.html_output.read_bytes()).hexdigest()} "
        f"{args.html_output.name}"
    )


if __name__ == "__main__":
    main()
