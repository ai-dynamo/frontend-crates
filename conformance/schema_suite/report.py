#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the pinned-main contracts, preserve evidence, and produce a standalone HTML report."""

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CATEGORIES = {
    "A": "Values & precision",
    "B": "Schema composition",
    "C": "References & scope",
    "D": "Containers & ownership",
    "E": "Native dialects",
    "F": "Streaming lifecycle",
    "G": "Grammar enforcement",
    "H": "Harness correctness",
}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(command, log, env=None, timeout=600):
    started = utc()
    before = time.monotonic()
    with log.open("w") as f:
        try:
            status = subprocess.run(
                command,
                cwd=ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout,
            ).returncode
        except subprocess.TimeoutExpired:
            status = 124
    return {
        "command": command,
        "started_utc": started,
        "ended_utc": utc(),
        "seconds": round(time.monotonic() - before, 3),
        "exit_code": status,
        "log": log.name,
    }


def render(out):
    metadata = json.loads((out / "metadata.json").read_text())
    results = [
        json.loads(line)
        for line in (out / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    catalogue = json.loads((out / "catalogue.json").read_text())
    counts = collections.Counter(r["status"] for r in results)
    group_status = {
        c["id"]: collections.Counter(
            r["status"] for r in results if r["group"] == c["id"]
        )
        for c in catalogue
    }
    metadata["counts"] = dict(counts)
    metadata["executions"] = sum(r.get("checks", 0) for r in results)
    metadata["failed_executions"] = sum(r.get("failed_checks", 0) for r in results)
    metadata["groups_with_failure"] = sum(
        any(v[k] for k in ["fail", "error", "not_run"]) for v in group_status.values()
    )
    metadata["groups_with_results"] = sum(bool(v) for v in group_status.values())
    payload = json.dumps(
        {
            "metadata": metadata,
            "results": results,
            "catalogue": catalogue,
            "categories": CATEGORIES,
        },
        ensure_ascii=False,
    ).replace("<", "\\u003c")
    page = HTML.replace("__DATA__", payload)
    (out / "report.html").write_text(page)
    (out / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Frontend parser schema correctness</title>
<style>
:root{color-scheme:light;--ink:#162126;--muted:#58676f;--border:#d8e0e3;--green:#236800;--red:#a82624;--bg:#f5f7f8}*{box-sizing:border-box}body{margin:0;font:14px/1.5 system-ui,sans-serif;color:var(--ink);background:var(--bg)}header{background:#172327;color:white;padding:32px max(24px,calc((100% - 1440px)/2));border-top:5px solid #76b900}header p{color:#cdd8db;max-width:1000px}h1{font-size:29px;margin:8px 0}h2{font-size:19px;margin:26px 0 12px}main{max-width:1488px;margin:auto;padding:0 24px 48px}a{color:#176192}header a{color:#b5dff3}code{font-family:ui-monospace,monospace;font-size:12px;overflow-wrap:anywhere}.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-weight:650;font-size:12px}.pass{background:#e5f2df;color:var(--green)}.fail,.error,.not_run{background:#fde8e7;color:var(--red)}.unavailable,.not_applicable{background:#edf0f2;color:#55616a}.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:22px 0}.card{background:white;border:1px solid var(--border);padding:18px;text-align:left;border-radius:7px;color:inherit;cursor:pointer}.card strong{display:block;font-size:30px}.card small{color:var(--muted)}.note{border-left:4px solid #83ae50;background:white;padding:14px 18px;margin:16px 0}.scroll{overflow:auto;background:white;border:1px solid var(--border);border-radius:6px}table{border-collapse:collapse;width:100%;text-align:left}th{background:#edf1f2;font-size:12px;text-transform:uppercase;letter-spacing:.04em}th,td{border-bottom:1px solid var(--border);padding:10px 12px;vertical-align:top}td.num{font-variant-numeric:tabular-nums}button,input,select{font:inherit}button{cursor:pointer}button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #277db1;outline-offset:2px}.matrix-button{border:0;background:none;padding:0;text-align:left;color:inherit}.filters{display:flex;flex-wrap:wrap;gap:10px;align-items:end;background:white;padding:14px;border:1px solid var(--border);border-radius:6px;position:sticky;top:0;z-index:1}label{display:flex;flex-direction:column;gap:4px;font-size:12px;font-weight:600}input,select{padding:8px;border:1px solid #adbcc3;border-radius:4px;background:white;max-width:260px}.details summary{cursor:pointer;color:#176192}.details pre{margin:6px 0 12px;background:#f5f7f8;border:1px solid #d8e0e3;border-radius:4px;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:430px;overflow:auto;font-size:12px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:12px}summary{font-weight:600}.subtle{color:var(--muted)}.pager{display:flex;gap:12px;align-items:center;padding:15px 0}.pager button,.reset{padding:7px 12px;background:white;border:1px solid #aebcc2;border-radius:4px}.count{font-size:12px;color:var(--muted)}.pill{padding:2px 6px;background:#eef1f3;border-radius:4px;font-size:11px}footer{margin-top:30px;color:var(--muted)}@media(max-width:900px){.cards{grid-template-columns:repeat(2,1fr)}.columns{grid-template-columns:1fr}header{padding:24px}main{padding:0 14px 30px}.filters{position:static}}
</style></head><body><header><div class="subtle" style="color:#a9c879">FRONTEND-CRATES · CPU CORRECTNESS BASELINE</div><h1>Schema and type coercion across parser generations</h1><p id="headline"></p><div id="revision"></div></header><main>
<div class="cards" id="cards"></div>
<div class="note"><strong>How to read these results.</strong> A row is one authored case on one parser surface. It can contain many chunking executions. Failure counts are not distinct bug counts. Unavailable implementations and non-applicable paths are excluded from the pass rate. Expectations are authored independently; current output is never used as the oracle. Production parser code is unchanged from the pinned commit. Corpus provenance: PRs <a href="https://github.com/ai-dynamo/frontend-crates/pull/215">#215</a>, #220, #223, #248, #251, #268–#271, #273–#275, #277, #280; the saved reproduction instructions contain the full group-to-PR mapping.</div>
<h2>Family × parser surface</h2><p class="subtle">Cells show passed / failed case rows. Click a cell to inspect its evidence. Native Unified is a separate implementation surface from v2 tool parsing.</p><div class="scroll"><table id="matrix"></table></div>
<details style="margin-top:24px"><summary id="catalogue-title">All 110 contract groups</summary><p class="subtle">Established regressions, explicit compatibility policies, and proposed correctness targets remain separately labeled. A red target is a missing capability, not automatically a regression introduced by main.</p><div class="scroll"><table id="catalogue"></table></div></details>
<h2>Case evidence</h2><div class="filters">
<label>Status<select id="status"><option value="">All statuses</option><option value="fail" selected>Fail</option><option value="pass">Pass</option><option value="error">Error</option><option value="unavailable">Unavailable</option><option value="not_applicable">Not applicable</option><option value="not_run">Not run</option></select></label>
<label>Family<select id="family"><option value="">All families</option></select></label><label>Surface<select id="surface"><option value="">All surfaces</option></select></label><label>Group<select id="group"><option value="">All groups</option></select></label><label>Search<input id="search" type="search" placeholder="Case ID, contract, or failure" autocomplete="off"></label><button class="reset" id="reset">Clear filters</button></div>
<div class="pager"><button id="prev">Previous</button><span id="page" aria-live="polite"></span><button id="next">Next</button><span id="matches" class="count"></span></div>
<div class="scroll"><table><thead><tr><th>Case</th><th>Family / surface</th><th>Result</th><th>Checks</th><th>Evidence</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2>Reproducibility and run record</h2><div class="scroll"><table id="runs"></table></div><details style="margin-top:16px"><summary>Complete provenance, versions and source hashes</summary><pre id="metadata" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre></details>
<footer>This self-contained report uses no external scripts, fonts, or services. Raw observations: <a href="results.jsonl">results.jsonl</a> · <a href="cases.json">authored cases</a> · <a href="summary.json">summary</a> · <a href="metadata.json">run metadata</a> · <a href="suite-source/README.md">reproduction instructions</a>. Historical canonical conformance captures were not rewritten.</footer>
</main><script>
const DATA=__DATA__;const M=DATA.metadata,R=DATA.results,C=DATA.catalogue;const el=id=>document.getElementById(id);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const fmt=n=>(n||0).toLocaleString();const bad=r=>['fail','error','not_run'].includes(r.status);const badge=s=>`<span class="badge ${esc(s)}">${esc(s.replaceAll('_',' '))}</span>`;
el('headline').textContent=`${C.length} named contract groups · ${fmt(R.length)} case/surface observations · ${fmt(M.executions)} executions including chunk boundaries. Baseline completed ${M.finished_utc}.`;
el('revision').innerHTML=`<a href="https://github.com/ai-dynamo/frontend-crates/commit/${esc(M.baseline_sha)}"><code>${esc(M.baseline_sha)}</code></a> · commit ${esc(M.commit_time)}<br><small>Main verified ${esc(M.main_checked_utc)} · test overlay SHA-256 ${esc(M.suite_sha256.slice(0,16))}</small>`;
const fail=R.filter(bad).length, pass=R.filter(r=>r.status==='pass').length;
const cardData=[['pass',pass,'Passed rows'],['fail',fail,'Failed / error rows'],['unavailable',M.counts.unavailable,'Unavailable rows'],['',M.groups_with_failure,'Groups with a failure'],['',M.executions,'Total executions']];
el('cards').innerHTML=cardData.map(([s,n,t])=>`<button class="card" data-status="${s}"><strong>${fmt(n)}</strong>${t}<br><small>${s==='pass'?(100*pass/Math.max(1,pass+fail)).toFixed(1)+'% of evaluated rows':s==='fail'?'Inspect expected versus actual':'Click to inspect'}</small></button>`).join('');
const families=[...new Set(R.map(r=>r.family))].sort(),surfaces=['v1_batch','v1_jail','v2_tool','unified'];
el('matrix').innerHTML='<thead><tr><th>Family</th>'+surfaces.map(s=>`<th>${s}</th>`).join('')+'</tr></thead><tbody>'+families.filter(f=>!['harness','generic'].includes(f)).map(f=>'<tr><td><strong>'+esc(f)+'</strong></td>'+surfaces.map(s=>{let rows=R.filter(r=>r.family===f&&r.surface===s),p=rows.filter(r=>r.status==='pass').length,n=rows.filter(bad).length,u=rows.filter(r=>r.status==='unavailable').length;return `<td><button class="matrix-button" data-family="${esc(f)}" data-surface="${s}">${p+n?`<span class="badge ${n?'fail':'pass'}">${p} / ${n}</span>`:u?badge('unavailable'):'—'}</button></td>`;}).join('')+'</tr>').join('')+'</tbody>';
el('catalogue-title').textContent=`All ${C.length} contract groups (${M.groups_with_results} with recorded results)`;
el('catalogue').innerHTML='<thead><tr><th>Group</th><th>Contract</th><th>Pass / fail / unavailable</th></tr></thead><tbody>'+C.map(c=>{let rs=R.filter(r=>r.group===c.id);return `<tr><td><button class="matrix-button" data-group="${c.id}">${c.id}</button></td><td>${esc(c.description)}</td><td>${rs.filter(r=>r.status==='pass').length} / ${rs.filter(bad).length} / ${rs.filter(r=>r.status==='unavailable').length}</td></tr>`}).join('')+'</tbody>';
for(const [id,values] of [['family',families],['surface',[...new Set(R.map(r=>r.surface))].sort()],['group',C.map(c=>c.id)]])for(const value of values){let option=document.createElement('option');option.value=value;option.textContent=value;el(id).append(option);}
let page=0,filtered=[],size=60;
function apply(){page=0;const q=el('search').value.toLowerCase();filtered=R.filter(r=>['status','family','surface','group'].every(k=>!el(k).value||r[k]===el(k).value)&&(!q||[r.id,r.contract,r.reason,JSON.stringify(r.failures||[])].join(' ').toLowerCase().includes(q)));draw();}
function draw(){let slice=filtered.slice(page*size,(page+1)*size);el('rows').replaceChildren();for(const r of slice){let tr=document.createElement('tr');tr.innerHTML=`<td><code>${esc(r.id)}</code><br><span class="pill">${esc(r.contract||'invariant')}</span></td><td>${esc(r.family)}<br><span class="subtle">${esc(r.surface)}</span></td><td>${badge(r.status)}</td><td class="num">${fmt(r.checks)}<br><span class="subtle">${fmt(r.failed_checks)} failed</span></td><td></td>`;let details=document.createElement('details');details.className='details';let summary=document.createElement('summary');summary.textContent=r.reason||r.failures?.[0]?.reason||(r.status==='pass'?'Inspect passing assertion':'Inspect case');details.append(summary);details.addEventListener('toggle',()=>{if(!details.open||details.dataset.loaded)return;details.dataset.loaded='1';for(const [title,value] of [['Native input',r.input],['Tool schema / init',{tools:r.tools,init:r.init}],['Expected',r.expected],['Failure evidence / actual output',(r.failures?.length?r.failures:r.actual)||r.reason||{expected_supported:r.expected_supported,actual_supported:r.actual_supported}],['Complete observation',r]]){if(value===undefined)continue;let label=document.createElement('strong');label.textContent=title;let pre=document.createElement('pre');pre.textContent=typeof value==='string'?value:JSON.stringify(value,null,2);details.append(label,pre);}});tr.lastElementChild.append(details);el('rows').append(tr);}el('page').textContent=`Page ${page+1} of ${Math.max(1,Math.ceil(filtered.length/size))}`;el('matches').textContent=`${fmt(filtered.length)} matching rows`;el('prev').disabled=page===0;el('next').disabled=(page+1)*size>=filtered.length;}
for(const id of ['status','family','surface','group'])el(id).addEventListener('change',apply);el('search').addEventListener('input',apply);el('prev').onclick=()=>{page--;draw()};el('next').onclick=()=>{page++;draw()};el('reset').onclick=()=>{for(const id of ['status','family','surface','group','search'])el(id).value='';apply()};
document.querySelectorAll('[data-family]').forEach(b=>b.onclick=()=>{el('family').value=b.dataset.family;el('surface').value=b.dataset.surface;el('status').value='';el('group').value='';apply();el('rows').scrollIntoView({block:'start'})});document.querySelectorAll('[data-group]').forEach(b=>b.onclick=()=>{el('group').value=b.dataset.group;el('status').value='';el('family').value='';el('surface').value='';apply();el('rows').scrollIntoView({block:'start'})});document.querySelectorAll('[data-status]').forEach(b=>b.onclick=()=>{el('status').value=b.dataset.status;apply()});
el('runs').innerHTML='<thead><tr><th>Command</th><th>Exit</th><th>Duration</th><th>Log</th></tr></thead><tbody>'+M.commands.map(c=>`<tr><td><code>${esc(c.command.join(' '))}</code></td><td>${c.exit_code}</td><td>${c.seconds==null?"not recorded":esc(c.seconds)+"s"}</td><td><a href="${esc(c.log)}">${esc(c.log)}</a></td></tr>`).join('')+'</tbody>';el('metadata').textContent=JSON.stringify(M,null,2);apply();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--xgrammar-python", default=sys.executable)
    ap.add_argument("--render-only", action="store_true")
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        print(json.dumps(render(out), indent=2))
        return
    subprocess.run(["git", "fetch", "origin", "main"], cwd=ROOT, check=True)
    baseline = git("rev-parse", "origin/main")
    checked = utc()
    if git("rev-parse", "HEAD") != baseline:
        raise SystemExit(
            "Checkout HEAD must equal freshly fetched origin/main. Test files may be overlaid; production code must be unchanged."
        )
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "parsers",
            "protocols",
            "Cargo.toml",
            "Cargo.lock",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    from cases import build_cases, CAPABILITIES
    from support import grammar_cases

    cases = build_cases()
    catalogue = json.loads((HERE / "catalogue.json").read_text())
    assert len(catalogue) == 110 and len({c["id"] for c in cases}) == len(cases)
    files = [
        ROOT / "conformance/tests/schema_coercion_contract.rs",
        *(
            HERE / name
            for name in [
                "cases.py",
                "support.py",
                "report.py",
                "catalogue.json",
                "README.md",
            ]
        ),
    ]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in files}
    suite_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    version = subprocess.check_output(
        [
            args.xgrammar_python,
            "-c",
            'import importlib.metadata; print(importlib.metadata.version("xgrammar"))',
        ],
        text=True,
    ).strip()
    if version != "0.2.7":
        raise SystemExit(f"Use pinned XGrammar 0.2.7; found {version}")
    metadata = {
        "baseline_sha": baseline,
        "commit_time": git("show", "-s", "--format=%cI", baseline),
        "commit_subject": git("show", "-s", "--format=%s", baseline),
        "main_checked_utc": checked,
        "started_utc": utc(),
        "repository": "ai-dynamo/frontend-crates",
        "worktree": str(ROOT),
        "suite_sha256": suite_hash,
        "suite_files": hashes,
        "cargo_lock_sha256": sha(ROOT / "Cargo.lock"),
        "rustc": subprocess.check_output(
            ["rustc", "--version"], cwd=ROOT, text=True
        ).strip(),
        "python": sys.version,
        "xgrammar_version": version,
        "xgrammar_python": args.xgrammar_python,
        "capabilities": CAPABILITIES,
        "commands": [],
    }
    (out / "cases.json").write_text(
        json.dumps(cases, indent=2, ensure_ascii=False) + "\n"
    )
    shutil.copyfile(HERE / "catalogue.json", out / "catalogue.json")
    (out / "suite-source").mkdir(exist_ok=True)
    for p in files:
        shutil.copyfile(p, out / "suite-source" / p.name)
    (out / "results.jsonl").write_text("")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    env = {
        **os.environ,
        "SCHEMA_SUITE_PYTHON": sys.executable,
        "SCHEMA_SUITE_XGRAMMAR_PYTHON": args.xgrammar_python,
        "SCHEMA_SUITE_RESULTS": str(out / "results.jsonl"),
    }
    command = [
        "cargo",
        "test",
        "--locked",
        "-p",
        "dynamo-conformance-fixtures-v2",
        "--test",
        "schema_coercion_contract",
        "--",
        "--test-threads=4",
    ]
    metadata["commands"].append(execute(command, out / "contracts.log", env))
    results = [json.loads(s) for s in (out / "results.jsonl").read_text().splitlines()]
    expected = []
    for c in cases:
        for surface in c.get("surfaces", ["v1_batch", "v1_jail", "v2_tool", "unified"]):
            expected.append((c["id"], c["group"], c["family"], surface))
    for i in range(1, 15):
        for c in grammar_cases(f"G{i:02}"):
            expected.append((c["id"], c["group"], c["family"], "grammar"))
    observed = {(r["id"], r["surface"]) for r in results}
    with (out / "results.jsonl").open("a") as f:
        for ident, group, family, surface in expected:
            if (ident, surface) not in observed:
                f.write(
                    json.dumps(
                        {
                            "id": ident,
                            "group": group,
                            "family": family,
                            "surface": surface,
                            "status": "not_run",
                            "reason": "Group did not produce an observation; inspect contracts.log",
                        }
                    )
                    + "\n"
                )
        for c in catalogue:
            if not any(r["group"] == c["id"] for r in results) and c["id"].startswith(
                "H"
            ):
                f.write(
                    json.dumps(
                        {
                            "id": c["id"] + ".not_run",
                            "group": c["id"],
                            "family": "harness",
                            "surface": "conformance",
                            "status": "not_run",
                            "reason": "No harness result; inspect contracts.log",
                        }
                    )
                    + "\n"
                )
    metadata["finished_utc"] = utc()
    metadata["test_summary"] = re.findall(
        r"test result:.*", (out / "contracts.log").read_text()
    )
    metadata["production_diff_empty"] = not git(
        "diff", "HEAD", "--", "parsers", "protocols", "Cargo.toml", "Cargo.lock"
    )
    assert hashes == {
        str(p.relative_to(ROOT)): sha(p) for p in files
    }, "Suite source changed during execution"
    metadata["artifact_sha256"] = {
        name: sha(out / name)
        for name in ["cases.json", "results.jsonl", "contracts.log"]
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = render(out)
    print(
        json.dumps(
            {
                "report": str(out / "report.html"),
                "sha": baseline,
                "counts": summary["counts"],
                "groups": summary["groups_with_results"],
                "test_summary": summary["test_summary"],
            },
            indent=2,
        )
    )
    # Report generation succeeds even for a red baseline; the captured cargo exit remains authoritative.


if __name__ == "__main__":
    main()
