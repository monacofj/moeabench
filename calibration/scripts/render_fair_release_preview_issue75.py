#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Create a versioned NONCERTIFIED FAIR #75 audit preview.

Separate the genuine current NSGA-II longitudinal experiment from historical
CSV archives whose generation-zero population is absent. The manifest ties
the report to exact candidate/GT hashes and explicit unsupported geometry.
The canonical calibration/audit_report.html is NEVER overwritten.
"""
import argparse
import html
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REPORTS=(ROOT/"calibration/reports").resolve()
RESOURCES=(ROOT/"moeabench/diagnostics/resources").resolve()
CANONICAL=(ROOT/"calibration/audit_report.html").resolve()

def _number(v):
    if v is None:
        return "Unavailable"
    f=float(v)
    if not math.isfinite(f):
        raise ValueError("Non-finite FAIR value")
    return f"{f:.7f}"

def render_versioned_preview(manifest,q,history):
    if manifest.get("status")!="NONCERTIFIED_EXPERIMENTAL_MANIFEST" or (
        manifest.get("for_publication") is not False or
        manifest.get("replaces_certified_baseline") is not False):
        raise ValueError("Only an expressly noncertified baseline manifest is allowed")
    if q.get("status")!="NONCERTIFIED_PAIRED_REAL_MOEA_Q_CANDIDATE" or (
        q.get("certified") is not False):
        raise ValueError("Only real-MOEA noncertified Q evidence is allowed")
    if (q.get("baseline_candidate_sha256") !=
        manifest.get("candidate_baseline_sha256") or
        q.get("baseline_frozen_sha256") !=
        manifest.get("source_baseline_sha256")):
        raise ValueError("Report manifest baseline hashes disagree with Q study")
    problem=q.get("problem")
    info=manifest.get("problems",{}).get(problem,{})
    if info.get("status")!="PROVISIONAL_NOT_CERTIFIED":
        raise ValueError("Problem lacks independent area candidate")
    if (q.get("reference_measure")!=info.get("reference_measure") or
        q.get("sample_seed")!=info.get("ideal_sampling_seed") or
        q.get("sample_count")!=info.get("ideal_sample_count") or
        q.get("frozen_gt_sha256")!=info.get("gt_sha256")):
        raise ValueError("Q provenance and candidate manifest differ")
    if (history.get("scope") != "file-presence preflight only, not calibration certification" or
        history.get("report_generation_safe") is not False):
        raise ValueError("Historical preflight must not claim report certification")
    if history.get("run_records",0)<1:
        raise ValueError("Missing historical archive inventory")
    if q.get("unrelated_fair_q_unchanged") is not True:
        raise ValueError("Other FAIR metrics changed or not verified")
    rows=[c for t in q["trials"] for c in t["checkpoints"]]
    if len(rows)!=q["all_checkpoint_count"]:
        raise ValueError("Q snapshot count mismatch")
    if len([c for c in rows if not c["both_endpoints_clipped"]]) != q["informative_checkpoint_count"]:
        raise ValueError("Q clipping classification mismatch")
    if not q["trials"] or not all(
        bool(t.get("P0_sha256")) and t.get("P0_points",0)>0 and
        t.get("headway_same_under_candidate") is True
        for t in q["trials"]):
        raise ValueError("Longitudinal Q evidence lacks real P0 provenance")
    esc=lambda x:html.escape(str(x),quote=True)
    details=[]
    for trial in q["trials"]:
        for c in trial["checkpoints"]:
            details.append(
                "<tr><td>"+esc(trial["optimizer_seed"])+"</td><td>"+
                esc(c["generation"])+"</td><td>"+
                esc(c["front_points"])+"</td><td>"+
                _number(c["q_frozen"])+"</td><td>"+
                _number(c["q_area_candidate"])+"</td><td>"+
                _number(c["absolute_q_shift"])+"</td><td>"+
                ("Excluded: saturated" if c["both_endpoints_clipped"] else "Informative")+
                "</td></tr>")
    coverage=[]
    for name,rec in sorted(manifest["problems"].items()):
        coverage.append("<tr><td>"+esc(name)+"</td><td>"+
                        esc(rec["status"])+"</td><td>"+
                        esc(rec.get("K_count","—"))+"</td></tr>")
    historical=(
        "The archive contains "+esc(history["run_records"])+" recorded runs, "+
        esc(history["final_front_files"])+" final-front files, and "+
        esc(history["with_true_generation_zero"])+" actual generation-zero snapshots. "+
        esc(history["headway_data_missing_runs"])+" runs lack the paired initial/final "+
        "evidence required for longitudinal Headway. Historic Q-scores are "+
        "<b>Unavailable</b> without independently certified, coordinate-compatible baselines."
    )
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>FAIR #75 — Versioned Candidate V1</title>
<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:auto;padding:2rem}
table{border-collapse:collapse;width:100%}td,th{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}
.notice{color:#9a3412;font-weight:bold}code{overflow-wrap:anywhere}</style></head><body>
<h1>FAIR #75 — versioned candidate audit preview V1</h1>
<p class="notice">NOT CERTIFIED. NOT the official calibration/audit_report.html.
No historical missing data were reconstructed. Do not publish as validated clinical Q.</p>
<h2>Provenance and scientific scope</h2>
<p>Manifest: <code>"""+esc(manifest["id"])+"""</code>;
source revision: <code>"""+esc(manifest["source_revision"])+"""</code>.</p>
<p>Frozen baseline SHA256: <code>"""+esc(manifest["source_baseline_sha256"])+"""</code>.
Candidate SHA256: <code>"""+esc(manifest["candidate_baseline_sha256"])+"""</code>.</p>
<p>Reference law: """+esc(info["reference_measure"])+""". Independent area sampler:
"""+esc(info["analytical_sampler"])+"""; seed """+esc(info["ideal_sampling_seed"])+""".
Frozen GT SHA256: <code>"""+esc(info["gt_sha256"])+"""</code>.</p>
<p>Only DTLZ1 and DTLZ2 (M=3) are included in this provisional intrinsic-area
candidate; other MOPs remain unsupported <i>for this candidate measure</i>.</p>
<h2>Real longitudinal NSGA-II experiment</h2>
<p>"""+esc(problem)+"""; population """+esc(q["K"])+""";
generations """+esc(q["generations"])+""";
ideal residue """+_number(q["area_ideal_res"])+""".
Genuine initial populations are identified by SHA256 per optimizer seed.
Headway and all unrelated FAIR and Q measures were independently checked unchanged
when switching between the frozen and candidate baselines.</p>
<p>Informative checkpoints: """+esc(q["informative_checkpoint_count"])+""";
both-endpoints-clipped checkpoints: """+esc(q["clipped_checkpoint_count"])+""";
max informative absolute Q-Closeness change: """+
_number(q["max_informative_absolute_q_shift"])+""".</p>
<table><thead><tr><th>Seed</th><th>Gen.</th><th>ND points</th><th>Q frozen</th>
<th>Q candidate</th><th>Absolute shift</th><th>Status</th></tr></thead>
<tbody>"""+ "".join(details)+"""</tbody></table>
<h2>Historical archived data (separate, uncalibrated)</h2>
<p>"""+historical+"""</p>
<p class="notice">No synthetic generation-zero populations; no guessed historical Q-scores.</p>
<h2>MOP-level candidate coverage</h2>
<table><thead><tr><th>MOP</th><th>Status</th><th>Provisional K records</th></tr>
</thead><tbody>"""+ "".join(coverage)+"""</tbody></table>
<p>Release blockers: approved sampling measure, candidate Q acceptability,
full regressions, historic report compatibility. These findings do not
authorize replacing the installed FAIR baseline or canonical audit report.</p>
</body></html>"""

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--q-study",type=Path,required=True)
    p.add_argument("--historical-preflight",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    out=a.output.resolve()
    if (out.exists() or out.suffix.lower()!=".html" or
        out.is_relative_to(REPORTS) or out.is_relative_to(RESOURCES) or out==CANONICAL):
        p.error("Refusing overwrite or certified report destination")
    data=[json.loads(pth.read_text()) for pth in (a.manifest,a.q_study,a.historical_preflight)]
    html_body=render_versioned_preview(*data)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(html_body,encoding="utf-8")
    print(f"NONCERTIFIED FAIR issue-75 candidate report: {out} ({len(html_body)} bytes)")
if __name__=="__main__":
    main()
