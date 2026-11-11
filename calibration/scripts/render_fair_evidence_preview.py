#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Render a clearly PROVISIONAL FAIR report from separately validated inputs.

Two incompatible historical contexts must never be silently pooled:
- Historical final-front CSVs: five uncalibrated FAIR physical measurements,
  with missing longitudinal HEADWAY and all Q scores marked unavailable.
- New reproducible MOEA pilot: genuine P0/Pf, calibrated GT and clinical
  scores against the explicitly labeled frozen or staged baselines.

This report is NOT the canonical moeabench audit_report and never writes to
the certified reports directory. Never infer unavailable values.
"""

import argparse
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRICS = ("CLOSENESS", "COVERAGE", "GAP", "REGULARITY", "BALANCE", "HEADWAY")


def _fmt(value):
    if value is None:
        return "<span class='missing'>Unavailable</span>"
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("A report metric is nonfinite")
    return f"{value:.6f}"


def _validate(pilot, archive):
    if pilot.get("status") != "provisional_longitudinal_pilot_not_certified":
        raise ValueError("Not a recognized independently validated longitudinal pilot")
    if pilot.get("headway_validated_from_true_P0") is not True:
        raise ValueError("A longitudinal pilot needs genuine P0 with validated Headway")
    if archive.get("status") != "provisional_noncertified":
        raise ValueError("Not a recognized provisional historical-data report")
    if archive.get("cannot_be_used_as_certified_audit_report") is not True:
        raise ValueError("Archive lacks noncertified protection")
    for key in METRICS:
        if key not in pilot["physical_metrics"]:
            raise ValueError(f"Missing pilot FAIR metric: {key}")
    for run in archive.get("runs", []):
        if run.get("q_scores") is not None:
            raise ValueError("Archive clinical Q-values lack certified reference provenance")
        for key in METRICS:
            if key not in run["metrics"]:
                raise ValueError(f"Missing archived FAIR metric: {key}")


def render_preview(pilot, archive):
    _validate(pilot, archive)
    escape = lambda s: html.escape(str(s), quote=True)
    config = pilot["experiment"]
    provenance = pilot["provenance"]
    comparison = pilot.get("candidate_closeness_comparison")
    rows = []
    for key in METRICS:
        value = pilot["physical_metrics"][key]
        q_value = pilot["clinical_q_scores_frozen_baseline"].get("Q_"+key)
        rows.append(
            "<tr><th scope='row'>"+key+"</th><td>"+_fmt(value)+"</td>"
            "<td>"+_fmt(q_value)+"</td></tr>"
        )
    pilot_table = "\n".join(rows)
    archive_rows = []
    for run in archive.get("runs", []):
        heading = f"{run['study']} / run {run['run']}"
        archive_rows.append(
            "<tr><th scope='row'>"+escape(heading)+"</th>"+
            "".join(f"<td>{_fmt(run['metrics'][key])}</td>" for key in METRICS)+
            "<td class='missing'>Unavailable</td></tr>"
        )

    comparison_html = ""
    if comparison is not None:
        if comparison.get("other_five_FAIR_metrics_and_Q_unchanged") is not True:
            raise ValueError("Unvalidated staged baseline comparison")
        comparison_html = (
            "<section><h2>Staged calibration comparison (same P0/Pf)</h2>"
            "<p>Candidate Closeness residual: <b>"+_fmt(comparison['ideal_res'])+
            "</b>. Closeness changed from <b>"+_fmt(comparison['old_closeness'])+
            "</b> to <b>"+_fmt(comparison['new_closeness'])+
            "</b>; Q-Closeness changed from <b>"+_fmt(comparison['old_q_closeness'])+
            "</b> to <b>"+_fmt(comparison['new_q_closeness'])+
            "</b>. All other FAIR and Q metrics were asserted unchanged.</p>"
            "<p>Candidate baseline SHA256: <code>"+
            escape(comparison["candidate_baseline_sha256"])+"</code></p></section>"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FAIR conformance — provisional evidence (issue 75)</title>
<style>
body{{font:16px/1.6 system-ui,sans-serif;max-width:1200px;margin:auto;padding:2rem;color:#17202a;background:#f8fafc}}
h1,h2{{line-height:1.25}} h1{{margin-bottom:.4rem}}
section{{padding:1.3rem;background:#fff;border:1px solid #dbe2e8;border-radius:10px;margin:1.3rem 0}}
table{{border-collapse:collapse;display:block;overflow-x:auto;width:100%}}
td,th{{border-bottom:1px solid #e2e8f0;padding:.5rem .8rem;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}
.missing{{color:#9a3412;font-weight:600}} .notice{{font-weight:700;color:#9a3412}}
code{{overflow-wrap:anywhere;word-break:break-word}}
</style></head><body>
<h1>FAIR conformance — provisional evidence</h1>
<p class="notice">NOT CERTIFIED — this is not the published audit_report.</p>
<p>Issue #75 / ADR 0051. Two different experimental datasets are presented
separately. Do not aggregate their metrics or infer quality scores for the archived data.</p>
<section><h2>1. New reproducible longitudinal DTLZ1 pilot</h2>
<p>MOEA: {escape(config['moea'])}; M={escape(config['M'])};
population={escape(config['population'])}; generations={escape(config['generations'])};
seed={escape(config['seed'])}. Initial population: {escape(pilot['sample_sizes']['P0'])};
final non-dominated front: {escape(pilot['sample_sizes']['Pf'])}.</p>
<p>GT source: {escape(provenance['gt_source'])}; baseline version:
{escape(provenance.get('baseline_version'))}. <b>Headway was independently
validated using the actual generation-zero population.</b></p>
<table><thead><tr><th>Metric</th><th>Physical FAIR</th><th>Q (frozen baseline)</th></tr></thead>
<tbody>{pilot_table}</tbody></table>
<p>Independent Headway = {_fmt(pilot['independent_headway'])};
GT SHA256: <code>{escape(provenance['gt_sha256'])}</code>;
P0 SHA256: <code>{escape(provenance['P0_sha256'])}</code>;
Pf SHA256: <code>{escape(provenance['Pf_sha256'])}</code>.</p>
</section>
{comparison_html}
<section><h2>2. Historical archived fronts — FAIR physical only</h2>
<p>The historical samples have no verified generation-zero population.
HEADWAY is unavailable, and no compatible clinical Q calibration has been
certified for these raw GT coordinates. Closeness is explicitly uncorrected
for unknown independent finite-GT residue.</p>
<table><thead><tr><th>Archived run</th>{''.join('<th>'+key+'</th>' for key in METRICS)}
<th>Q-scores</th></tr></thead><tbody>{''.join(archive_rows)}</tbody></table>
<p>File-processing errors: {len(archive.get('errors', []))}. This subsection does
not reuse the historical report's fabricated initial populations.</p>
</section>
<footer><p>Scientific release gate: independent calibration validation, full data
provenance, regression suite, and explicitly versioned replacement report.
No official baseline or published audit file has been replaced.</p></footer>
</body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.suffix.lower() != ".html":
        parser.error("Expected an HTML output")
    if out.is_relative_to((ROOT/"calibration/reports").resolve()) or out == (ROOT/"calibration/audit_report.html").resolve():
        parser.error("Provisional preview must not overwrite or reside among certified audit files")
    if out.exists():
        parser.error("Refusing to overwrite an existing report")
    with args.pilot.open(encoding="utf-8") as fh:
        pilot = json.load(fh)
    with args.archive.open(encoding="utf-8") as fh:
        archive = json.load(fh)
    content = render_preview(pilot, archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Provisional FAIR evidence preview: {out} ({len(content)} bytes)")


if __name__ == "__main__":
    main()
