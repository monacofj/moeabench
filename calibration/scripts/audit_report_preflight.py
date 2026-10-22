#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Non-destructive preflight for regenerating calibration/audit_report.html.

A longitudinal FAIR HEADWAY needs P_initial and P_final from the SAME run.
Historical audit scripts instead fabricated a random P_initial in the absence
of generation-zero data. This preflight records the actual saved inputs and
makes missing requirements explicit; it never invents initial populations.
It does NOT compute or certify Q-scores.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
RUN_PATTERN = re.compile(
    r"^(?P<study>.+)_standard_run(?P<run>\d{2})(?:_gen(?P<gen>\d+))?\.csv$"
)


def inspect_inputs(data_dir: Path, gt_dir: Path, baseline_csv: Path) -> dict:
    groups = defaultdict(lambda: {"final": False, "initial": False, "generations": []})
    ignored = 0
    for path in sorted(Path(data_dir).glob("*.csv")):
        match = RUN_PATTERN.fullmatch(path.name)
        if match is None:
            ignored += 1
            continue
        key = (match["study"], match["run"])
        rec = groups[key]
        if match["gen"] is None:
            rec["final"] = True
        else:
            generation = int(match["gen"])
            rec["generations"].append(generation)
            if generation == 0:
                rec["initial"] = True

    records = []
    for (study, run), rec in sorted(groups.items()):
        gens = sorted(set(rec["generations"]))
        records.append({
            "study": study,
            "run": run,
            "final_present": rec["final"],
            "generation_zero_present": rec["initial"],
            "earliest_saved_generation": gens[0] if gens else None,
            "saved_generation_count": len(gens),
            "headway_data_ready": rec["final"] and rec["initial"],
        })

    studies = sorted({item["study"] for item in records})
    gt_files = sorted(Path(gt_dir).glob("*_3_optimal.csv"))
    complete = sum(r["headway_data_ready"] for r in records)
    finals = sum(r["final_present"] for r in records)
    return {
        "scope": "file-presence preflight only, not calibration certification",
        "calibration_data_dir": str(data_dir),
        "baseline_csv": {"path": str(baseline_csv), "exists": baseline_csv.is_file()},
        "studies": len(studies),
        "run_records": len(records),
        "final_front_files": finals,
        "with_true_generation_zero": sum(r["generation_zero_present"] for r in records),
        "headway_data_ready_runs": complete,
        "headway_data_missing_runs": len(records) - complete,
        "ground_truth_3_objective_files": len(gt_files),
        "other_csv_files": ignored,
        "report_generation_safe": bool(records and finals == len(records)
                                       and complete == len(records)
                                       and baseline_csv.is_file()),
        "runs": records,
        "notes": [
            "Do not substitute a synthetic random initial population for HEADWAY.",
            "Check GT normalization, provenance and baseline version separately.",
            "The legacy audit script contains removed FAIR API calls and is not validated.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "calibration/data/calibration_data")
    parser.add_argument("--gt-dir", type=Path, default=ROOT / "calibration/data/ground_truth")
    parser.add_argument("--baseline-csv", type=Path, default=ROOT / "calibration/reports/baselines.csv")
    parser.add_argument("--output", type=Path, help="Optional summary JSON path")
    args = parser.parse_args()
    report = inspect_inputs(args.data_dir, args.gt_dir, args.baseline_csv)
    summary = {k:v for k,v in report.items() if k != "runs"}
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
