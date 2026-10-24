#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""ADR 0049: independent, bounded Q_HEADWAY anchor validation (issue #75).

The physical statistic uses true paired P0/Pf and is not clipped. The Q
interpretation is analytically anchored at Headway=0 (Q=1) and Headway=1
(Q=0); scores for regression Headway>1 are clipped to zero. This explicitly
does not invent a random-sample Headway ECDF or replace any certified file.
"""

import argparse
import json
from pathlib import Path
import sys

# Preserve direct-CLI support (python calibration/scripts/...). Installed
# moeabench alone does not make the calibration namespace importable.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial import cKDTree

from moeabench.diagnostics import fair, qscore
from calibration.scripts.audit_longitudinal_pilot import run_pilot

def _gd95(points, reference):
    return float(np.percentile(cKDTree(reference).query(points, k=1)[0], 95))


def independent_ratio(initial, final, reference):
    """Independent nearest-GT reference oracle for ADR 0049."""
    before = _gd95(initial, reference)
    after = _gd95(final, reference)
    return after / before if before > 1e-12 else (1.0 if after > 1e-12 else 0.0)


def expected_q(ratio):
    """Independent ADR 0049 log-linear anchor oracle."""
    if np.isnan(ratio):
        return float("nan")
    return float(1 - np.clip(np.log1p(max(0.0, ratio))/np.log(2.0), 0, 1))


def validate_analytic_controls():
    t = np.linspace(0.0, 1.0, 41)
    gt = np.column_stack((t, 1-t, np.zeros_like(t)))
    base = gt[::4].copy()
    p0 = base + np.array([0.0, 0.0, 0.40])
    cases = {
        "perfect": (base, 0.0, 1.0),
        "improved": (base + np.array([0.0, 0.0, 0.10]), 0.25, expected_q(0.25)),
        "unchanged": (p0.copy(), 1.0, 0.0),
        "regressed": (base + np.array([0.0, 0.0, 0.80]), 2.0, 0.0),
    }
    report = {}
    for name, (pf, expected_headway, expected_quality) in cases.items():
        calculated = float(fair.headway(
            pf, ref=gt, s_k=0.1, initial_data=p0,
        ).value)
        oracle = independent_ratio(p0, pf, gt)
        q = float(qscore.q_headway(calculated).value)
        assert np.isclose(calculated, oracle, atol=1e-12, rtol=0)
        assert np.isclose(calculated, expected_headway, atol=1e-12, rtol=0)
        assert np.isclose(q, expected_quality, atol=1e-12, rtol=0)
        report[name] = {"headway": calculated, "q_headway": q}

    # No synthetic P0 is allowed when a caller passes only a final front.
    missing = float(fair.headway(
        cases["improved"][0], ref=gt, s_k=0.1,
    ).value)
    missing_q = float(qscore.q_headway(missing).value)
    assert np.isnan(missing) and np.isnan(missing_q)
    report["missing_P0"] = {"headway": None, "q_headway": None}

    # A zero-error initial reference is a specifically defined edge case.
    assert fair.headway(base, ref=gt, s_k=0.1, initial_data=base).value == 0.0
    assert fair.headway(p0, ref=gt, s_k=0.1, initial_data=base).value == 1.0
    return report


def validate_real_pairs(seeds, *, population=32, generations=20):
    trials = []
    for seed in seeds:
        pilot = run_pilot(seed=seed, population=population, generations=generations)
        measured = float(pilot["physical_metrics"]["HEADWAY"])
        reported = float(pilot["clinical_q_scores_frozen_baseline"]["Q_HEADWAY"])
        expected = expected_q(measured)
        assert np.isclose(measured, pilot["independent_headway"], atol=1e-12, rtol=0)
        assert np.isclose(reported, expected, atol=1e-12, rtol=0)
        assert pilot["headway_validated_from_true_P0"]
        trials.append({
            "seed": int(seed),
            "P0_sha256": pilot["provenance"]["P0_sha256"],
            "Pf_sha256": pilot["provenance"]["Pf_sha256"],
            "GT_sha256": pilot["provenance"]["gt_sha256"],
            "P0_points": pilot["sample_sizes"]["P0"],
            "Pf_points": pilot["sample_sizes"]["Pf"],
            "headway": measured,
            "q_headway": reported,
        })
    return trials


def validate(seeds=(7, 8), *, population=32, generations=20):
    return {
        "status": "PROVISIONAL_ADR0049_ANCHOR_VALIDATION",
        "headway_physical": "GD95(Pfinal->GT)/GD95(Pinitial->GT)",
        "quality_mapping": "1-clip(log1p(headway)/log(2),0,1)",
        "analytic_anchors": {"perfect": 0.0, "no_progress": 1.0},
        "no_empirical_distribution_inferred": True,
        "analytic_controls": validate_analytic_controls(),
        "real_moea_trials": validate_real_pairs(
            seeds, population=population, generations=generations,
        ),
        "certified_baseline_replaced": False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,action="append",dest="seeds",required=True)
    p.add_argument("--population",type=int,default=32)
    p.add_argument("--generations",type=int,default=20)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    out=args.output.resolve()
    if out.is_relative_to((ROOT/"calibration/reports").resolve()) or out.is_relative_to(
        (ROOT/"moeabench/diagnostics/resources").resolve()
    ):
        p.error("Refusing to place provisional validation in certified output directories")
    if out.exists():
        p.error("Refusing to overwrite a previous validation artifact")
    result=validate(tuple(args.seeds),population=args.population,
                    generations=args.generations)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
