#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reproducible longitudinal FAIR pilot with genuine generation-zero data.

Unlike the legacy audit_calibration.py, this protocol executes an MOEA and
retains P0 from the *same run* as Pf. It independently checks HEADWAY against
GD95(Pf,GT)/GD95(P0,GT) on calibrated coordinates. The result is a staged
research pilot, NOT a replacement for the 985 legacy runs or a certified
audit_report. Historical v0.14.0 clinical baselines remain read-only.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import moeabench as mb
from moeabench.diagnostics import baselines

ROOT = Path(__file__).resolve().parents[2]
CERTIFIED_REPORT_DIR = (ROOT / "calibration" / "reports").resolve()


def _digest(values):
    values = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(values.tobytes()).hexdigest()


def independent_headway(population_zero, population_final, gt):
    """Reference-only mathematical oracle, without calling FAIR/Headway."""
    def gd95(pop):
        distances = cKDTree(gt).query(pop, k=1)[0]
        return float(np.percentile(distances, 95))
    d0 = gd95(population_zero)
    df = gd95(population_final)
    if d0 <= 1e-12:
        return 1.0 if df > 1e-12 else 0.0
    return df / d0


def run_pilot(*, seed=1, generations=60, population=64, candidate_baseline=None):
    if generations < 2 or population < 8 or population % 4:
        raise ValueError("Require generations>=2 and population>=8 divisible by 4")
    exp = mb.experiment()
    exp.mop = mb.mops.DTLZ1(M=3)
    exp.moea = mb.moeas.NSGA2(population=population, generations=generations)
    exp.run(seed=seed, silent=True)

    old = mb.clinic.audit(exp)
    if old.quality is None or old.fr is None:
        raise RuntimeError("Registered clinical reference is unavailable")
    ctx = old.diagnostic_context
    gt = np.asarray(ctx["GT"], dtype=np.float64)
    initial = np.asarray(ctx["P_initial"], dtype=np.float64)
    final = np.asarray(ctx["P_final"], dtype=np.float64)
    if len(initial) == 0 or len(final) == 0:
        raise RuntimeError("Experiment failed to preserve P0 or Pf")
    if ctx.get("gt_source") != "registered":
        raise RuntimeError("Expected the exact registered calibration GT")
    if ctx.get("normalization") is None:
        raise RuntimeError("Missing calibration-coordinate provenance")

    observed = float(old.fr.metrics["HEADWAY"].value)
    expected = independent_headway(initial, final, gt)
    if not np.isclose(observed, expected, atol=1e-12, rtol=1e-12):
        raise AssertionError(f"Headway disagrees with independent GD95 ratio: {observed} vs {expected}")

    normalized = ctx["normalization"]
    raw_zero = np.asarray(exp.pop(0).objectives, dtype=np.float64)
    raw_final = np.asarray(exp.front(-1), dtype=np.float64)
    scale = np.asarray(normalized["nadir"]) - np.asarray(normalized["ideal"])
    shift = np.asarray(normalized["ideal"])
    np.testing.assert_allclose((raw_zero-shift)/scale, initial, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose((raw_final-shift)/scale, final, atol=1e-12, rtol=1e-12)

    physical = {key:float(value.value) for key,value in old.fr.metrics.items()}
    q_scores = {key:float(value.value) for key,value in old.quality.scores.items()}
    if not all(np.isfinite(v) for v in [*physical.values(), *q_scores.values()]):
        raise RuntimeError("Pilot produced nonfinite FAIR/Q results")

    candidate_comparison = None
    if candidate_baseline is not None:
        candidate_baseline = Path(candidate_baseline)
        if not candidate_baseline.is_file():
            raise FileNotFoundError(candidate_baseline)
        with baselines.use_baselines(str(candidate_baseline)):
            staged = mb.clinic.audit(exp)
            if staged.quality is None:
                raise RuntimeError("Candidate baseline is incompatible with this GT")
            epsilon = float(
                baselines.get_baseline_data(
                    "DTLZ1", int(ctx["k"]), "closeness"
                )["ideal_res"]
            )

        original_u = np.asarray(old.fr.metrics["CLOSENESS"].raw_data)
        corrected_u = np.asarray(staged.fr.metrics["CLOSENESS"].raw_data)
        np.testing.assert_allclose(
            corrected_u, np.maximum(0, original_u - epsilon),
            atol=1e-12, rtol=0,
        )
        # This comparison uses exactly the same P0, Pf and GT. Every metric
        # except Closeness (and its derived Q) must be unchanged.
        other_physical = ("COVERAGE", "GAP", "REGULARITY", "BALANCE", "HEADWAY")
        other_quality = ("Q_COVERAGE", "Q_GAP", "Q_REGULARITY",
                         "Q_BALANCE", "Q_HEADWAY")
        for key in other_physical:
            np.testing.assert_allclose(
                old.fr.metrics[key].value, staged.fr.metrics[key].value,
                atol=1e-12, rtol=0,
            )
        for key in other_quality:
            np.testing.assert_allclose(
                old.quality.scores[key].value, staged.quality.scores[key].value,
                atol=1e-12, rtol=0,
            )
        candidate_comparison = {
            "candidate_baseline_sha256": hashlib.sha256(
                candidate_baseline.read_bytes()
            ).hexdigest(),
            "ideal_res": epsilon,
            "old_closeness": float(old.fr.metrics["CLOSENESS"].value),
            "new_closeness": float(staged.fr.metrics["CLOSENESS"].value),
            "old_q_closeness": float(old.quality.scores["Q_CLOSENESS"].value),
            "new_q_closeness": float(staged.quality.scores["Q_CLOSENESS"].value),
            "other_five_FAIR_metrics_and_Q_unchanged": True,
        }

    return {
        "status": "provisional_longitudinal_pilot_not_certified",
        "experiment": {
            "mop": "DTLZ1", "M": 3, "moea": "NSGA2",
            "seed": int(seed), "generations": int(generations),
            "population": int(population),
        },
        "provenance": {
            "gt_source": ctx["gt_source"],
            "gt_sha256": _digest(gt),
            "registered_gt_points": int(len(gt)),
            "P0_sha256": _digest(initial),
            "Pf_sha256": _digest(final),
            "P0_raw_sha256": _digest(raw_zero),
            "Pf_raw_sha256": _digest(raw_final),
            "normalization": "calibration_ideal_nadir",
            "baseline_version": old.reproducibility.get("baseline_version") if old.reproducibility else None,
        },
        "sample_sizes": {"P0":len(initial), "Pf":len(final), "K":int(ctx["k"])},
        "physical_metrics": physical,
        "clinical_q_scores_frozen_baseline": q_scores,
        "independent_headway": expected,
        "headway_validated_from_true_P0": True,
        "candidate_closeness_comparison": candidate_comparison,
        "notes": "A newly executed reproducible study; NOT a reconstruction of a historical run.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-baseline", type=Path,
                        help="Optional staged, noncertified baseline for paired comparison")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(CERTIFIED_REPORT_DIR) or output == (ROOT/"calibration/audit_report.html").resolve():
        parser.error("Pilot cannot be written among certified audit artifacts")
    if output.exists():
        parser.error("Refusing to overwrite existing pilot")
    result = run_pilot(seed=args.seed, generations=args.generations,
                       population=args.population,
                       candidate_baseline=args.candidate_baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
