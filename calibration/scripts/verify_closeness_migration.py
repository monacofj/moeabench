#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Strict non-destructive verification of a staged FAIR Closeness migration.

Issue #75: applying a finite-GT ideal residue may update only the matching
problem's 'closeness' records. Other metrics, K values, problem records,
metadata and reference arrays must be byte-for-byte equivalent as decoded
JSON values. The check also independently replays the clipping transform
on the archived ECDF without calling the migration implementation.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def verify(original, candidate, changed_problems):
    changed = set(changed_problems)
    if not changed or not changed.issubset(original.get("problems", {})):
        raise ValueError("Unknown or empty target problem list")
    if set(original) != set(candidate):
        raise ValueError("Top-level schema changed")
    for field in original:
        if field != "problems" and original[field] != candidate[field]:
            raise ValueError(f"Top-level metadata changed: {field}")
    if set(original["problems"]) != set(candidate["problems"]):
        raise ValueError("Problem set changed")

    records = {}
    for name, before_problem in original["problems"].items():
        after_problem = candidate["problems"][name]
        if name not in changed:
            if before_problem != after_problem:
                raise ValueError(f"Untargeted problem changed: {name}")
            continue
        if set(before_problem) != set(after_problem):
            raise ValueError(f"Problem keys changed: {name}")
        result = {}
        for key, before_k in before_problem.items():
            after_k = after_problem[key]
            if not key.isdigit() or not isinstance(before_k, dict) or "closeness" not in before_k:
                if before_k != after_k:
                    raise ValueError(f"Noncalibration record changed: {name}/{key}")
                continue
            if set(before_k) != set(after_k):
                raise ValueError(f"Metric set changed: {name}/{key}")
            for metric, value in before_k.items():
                if metric != "closeness" and value != after_k[metric]:
                    raise ValueError(f"Unrelated FAIR metric changed: {name}/{key}/{metric}")
            old = before_k["closeness"]
            new = after_k["closeness"]
            method = new.get("ideal_res_method")
            allowed_methods = (
                "independent_analytical_pf",
                "uniform_normalized_objective_pf_area_v1",
            )
            if method not in allowed_methods:
                raise ValueError(f"Residue provenance missing: {name}/{key}")
            extra_fields = set()
            if method == "uniform_normalized_objective_pf_area_v1":
                if new.get("reference_measure") != "uniform_intrinsic_area_normalized_objective_pf":
                    raise ValueError(f"Area measure provenance invalid: {name}/{key}")
                if not isinstance(new.get("ideal_res_sampler"), str) or not new["ideal_res_sampler"]:
                    raise ValueError(f"Area sampler provenance missing: {name}/{key}")
                if (not isinstance(new.get("ideal_res_sampling_seed"), int) or
                        new["ideal_res_sampling_seed"] < 0):
                    raise ValueError(f"Area RNG provenance invalid: {name}/{key}")
                if (not isinstance(new.get("ideal_res_sample_count"), int) or
                        new["ideal_res_sample_count"] < 2):
                    raise ValueError(f"Area sample count invalid: {name}/{key}")
                extra_fields = {
                    "reference_measure", "ideal_res_sampler",
                    "ideal_res_sampling_seed", "ideal_res_sample_count",
                }
            eps = float(new["ideal_res"])
            if not np.isfinite(eps) or eps <= 0:
                raise ValueError(f"Invalid positive analytical residue: {name}/{key}")
            expected = np.maximum(0.0, np.asarray(old["rand_ecdf"], dtype=float) - eps)
            np.testing.assert_allclose(new["rand_ecdf"], expected, atol=1e-12, rtol=0)
            np.testing.assert_allclose(
                new["rand50"], max(0.0, float(old["rand50"]) - eps), atol=1e-12, rtol=0
            )
            if "p95" in old:
                np.testing.assert_allclose(
                    new["p95"], max(0.0, float(old["p95"]) - eps), atol=1e-12, rtol=0
                )
            for attr, value in old.items():
                if attr not in ({"ideal_res", "ideal_res_method", "rand_ecdf", "rand50", "p95"} | extra_fields):
                    if new.get(attr) != value:
                        raise ValueError(f"Unexpected closeness field modified: {name}/{key}/{attr}")
            allowed_new = {"ideal_res_method", "rand_ecdf", "rand50",
                           "p95", "ideal_res"} | extra_fields
            if set(new).difference(set(old)) - allowed_new:
                raise ValueError(f"Unrecognized new Closeness fields: {name}/{key}")
            result[key] = {"epsilon": eps, "old_rand50": old["rand50"],
                           "new_rand50": new["rand50"],
                           "ideal_res_method": method}
        if not result:
            raise ValueError(f"No calibration K records migrated: {name}")
        records[name] = result

    return {
        "status": "PASS_PROVISIONAL_CLOSENESS_ONLY",
        "targeted": sorted(changed),
        "untouched_problem_count": len(original["problems"]) - len(changed),
        "k_counts": {name:len(data) for name, data in records.items()},
        "k_records": records,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--problem",action="append",dest="problems",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    with a.original.open(encoding="utf-8") as fh: old=json.load(fh)
    with a.candidate.open(encoding="utf-8") as fh: new=json.load(fh)
    result=verify(old,new,a.problems)
    target=a.output.resolve()
    if target.exists():
        p.error("Refusing to overwrite migration verification")
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"targeted":result["targeted"],
                      "untouched":result["untouched_problem_count"],
                      "k_counts":result["k_counts"]},indent=2))


if __name__=="__main__":
    main()
