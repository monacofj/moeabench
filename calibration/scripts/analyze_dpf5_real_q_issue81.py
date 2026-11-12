#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Descriptive paired analysis; no statistical acceptance/release rule.

Calibration columns share fixed authentic populations. Optimizer-seed rows
are summarized separately within each algorithm, K and generation budget.
No rank comparisons are made between different K. Exact ties remain ties.
"""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from calibration.scripts.study_dpf5_real_q_full_calibrations_issue81 import q_from_raw


def summary(values):
    v = np.asarray(values, dtype=float)
    return dict(min=float(v.min()), max=float(v.max()), mean=float(v.mean()),
                range=float(np.ptp(v)), sample_sd=float(v.std(ddof=1)) if v.size > 1 else None)


def pair_analysis(rows):
    pairs = []
    for a, b in combinations(range(len(rows)), 2):
        x, y = rows[a], rows[b]
        gap = np.asarray(x['candidate_q']) - y['candidate_q']
        historic = float(x['historical_q'] - y['historical_q'])
        pooled = float(x['pooled_reference_q_experimental'] - y['pooled_reference_q_experimental'])
        pairs.append(dict(a=a, b=b, candidate_gap_min=float(gap.min()),
                          candidate_gap_max=float(gap.max()), historical_gap=historic,
                          pooled_gap=pooled, exact_tie_columns=int(np.sum(gap == 0)),
                          strict_reversal_across_calibrations=bool(gap.min() < 0 < gap.max()),
                          strict_reversal_vs_historical=bool(np.any(gap * historic < 0)),
                          strict_reversal_pool_vs_historical=bool(pooled * historic < 0)))
    return pairs


def analyze(report):
    result = {}
    for k, cell in report['results_by_K'].items():
        rows = cell['real_runs']
        per_run = []
        for row in rows:
            q = np.asarray(row['candidate_q'])
            r = dict(optimizer=row['optimizer'], seed=row['seed'], generations=row.get('generations', report['generations']),
                     historical_q=row['historical_q'], calibration_q=summary(q),
                     signed_shift_vs_historical=summary(q - row['historical_q']),
                     pooled_q=row['pooled_reference_q_experimental'],
                     pooled_minus_individual_mean=row['pooled_minus_single_mean_q'])
            if 'raw_normalized_distances' in row:
                raw = row['raw_normalized_distances']
                pool = cell['experimental_pooled_reference']
                r['epsilon_only_fixed_pooled_ecdf'] = summary([
                    q_from_raw(raw, c['epsilon'], pool['ecdf']) for c in cell['replicates']])
                r['ecdf_only_fixed_mean_epsilon'] = summary([
                    q_from_raw(raw, pool['epsilon_mean'], c['ecdf']) for c in cell['replicates']])
            per_run.append(r)
        groups = {}
        for algorithm, budget in sorted({(r['optimizer'], r.get('generations', report['generations'])) for r in rows}):
            group = [r for r in rows if r['optimizer'] == algorithm and r.get('generations', report['generations']) == budget]
            matrix = np.asarray([r['candidate_q'] for r in group])
            groups[f'{algorithm}/{budget}'] = dict(
                optimizer_seeds=[r['seed'] for r in group],
                optimizer_seed_spread_at_each_fixed_calibration=[summary(v) for v in matrix.T],
                optimizer_seed_spread_historical=summary([r['historical_q'] for r in group]),
                optimizer_seed_spread_pooled=summary([r['pooled_reference_q_experimental'] for r in group]))
        pairs = pair_analysis(rows)
        result[k] = dict(per_run=per_run, optimizer_variability_by_algorithm_budget=groups,
                         pairwise_comparisons=pairs,
                         strict_reversals_across_calibrations=sum(p['strict_reversal_across_calibrations'] for p in pairs),
                         strict_reversals_vs_historical=sum(p['strict_reversal_vs_historical'] for p in pairs),
                         pairs_with_any_exact_tie=sum(p['exact_tie_columns'] > 0 for p in pairs),
                         candidate_zero_scores=int(np.sum(np.asarray([r['candidate_q'] for r in rows]) == 0)),
                         candidate_one_scores=int(np.sum(np.asarray([r['candidate_q'] for r in rows]) == 1)))
    return dict(status='DESCRIPTIVE_NONCERTIFIED', results_by_K=result,
                limitation='Finite observed ranges/SD are not confidence intervals; no acceptance threshold. Groups share calibration columns; budgets may share trajectory prefixes. Generation counts are not equal objective-evaluation budgets across engines.')


def verify_saved_evidence(report):
    """Rebuild full calibrations and audit saved authentic arrays independently."""
    import moeabench as mb
    from scipy.spatial import cKDTree
    from scipy.stats import wasserstein_distance
    from calibration.scripts.validate_dpf5_closeness_protocol_issue81 import frozen_reference, _sha
    from calibration.scripts.study_dpf5_real_q_full_calibrations_issue81 import make_replicate, _baseline_snapshot, pooled_candidate
    from calibration.scripts.generate_baselines import estimate_gt_normals, generate_half_normal_blur
    _, gt, ideal, nadir, model = frozen_reference()
    tree = cKDTree(gt)
    normals = estimate_gt_normals(gt)
    seen = set()
    runs_checked = calibrations_checked = 0
    max_scipy_q_error = 0.0
    compression = {}
    for path, expected in report['protected_artifact_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    for k, cell in report['results_by_K'].items():
        raw_references = []
        compression[k] = []
        for saved in cell['replicates']:
            seeds = {saved['seeds']['ideal'], saved['seeds']['blur'], *saved['fit_reserved_seeds']}
            assert len(seeds) == 8 and not seen.intersection(seeds)
            seen.update(seeds)
            rebuilt = make_replicate(model=model, gt=gt, ideal=ideal, nadir=nadir,
                                     tree=tree, normals=normals, k=int(k), scale=cell['s_K'],
                                     replicate=saved['replicate'], ideal_size=report['ideal_size'])
            for key, value in rebuilt.items():
                if key == 'ecdf':
                    np.testing.assert_array_equal(value, saved[key])
                else:
                    assert value == saved[key], (k, saved['replicate'], key)
            blur = generate_half_normal_blur(gt, normals, saved['sigma'], np.random.RandomState(saved['seeds']['blur']))
            raw_references.append(np.maximum(0., tree.query(blur)[0] / cell['s_K'] - saved['epsilon']))
            calibrations_checked += 1
        epsilon, ecdf = pooled_candidate(cell['replicates'])
        assert epsilon == cell['experimental_pooled_reference']['epsilon_mean']
        np.testing.assert_array_equal(ecdf, cell['experimental_pooled_reference']['ecdf'])
        for row in cell['real_runs']:
            p0, pf = np.asarray(row['P0_normalized']), np.asarray(row['Pf_normalized'])
            assert _sha(p0) == row['P0_sha256'] and _sha(pf) == row['Pf_sha256']
            physical = tree.query(pf)[0] / cell['s_K']
            np.testing.assert_array_equal(physical, row['raw_normalized_distances'])
            audit = mb.clinic.audit(pf, ground_truth=gt, initial_data=p0, problem='DPF5', k=int(k))
            assert _baseline_snapshot(audit) == {name: tuple(v) for name, v in row['non_closeness_snapshot'].items()}
            assert float(audit.quality.scores['Q_CLOSENESS'].value) == row['historical_q']
            np.testing.assert_array_equal([q_from_raw(physical, c['epsilon'], c['ecdf'])
                                           for c in cell['replicates']], row['candidate_q'])
            assert q_from_raw(physical, epsilon, ecdf) == row['pooled_reference_q_experimental']
            # Independent numerical oracle for the unchanged runtime W1 implementation.
            for c, expected in [(c, q) for c, q in zip(cell['replicates'], row['candidate_q'])] + [
                    (dict(epsilon=epsilon, ecdf=ecdf), row['pooled_reference_q_experimental'])]:
                u = np.maximum(0., physical - c['epsilon'])
                d_i = float(np.mean(u))
                d_r = float(wasserstein_distance(u, c['ecdf']))
                delta = float(np.mean(c['ecdf']))
                oracle = 0. if d_i >= delta else (1. if d_i + d_r <= 1e-12 else d_r / (d_i + d_r))
                error = abs(oracle - expected)
                max_scipy_q_error = max(max_scipy_q_error, error)
                assert error < 1e-12  # numerical equality, not scientific acceptance
            full_q = [q_from_raw(physical, c['epsilon'], raw)
                      for c, raw in zip(cell['replicates'], raw_references)]
            compression[k].append(dict(optimizer=row['optimizer'], generations=row['generations'], seed=row['seed'],
                individual_stored200_minus_full3367=summary(np.asarray(row['candidate_q']) - full_q),
                pooled200_minus_uncompressed_stored_quantile_mixture=row['pooled_reference_q_experimental'] - q_from_raw(
                    physical, epsilon, np.concatenate([c['ecdf'] for c in cell['replicates']])),
                pooled200_minus_mixture_of_all_corrected_raw_draws=row['pooled_reference_q_experimental'] - q_from_raw(
                    physical, epsilon, np.concatenate(raw_references))))
            runs_checked += 1
    return dict(max_absolute_q_error_vs_scipy=max_scipy_q_error,
                compression_sensitivity_by_K=compression,
                protected_file_hashes_checked=len(report['protected_artifact_sha256']),
                full_calibrations_rebuilt_exactly=calibrations_checked,
                saved_runs_reaudited_exactly=runs_checked)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--verify', action='store_true', help='Rebuild references and re-audit saved populations')
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    # This tool writes research evidence only, with the same protections as producer.
    from calibration.scripts.validate_dpf5_closeness_protocol_issue81 import RESOURCE_DIR, REPORTS, OFFICIAL
    dest = args.output.resolve()
    if dest.exists() or dest.is_relative_to(RESOURCE_DIR) or dest.is_relative_to(REPORTS) or dest == OFFICIAL:
        p.error('Refusing overwrite or protected path')
    report = json.loads(args.input.read_text())
    result = analyze(report)
    if args.verify:
        result['evidence_verification'] = verify_saved_evidence(report)
    result['input_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
