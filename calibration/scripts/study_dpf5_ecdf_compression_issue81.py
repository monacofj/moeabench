#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Paired ECDF representation study: reuse saved calibrations and populations.

No optimization, sigma fitting, ideal sampling, or reference aggregation.
Full means all 3367 corrected negative distances with their original weights.
All reduced representations also have equal weights, preserving the runtime API.
"""
import argparse
import csv
import gzip
import hashlib
import json
import io
import platform
import sys
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance
from calibration.scripts.generate_baselines import estimate_gt_normals, generate_half_normal_blur
from calibration.scripts.validate_dpf5_closeness_protocol_issue81 import (
    frozen_reference, _sha, RESOURCE_DIR, REPORTS, OFFICIAL,
)
from calibration.scripts.study_dpf5_real_q_full_calibrations_issue81 import q_from_raw

EVIDENCE = ROOT / 'calibration/experimental/issue81/2026-11-12'
RESOLUTIONS = (200, 400, 800, 1600, 3367)
METHODS = {
    'compatibility_floor': 'z[floor(j*(n-1)/(m-1))], j=0..m-1; actual np.linspace(...).astype(int); mass 1/m',
    'endpoint_linear': 'linear interpolation of z at h=j*(n-1)/(m-1); mass 1/m (type-7 endpoint grid)',
    'midpoint_inverse': 'Q_emp((j+1/2)/m)=z[ceil(n*(j+1/2)/m)-1]; mass 1/m; L1-optimal quantile-bin median',
    'bin_mean': 'm*integral_[j/m,(j+1)/m] Q_emp(p) dp; exact integer overlap of empirical quantile cells; mass 1/m; preserves mean, L2-optimal bin constant',
}


def represent(values, m, method):
    z = np.sort(np.asarray(values, dtype=np.float64))
    n = len(z)
    if (z.ndim != 1 or n < 2 or not np.all(np.isfinite(z)) or
            np.any(z < 0) or not isinstance(m, int) or not 2 <= m <= n or method not in METHODS):
        raise ValueError('Expected finite nonnegative empirical law and 2 <= m <= n')
    if m == n:
        return z.copy()  # every definition is the exact empirical law at m=n
    if method == 'compatibility_floor':
        return z[np.linspace(0, n - 1, m).astype(int)]
    if method == 'endpoint_linear':
        return np.interp(np.linspace(0, n - 1, m), np.arange(n), z)
    if method == 'midpoint_inverse':
        # ceil(n*(2*j+1)/(2*m))-1, with exact integer arithmetic.
        ranks = (n * (2 * np.arange(m) + 1) + 2 * m - 1) // (2 * m) - 1
        return z[ranks]
    result = []
    for j in range(m):
        left, right = j * n, (j + 1) * n
        indices = np.arange(left // m, (right + m - 1) // m)
        overlap = np.minimum(right, (indices + 1) * m) - np.maximum(left, indices * m)
        result.append(float(np.dot(z[indices], overlap / n)))
    return np.asarray(result)


def stats(values):
    v = np.asarray(values, dtype=float).ravel()
    if not len(v) or not np.all(np.isfinite(v)):
        raise ValueError('Nonempty finite values required')
    return dict(n=int(len(v)), min=float(v.min()), max=float(v.max()), mean=float(v.mean()),
                rms=float(np.sqrt(np.mean(v*v))),
                quantiles={str(p): float(np.quantile(v, p)) for p in (0, .1, .25, .5, .75, .9, .95, .99, 1)})


def errors(q, full):
    e = np.asarray(q) - full
    valid = np.asarray(full) != 0
    return dict(signed=stats(e), absolute=stats(abs(e)),
                relative_absolute=stats(abs(e[valid] / full[valid])) if np.any(valid) else None,
                relative_undefined_full_zero=int(np.sum(~valid)))


def numerical_sign(a, b):
    """One-ULP diagnostic only; not a practical/scientific acceptance tolerance."""
    a, b = np.asarray(a), np.asarray(b)
    gap = a - b
    ulp = np.maximum(abs(np.spacing(a)), abs(np.spacing(b)))
    return np.where(abs(gap) <= ulp, 0, np.sign(gap)).astype(int)


def pairwise(q, full, historical=None):
    records = []
    for a, b in combinations(range(len(q)), 2):
        gap, baseline = q[a] - q[b], full[a] - full[b]
        sign, target = numerical_sign(q[a], q[b]), numerical_sign(full[a], full[b])
        historic = None if historical is None else float(historical[a]-historical[b])
        records.append(dict(a=a, b=b, gaps=gap.tolist(), full_gaps=baseline.tolist(),
            historical_gap=historic, strict_reversal_vs_historical=False if historic is None else bool(np.any(sign * np.sign(historic) < 0)),
            strict_reversal_across_calibrations=bool(np.any(sign < 0) and np.any(sign > 0)),
            full_strict_reversal_across_calibrations=bool(np.any(target < 0) and np.any(target > 0)),
            disagreement_replicates=np.flatnonzero(sign * target < 0).tolist(),
            exact_tie_replicates=np.flatnonzero(gap == 0).tolist(),
            numerical_tie_nonexact_replicates=np.flatnonzero((sign == 0) & (gap != 0)).tolist(),
            raw_sign_disagreement_replicates=np.flatnonzero(gap * baseline < 0).tolist()))
    return dict(pairs=records,
                strict_reversing_pairs=sum(r['strict_reversal_across_calibrations'] for r in records),
                full_strict_reversing_pairs=sum(r['full_strict_reversal_across_calibrations'] for r in records),
                strict_reversal_vs_historical_pairs=sum(r['strict_reversal_vs_historical'] for r in records),
                disagreeing_pairs=sum(bool(r['disagreement_replicates']) for r in records),
                disagreeing_pair_calibration_cells=sum(len(r['disagreement_replicates']) for r in records),
                exact_tie_cells=sum(len(r['exact_tie_replicates']) for r in records),
                numerical_tie_nonexact_cells=sum(len(r['numerical_tie_nonexact_replicates']) for r in records))


def variance_identity(q, full):
    e = q - full
    centered_full = full - full.mean(axis=1, keepdims=True)
    centered_e = e - e.mean(axis=1, keepdims=True)
    vf, ve = np.mean(centered_full**2, axis=1), np.mean(centered_e**2, axis=1)
    covariance = np.mean(centered_full * centered_e, axis=1)
    vq = np.var(q, axis=1)
    return dict(full_calibration_variance=vf.tolist(), representation_error_variance=ve.tolist(),
                covariance=covariance.tolist(), represented_variance=vq.tolist(),
                identity_max_residual=float(np.max(abs(vq - (vf + ve + 2*covariance)))),
                full_calibration_ranges=np.ptp(full, axis=1).tolist(),
                represented_calibration_ranges=np.ptp(q, axis=1).tolist(),
                centered_representation_error_ranges=np.ptp(e, axis=1).tolist())


def scipy_q(raw, epsilon, reference):
    u = np.maximum(0., np.asarray(raw) - epsilon)
    di, dr, delta = float(np.mean(u)), wasserstein_distance(u, reference), float(np.mean(reference))
    return 0. if di >= delta else (1. if di + dr <= 1e-12 else float(dr / (di + dr)))


def reconstruct(evidence):
    manifest = json.loads((evidence/'manifest.json').read_text())
    payloads = {}
    for name in ('expanded-real-q.json.gz', 'expanded-analysis.json.gz'):
        compressed = (evidence/name).read_bytes()
        if hashlib.sha256(compressed).hexdigest() != manifest['files'][name]['gzip_sha256']:
            raise AssertionError('Compressed source hash mismatch')
        raw = gzip.decompress(compressed)
        if hashlib.sha256(raw).hexdigest() != manifest['files'][name]['uncompressed_sha256']:
            raise AssertionError('Original source hash mismatch')
        payloads[name] = json.loads(raw)
    saved, previous = payloads['expanded-real-q.json.gz'], payloads['expanded-analysis.json.gz']
    for path, expected in saved['protected_artifact_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != expected:
            raise AssertionError('Protected artifact mismatch: ' + path)
    _, gt, ideal, nadir, _ = frozen_reference()
    if _sha(gt) != saved['frozen_GT_sha256'] or _sha(np.stack([ideal, nadir])) != saved['normalization_sha256']:
        raise AssertionError('Frozen geometry mismatch')
    generator = ROOT/'calibration/scripts/generate_baselines.py'
    if hashlib.sha256(generator.read_bytes()).hexdigest() != saved['source_sha256'][str(generator.relative_to(ROOT))]:
        raise AssertionError('Blur implementation differs from original')
    normals, tree = estimate_gt_normals(gt), cKDTree(gt)
    references, previous_full = {}, {}
    for k, cell in saved['results_by_K'].items():
        refs = []
        for cal in cell['replicates']:
            # Replay the original final draw at the SAVED sigma; never refit.
            blur = generate_half_normal_blur(gt, normals, cal['sigma'], np.random.RandomState(cal['seeds']['blur']))
            if _sha(blur) != cal['blur_objectives_sha256']:
                raise AssertionError('Replay did not reproduce original blur bytes')
            raw = tree.query(blur)[0] / cell['s_K']
            if float(np.median(raw)) != cal['held_out_normalized_median']:
                raise AssertionError('Held-out median mismatch')
            corrected = np.maximum(0., raw - cal['epsilon'])
            compat = represent(corrected, 200, 'compatibility_floor')
            if _sha(compat) != cal['ecdf_sha256']:
                raise AssertionError('Compatibility ECDF hash mismatch')
            np.testing.assert_array_equal(compat, cal['ecdf'])
            refs.append(np.sort(corrected))
        references[k] = np.asarray(refs)
        for row in cell['real_runs']:
            if _sha(row['P0_normalized']) != row['P0_sha256'] or _sha(row['Pf_normalized']) != row['Pf_sha256']:
                raise AssertionError('Original population hash mismatch')
            np.testing.assert_array_equal(tree.query(row['Pf_normalized'])[0] / cell['s_K'], row['raw_normalized_distances'])
        previous_full[k] = np.asarray([r['full3367_candidate_q'] for r in previous['evidence_verification']['compression_sensitivity_by_K'][k]])
    return saved, references, previous_full, manifest


def study(evidence=EVIDENCE):
    saved, refs, previous_full, manifest = reconstruct(evidence)
    result, qs, all_pairs, max_oracle_error = {}, [], {}, 0.
    for k, cell in saved['results_by_K'].items():
        runs, calibrations = cell['real_runs'], cell['replicates']
        full = np.asarray([[q_from_raw(row['raw_normalized_distances'], c['epsilon'], refs[k][j])
                            for j, c in enumerate(calibrations)] for row in runs])
        np.testing.assert_array_equal(full, previous_full[k])
        historical=np.asarray([r['historical_q'] for r in runs])
        base_pairs = pairwise(full, full, historical)
        closest = sorted(base_pairs['pairs'], key=lambda p: min(abs(np.asarray(p['gaps']))))[:10]
        labels = [{key: r[key] for key in ('optimizer', 'generations', 'seed', 'historical_q', 'P0_sha256', 'Pf_sha256')} for r in runs]
        groups = {}
        for algorithm, budget in sorted({(r['optimizer'], r['generations']) for r in runs}):
            rows = [i for i, r in enumerate(runs) if r['optimizer']==algorithm and r['generations']==budget]
            groups[f'{algorithm}/{budget}'] = dict(run_indices=rows, full_optimizer_seed_range_by_calibration=np.ptp(full[rows], axis=0).tolist())
        data = dict(run_labels=labels, full_q=full.tolist(), full_calibration_range=stats(np.ptp(full, axis=1)),
                    full_pairwise_summary={key: value for key, value in base_pairs.items() if key != 'pairs'},
                    closest_ten_pairs=closest, optimizer_seed_spread=groups,
                    bin_mean_population_quantile_alignment={str(m): all(m % len(row['raw_normalized_distances']) == 0 for row in runs) for m in RESOLUTIONS},
                    methods={})
        all_pairs[k] = {'full': base_pairs['pairs']}
        for method in METHODS:
            entries, matrices = {}, []
            for m in RESOLUTIONS:
                laws = [represent(z, m, method) for z in refs[k]]
                q = np.asarray([[q_from_raw(row['raw_normalized_distances'], c['epsilon'], laws[j])
                                 for j, c in enumerate(calibrations)] for row in runs])
                oracle_matrix = np.empty_like(q)
                for i, row in enumerate(runs):
                    for j, c in enumerate(calibrations):
                        oracle = scipy_q(row['raw_normalized_distances'], c['epsilon'], laws[j])
                        oracle_matrix[i,j] = oracle
                        max_oracle_error = max(max_oracle_error, abs(oracle - q[i,j]))
                        qs.append(dict(K=int(k), method=method, resolution=m, run_index=i,
                            optimizer=row['optimizer'], generations=row['generations'], seed=row['seed'],
                            calibration=j, epsilon=c['epsilon'], q=float(q[i,j]), full_q=float(full[i,j]),
                            signed_error=float(q[i,j]-full[i,j]), absolute_error=float(abs(q[i,j]-full[i,j])),
                            relative_absolute_error=float(abs(q[i,j]-full[i,j])/full[i,j]) if full[i,j]!=0 else None))
                if method == 'compatibility_floor' and m == 200:
                    np.testing.assert_array_equal(q, [r['candidate_q'] for r in runs])
                if m == 3367:
                    np.testing.assert_array_equal(q, full)
                pairs = pairwise(q, full, historical)
                oracle_sign_disagreements = sum(int(np.sum(numerical_sign(q[a],q[b]) != numerical_sign(oracle_matrix[a],oracle_matrix[b]))) for a,b in combinations(range(len(q)),2))
                all_pairs[k][f'{method}/{m}'] = pairs['pairs']
                entries[str(m)] = dict(errors=errors(q, full), oracle_pair_sign_disagreement_cells=oracle_sign_disagreements,
                    ranking={key: value for key,value in pairs.items() if key != 'pairs'},
                    variability=variance_identity(q, full),
                    reference_wasserstein_error=stats([wasserstein_distance(law,z) for law,z in zip(laws,refs[k])]),
                    reference_mean_error=stats([np.mean(law)-np.mean(z) for law,z in zip(laws,refs[k])]),
                    optimizer_seed_ranges={label: np.ptp(q[g['run_indices']],axis=0).tolist() for label,g in groups.items()})
                matrices.append(q)
            # Pointwise violations are counted exactly, without treating them as independent trials.
            entries['convergence'] = [dict(from_resolution=a, to_resolution=b,
                cells_with_increased_absolute_error=int(np.sum(abs(y-full)>abs(x-full))),
                cells_with_increase_beyond_two_score_ulps=int(np.sum((abs(y-full)-abs(x-full)) > (abs(np.spacing(x))+abs(np.spacing(y))))),
                max_absolute_error_change=float(np.max(abs(y-full)-abs(x-full))))
                for a,b,x,y in zip(RESOLUTIONS[:-1],RESOLUTIONS[1:],matrices[:-1],matrices[1:])]
            data['methods'][method] = entries
        result[k] = data
    if max_oracle_error >= 1e-12:
        raise AssertionError('Numerical W1 oracle disagreement')
    # Recheck protected hashes after all calculations.
    for path, expected in saved['protected_artifact_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != expected:
            raise AssertionError('Protected artifact mutated')
    global_summary = {}
    for method in METHODS:
        global_summary[method] = {}
        for m in RESOLUTIONS:
            subset=[r for r in qs if r['method']==method and r['resolution']==m]
            error=errors(np.asarray([r['q'] for r in subset]),np.asarray([r['full_q'] for r in subset]))
            global_summary[method][str(m)] = dict(errors=error,
                strict_reversing_pairs=sum(c['methods'][method][str(m)]['ranking']['strict_reversing_pairs'] for c in result.values()),
                disagreeing_pair_calibration_cells=sum(c['methods'][method][str(m)]['ranking']['disagreeing_pair_calibration_cells'] for c in result.values()))
    report = dict(status='NONCERTIFIED_ECDF_COMPRESSION_SENSITIVITY', resolutions=list(RESOLUTIONS),
        methods=METHODS, full_reference='3367 original corrected rows, multiplicities retained; no new calibration',
        input_manifest=manifest, original_environment=saved['environment'],
        global_summary=global_summary,
        calibrations_by_K={k:[dict({key:value for key,value in c.items() if key!='ecdf'},
                                  full_corrected_sorted_sha256=_sha(refs[k][i]))
                             for i,c in enumerate(cell['replicates'])] for k,cell in saved['results_by_K'].items()},
        environment=dict(python=platform.python_version(), **{n: version(n) for n in ('numpy','scipy','moeabench')}),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       (Path(__file__),ROOT/'moeabench/diagnostics/qscore.py',ROOT/'calibration/scripts/generate_baselines.py')},
        verification=dict(replayed_original_blurs=24, exact_saved_200_ecdfs=24, exact_saved_populations=48,
                          exact_previous_full_q_cells=384, protected_hashes=len(saved['protected_artifact_sha256']),
                          new_optimizations=0, new_sigma_fits=0, new_ideal_samples=0, max_scipy_q_discrepancy=max_oracle_error),
        numerical_ties='Exact zeros separately; nonzero gap <= max(ULP(q_a),ULP(q_b)) is numerical-resolution diagnostic only, no practical acceptance tolerance.',
        dependence='384 run/calibration cells share 48 populations and 24 calibrations. Methods/resolutions are paired, not new independent samples. Descriptive variance identity uses divisor 8; no uncertainty inference.',
        relative_error='abs(q_m-q_full)/abs(q_full); undefined at q_full=0, excluded and counted explicitly',
        results_by_K=result)
    return report, qs, all_pairs, refs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, default=EVIDENCE)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    dest = args.output_dir.resolve()
    if dest.exists() or dest.is_relative_to(RESOURCE_DIR) or dest.is_relative_to(REPORTS) or dest == OFFICIAL or dest.is_relative_to(OFFICIAL):
        p.error('Refusing existing/protected output directory')
    report, rows, pairs, refs = study(args.evidence)
    dest.mkdir(parents=True)
    (dest/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    buffer = io.StringIO(newline='')
    w = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
    (dest/'q-by-cell.csv.gz').write_bytes(gzip.compress(buffer.getvalue().encode(),mtime=0))
    with (dest/'method-comparison.csv').open('w') as f:
        fields=['method','resolution','mean_absolute_error','p95_absolute_error','max_absolute_error',
                'mean_relative_absolute_error','max_relative_absolute_error','strict_reversing_pairs',
                'disagreeing_pair_calibration_cells']
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n'); w.writeheader()
        for method, cells in report['global_summary'].items():
            for m, cell in cells.items():
                e=cell['errors']
                w.writerow(dict(method=method,resolution=m,mean_absolute_error=e['absolute']['mean'],
                    p95_absolute_error=e['absolute']['quantiles']['0.95'],max_absolute_error=e['absolute']['max'],
                    mean_relative_absolute_error=e['relative_absolute']['mean'],max_relative_absolute_error=e['relative_absolute']['max'],
                    strict_reversing_pairs=cell['strict_reversing_pairs'],disagreeing_pair_calibration_cells=cell['disagreeing_pair_calibration_cells']))
    (dest/'pairwise.json.gz').write_bytes(gzip.compress((json.dumps(pairs, sort_keys=True)+'\n').encode(),mtime=0))
    np.savez_compressed(dest/'reconstructed-full-references.npz', **{'K'+k: v for k,v in refs.items()})
    manifest = {f.name:dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size) for f in sorted(dest.iterdir())}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for method in METHODS:
        for m in RESOLUTIONS:
            maximum=max(report['results_by_K'][k]['methods'][method][str(m)]['errors']['absolute']['max'] for k in report['results_by_K'])
            print(method,m,'max |Q-Qfull|',maximum)


if __name__ == '__main__':
    main()
