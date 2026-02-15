# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
MoeaBench Calibration Auditor (v0.9.0)
=====================================

This script acts as the "Analytical Intelligence" layer. It processes raw 
experimental data, calculates all Clinical Metrology metrics (Q-Scores), 
and generates a consolidated JSON audit file.

This separation allows the Visual Report generator to remain lightweight 
and purely focused on rendering.

Output:
-------
- tests/calibration_audit_v0.9.json
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

# Ensure local MoeaBench is importable
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

# Paths
DATA_DIR = os.path.join(PROJ_ROOT, "tests/calibration_data")
GT_DIR = os.path.join(PROJ_ROOT, "tests/ground_truth")
BASELINE_FILE = os.path.join(PROJ_ROOT, "tests/baselines_v0.8.0.csv")
AUDIT_JSON = os.path.join(PROJ_ROOT, "tests/calibration_audit_v0.9.json")

def _load_nd_points(csv_path):
    pts = pd.read_csv(csv_path).values
    if pts.shape[1] > 3:
        pts = pts[:, :3]
    if pts.shape[1] != 3:
        raise ValueError(f"Expected 3 objectives, got {pts.shape[1]}")
    try:
        from MoeaBench.core.utils import is_non_dominated
        pts = pts[is_non_dominated(pts)]
    except Exception:
        pass
    return pts

def _aggregate_clinical(mop_name, alg, F_opt):
    import MoeaBench.diagnostics.fair as fair
    import MoeaBench.diagnostics.qscore as qscore
    import MoeaBench.diagnostics.baselines as base
    from MoeaBench.metrics.GEN_igdplus import GEN_igdplus
    from MoeaBench.metrics.GEN_gdplus import GEN_gdplus

    pattern = os.path.join(DATA_DIR, f"{mop_name}_{alg}_standard_run[0-9][0-9].csv")
    run_files = sorted(glob.glob(pattern))
    
    # Fair Metric Accumulators (Physics)
    fair_denoise_vals, fair_cov_vals, fair_gap_vals, fair_reg_vals, fair_bal_vals = [], [], [], [], []
    u_closeness_vals = [] # List of arrays

    # Bucket FAIR samples by snapped K to compute distributional Q (Wasserstein)
    fair_by_k = {
        'denoise': defaultdict(list),
        'closeness': defaultdict(list), # Stores flat list of all u samples for this K
        'cov': defaultdict(list),
        'gap': defaultdict(list),
        'reg': defaultdict(list),
        'bal': defaultdict(list),
    }
    
    # Metadata/Extra
    k_used_vals, k_raw_vals = [], []
    igd_p_vals, gd_p_vals = [], []
    
    # Validation Counters
    n_undefined_baseline = 0
    n_k_fail = 0

    s_gt = base.get_resolution_factor(F_opt)
    s_fit = 1.0 # Default

    for rf in run_files:
        if "_gen" in os.path.basename(rf): continue
        try:
            F_run = _load_nd_points(rf)
            K_raw = len(F_run)
            
            if K_raw >= 10:
                K_target = base.snap_k(K_raw)
            else:
                n_k_fail += 1; continue
                
            # Calculate s_K (Scale) for this K
            s_fit = base.get_resolution_factor_k(F_opt, K_target, seed=0)

            # Load Baselines
            try:
                # Fetch ECDF values internally if needed, or just values for summary
                f_u, f_r = base.get_baseline_values(mop_name, K_target, "denoise")
                c_u, c_r = base.get_baseline_values(mop_name, K_target, "cov")
                g_u, g_r = base.get_baseline_values(mop_name, K_target, "gap")
                u_u, u_r = base.get_baseline_values(mop_name, K_target, "reg")
                b_u, b_r = base.get_baseline_values(mop_name, K_target, "bal")
                # For closeness, we may need get_baseline_ecdf later in _q_and_dists_weighted
            except base.UndefinedBaselineError:
                n_undefined_baseline += 1; continue

            # Subsample for Fair metrics
            P_eval = base.get_ref_uk(F_run, K_target, seed=0) if len(F_run) > K_target else F_run
            U_ref = base.get_ref_uk(F_opt, K_target, seed=0)
            C_cents, _ = base.get_ref_clusters(F_opt, c=32, seed=0)
            
            # Hist Ref
            d_u = base.cdist(U_ref, C_cents)
            lab_u = np.argmin(d_u, axis=1)
            hist_ref = np.bincount(lab_u, minlength=len(C_cents)).astype(float)
            hist_ref /= np.sum(hist_ref)

            # Metrics
            # Using s_fit (s_K) instead of s_gt for Clinical consistency (ADR 0026)
            fair_f = fair.compute_fair_fit(P_eval, F_opt, s_fit=s_fit) # Renamed param
            fair_c = fair.compute_fair_coverage(P_eval, F_opt)
            fair_g = fair.compute_fair_gap(P_eval, F_opt)
            fair_r = fair.compute_fair_regularity(P_eval, U_ref)
            fair_b = fair.compute_fair_balance(P_eval, C_cents, hist_ref)
            
            # New: Closeness distribution
            u_dist = fair.compute_fair_closeness_distribution(P_eval, F_opt, s_fit=s_fit)
            
            # Store FAIR values (per-run) and also bucket by K for distributional Q
            fair_denoise_vals.append(fair_f); fair_cov_vals.append(fair_c); fair_gap_vals.append(fair_g); fair_reg_vals.append(fair_r); fair_bal_vals.append(fair_b)
            u_closeness_vals.append(u_dist)
            
            fair_by_k["denoise"][K_target].append(fair_f)
            fair_by_k["closeness"][K_target].extend(u_dist.tolist()) # Accumulate points for distributional W1
            fair_by_k["cov"][K_target].append(fair_c)
            fair_by_k["gap"][K_target].append(fair_g)
            fair_by_k["reg"][K_target].append(fair_r)
            fair_by_k["bal"][K_target].append(fair_b)
            k_used_vals.append(K_target); k_raw_vals.append(K_raw)
            igd_p_vals.append(GEN_igdplus([F_run], F_opt).evaluate()[0])
            gd_p_vals.append(GEN_gdplus([F_run], F_opt).evaluate()[0])
            
            # Record s_fit for the report
            metrics_entry = {"s_fit": s_fit}
            
        except Exception: 
            import traceback; traceback.print_exc()
            continue
    n_runs = len(fair_denoise_vals)
    def _med(lst): return float(np.nanmedian(lst)) if len(lst) else np.nan
    def _avg(lst): return float(np.mean(lst)) if len(lst) else np.nan
    # Distributional Q via Wasserstein-1 against baseline random ECDF and a practical-ideal sample
    def _get_practical_ideal_samples(k: int):
        s_k = base.get_resolution_factor_k(F_opt, k, seed=0)
        U_ref = base.get_ref_uk(F_opt, k, seed=0)
        C_cents, _ = base.get_ref_clusters(F_opt, c=32, seed=0)
        d_u = base.cdist(U_ref, C_cents)
        lab_u = np.argmin(d_u, axis=1)
        hist_ref = np.bincount(lab_u, minlength=len(C_cents)).astype(float)
        hist_ref /= np.sum(hist_ref)

        N_IDEAL = 30
        vals = {'denoise': [], 'closeness': [], 'cov': [], 'gap': [], 'reg': [], 'bal': []}
        for i in range(N_IDEAL):
            pop_uni = base.get_ref_uk(F_opt, k, seed=100+i)
            vals['denoise'].append(fair.compute_fair_fit(pop_uni, F_opt, s_fit=s_k))
            vals['closeness'].extend(fair.compute_fair_closeness_distribution(pop_uni, F_opt, s_fit=s_k).tolist())
            vals['cov'].append(fair.compute_fair_coverage(pop_uni, F_opt))
            vals['gap'].append(fair.compute_fair_gap(pop_uni, F_opt))
            vals['reg'].append(fair.compute_fair_regularity(pop_uni, U_ref))
            vals['bal'].append(fair.compute_fair_balance(pop_uni, C_cents, hist_ref))
        return s_k, {m: np.asarray(v, float) for m, v in vals.items()}

    def _q_and_dists_weighted(metric: str):
        """Compute distributional Q and diagnostic distances (weighted across K).

        Returns (q, d_ideal, d_rand, delta), where:
          - d_ideal = W1(D_F, D_GT)
          - d_rand  = W1(D_F, D_R)
          - delta   = W1(D_GT, D_R)
        """
        total = 0
        acc_q = 0.0
        acc_di = 0.0
        acc_dr = 0.0
        acc_delta = 0.0
        for k, samples in fair_by_k[metric].items():
            if len(samples) == 0:
                continue
            try:
                _, _, rand_ecdf = base.get_baseline_ecdf(mop_name, k, metric)
            except base.UndefinedBaselineError:
                continue
            s_k, ideal_samps = _get_practical_ideal_samples(k)
            f_s = np.asarray(samples, float)
            i_s = ideal_samps[metric]
            r_s = np.asarray(rand_ecdf, float)

            # Q-score
            q_k = qscore.compute_q_wasserstein(f_s, i_s, r_s)

            # Diagnostics (same W1 used internally)
            d_i = qscore._wasserstein_1d(f_s, i_s)
            d_r = qscore._wasserstein_1d(f_s, r_s)
            delta = qscore._wasserstein_1d(i_s, r_s)
            w = len(samples)
            total += w
            acc_q += float(q_k) * w
            acc_di += float(d_i) * w
            acc_dr += float(d_r) * w
            acc_delta += float(delta) * w
        if total <= 0:
            return float('nan'), float('nan'), float('nan'), float('nan')
        return acc_q / total, acc_di / total, acc_dr / total, acc_delta / total

    q_denoise, di_denoise, dr_denoise, de_denoise = _q_and_dists_weighted('denoise')
    q_closeness, di_closeness, dr_closeness, de_closeness = _q_and_dists_weighted('closeness')
    q_cov, di_cov, dr_cov, de_cov = _q_and_dists_weighted('cov')
    q_gap, di_gap, dr_gap, de_gap = _q_and_dists_weighted('gap')
    q_reg, di_reg, dr_reg, de_reg = _q_and_dists_weighted('reg')
    q_bal, di_bal, dr_bal, de_bal = _q_and_dists_weighted('bal')

    mq = {'denoise': q_denoise, 'closeness': q_closeness, 'cov': q_cov, 'gap': q_gap, 'reg': q_reg, 'bal': q_bal}
    md = {
        'denoise': {'d_ideal': di_denoise, 'd_rand': dr_denoise, 'delta': de_denoise},
        'closeness': {'d_ideal': di_closeness, 'd_rand': dr_closeness, 'delta': de_closeness},
        'cov': {'d_ideal': di_cov, 'd_rand': dr_cov, 'delta': de_cov},
        'gap': {'d_ideal': di_gap, 'd_rand': dr_gap, 'delta': de_gap},
        'reg': {'d_ideal': di_reg, 'd_rand': dr_reg, 'delta': de_reg},
        'bal': {'d_ideal': di_bal, 'd_rand': dr_bal, 'delta': de_bal},
    }
    mf = {"denoise": _med(fair_denoise_vals), "cov": _med(fair_cov_vals), "gap": _med(fair_gap_vals), "reg": _med(fair_reg_vals), "bal": _med(fair_bal_vals)}
    if u_closeness_vals:
        # For fair_closeness (physics), we can show the median of all point-distances across all runs
        mf["closeness"] = _med(np.concatenate(u_closeness_vals))
    else:
        mf["closeness"] = np.nan

    # Anchors shown in report: (good) median of practical-ideal sample, (bad) baseline rand50, at reference K
    k_ref = int(np.nanmedian(k_used_vals)) if len(k_used_vals) else 10
    s_k_ref, ideal_ref = _get_practical_ideal_samples(k_ref)

    def _rand50(metric: str) -> float:
        _, r50 = base.get_baseline_values(mop_name, k_ref, metric)
        return float(r50)

    mi = {m: float(np.nanmedian(ideal_ref[m])) for m in ['denoise','closeness','cov','gap','reg','bal']}
    mr = {m: _rand50(m) for m in ['denoise','closeness','cov','gap','reg','bal']}
    
    q_worst = np.nanmin(list(mq.values()))
    # Strict fallback for all-NaN case
    if np.isnan(q_worst): q_worst = 0.0
    
    return {
        "n_runs": n_runs, "n_valid": n_runs - n_k_fail, "n_k_fail": n_k_fail,
        "denoise": {"q": mq["denoise"], "fair": mf["denoise"], "anchor_good": mi["denoise"], "anchor_bad": mr["denoise"], **md["denoise"]},
        "closeness": {"q": mq["closeness"], "fair": mf["closeness"], "anchor_good": mi["closeness"], "anchor_bad": mr["closeness"], **md["closeness"]},
        "cov": {"q": mq["cov"], "fair": mf["cov"], "anchor_good": mi["cov"], "anchor_bad": mr["cov"], **md["cov"]},
        "gap": {"q": mq["gap"], "fair": mf["gap"], "anchor_good": mi["gap"], "anchor_bad": mr["gap"], **md["gap"]},
        "reg": {"q": mq["reg"], "fair": mf["reg"], "anchor_good": mi["reg"], "anchor_bad": mr["reg"], **md["reg"]},
        "bal": {"q": mq["bal"], "fair": mf["bal"], "anchor_good": mi["bal"], "anchor_bad": mr["bal"], **md["bal"]},
        "k_used": _med(k_used_vals), "k_raw": _med(k_raw_vals),
        "igd_p": {"mean": _avg(igd_p_vals), "std": float(np.std(igd_p_vals)) if igd_p_vals else 0},
        "gd_p": {"mean": _avg(gd_p_vals), "std": float(np.std(gd_p_vals)) if gd_p_vals else 0},
        "s_fit": float(s_k_ref) # The macroscopic ruler (s_K)
    }

def _audit_problem_alg(mop_name, alg, F_opt, mop_df):
    """Worker function for parallel audit."""
    print(f"  - {alg}")
    # Final stats from baseline CSV
    alg_stats = mop_df[(mop_df['Algorithm'] == alg) & (mop_df['Intensity'] == 'standard')]
    stats_row = alg_stats.iloc[0].to_dict() if not alg_stats.empty else {}
    
    # Clinical Aggregation (Heavy)
    clinical = _aggregate_clinical(mop_name, alg, F_opt) if F_opt is not None else {}
    
    # Representative Final Front
    final_file = os.path.join(DATA_DIR, f"{mop_name}_{alg}_standard_run00.csv")
    F_obs_pts = _load_nd_points(final_file) if os.path.exists(final_file) else np.array([])
    F_obs = F_obs_pts.tolist()
    
    # CDF Data for validation plot
    cdf_dists = []
    point_dists = []
    if F_opt is not None and len(F_obs) > 0:
        from scipy.spatial.distance import cdist
        dists = cdist(np.array(F_obs), F_opt, metric='euclidean')
        raw_d = np.min(dists, axis=1)
        point_dists = raw_d.tolist()
        cdf_dists = np.sort(raw_d).tolist()

    # Convergence History
    history = {"gens": [], "igd": [], "hv_rel": []}
    if F_opt is not None:
        try:
            from pymoo.indicators.igd import IGD
            metric_igd = IGD(F_opt, zero_to_one=True)
            from pymoo.indicators.hv import Hypervolume
            ideal, nadir = np.min(F_opt, axis=0), np.max(F_opt, axis=0)
            hv_engine = Hypervolume(ref_point=np.array([1.1]*3), norm_ref_point=False, zero_to_one=True, ideal=ideal, nadir=nadir)
            hv_opt = hv_engine.do(F_opt)
            
            for g in range(100, 1100, 100):
                snap_file = os.path.join(DATA_DIR, f"{mop_name}_{alg}_standard_run00_gen{g}.csv")
                if os.path.exists(snap_file):
                    F_snap = _load_nd_points(snap_file)
                    history["gens"].append(g)
                    history["igd"].append(float(metric_igd.do(F_snap)))
                    hv_v = hv_engine.do(F_snap)
                    history["hv_rel"].append((hv_v / hv_opt) * 100 if hv_opt > 0 else 0)
        except Exception as e:
            print(f"Warning: History calculation failed for {mop_name}_{alg}: {e}")

    return alg, {
        "stats": stats_row,
        "clinical": clinical,
        "final_front": F_obs,
        "cdf_dists": cdf_dists,
        "point_dists": point_dists,
        "history": history
    }

def run_audit():
    if not os.path.exists(BASELINE_FILE): return print("Baseline CSV not found.")
    df_base = pd.read_csv(BASELINE_FILE)
    mops = sorted(df_base['MOP'].unique())
    
    # Load existing data for RESUME capability
    audit_data = {"problems": {}}
    if os.path.exists(AUDIT_JSON):
        try:
            with open(AUDIT_JSON, "r") as f:
                audit_data = json.load(f)
            print(f"Resuming audit. {len(audit_data['problems'])} problems already in JSON.")
        except:
            print("Could not load existing audit JSON. Starting fresh.")

    from concurrent.futures import ProcessPoolExecutor

    for mop_name in mops:
        if mop_name in audit_data["problems"] and audit_data["problems"][mop_name].get("algorithms"):
            # Deep check for completeness
            if len(audit_data["problems"][mop_name]["algorithms"]) >= 3:
                print(f"Problem {mop_name} already audited. Skipping.")
                continue

        print(f"Auditing {mop_name}...")
        mop_df = df_base[df_base['MOP'] == mop_name]
        
        # Load GT
        gt_file = os.path.join(GT_DIR, f"{mop_name}_3_optimal.csv")
        F_opt = pd.read_csv(gt_file, header=None).values if os.path.exists(gt_file) else None
        
        audit_data["problems"][mop_name] = {"utopia": None, "nadir": None}
        if F_opt is not None:
            audit_data["problems"][mop_name]["utopia"] = np.min(F_opt, axis=0).tolist()
            audit_data["problems"][mop_name]["nadir"] = np.max(F_opt, axis=0).tolist()
            # Representative GT (downsampled for 3D plot)
            audit_data["problems"][mop_name]["gt_points"] = F_opt[np.random.choice(len(F_opt), min(1000, len(F_opt)), replace=False)].tolist()

        audit_data["problems"][mop_name]["algorithms"] = {}
        algs = sorted(mop_df['Algorithm'].unique())
        
        # USE PROCESS POOL FOR ALGORITHMS
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(_audit_problem_alg, mop_name, alg, F_opt, mop_df) for alg in algs]
            for future in futures:
                alg_name, alg_res = future.result()
                audit_data["problems"][mop_name]["algorithms"][alg_name] = alg_res

        # Incremental Save
        with open(AUDIT_JSON, "w") as f: json.dump(audit_data, f)
        print(f"Incremental save: {mop_name} completed.")

    print(f"Audit complete: {AUDIT_JSON}")

if __name__ == "__main__":
    run_audit()
