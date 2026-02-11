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
from collections import Counter

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
    
    # Q-Score Accumulators
    fit_vals, cov_vals, gap_vals, reg_vals, bal_vals = [], [], [], [], []
    
    # Fair Metric Accumulators (Physics)
    fair_fit_vals, fair_cov_vals, fair_gap_vals, fair_reg_vals, fair_bal_vals = [], [], [], [], []
    
    # Baseline Accumulators
    ideal_fit_vals, rand_fit_vals = [], []
    ideal_cov_vals, rand_cov_vals = [], []
    ideal_gap_vals, rand_gap_vals = [], []
    ideal_reg_vals, rand_reg_vals = [], []
    ideal_bal_vals, rand_bal_vals = [], []
    
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
                f_u, f_r = base.get_baseline_values(mop_name, K_target, "fit")
                c_u, c_r = base.get_baseline_values(mop_name, K_target, "cov")
                g_u, g_r = base.get_baseline_values(mop_name, K_target, "gap")
                u_u, u_r = base.get_baseline_values(mop_name, K_target, "reg")
                b_u, b_r = base.get_baseline_values(mop_name, K_target, "bal")
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

            q_f = qscore.compute_q_fit(fair_f, mop_name, K_target)
            q_c = qscore.compute_q_coverage(fair_c, mop_name, K_target)
            q_g = qscore.compute_q_gap(fair_g, mop_name, K_target)
            q_r = qscore.compute_q_regularity(fair_r, mop_name, K_target)
            q_b = qscore.compute_q_balance(fair_b, mop_name, K_target)
            
            # Store
            fit_vals.append(q_f); cov_vals.append(q_c); gap_vals.append(q_g); reg_vals.append(q_r); bal_vals.append(q_b)
            fair_fit_vals.append(fair_f); fair_cov_vals.append(fair_c); fair_gap_vals.append(fair_g); fair_reg_vals.append(fair_r); fair_bal_vals.append(fair_b)
            ideal_fit_vals.append(0.0); rand_fit_vals.append(f_r)
            ideal_cov_vals.append(c_u); rand_cov_vals.append(c_r)
            ideal_gap_vals.append(g_u); rand_gap_vals.append(g_r)
            ideal_reg_vals.append(u_u); rand_reg_vals.append(u_r)
            ideal_bal_vals.append(b_u); rand_bal_vals.append(b_r)
            k_used_vals.append(K_target); k_raw_vals.append(K_raw)
            igd_p_vals.append(GEN_igdplus([F_run], F_opt).evaluate()[0])
            gd_p_vals.append(GEN_gdplus([F_run], F_opt).evaluate()[0])
            
            # Record s_fit for the report
            metrics_entry = {"s_fit": s_fit}
            
        except Exception: 
            import traceback; traceback.print_exc()
            continue

    n_runs = len(fit_vals)
    def _med(lst): return float(np.nanmedian(lst)) if len(lst) else np.nan
    def _avg(lst): return float(np.mean(lst)) if len(lst) else np.nan
        
    mq = {"fit": _med(fit_vals), "cov": _med(cov_vals), "gap": _med(gap_vals), "reg": _med(reg_vals), "bal": _med(bal_vals)}
    mf = {"fit": _med(fair_fit_vals), "cov": _med(fair_cov_vals), "gap": _med(fair_gap_vals), "reg": _med(fair_reg_vals), "bal": _med(fair_bal_vals)}
    mi = {"fit": _med(ideal_fit_vals), "cov": _med(ideal_cov_vals), "gap": _med(ideal_gap_vals), "reg": _med(ideal_reg_vals), "bal": _med(ideal_bal_vals)}
    mr = {"fit": _med(rand_fit_vals), "cov": _med(rand_cov_vals), "gap": _med(rand_gap_vals), "reg": _med(rand_reg_vals), "bal": _med(rand_bal_vals)}
    
    summary_list = []
    # Fail-Closed NaN Check
    if np.isnan(mq["fit"]) or mq["fit"] < 0.67: summary_list.append("Insufficient Precision (Proximity to Front)")
    if np.isnan(mq["cov"]) or mq["cov"] < 0.67: summary_list.append("Limited Coverage (Reduced Extent)")
    if np.isnan(mq["gap"]) or mq["gap"] < 0.67: summary_list.append("Significant Topological Interruptions (Gaps)")
    if np.isnan(mq["reg"]) or mq["reg"] < 0.67: summary_list.append("Uneven Distribution Pattern (Irregularity)")
    if np.isnan(mq["bal"]) or mq["bal"] < 0.67: summary_list.append("Structural Bias (Non-Uniform Objective Coverage)")
    
    summary_text = "; ".join(summary_list) if summary_list else "Excellent Performance (All Dimensions)"
    
    q_worst = np.nanmin(list(mq.values()))
    # Strict fallback for all-NaN case
    if np.isnan(q_worst): q_worst = 0.0
    
    verdict = "RESEARCH" if q_worst >= 0.67 else ("INDUSTRY" if q_worst >= 0.34 else "FAIL")
    # Correct condition for undefined baseline
    if n_runs == 0 and n_undefined_baseline > 0: verdict = "UNDEFINED_BASELINE"

    return {
        "n_runs": n_runs, "n_valid": n_runs - n_k_fail, "n_k_fail": n_k_fail,
        "verdict": verdict, "summary": summary_text,
        "fit": {"q": mq["fit"], "fair": mf["fit"], "ideal": mi["fit"], "rand": mr["fit"]},
        "cov": {"q": mq["cov"], "fair": mf["cov"], "ideal": mi["cov"], "rand": mr["cov"]},
        "gap": {"q": mq["gap"], "fair": mf["gap"], "ideal": mi["gap"], "rand": mr["gap"]},
        "reg": {"q": mq["reg"], "fair": mf["reg"], "ideal": mi["reg"], "rand": mr["reg"]},
        "bal": {"q": mq["bal"], "fair": mf["bal"], "ideal": mi["bal"], "rand": mr["bal"]},
        "k_used": _med(k_used_vals), "k_raw": _med(k_raw_vals),
        "igd_p": {"mean": _avg(igd_p_vals), "std": float(np.std(igd_p_vals)) if igd_p_vals else 0},
        "gd_p": {"mean": _avg(gd_p_vals), "std": float(np.std(gd_p_vals)) if gd_p_vals else 0},
        "s_fit": s_fit if 's_fit' in locals() else 1.0 # The macroscopic ruler (s_K)
    }

def run_audit():
    if not os.path.exists(BASELINE_FILE): return print("Baseline CSV not found.")
    df_base = pd.read_csv(BASELINE_FILE)
    mops = sorted(df_base['MOP'].unique())
    audit_data = {"problems": {}}

    for mop_name in mops:
        print(f"Auditing {mop_name}...")
        mop_df = df_base[df_base['MOP'] == mop_name]
        
        # Load GT
        gt_file = os.path.join(GT_DIR, f"{mop_name}_3_optimal.csv")
        F_opt = pd.read_csv(gt_file, header=None).values if os.path.exists(gt_file) else None
        
        audit_data["problems"][mop_name] = {"ideal": None, "nadir": None}
        if F_opt is not None:
            audit_data["problems"][mop_name]["ideal"] = np.min(F_opt, axis=0).tolist()
            audit_data["problems"][mop_name]["nadir"] = np.max(F_opt, axis=0).tolist()
            # Representative GT (downsampled for 3D plot)
            audit_data["problems"][mop_name]["gt_points"] = F_opt[np.random.choice(len(F_opt), min(1000, len(F_opt)), replace=False)].tolist()

        audit_data["problems"][mop_name]["algorithms"] = {}
        algs = sorted(mop_df['Algorithm'].unique())
        for alg in algs:
            print(f"  - {alg}")
            # Final stats from baseline CSV
            alg_stats = mop_df[(mop_df['Algorithm'] == alg) & (mop_df['Intensity'] == 'standard')]
            stats_row = alg_stats.iloc[0].to_dict() if not alg_stats.empty else {}
            
            # Clinical Aggregation (Heavy)
            clinical = _aggregate_clinical(mop_name, alg, F_opt) if F_opt is not None else {}
            
            # Representative Final Front
            final_file = os.path.join(DATA_DIR, f"{mop_name}_{alg}_standard_run00.csv")
            F_obs = _load_nd_points(final_file).tolist() if os.path.exists(final_file) else []
            
            # CDF Data for validation plot
            cdf_dists = []
            point_dists = []
            if F_opt is not None and F_obs:
                from scipy.spatial.distance import cdist
                dists = cdist(np.array(F_obs), F_opt, metric='euclidean')
                # Save RAW distances for 3D point coloring (aligned to front)
                raw_d = np.min(dists, axis=1)
                point_dists = raw_d.tolist()
                # Save SORTED distances for CDF Plot
                cdf_dists = np.sort(raw_d).tolist()

            # Convergence History
            history = {"gens": [], "igd": [], "hv_rel": []}
            if F_opt is not None:
                from pymoo.indicators.igd import IGD
                metric_igd = IGD(F_opt, zero_to_one=True)
                # HV Setup
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

            audit_data["problems"][mop_name]["algorithms"][alg] = {
                "stats": stats_row,
                "clinical": clinical,
                "final_front": F_obs,
                "cdf_dists": cdf_dists,
                "point_dists": point_dists, # New field for correct 3D coloring
                "history": history
            }

    with open(AUDIT_JSON, "w") as f: json.dump(audit_data, f)
    print(f"Audit complete: {AUDIT_JSON}")

if __name__ == "__main__":
    run_audit()
