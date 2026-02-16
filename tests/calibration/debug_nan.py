# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import numpy as np
import pandas as pd
import json

# Ensure local MoeaBench is importable
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import MoeaBench.diagnostics.fair as fair
import MoeaBench.diagnostics.qscore as qscore
import MoeaBench.diagnostics.baselines as base

DATA_DIR = os.path.join(PROJ_ROOT, "tests/calibration_data")
GT_DIR = os.path.join(PROJ_ROOT, "tests/ground_truth")
BASELINE_FILE = os.path.join(PROJ_ROOT, "MoeaBench/diagnostics/resources/baselines_v4.json")

def load_nd_points(csv_path):
    pts = pd.read_csv(csv_path).values
    if pts.shape[1] > 3:
        pts = pts[:, :3]
    return pts


def debug():
    import glob
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_standard_run00.csv")))
    
    print(f"Found {len(files)} files.")
    
    with open(BASELINE_FILE, 'r') as f:
        baselines = json.load(f)

    for run_file in files:
        filename = os.path.basename(run_file)
        parts = filename.split('_')
        mop_name = parts[0]
        alg_name = parts[1]
        
        # Load GT
        gt_file = os.path.join(GT_DIR, f"{mop_name}_3_optimal.csv")
        if not os.path.exists(gt_file):
            # Try 2 objectives if 3 not found, or skip
            gt_file_2 = os.path.join(GT_DIR, f"{mop_name}_2_optimal.csv")
            if os.path.exists(gt_file_2):
                 gt_file = gt_file_2
            else:
                # print(f"Skipping {mop_name}: GT not found")
                continue
                
        F_opt = pd.read_csv(gt_file, header=None).values
        F_run = load_nd_points(run_file)
        
        if len(F_run) < 10:
            print(f"Skipping {mop_name} {alg_name}: Not enough points ({len(F_run)})")
            continue

        K_target = 50
        
        try:
            # S_FIT
            s_fit = base.get_resolution_factor_k(F_opt, K_target, seed=0)
            
            # Subsample
            P_eval = base.get_ref_uk(F_run, K_target, seed=0) if len(F_run) > K_target else F_run
            U_ref = base.get_ref_uk(F_opt, K_target, seed=0)
            C_cents, _ = base.get_ref_clusters(F_opt, c=32, seed=0)
            
            d_u = base.cdist(U_ref, C_cents)
            lab_u = np.argmin(d_u, axis=1)
            hist_ref = np.bincount(lab_u, minlength=len(C_cents)).astype(float)
            hist_ref /= np.sum(hist_ref)

            # Fair Metrics
            fair_f = fair.compute_fair_fit(P_eval, F_opt, s_fit)
            fair_c = fair.compute_fair_coverage(P_eval, F_opt)
            fair_g = fair.compute_fair_gap(P_eval, F_opt)
            fair_r = fair.compute_fair_regularity(P_eval, U_ref)
            fair_b = fair.compute_fair_balance(P_eval, C_cents, hist_ref)

            if np.isnan(fair_f) or np.isnan(fair_c) or np.isnan(fair_g) or np.isnan(fair_r) or np.isnan(fair_b):
                print(f"NAN DETECTED in {mop_name} {alg_name}!")
                print(f"  fair_f: {fair_f}")
                print(f"  fair_c: {fair_c}")
                print(f"  fair_g: {fair_g}")
                print(f"  fair_r: {fair_r}")
                print(f"  fair_b: {fair_b}")
                print(f"  s_fit: {s_fit}")
        except Exception as e:
            print(f"ERROR in {mop_name} {alg_name}: {e}")


if __name__ == "__main__":
    debug()
