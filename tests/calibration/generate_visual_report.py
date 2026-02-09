# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
MoeaBench v0.8.0 Visual Calibration Generator
===============================================

This script processes the full calibration trace to create an interactive 
HTML report. It combines 3D Pareto front visualizations with dynamic 
convergence curves (IGD/HV) based on snapshots.

Key Features:
---------------------------
- 3D Overlays: Projection of algorithms onto the "Mathematical Truth" (Ground Truth).
- Convergence Timeplots: IGD and Evolutive HV graphs throughout generations.
- Plotly Interactivity: Zoom, rotation, and point inspection directly in the browser.
- Consistent Normalization: Uses the same Ideal/Nadir bounds as v0.8.0.

Output:
------
- tests/CALIBRATION_v0.8.0.html

Usage:
----
python tests/calibration/generate_visual_report.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from collections import Counter
from plotly.subplots import make_subplots

# Ensure local MoeaBench is importable
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

# Paths
DATA_DIR = os.path.join(PROJ_ROOT, "tests/calibration_data")
GT_DIR = os.path.join(PROJ_ROOT, "tests/ground_truth")
BASELINE_FILE = os.path.join(PROJ_ROOT, "tests/baselines_v0.8.0.csv")
OUTPUT_HTML = os.path.join(PROJ_ROOT, "tests/CALIBRATION_v0.8.0.html")

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
    import MoeaBench.mops as mops
    import MoeaBench.clinic.indicators as clinic
    import MoeaBench.clinic.baselines as base
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    def is_non_dominated(pts):
        if len(pts) == 0: return []
        fronts = NonDominatedSorting().do(pts)
        if len(fronts) > 0:
            return pts[fronts[0]]
        return pts

    def _apply_k_policy(P, K_target):
        # Strict K-Policy:
        # If |P| > K, downsample with FPS (seed 0).
        # If |P| <= K, return P (baselines must exist for this size).
        if len(P) > K_target:
            return base.get_ref_uk(P, K_target, seed=0)
        return P

    pattern = os.path.join(DATA_DIR, f"{mop_name}_{alg}_standard_run[0-9][0-9].csv")
    run_files = sorted(glob.glob(pattern))
    
    # Quality Score Accumulators
    fit_vals = []
    cov_vals = []
    density_vals = []
    regularity_vals = []
    balance_vals = []
    
    # Validation Counters
    n_undefined_baseline = 0
    n_k_fail = 0
    k_used_list = []


    # Pre-compute Reference Objects for Q-Scores
    # We need to determine the Grid K first? No, specific to each run size?
    # Actually K-Policy says: select K based on |ND|.
    # But usually all runs have same N. Let's do it per-run to be safe.
    
    # Cache resolution factor for FIT
    s_gt = base.get_resolution_factor(F_opt)

    for rf in run_files:
        if "_gen" in os.path.basename(rf):
            continue
        try:
            F_run = _load_nd_points(rf)
            K_raw = len(F_run)
            
            # K-Selection Policy (Grid)
            if K_raw >= 50:
                # Max grid value <= K_raw
                grid = [50, 100, 150, 200]
                K_target = max([k for k in grid if k <= K_raw])
            elif K_raw >= 10:
                K_target = K_raw # Path B
            else:
                # Path C: Insufficient Data -> Absolute Failure (Quality=0.0)
                n_k_fail += 1
                fit_vals.append(0.0)
                cov_vals.append(0.0)
                density_vals.append(0.0)
                regularity_vals.append(0.0)
                balance_vals.append(0.0)
                continue
                
            k_used_list.append(K_target)
                
            # Downsample if needed
            P_eval = _apply_k_policy(F_run, K_target)
            
            # Load Baselines (Fail-Closed)
            # Need pairs (uni50, rand50) for 5 metrics
            try:
                # 1. FIT
                f_u, f_r = base.get_baseline_values(mop_name, K_target, "fit")
                # 2. COVERAGE
                c_u, c_r = base.get_baseline_values(mop_name, K_target, "coverage")
                # 3. GAP
                g_u, g_r = base.get_baseline_values(mop_name, K_target, "gap")
                # 4. UNIFORMITY
                u_u, u_r = base.get_baseline_values(mop_name, K_target, "uniformity")
                # 5. BALANCE
                b_u, b_r = base.get_baseline_values(mop_name, K_target, "balance")
            except ValueError:
                n_undefined_baseline += 1
                fit_vals.append(np.nan)
                cov_vals.append(np.nan)
                density_vals.append(np.nan)
                regularity_vals.append(np.nan)
                balance_vals.append(np.nan)
                continue

            # Generate References needed for Uni/Bal
            # U_K (FPS of GT, Seed 0)
            U_ref = base.get_ref_uk(F_opt, K_target, seed=0)
            
            # Clusters (KMeans of GT, Seed 0)
            C_cents, _ = base.get_ref_clusters(F_opt, c=32, seed=0)
            # Reference Histogram for Balance (U_K distribution)
            d_u = base.cdist(U_ref, C_cents)
            lab_u = np.argmin(d_u, axis=1)
            hist_ref = np.bincount(lab_u, minlength=len(C_cents)).astype(float)
            hist_ref /= np.sum(hist_ref)

            # Compute Quality Scores
            q_f = clinic.fit_quality(P_eval, F_opt, s_gt, f_u, f_r)
            q_c = clinic.coverage_quality(P_eval, F_opt, c_u, c_r)
            q_d = clinic.density_quality(P_eval, F_opt, g_u, g_r)
            q_r = clinic.regularity_quality(P_eval, U_ref, u_u, u_r)
            q_b = clinic.balance_quality(P_eval, C_cents, hist_ref, b_u, b_r)
            
            fit_vals.append(q_f)
            cov_vals.append(q_c)
            density_vals.append(q_d)
            regularity_vals.append(q_r)
            balance_vals.append(q_b)
            
        except Exception as e:
            # print(f"Run Error: {e}")
            continue

    n_runs = len(fit_vals)
    
    # Aggregation (Median)
    med_fit = float(np.nanmedian(fit_vals)) if n_runs else np.nan
    med_cov = float(np.nanmedian(cov_vals)) if n_runs else np.nan
    med_density = float(np.nanmedian(density_vals)) if n_runs else np.nan
    med_regularity = float(np.nanmedian(regularity_vals)) if n_runs else np.nan
    med_balance = float(np.nanmedian(balance_vals)) if n_runs else np.nan
    
    # Verdict Logic (Weakest Link)
    # Thresholds: Research <= 0.33, Industry <= 0.66, Fail > 0.66
    verdict = "UNDEFINED"
    summary_list = []
    
    if n_undefined_baseline > 0 and n_runs == n_undefined_baseline:
        verdict = "UNDEFINED_BASELINE"
        summary_list.append("Missing Reference")
    elif np.isfinite([med_fit, med_cov, med_density, med_regularity, med_balance]).all():
        min_q = min(med_fit, med_cov, med_density, med_regularity, med_balance)
        if min_q >= 0.67:
            verdict = "RESEARCH"
        elif min_q >= 0.34:
            verdict = "INDUSTRY"
        else:
            verdict = "FAIL"
            
        # Summary Construction (Flags for non-optimal dimensions)
        if med_fit < 0.67: summary_list.append("Poor Fit")
        if med_cov < 0.67: summary_list.append("Low Cov")
        if med_density < 0.67: summary_list.append("Low Density")
        if med_regularity < 0.67: summary_list.append("Irregular")
        if med_balance < 0.67: summary_list.append("Imbalanced")
        
    summary_text = ", ".join(summary_list) if summary_list else "Optimal"

    return {
        "n_runs": n_runs,
        "n_valid": n_runs - n_k_fail,
        "n_k_fail": n_k_fail,
        "n_undefined": n_undefined_baseline,
        "fit": med_fit,
        "cov": med_cov,
        "density": med_density,
        "regularity": med_regularity,
        "balance": med_balance,
        "verdict": verdict,
        "summary": summary_text
    }



def generate_visual_report():
    if not os.path.exists(BASELINE_FILE):
        print("Baseline CSV not found. Run analysis first.")
        return

    df_base = pd.read_csv(BASELINE_FILE)
    
    # Custom Sort: DTLZ before DPF, then alphabetical
    def mop_sort_key(name):
        if name.startswith('DTLZ'): return (0, name)
        if name.startswith('DPF'): return (1, name)
        return (2, name)
        
    mops = sorted(df_base['MOP'].unique(), key=mop_sort_key)
    
    html_content = [
        "<html><head><title>MoeaBench v0.8.0 Calibration</title>",
        "<script type='text/x-mathjax-config'>MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\\\(','\\\\)']]}});</script>",
        "<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML'></script>",
        "<style>body { font-family: 'Inter', system-ui, -apple-system, sans-serif; margin: 0; padding: 40px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); line-height: 1.6; color: #1e293b; min-height: 100vh; }",
        ".header-container { max-width: 95%; margin: 0 auto 50px auto; text-align: center; }",
        "h1 { color: #0f172a; font-size: 2.5em; letter-spacing: -0.02em; font-weight: 800; margin-bottom: 10px; background: linear-gradient(90deg, #3498db, #2c3e50); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }",
        "h2 { color: #334155; margin-top: 40px; font-size: 1.5em; font-weight: 700; border-left: 5px solid #3498db; padding-left: 15px; margin-bottom: 20px; }",
        "h3 { color: #475569; margin-top: 25px; font-weight: 600; font-size: 1.1em; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }",
        ".mop-section { background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); padding: 40px; border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.3); box-shadow: 0 10px 30px rgba(0,0,0,0.04); margin: 0 40px 60px 40px; }",
        ".metrics-footer { font-size: 0.85em; color: #64748b; margin-top: 25px; font-family: 'JetBrains Mono', 'Fira Code', monospace; background: rgba(0,0,0,0.02); padding: 20px; border-radius: 12px; border: 1px solid rgba(0,0,0,0.05); }",
        ".intro-box { background: white; padding: 40px; border-radius: 16px; margin-bottom: 50px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); max-width: 95%; margin-left: auto; margin-right: auto; }",
        ".note-box { background: #fffcf0; padding: 20px; border-radius: 10px; margin-top: 25px; border-left: 5px solid #f1c40f; font-size: 0.95em; color: #856404; }",
        "table { width: 100%; border-collapse: separate; border-spacing: 0; margin-top: 25px; margin-bottom: 35px; background: white; border-radius: 12px; border: 1px solid #e2e8f0; table-layout: fixed; }",
        "th { background: #f8fafc; color: #475569; font-weight: 600; text-align: left; padding: 12px 10px; border-bottom: 1px solid #e2e8f0; text-transform: uppercase; font-size: 0.7em; letter-spacing: 0.05em; }",
        "td { padding: 12px 10px; border-bottom: 1px solid #f1f5f9; font-size: 0.85em; }",
        ".nowrap { white-space: nowrap; }",
        "tr:last-child td { border-bottom: none; }",
        "tr:hover td { background-color: #f8fafc; }",
        ".diag-badge { padding: 3px 8px; border-radius: 4px; font-size: 0.7em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.02em; display: inline-block; margin-bottom: 4px; margin-right: 4px; }",
        ".diag-optimal { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }", # Green for Optimal (Research Tier)
        ".diag-failure { background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }",
        ".diag-warning { background: #fef9c3; color: #a16207; border: 1px solid #fef08a; }",
        ".diag-shadow { background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }",
        ".verdict-pass { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }", # Green for PASS
        ".verdict-fail { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }", # Red for FAIL
        ".diag-rationale { font-size: 0.8em; color: #64748b; font-style: italic; display: block; line-height: 1.3; margin-top: 4px; }",
        "code { background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.9em; color: #0f172a; }",
        "ul { padding-left: 20px; }",
        "li { margin-bottom: 10px; }",
        "</style></head><body>",
        "<div class='header-container'>",
        "<h1>MoeaBench v0.8.0 Technical Calibration Report</h1>",
        "<p>Scientific Performance Audit and Convergence Metrics</p>",
        "</div>",
        
        "<div class='intro-box'>",
        "<h2>1. Methodology & Experimental Context</h2>",
        "<p>This report serves as the official scientific audit for <b>MoeaBench v0.8.0</b>. The objective is to validate and calibrate the numerical integrity and topological fidelity of the framework's core algorithms against established mathematical benchmarks (Ground Truth).</p>",
        
        "<h3>Experimental Setup</h3>",
        "<ul>",
        "<li><b>Population Framework:</b> All algorithms used a population size of <code>N=200</code>.</li>",
        "<li><b>Evolutionary Budget:</b> Runs were executed for exactly <code>1000 generations</code>.</li>",
        "<li><b>Statistical Baseline:</b> Metrics in the summary tables are derived from <b>30 independent runs</b> to ensure stochastical significance.</li>",
        "<li><b>Objective Space:</b> All problems were configured with <code>M=3</code> objectives for volumetric analysis.</li>",
        "<li><b>Normalization Strategy:</b> We enforce <b>Strict Theoretical Normalization</b>. The [0, 1] range is mapped exclusively from the theoretical <i>Ideal</i> and <i>Nadir</i> points of the Ground Truth, not from observed data.</li>",
        "</ul>",
        
        "<h2>2. Metric Glossary & Interpretation</h2>",
        "<ul>",
        "<li><b>IGD (Inverted Generational Distance):</b> Measures both convergence (proximity) and diversity (spread). <i>Lower is strictly better.</i></li>",
        "<li><b>EMD (Uniform):</b> Earth Mover's Distance against a uniformized GT reference. Certification uses <b>EMD_eff</b> (ratio to calibrated floor for the same K), not a fixed absolute cutoff.</li>",
        "<li><b>Purity (GD_p95):</b> 95th-percentile distance from population to GT. Certification uses <b>Purity_eff</b> (ratio to calibrated floor).</li>",
        "<li><b>H_raw:</b> The absolute hypervolume calculated with Reference Point 1.1.</li>",
        r"<li><b>H_ratio:</b> $H_{raw} / RefBox$. Search area coverage (volume). Strictly $\le 1.0$.</li>",
        r"<li><b>H_rel:</b> $H_{raw} / H_{GT}$. Convergence to optimal front. Can exceed 100% due to saturation.</li>",
        "</ul>",
        
        "<h2>3. Clinical Quality Matrix: Interpretation</h2>",
        "<div class='note-box'>",
        "<strong>Quality Score Scale ($[0, 1]$):</strong> This matrix employs a <strong>High-is-Better</strong> scale. <br>",
        "&bull; <strong>1.00 (Optimal):</strong> Performance is statistically indistinguishable from a perfectly uniform sampling of the Ground Truth. <br>",
        "&bull; <strong>0.00 (Failure):</strong> Performance is no better than random sampling within the objective space manifold.",
        "</div>",
        
        "<ul style='margin-top: 20px;'>",
        "<li><b>FIT:</b> Measures the proximity of the population to the mathematical surface. High quality (near 1.0) indicates that points are exactly on the front.</li>",
        "<li><b>COVERAGE:</b> Evaluates the global spread. High quality indicates the algorithm found the entire extent of the Pareto front without skipping regions.</li>",
        r"<li><b>DENSITY ($Q_{density}$):</b> Inverse of the largest gap size ($IGD_{95}$). High quality assures the front is densely populated without significant interruptions in the manifold.</li>",
        r"<li><b>REGULARITY ($Q_{reg}$):</b> Inverse of the Wasserstein distance to a theoretical uniform lattice. High quality indicates equidistant spacing between solutions.</li>",
        r"<li><b>BALANCE ($Q_{bal}$):</b> Inverse of the Jensen-Shannon divergence from the ideal cluster distribution. High quality guarantees unbiased coverage across all trade-off regions.</li>",
        "</ul>",

        "<h3>3.1. Numerical Definition & Certification</h3>",
        "<p>The Quality Score $Q$ is calculated by normalizing raw metrics against two strict baselines: the <b>Optimal Uniform Sampling ($U_{ref}$)</b> and a <b>Random Sampling ($R_{ref}$)</b> of the Ground Truth.</p>",
        
        r"<div style='background:#f8fafc; padding:10px; border-left:4px solid #475569; margin: 10px 0;'>",
        r"<strong>Normalization Formula:</strong><br>",
        r"$$ Q(m) = 1.0 - \text{clip}\left( \frac{m_{obs} - m_{optimal}}{m_{random} - m_{optimal}}, 0, 1 \right) $$",
        r"<br>Where $m_{optimal}$ is the median metric of 30 theoretical uniform sets, and $m_{random}$ is the median of 30 random sets.",
        r"</div>",

        "<p><strong>Certification Terciles:</strong> The final verdict is determined by the <i>minimum</i> quality across all 5 dimensions (Weakest Link Principle).</p>",
        "<ul>",
        r"<li><span class='diag-badge verdict-pass'>RESEARCH TIER</span> ($Q_{min} \ge 0.67$): Algorithm is statistically indistinguishable from the theoretical optimum in all aspects.</li>",
        r"<li><span class='diag-badge diag-warning'>INDUSTRY TIER</span> ($0.34 \le Q_{min} < 0.67$): Algorithm is robust and better than random, but shows measurable deviation from perfection (e.g., slight irregularity).</li>",
        r"<li><span class='diag-badge verdict-fail'>FAILURE</span> ($Q_{min} \le 0.33$): Algorithm failed to converge or produced a distribution equivalent to (or worse than) random guessing in at least one dimension.</li>",
        "</ul>",
        "<li><b>T-conv (Stabilization):</b> The generation where the algorithm reaches a stable state (within 5% of its final IGD value).</li>",
        "<li><b>Time (s):</b> Average wall-clock execution time per run on the reference hardware.</li>",
        "</ul>",
        
        "<div class='note-box'>",
        "<strong>Scientific Note: The Discretization Effect & Negative H_diff</strong><br>",
        "In cases of near-perfect convergence, you may observe an H_rel exceeding 100%.<br>",
        "<ul><li><b>Cause:</b> The 'Ground Truth' is a discrete sample (2k points for DTLZ, 10k for DPF). If an algorithm fills the gaps between these reference points, its volume can mathematically exceed the reference's volume.</li>",
        "<li><b>Interpretation:</b> This indicates <b>performance saturation</b>. The algorithm has found a distribution that is strictly numerically superior to the discrete baseline.</li></ul>",
        "</div>",
        "</div>"
    ]

    for mop_name in mops:
        print(f"Processing {mop_name}...")
        mop_df = df_base[df_base['MOP'] == mop_name]
        
        # Load Ground Truth (Calibration is M=3)
        gt_file = os.path.join(GT_DIR, f"{mop_name}_3_optimal.csv")
        if os.path.exists(gt_file):
            F_opt = pd.read_csv(gt_file, header=None).values
            # Theoretical Bounds (Strict)
            ideal = np.min(F_opt, axis=0)
            nadir = np.max(F_opt, axis=0)
            
            # Use Pymoo to calc theoretical Max HV
            from pymoo.indicators.hv import Hypervolume
            hv_metric = Hypervolume(ref_point=np.array([1.1]*3),
                                    norm_ref_point=False,
                                    zero_to_one=True,
                                    ideal=ideal,
                                    nadir=nadir)
            hv_opt = hv_metric.do(F_opt)
        else:
            F_opt = None
            # Fallback to CSV if GT missing
            row0 = mop_df.iloc[0]
            ideal = np.array([row0['Ideal_1'], row0['Ideal_2'], row0['Ideal_3']])
            nadir = np.array([row0['Nadir_1'], row0['Nadir_2'], row0['Nadir_3']])
            hv_opt = row0['HV_opt']

        # Layout: 2 rows (3D + Convergence | CDF Analysis)
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.6, 0.4],
            row_heights=[0.6, 0.4],
            specs=[[{'type': 'scene', 'rowspan': 2}, {'type': 'xy', 'secondary_y': True}],
                   [None, {'type': 'xy'}]],
            subplot_titles=(f"Final Pareto Front (M=3)", "Convergence History", 
                            "Validation: Distance-to-GT CDF (Density)")
        )

        # 1. Add Ground Truth to 3D plot (Lowest Priority, subtle)
        if F_opt is not None:
            print(f"    - Trace: Ground Truth Group (GT)")
            try:
                fig.add_trace(go.Scatter3d(
                    x=F_opt[:,0], y=F_opt[:,1], z=F_opt[:,2],
                    mode='markers',
                    marker=dict(
                        size=2.0, 
                        color='#475569', # Slate-600: Strong Visibility
                        opacity=0.4
                    ),
                    name='Ground Truth',
                    legend='legend',
                    legendgroup='GT',
                    showlegend=True
                ), row=1, col=1)
            except Exception as e:
                print(f"      ERROR adding GT trace: {e}")

        algs = sorted(mop_df['Algorithm'].unique())
        # Scientific Aesthetics - Vibrant High-Res
        colors_rgba = {
            'NSGA2': 'rgba(99, 102, 241, 0.95)', # Indigo
            'NSGA3': 'rgba(5, 150, 105, 0.95)',  # Emerald
            'MOEAD': 'rgba(217, 119, 6, 0.95)'   # Amber
        }
        colors_solid = {
            'NSGA2': '#6366f1', 
            'NSGA3': '#059669', 
            'MOEAD': '#d97706'
        }
        
        # Store metrics for HTML table
        mop_metrics = []

        for alg in algs:
            # Get stats from baseline df
            alg_stats = mop_df[(mop_df['Algorithm'] == alg) & (mop_df['Intensity'] == 'standard')]
            
            # Initialize metrics with defaults
            igd_mean = 0.0
            igd_std = 0.0
            emd_val = 0.0
            gd_mean = 0.0
            gd_std = 0.0
            sp_mean = 0.0
            sp_std = 0.0
            hv_raw = 0.0
            hv_ratio = 0.0
            hv_rel_stat = 0.0
            time_avg = 0.0
            diag_res = None
            
            # Get Clinical Floors (N=200 fixed for this report)
            if not alg_stats.empty:
                row = alg_stats.iloc[0]
                igd_mean = row['IGD_mean']
                igd_std = row['IGD_std']
                gd_mean = row['GD_mean']
                gd_std = row['GD_std']
                sp_mean = row['SP_mean']
                sp_std = row['SP_std']
                hv_raw = row['H_raw'] 
                hv_ratio = row['H_ratio']
                hv_rel_stat = row['H_rel']
                time_avg = row['Time_sec']

            
            # Note: We re-calculate dynamic HV (rel) from the final snapshot (Gen 1000) for strictness
            hv_rel_final = 0.0

            # Final Front (from standard intensity)
            final_file = os.path.join(DATA_DIR, f"{mop_name}_{alg}_standard_run00.csv")
            F_obs = None
            if os.path.exists(final_file):
                print(f"    - Trace: {alg} 3D Final")
                try:
                    F_obs = _load_nd_points(final_file)

                    jitter_scale = 0.005
                    F_jitter = F_obs + np.random.normal(0, jitter_scale, F_obs.shape)

                    diag_res = None
                    min_dists = np.array([])
                    gt_min_dists = np.array([])
                    if F_opt is not None:
                        from MoeaBench.diagnostics.auditor import audit
                        diag_res = audit(F_obs, ground_truth=F_opt)
                        min_dists = np.array(diag_res.metrics.get('min_dists', []))
                        gt_min_dists = np.array(diag_res.metrics.get('gt_min_dists', []))

                    if len(min_dists) == len(F_obs):
                        valid = np.isfinite(min_dists)
                        if not np.any(valid):
                            valid[:] = True
                        threshold = np.nanpercentile(min_dists[valid], 50) if np.any(valid) else 0.0
                        good_mask = valid & (min_dists <= threshold)
                        bad_mask = valid & (min_dists > threshold)
                        if not np.any(good_mask):
                            good_mask = valid
                            bad_mask = np.zeros_like(valid, dtype=bool)
                    else:
                        good_mask = np.ones(len(F_obs), dtype=bool)
                        bad_mask = np.zeros(len(F_obs), dtype=bool)

                    good_points = F_jitter[good_mask]
                    bad_points = F_jitter[bad_mask]
                    alg_color = colors_rgba.get(alg, 'rgba(0,0,0,0.9)')
                    tag_warning = ""

                    if len(good_points):
                        fig.add_trace(go.Scatter3d(
                            x=good_points[:,0], y=good_points[:,1], z=good_points[:,2],
                            mode='markers',
                            marker=dict(
                                symbol='circle',
                                size=4.2,
                                color=alg_color,
                                opacity=0.8,
                                line=dict(width=0)
                            ),
                            name=f'{alg} (Close | filled)',
                            legend='legend',
                            legendgroup=alg,
                            showlegend=True
                        ), row=1, col=1)
                    if len(bad_points):
                        fig.add_trace(go.Scatter3d(
                            x=bad_points[:,0], y=bad_points[:,1], z=bad_points[:,2],
                            mode='markers',
                            marker=dict(
                                symbol='circle',
                                size=5.1,
                                color='rgba(0,0,0,0)',
                                opacity=0.65,
                                line=dict(width=1.8, color=alg_color)
                            ),
                            name=f'{alg} (Far | hollow)',
                            legend='legend',
                            legendgroup=alg,
                            showlegend=True
                        ), row=1, col=1)

                    if diag_res is not None and len(gt_min_dists):
                        # 2. Use Auditor to get Standardized Metrics and Distributions
                        
                        # A. 3D Coverage Heatmap (Gap Map)
                        cmax = float(np.nanpercentile(gt_min_dists, 95)) if np.isfinite(gt_min_dists).any() else 1.0
                        fig.add_trace(go.Scatter3d(
                            x=F_opt[:,0], y=F_opt[:,1], z=F_opt[:,2],
                            mode='markers',
                            marker=dict(
                                size=2.5,
                                color=gt_min_dists,
                                colorscale='Reds',
                                cmin=0, cmax=cmax,
                                opacity=0.8,
                                showscale=True,
                                colorbar=dict(title='Dist to GT', thickness=15)
                            ),
                            name=f'Gap Map ({alg})',
                            legend='legend',
                            legendgroup=alg,
                            visible='legendonly',
                            showlegend=True
                        ), row=1, col=1)

                        # B. CDF Analysis trace
                        min_dists_sorted = np.sort(min_dists)
                        y_cdf = np.arange(len(min_dists_sorted)) / float(len(min_dists_sorted))
                        
                        fig.add_trace(go.Scatter(
                            x=min_dists_sorted, y=y_cdf,
                            mode='lines',
                            line=dict(color=colors_solid.get(alg, 'black'), width=2),
                            name=f'{alg}',
                            legend='legend3',
                            legendgroup=alg,
                            showlegend=True,
                            hoverinfo='x+y+name'
                        ), row=2, col=2)
                        
                        # Add Decision Threshold Lines (Scientific Reference)
                        if alg == algs[0]:
                             # 95% Population Fraction Base Line (Reference)
                             fig.add_trace(go.Scatter(
                                x=[0, (np.max(min_dists_sorted) if len(min_dists_sorted) else 1.0)*1.5], y=[0.95, 0.95],
                                mode='lines',
                                line=dict(color='#64748b', dash='dot', width=0.8),
                                name='95% Target',
                                legendgroup='Threshold',
                                showlegend=False,
                                hoverinfo='skip'
                             ), row=2, col=2)
                        
                             # NOTE: We intentionally avoid drawing absolute distance gates here.
                             # The clinical auditor uses calibrated scores (0..1) rather than
                             # "distance <= X" lines, so those would be misleading/magic-numbery.


                except Exception as e:
                    print(f"      ERROR adding {alg} 3D trace: {e}")

            # Convergence History (Snapshots)
            gens = []
            igd_vals = []
            hv_rels = []
            
            # Use Pymoo IGD object for consistency
            metric_igd = None
            if F_opt is not None:
                from pymoo.indicators.igd import IGD
                metric_igd = IGD(F_opt, zero_to_one=True)

            F_snap = None
            for g in range(100, 1100, 100):
                snap_file = os.path.join(DATA_DIR, f"{mop_name}_{alg}_standard_run00_gen{g}.csv")
                if os.path.exists(snap_file):
                    F_snap = pd.read_csv(snap_file).values
                    if F_snap.shape[1] > 3: F_snap = F_snap[:, :3]
                    
                    # CRITICAL FIX: Filter Snapshots too
                    try:
                        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
                        if len(F_snap) > 0:
                            nds = NonDominatedSorting()
                            fronts = nds.do(F_snap)
                            if len(fronts) > 0:
                                F_snap = F_snap[fronts[0]]
                    except Exception:
                        pass # If NDS fails, use raw (better than crash)
                    
                    igd = float(metric_igd.do(F_snap)) if metric_igd else 0
                    
                    # Strict Theoretical HV Calculation
                    hv_calc = Hypervolume(ref_point=np.array([1.1]*3),
                                          norm_ref_point=False,
                                          zero_to_one=True,
                                          ideal=ideal,
                                          nadir=nadir)
                    hv_val = hv_calc.do(F_snap)
                    
                    # Ensure HV Rel
                    hv_rel_val = (hv_val / hv_opt) * 100 if hv_opt > 0 else 0
                    
                    gens.append(g)
                    igd_vals.append(igd)
                    hv_rels.append(hv_rel_val)

            # Calculate T-conv (stability point) and EMD (Topo Error)
            t_conv = "-"
            clinical_agg = {
                "n_runs": 0,
                "n_valid": 0,
                "n_k_fail": 0,
                "fit": np.nan,
                "cov": np.nan,
                "density": np.nan,
                "regularity": np.nan,
                "balance": np.nan,
                "verdict": "UNDEFINED",
                "summary": "-"
            }
            if igd_vals:
                final_igd = igd_vals[-1]
                hv_rel_final = hv_rels[-1]
                
                # Topological Error (EMD) between final snapshot and GT
                if diag_res is not None:
                    # The auditor stores topological error as 'shape' or 'emd'
                    emd_val = float(diag_res.metrics.get('emd', diag_res.metrics.get('shape', emd_val)))
                
                if emd_val == 0.0 and F_opt is not None and F_obs is not None:
                    try:
                        import MoeaBench as mb
                        # Fallback: Calculate axis-wise Wasserstein distance (EMD)
                        res = mb.stats.topo_distribution(F_obs, F_opt, method='emd')
                        emd_val = float(np.mean(list(res.results.values())))
                    except Exception:
                        pass
                
                for g_idx, g_val in enumerate(igd_vals):
                    if g_val <= 1.05 * final_igd:
                        t_conv = str(gens[g_idx])
                        break

            if F_opt is not None:
                clinical_agg = _aggregate_clinical(mop_name, alg, F_opt)

            # Variability indicator (robust): IQR / median on score_global across runs.
            # Updated: Use FIT score stability as proxy? Or just mark STABLE
            # For now, simplistic
            consistency_flag = "STABLE"

            mop_metrics.append({
                "alg": alg,
                "igd": f"{igd_mean:.4f}",
                "gd": f"{gd_mean:.4e} &plusmn; {gd_std:.1e}",
                "sp": f"{sp_mean:.4e} &plusmn; {sp_std:.1e}",
                "emd": f"{emd_val:.4f}",
                "h_raw": f"{hv_raw:.4f}",
                "h_ratio": f"{hv_ratio:.4f}",
                "h_rel": f"{hv_rel_stat * 100:.2f}%",
                "time": f"{time_avg:.2f}",
                "t_conv": t_conv,
                "clinical_agg": clinical_agg,
                "consistency": consistency_flag
            })

            if gens:
                print(f"    - Trace: {alg} Convergence Lines")
                try:
                    # Plot IGD (Primary Y)
                    fig.add_trace(go.Scatter(
                        x=gens, y=igd_vals,
                        mode='lines+markers',
                        line=dict(color=colors_solid.get(alg, 'black'), shape='spline'),
                        marker=dict(
                            size=6,
                            color=colors_rgba.get(alg, 'rgba(0,0,0,0.75)'),
                            line=dict(width=0) # Final Zen: No border
                        ),
                        name=f'{alg}',
                        legend='legend2',
                        legendgroup=alg,
                        showlegend=True
                    ), row=1, col=2)
                    
                    # Plot HV Ratio (Secondary Y)
                    fig.add_trace(go.Scatter(
                        x=gens, y=hv_rels,
                        mode='lines+markers',
                        line=dict(color=colors_solid.get(alg, 'black'), dash='dash', shape='spline'),
                        marker=dict(
                            size=6,
                            symbol='diamond',
                            color=colors_rgba.get(alg, 'rgba(0,0,0,0.75)'),
                            line=dict(width=0) # Final Zen: No border
                        ),
                        name=f'{alg} HV%',
                        legend='legend2',
                        legendgroup=alg,
                        showlegend=True
                    ), row=1, col=2, secondary_y=True)
                except Exception as e:
                    print(f"      ERROR adding {alg} convergence traces: {e}")
 
        # Clinical 3D Composition
        fig.update_layout(
            height=900,
            template='plotly_white',
            dragmode='turntable',
            scene=dict(
                xaxis=dict(title='f1', range=[0, 1.1*nadir[0]], gridcolor="#f1f5f9", showbackground=False, zerolinecolor="#e2e8f0"),
                yaxis=dict(title='f2', range=[0, 1.1*nadir[1]], gridcolor="#f1f5f9", showbackground=False, zerolinecolor="#e2e8f0"),
                zaxis=dict(title='f3', range=[0, 1.1*nadir[2]], gridcolor="#f1f5f9", showbackground=False, zerolinecolor="#e2e8f0"),
                camera=dict(eye=dict(x=1.7, y=1.7, z=1.5)),
                aspectmode='cube',
                domain=dict(x=[0, 0.6], y=[0, 1])
            ),
            margin=dict(l=0, r=20, b=0, t=60),
            legend=dict(
                orientation="v", yanchor="top", y=0.98, xanchor="left", x=0.02,
                bgcolor="rgba(255,255,255,0.8)", bordercolor="#e2e8f0", borderwidth=1,
                font=dict(size=10), tracegroupgap=5
            ),
            legend2=dict(
                orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98,
                bgcolor="rgba(255,255,255,0.8)", bordercolor="#e2e8f0", borderwidth=1,
                font=dict(size=10), tracegroupgap=5
            ),
            legend3=dict(
                orientation="v", yanchor="top", y=0.48, xanchor="right", x=0.98,
                bgcolor="rgba(255,255,255,0.8)", bordercolor="#e2e8f0", borderwidth=1,
                font=dict(size=10), tracegroupgap=5, title=dict(text="Algorithms", font=dict(size=11, weight='bold'))
            ),
            legend4=dict(
                orientation="v", yanchor="top", y=0.48, xanchor="left", x=0.62,
                bgcolor="rgba(255,255,255,0.8)", bordercolor="#e2e8f0", borderwidth=1,
                font=dict(size=10), tracegroupgap=5, title=dict(text="Audit Tiers", font=dict(size=11, weight='bold'))
            )
        )
        fig.update_yaxes(title_text="IGD (Log)", secondary_y=False, row=1, col=2, type="log", gridcolor="#f8fafc")
        fig.update_yaxes(title_text="H_rel %", secondary_y=True, row=1, col=2, range=[0, 115], showgrid=False)
        fig.update_xaxes(title_text="Generations", row=1, col=2, gridcolor="#f8fafc")

        # CDF Axis
        fig.update_xaxes(title_text="Dist to GT (Euclidean)", row=2, col=2, gridcolor="#f8fafc")
        fig.update_yaxes(title_text="CDF (Frac of Pop)", row=2, col=2, range=[0, 1.05], gridcolor="#f8fafc")

        # Convert to HTML
        div = fig.to_html(full_html=False, include_plotlyjs='cdn' if mop_name == mops[0] else False)
        
        html_content.append(f"<div class='mop-section'>")
        html_content.append(f"<h2>{mop_name} Benchmark Analysis</h2>")
        
        # 1. Build numerical metrics table
        metrics_table = ["<table>",
                 "<colgroup>",
                 "<col style='width: 90px'>",  # Alg
                 "<col style='width: 170px'>", # IGD
                 "<col style='width: 170px'>", # GD
                 "<col style='width: 170px'>", # SP
                 "<col style='width: 80px'>",  # EMD
                 "<col style='width: 80px'>",  # H_raw
                 "<col style='width: 80px'>",  # H_ratio
                 "<col style='width: 80px'>",  # H_rel
                 "<col style='width: 80px'>",  # Time
                 "<col style='width: 90px'>",  # Stabil
                 "</colgroup>",
                 "<tr>",
                 "<th>Algorithm</th>",
                 "<th>IGD (Mean &plusmn; Std)</th>",
                 "<th>GD (Mean &plusmn; Std)</th>",
                 "<th>SP (Mean &plusmn; Std)</th>",
                 "<th>EMD (Wasserstein)</th>",
                 "<th>H_raw</th>",
                 "<th>H_ratio</th>",
                 "<th>H_rel</th>",
                 "<th>Time(s)</th>",
                 "<th>Stabil.</th></tr>"]
        
        for m in mop_metrics:
            metrics_table.append(f"<tr><td style='font-weight: bold; color: {colors_solid.get(m['alg'], 'black')}'>{m['alg']}</td>")
            metrics_table.append(f"<td class='nowrap'>{m['igd']}</td>")
            metrics_table.append(f"<td class='nowrap'>{m['gd']}</td>")
            metrics_table.append(f"<td class='nowrap'>{m['sp']}</td>")
            metrics_table.append(f"<td class='nowrap'>{m['emd']}</td>")
            metrics_table.append(f"<td class='nowrap'>{m['h_raw']}</td>")
            metrics_table.append(f"<td class='nowrap'>{m['h_ratio']}</td>")
            metrics_table.append(f"<td class='nowrap'>{m['h_rel']}</td>")
            metrics_table.append(f"<td class='nowrap'>{m['time']}</td>")
            metrics_table.append(f"<td class='nowrap'>Gen {m['t_conv']}</td></tr>")
        metrics_table.append("</table>")

        # 2. Build Quality Matrix table (Aggregated across runs)
        matrix_table = [
            "<h3>Clinical Quality Matrix</h3>",
            "<p class='diag-rationale' style='margin: 4px 0 12px 0;'>"
            "Clinical certification is aggregated over all available <code>standard_runXX</code> files "
            "for each algorithm. The scores represent normalized <b>Quality</b>: 1.0 (Optimal) to 0.0 (Random)."
            "</p>",
            "<table style='margin-top:0'>",
            "<thead><tr>",
            "<th>Algorithm</th>",
            "<th>FIT</th>",
            "<th>COVERAGE</th>",
            "<th>DENSITY</th>",
            "<th>REGULARITY</th>",
            "<th>BALANCE</th>",
            "<th>SUMMARY</th>",
            "<th>RUNS</th>",
            "<th>CERTIFICATION</th>",
            "</tr></thead>"
        ]

        for m in mop_metrics:
            matrix_table.append(f"<tr><td style='font-weight: bold; color: {colors_solid.get(m['alg'], 'black')}'>{m['alg']}</td>")

            agg = m.get("clinical_agg", {})
            
            # 5 Quality Scores
            for col in ["fit", "cov", "density", "regularity", "balance"]:
                val = agg.get(col, np.nan)
                if not np.isfinite(val):
                    matrix_table.append("<td><span class='diag-badge diag-shadow'>N/A</span></td>")
                else:
                    # Color Mapping (High is Green)
                    if val >= 0.67: cls = "diag-optimal"
                    elif val >= 0.34: cls = "diag-warning"
                    else: cls = "diag-failure"
                    matrix_table.append(f"<td><span class='diag-badge {cls}'>{val:.2f}</span></td>")
            
            # Summary
            s = agg.get("summary", "-")
            matrix_table.append(f"<td style='font-size:0.8em; color:#64748b'>{s}</td>")

            n_runs = int(agg.get('n_runs', 0))
            n_valid = int(agg.get('n_valid', 0))
            n_fail = int(agg.get('n_k_fail', 0))
            
            # Detailed Breakdown in Runs Column
            if n_fail > 0:
                run_str = f"{n_valid}/{n_runs} <span style='font-size:0.8em; color:#ef4444'>({n_fail} Fail)</span>"
            else:
                run_str = f"{n_runs}"
            matrix_table.append(f"<td>{run_str}</td>")

            # Verdict
            v = agg.get("verdict", "UNDEFINED")
            if v == "RESEARCH": v_cls = "verdict-pass"
            elif v == "INDUSTRY": v_cls = "diag-warning"
            elif "UNDEFINED" in v: v_cls = "diag-shadow"
            else: v_cls = "verdict-fail"
            matrix_table.append(f"<td><span class='diag-badge {v_cls}'>{v}</span></td>")
            
            matrix_table.append("</tr>")
        
        matrix_table.append("</table>")
        
        html_content.append("".join(metrics_table))
        
        # Formatting bounds as readable tuples
        ideal_str = "(" + ", ".join([f"{v:.3f}" for v in ideal]) + ")"
        nadir_str = "(" + ", ".join([f"{v:.3f}" for v in nadir]) + ")"
        
        html_content.append(f"<div class='metrics-footer'><strong>Theoretical Reference:</strong><br>Ideal Point: {ideal_str}<br>Nadir Point: {nadir_str}<br>Sampled Reference HV: {hv_opt:.6f}</div>")
        html_content.append("<p class='diag-rationale' style='margin-top: 12px;'><strong>Visual Semantics:</strong> Filled points mark solutions close to the Ground Truth, while hollow markers highlight points that remain far from the GT surface.</p>")
        
        # Primary Visualization
        html_content.append(div)
        
        # Matrix at bottom
        html_content.append("".join(matrix_table))
        
        html_content.append(f"</div>")

    html_content.append("</body></html>")
    
    with open(OUTPUT_HTML, "w") as f:
        f.write("\n".join(html_content))
    
    print(f"\nSuccess! Interactive report generated at: {OUTPUT_HTML}")

if __name__ == "__main__":
    generate_visual_report()
