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
    from MoeaBench.diagnostics.auditor import audit
    from MoeaBench.diagnostics.enums import DiagnosticProfile
    import MoeaBench.mops as mops

    pattern = os.path.join(DATA_DIR, f"{mop_name}_{alg}_standard_run[0-9][0-9].csv")
    run_files = sorted(glob.glob(pattern))
    statuses = []
    profile_pass = {p.name: 0 for p in DiagnosticProfile}
    invalid_dim = 0
    invalid_other = 0
    score_globals = []
    score_covs = []
    score_pures = []
    score_shapes = []

    for rf in run_files:
        if "_gen" in os.path.basename(rf):
            continue
        try:
            F_run = _load_nd_points(rf)
            mop_obj = None
            try:
                mop_cls = getattr(mops, mop_name, None)
                mop_obj = mop_cls() if mop_cls else None
            except Exception:
                mop_obj = None

            class DummyRun:
                def __init__(self, objs, prob):
                    self.last_pop = type('P', (), {'objectives': objs})
                    self.experiment = type('E', (), {'mop': prob})

            run = DummyRun(F_run, mop_obj)
            d = audit(run, ground_truth=F_opt)
            if d.status.name in ["UNDEFINED_INPUT", "UNDEFINED_BASELINE", "UNDEFINED"]:
                invalid_other += 1
                continue
            statuses.append(d.status.name)
            for p in DiagnosticProfile:
                if d.verdicts.get(p.name, "FAIL") == "PASS":
                    profile_pass[p.name] += 1
            score_globals.append(float(d.metrics.get("score_global", np.nan)))
            score_covs.append(float(d.metrics.get("score_cov", np.nan)))
            score_pures.append(float(d.metrics.get("score_pure", np.nan)))
            score_shapes.append(float(d.metrics.get("score_shape", np.nan)))
        except ValueError:
            invalid_dim += 1
            continue
        except Exception:
            invalid_other += 1
            continue

    n_runs = len(statuses)
    status_mode = "UNDEFINED"
    if n_runs:
        # Prefer a robust, score-based aggregate diagnosis over "mode of run-level statuses".
        # This matches the report narrative: diagnosis is statistical across runs, not run00.
        cov_med = float(np.nanmedian(score_covs)) if len(score_covs) else np.nan
        pure_med = float(np.nanmedian(score_pures)) if len(score_pures) else np.nan
        shape_med = float(np.nanmedian(score_shapes)) if len(score_shapes) else np.nan
        # Research-tier threshold on calibrated score (interpretable: closest to quasi-uniform-at-K).
        if np.isfinite(cov_med) and np.isfinite(pure_med) and np.isfinite(shape_med):
            if max(cov_med, pure_med, shape_med) <= 0.25:
                status_mode = "IDEAL_FRONT"
            else:
                comps = {"cov": cov_med, "pure": pure_med, "shape": shape_med}
                dom = max(comps, key=lambda k: comps[k])
                if dom == "cov":
                    status_mode = "GAPPED_COVERAGE" if shape_med <= 0.50 else "COLLAPSED_FRONT"
                elif dom == "pure":
                    status_mode = "NOISY_POPULATION"
                else:
                    status_mode = "BIASED_SPREAD" if (cov_med <= 0.50 and pure_med <= 0.50) else "DISTORTED_COVERAGE"
                if float(np.nanmedian(score_globals)) >= 0.99:
                    status_mode = "SEARCH_FAILURE"
        else:
            status_mode = Counter(statuses).most_common(1)[0][0]

    pass_rates = {}
    for p, c in profile_pass.items():
        pass_rates[p] = (100.0 * c / n_runs) if n_runs else 0.0

    return {
        "n_runs": n_runs,
        "n_total": len(run_files),
        "n_invalid_dim": invalid_dim,
        "n_invalid_other": invalid_other,
        "status_mode": status_mode,
        "pass_rates": pass_rates,
        "score_global_median": float(np.nanmedian(score_globals)) if len(score_globals) else np.nan,
        "score_global_q25": float(np.nanpercentile(score_globals, 25)) if len(score_globals) else np.nan,
        "score_global_q75": float(np.nanpercentile(score_globals, 75)) if len(score_globals) else np.nan,
        "score_cov_median": float(np.nanmedian(score_covs)) if len(score_covs) else np.nan,
        "score_pure_median": float(np.nanmedian(score_pures)) if len(score_pures) else np.nan,
        "score_shape_median": float(np.nanmedian(score_shapes)) if len(score_shapes) else np.nan,
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
        ".diag-optimal { background: #dbfcfe; color: #0284c7; border: 1px solid #bae6fd; }", # Blue/Cyan for Geometry Ideal
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
                "status_mode": "UNDEFINED",
                "pass_rates": {"EXPLORATORY": 0.0, "INDUSTRY": 0.0, "STANDARD": 0.0, "RESEARCH": 0.0},
                "score_global_median": np.nan,
                "score_global_q25": np.nan,
                "score_global_q75": np.nan,
                "score_cov_median": np.nan,
                "score_pure_median": np.nan,
                "score_shape_median": np.nan,
            }
            if igd_vals:
                final_igd = igd_vals[-1]
                hv_rel_final = hv_rels[-1]
                
                # Topological Error (EMD) between final snapshot and GT
                if diag_res is not None:
                    emd_val = float(diag_res.metrics.get('emd', emd_val))
                elif F_opt is not None and F_snap is not None:
                    import MoeaBench as mb
                    # Use internal topo_distribution with EMD method
                    # This calculates axis-wise Wasserstein distance (EMD)
                    res = mb.stats.topo_distribution(F_snap, F_opt, method='emd')
                    emd_val = np.mean(list(res.results.values()))
                
                for g_idx, g_val in enumerate(igd_vals):
                    if g_val <= 1.05 * final_igd:
                        t_conv = str(gens[g_idx])
                        break

            if F_opt is not None:
                clinical_agg = _aggregate_clinical(mop_name, alg, F_opt)

            # Variability indicator (robust): IQR / median on score_global across runs.
            sg_med = clinical_agg["score_global_median"]
            iqr = clinical_agg["score_global_q75"] - clinical_agg["score_global_q25"]
            var_ratio = (iqr / sg_med) if (np.isfinite(iqr) and np.isfinite(sg_med) and sg_med > 1e-12) else np.nan
            consistency_flag = "STABLE" if (np.isnan(var_ratio) or var_ratio <= 0.60) else "VARIABLE"

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

        # 2. Build Certification Matrix table (Aggregated across runs)
        matrix_table = ["<h3>Certification & Pathology Matrix</h3>",
                        "<p class='diag-rationale' style='margin: 4px 0 12px 0;'>"
                        "Clinical certification is aggregated over all available <code>standard_runXX</code> files "
                        "for each algorithm (status mode + pass-rate by profile). The <code>run00</code> layer is visualization/debug only."
                        "</p>",
                        "<p class='diag-rationale' style='margin: 0 0 16px 0;'>"
                        "Status is diagnostic and invariant: it tells you which component dominates the front (coverage/purity/shape) "
                        "across 30 runs. Profiles are merely pass/fail gates on <code>score_global</code>. Thus a high pass-rate "
                        "in Exploratory/Industry with a <code>BIASED_SPREAD</code> status signals “geometria boa, falta uniformidade”, "
                        "and the report calls it out explicitly instead of changing the status."
                        "</p>",
                        "<table>",
                        "<tr><th>Algorithm</th>",
                        "<th>Geometry Status</th><th>Median scores</th>"]
        for p_name in ["EXPLORATORY", "INDUSTRY", "STANDARD", "RESEARCH"]:
            matrix_table.append(f"<th>{p_name} pass-rate</th>")
        matrix_table.append("<th>Runs</th><th>Skipped</th><th>Variability</th></tr>")

        for m in mop_metrics:
            matrix_table.append(f"<tr><td style='font-weight: bold; color: {colors_solid.get(m['alg'], 'black')}'>{m['alg']}</td>")

            agg = m.get("clinical_agg", {})
            status_name = agg.get("status_mode", "UNDEFINED")
            if status_name in ["IDEAL_FRONT", "SUPER_SATURATION"]:
                status_cls = "diag-optimal"
            elif status_name in ["SEARCH_FAILURE", "COLLAPSED_FRONT", "UNDEFINED_BASELINE"]:
                status_cls = "diag-failure"
            elif status_name == "UNDEFINED":
                status_cls = "diag-shadow"
            else:
                status_cls = "diag-warning"
            matrix_table.append(f"<td><span class='diag-badge {status_cls}'>{status_name}</span></td>")

            sg = agg.get("score_global_median", np.nan)
            sc = agg.get("score_cov_median", np.nan)
            sp = agg.get("score_pure_median", np.nan)
            ss = agg.get("score_shape_median", np.nan)
            if all(np.isfinite([sg, sc, sp, ss])):
                matrix_table.append(f"<td class='nowrap'><code>g={sg:.2f}</code> <code>cov={sc:.2f}</code> <code>pure={sp:.2f}</code> <code>shape={ss:.2f}</code></td>")
            else:
                matrix_table.append("<td class='nowrap'><span class='diag-badge diag-shadow'>n/a</span></td>")

            for p_name in ["EXPLORATORY", "INDUSTRY", "STANDARD", "RESEARCH"]:
                rate = float(agg.get("pass_rates", {}).get(p_name, 0.0))
                v_cls = "verdict-pass" if rate >= 50.0 else "verdict-fail"
                matrix_table.append(f"<td><span class='diag-badge {v_cls}'>{rate:.1f}%</span></td>")
            n_runs = int(agg.get('n_runs', 0))
            n_total = int(agg.get('n_total', n_runs))
            matrix_table.append(f"<td>{n_runs}/{n_total}</td>")
            skipped = int(agg.get('n_invalid_dim', 0)) + int(agg.get('n_invalid_other', 0))
            matrix_table.append(f"<td>{skipped}</td>")
            cons = m.get("consistency", "STABLE")
            c_cls = "diag-shadow" if cons == "STABLE" else "diag-warning"
            matrix_table.append(f"<td><span class='diag-badge {c_cls}'>{cons}</span></td>")
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
