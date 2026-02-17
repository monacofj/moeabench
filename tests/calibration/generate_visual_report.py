# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
MoeaBench Calibration Visual Report Generator (v0.9.0)
=====================================================

Decoupled Renderer: Consumes pre-calculated audit data from JSON.
"""

import os
import sys
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Paths
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
AUDIT_JSON = os.path.join(PROJ_ROOT, "tests/calibration_audit_v0.9.json")
OUTPUT_HTML = os.path.join(PROJ_ROOT, "tests/CALIBRATION_v0.9.html")

# Ensure project root in path for imports
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from MoeaBench.diagnostics import qscore


def generate_visual_report():
    if not os.path.exists(AUDIT_JSON):
        print(f"Error: Audit file {AUDIT_JSON} not found. Run audit_calibration.py first.")
        return

    with open(AUDIT_JSON, "r") as f:
        audit_data = json.load(f)

    problems = audit_data["problems"]
    
    # Custom Sort Order: DTLZ first, then DPF (or others)
    def mop_sort_key(name):
        if "DTLZ" in name:
            return (0, name) # Priority 0
        return (1, name) # Priority 1
        
    mops = sorted(problems.keys(), key=mop_sort_key)

    html_content = [
        "<html><head><title>MoeaBench v0.9.0 Calibration</title>",
        "<style>",
        "body { font-family: 'Inter', system-ui, sans-serif; background: #f8fafc; color: #1e293b; margin: 0; padding: 2rem; }",
        "h1 { color: #0f172a; border-left: 5px solid #6366f1; padding-left: 1rem; margin-bottom: 0; }",
        ".mop-section { background: white; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); padding: 1.5rem; margin-top: 2rem; border: 1px solid #e2e8f0; }",
        "table { width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.9rem; }",
        "th { background: #f1f5f9; text-align: left; padding: 0.75rem; border-bottom: 2px solid #e2e8f0; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; }",
        "td { padding: 0.75rem; border-bottom: 1px solid #f1f5f9; }",
        ".diag-badge { display: inline-block; padding: 0.25rem 0.5rem; border-radius: 6px; font-weight: 600; text-align: center; min-width: 60px; white-space: nowrap; }",
        ".diag-optimal { background: #dcfce7; color: #166534; }",
        ".diag-warning { background: #fef9c3; color: #854d0e; }",
        ".diag-failure { background: #fee2e2; color: #991b1b; }",
        ".matrix-summary { font-style: italic; color: #64748b; font-size: 0.85rem; min-width: 200px; white-space: normal; }",
        "th, td:not(.matrix-summary) { white-space: normal; }",
        "</style></head><body>",
        "<h1>MoeaBench v0.9.0 Technical Calibration Report</h1>",
        "<p>This report serves as the official scientific audit for <b>MoeaBench v0.9.0</b>. It implements the <i>Clinical Metrology</i> standard (ADR 0026) for objective framework certification.</p>",
        "<div class='didactic-box' style='background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 30px;'>",
        "<h2>1. Methodology & Experimental Context</h2>",
        "<p>The goal is to certify algorithmic performance against rigorous mathematical <b>Ground Truth (GT)</b> benchmarks.</p>",
        "<ul>",
        "<li><b>Population:</b> Algorithms ran with <i>N=200</i>. Q-Scores reflect the performance of the effective non-dominated population ($K \\le 200$).</li>",
        "<li><b>Scale Invariance:</b> All distance-based metrics are normalized by the <i>Utopia</i> and <i>Nadir</i> points of the theoretical manifold.</li>",
        "<li><b>Statistical Validity:</b> Results aggregated from <b>30 independent runs</b> to filter stochastic noise.</li>",
        "</ul>",

        "<h2>2. Hierarchy of Evidence</h2>",
        "<p>The data is presented in a linear progression: from raw geometry to resolution-corrected \"Fair\" metrics, and finally to calibrated clinical Q-Scores.</p>",
        
        "<h3>Tier 1: Raw Performance Diagnostics</h3>",
        "<p>First, we measure simple Euclidean errors to assess the absolute geometry of the solution set relative to the Reference Set.</p>",
        "<table class='metric-table'>",
        "<tr><th>Metric</th><th>Scientific Concept & Conceptual Calculation</th></tr>",
        "<tr><td><b>GD</b></td><td><b>Generational Distance:</b> The average distance from your solutions to the nearest point in the Reference Set.<br><i>Concept:</i> Pure convergence error.</td></tr>",
        "<tr><td><b>IGD</b></td><td><b>Inverted Generational Distance:</b> The average distance from every Reference Point to your nearest solution.<br><i>Concept:</i> Convergence + Diversity.</td></tr>",
        "<tr><td><b>GD+</b></td><td><b>GD (Pareto-Compliant):</b> Modified GD using Pareto-compliant distance.<br><i>Concept:</i> Mathematical rigor for weak-dominance.</td></tr>",
        "<tr><td><b>IGD+</b></td><td><b>IGD (Pareto-Compliant):</b> The gold standard for raw performance.<br><i>Concept:</i> Ensures the population effectively \"covers\" the target manifold.</td></tr>",
        "<tr><td><b>SP</b></td><td><b>Spacing:</b> Standard deviation of neighbor distances.<br><i>Concept:</i> Measures \"Clumping\" vs Uniformity.</td></tr>",
        "<tr><td><b>NEAR@1 / @2</b></td><td><b>High-Precision Count:</b> % of population with error < 1% or 2% of resolution.<br><i>Concept:</i> Density of the \"Elite\" population.</td></tr>",
        "<tr><td><b>H_rel</b></td><td><b>Relative Hypervolume:</b> Dominated volume normalized by Ideal volume.<br><i>Concept:</i> Fraction of the trade-off space captured.</td></tr>",
        "<tr><td><b>Stabil.</b></td><td><b>Stabilization Gen:</b> Earliest generation hitting 99.9% of final quality.<br><i>Concept:</i> Learning efficiency.</td></tr>",
        "</table>",

        "<h3>Tier 2: The Physical \"Fair\" Metrics</h3>",
        "<p>Before assigning a grade, we compute the <b>Physical Metrics</b>. These are \"Fair\" because they are robust to outliers and, where applicable, normalized by the problem's resolution (s_K).</p>",
        "<table class='metric-table'>",
        "<tr><th>Fair Metric</th><th>Physical Definition (Measurement)</th></tr>",
        "<tr><td><b>Fair Denoise</b></td><td><b>Convergence Depth:</b> The 95th percentile distance to the Ground Truth, divided by resolution (s_K).<br><i>Meaning:</i> \"How many pixels away is the front?\"</td></tr>",
        "<tr><td><b>Fair Closeness</b></td><td><b>Proximity Distribution:</b> The set of all distances to Ground Truth, divided by s_K.<br><i>Meaning:</i> The complete distribution of convergence errors.</td></tr>",
        "<tr><td><b>Fair Coverage</b></td><td><b>Extent Error:</b> The mean distance from the Ground Truth to the Front (IGD Mean).<br><i>Meaning:</i> The average gap left uncovered.</td></tr>",
        "<tr><td><b>Fair Gap</b></td><td><b>Continuity Breach:</b> The 95th percentile of the IGD (Worst Hole).<br><i>Meaning:</i> The size of the largest gap in the front.</td></tr>",
        "<tr><td><b>Fair Regularity</b></td><td><b>Uniformity Mismatch:</b> Wasserstein distance between the nearest-neighbor distribution and a perfect lattice.<br><i>Meaning:</i> How much the grid spacing deviates from ideal.</td></tr>",
        "<tr><td><b>Fair Balance</b></td><td><b>Spread Bias:</b> Jensen-Shannon divergence of cluster variation.<br><i>Meaning:</i> Does the algorithm ignore certain regions?</td></tr>",
        "</table>",

        "<h3>Tier 3: The Clinical Quality Matrix (Q-Scores)</h3>",
        "<p>The <b>Q-Scores</b> [0.0 to 1.0] are the final clinical grades. Each cell in the matrix displays a score derived from a <b>Calibration Console</b>.</p>",

        "<h4>3.1 How to Read the Console (f, g, b)</h4>",
        "<table class='console-table'>",
        "<tr><th>Symbol</th><th>Definition</th><th>Physical Meaning</th></tr>",
        "<tr><td><b>f</b></td><td><b>Fair Measured</b></td><td>The value actually measured for your algorithm (from Tier 2).</td></tr>",
        "<tr><td><b>g</b></td><td><b>Goal / Good</b></td><td>The value of the <b>Ideal State</b> (e.g., Ground Truth = 0.0).</td></tr>",
        "<tr><td><b>b</b></td><td><b>Baseline / Bad</b></td><td>The value of the <b>Failure State</b> (e.g., Random Guessing or Gaussian Blur).</td></tr>",
        "<tr><td><b>s</b></td><td><b>Resolution</b></td><td>The physical \"pixel size\" (s_K) used for normalization.</td></tr>",
        "</table>",
        "<p><i>Key Insight:</i> The Q-Score is simply the position of <code>f</code> normalized between <code>b</code> (0.0) and <code>g</code> (1.0).</p>",

        "<h4>3.2 Q-Score Definitions & Baselines</h4>",
        "<table class='metric-table'>",
        "<tr><th>Q-Score</th><th>Concept</th><th>Baseline Failure (Q=0)</th><th>Ideal State (Q=1)</th></tr>",
        "<tr><td><b>Denoise</b></td><td><b>Entropic Progress:</b> Did the algorithm reduce the initial randomness?</td><td><b>Random Guessing:</b> Points scattered blindly in the objective space.</td><td><b>Noise-Free:</b> Complete removal of entropic noise.</td></tr>",
        "<tr><td><b>Closeness</b></td><td><b>Precision vs. Blur:</b> How sharp is the convergence?</td><td><b>Gaussian Blur:</b> Ground Truth \"inflated\" by statistical noise.</td><td><b>Discrete Truth:</b> Perfect, sharp convergence.</td></tr>",
        "<tr><td><b>Coverage</b></td><td><b>Extent:</b> Does the population stretch to the edges?</td><td><b>Random Extent:</b> Coverage limited by random chance.</td><td><b>Full Manifold:</b> Limits reached perfectly.</td></tr>",
        "<tr><td><b>Gap</b></td><td><b>Continuity:</b> Are there large holes?</td><td><b>Random Gaps:</b> Holes expected from random sampling.</td><td><b>Seamless:</b> No gaps larger than resolution.</td></tr>",
        "<tr><td><b>Regularity</b></td><td><b>Uniformity:</b> Are points evenly spaced?</td><td><b>Random Spacing:</b> Poisson-like clumping.</td><td><b>Crystalline:</b> Perfect lattice grid.</td></tr>",
        "<tr><td><b>Balance</b></td><td><b>Fairness:</b> Is the spread equitable?</td><td><b>Biased Spread:</b> Random clustering in specific regions.</td><td><b>Equitable:</b> Perfect mass distribution.</td></tr>",
        "</table>",

        "<h3>Tier 4: Structural Biopsy (Marker Grammar)</h3>",
        "<p>The 3D Pareto Front uses specific markers to indicate <b>point-wise quality</b> based on the local Q-Score:</p>",
        "<ul>",
        "<li>● <b>Solid Circle:</b> High precision ($Q \ge 0.5$). Points effectively converging to the manifold.</li>",
        "<li>○ <b>Hollow Circle:</b> Standard precision ($0 \le Q < 0.5$). Converged, but with residual error approaching the noise floor.</li>",
        "<li>◇ <b>Diamond Open:</b> Failure ($Q < 0$). Points statistically indistinguishable from random noise or Gaussian blur.</li>",
        "</ul>",
        "</div>"
    ]

    colors_solid = {'NSGA2': '#6366f1', 'NSGA3': '#059669', 'MOEAD': '#d97706'}

    for mop_name in mops:
        problem = problems[mop_name]
        algs = sorted(problem["algorithms"].keys())
        
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.6, 0.4],
            row_heights=[0.6, 0.4],
            specs=[[{'type': 'scene', 'rowspan': 2}, {'type': 'xy', 'secondary_y': True}],
                   [None, {'type': 'xy'}]],
            subplot_titles=(f"Final Pareto Front (M=3)", "Convergence History", 
                            "Validation: Distance-to-GT CDF (Gap Analysis)")
        )

        # Ground Truth
        if "gt_points" in problem and problem["gt_points"]:
            gt = np.array(problem["gt_points"])
            fig.add_trace(go.Scatter3d(
                x=gt[:,0], y=gt[:,1], z=gt[:,2],
                mode='markers', marker=dict(size=2, color='#475569', opacity=0.3),
                name='Ground Truth', legendgroup='GT'
            ), row=1, col=1)

        mop_metrics = []
        for alg in algs:
            data = problem["algorithms"][alg]
            stats = data["stats"]
            clinical = data["clinical"]
            
            # 3D Front (Clinical Differentiation: Solid / Hollow / X)
            if data["final_front"]:
                front = np.array(data["final_front"])
                
                # Use raw point distances (aligned) if available, else fallback to sorted (buggy but fallback)
                if "point_dists" in data and data["point_dists"]:
                    dists = np.array(data["point_dists"])
                elif data["cdf_dists"]:
                    print(f"Warning: Using sorted cdf_dists for {alg} (Visual artifacts likely)")
                    dists = np.array(data["cdf_dists"])
                else:
                    dists = np.zeros(len(front))
                
                # Fetch DENOISE/CLOSENESS baselines (already normalized in v4)
                denoise_data = clinical.get("denoise", {})
                closeness_data = clinical.get("closeness", {})
                s_fit = clinical.get("s_fit", 1.0) # The new s_K scaling factor

                # Calculate Q-Score using s_fit normalization (Harmonized with QScore.py)
                try:
                    # K discretization to match Auditor (Fixed K-Target Policy)
                    from MoeaBench.diagnostics import baselines as base
                    K_raw = len(front)
                    
                    # Strict Snap (Floor) matches Auditor
                    K_target = base.snap_k(K_raw)

                    q_points = qscore.q_closeness_points(dists, problem=mop_name, k=K_target, s_k=s_fit)
                except Exception as e:
                    print(f"Warning: Q-Point calc failed for {alg} on {mop_name}: {e}")
                    q_points = np.zeros_like(dists)
                
                
                mask_solid = q_points >= 0.5
                mask_hollow = (q_points >= 0.0) & (q_points < 0.5)
                mask_diamond = q_points < 0.0
                
                # Trace 1: Precision (Solid Circle)
                if np.any(mask_solid):
                    pts = front[mask_solid]
                    fig.add_trace(go.Scatter3d(
                        x=pts[:,0], y=pts[:,1], z=pts[:,2],
                        mode='markers', marker=dict(size=4, color=colors_solid.get(alg, 'black'), opacity=1.0),
                        name=f'{alg}', legendgroup=alg
                    ), row=1, col=1)

                # Trace 2: Standard (Hollow Circle)
                if np.any(mask_hollow):
                    pts = front[mask_hollow]
                    fig.add_trace(go.Scatter3d(
                        x=pts[:,0], y=pts[:,1], z=pts[:,2],
                        mode='markers',
                        marker=dict(
                            size=4, symbol='circle-open',
                            line=dict(color=colors_solid.get(alg, 'black'), width=2),
                            opacity=0.8
                        ),
                        name=f'{alg}', legendgroup=alg, showlegend=False
                    ), row=1, col=1)

                # Trace 3: Failure (Diamond Open)
                if np.any(mask_diamond):
                    pts = front[mask_diamond]
                    fig.add_trace(go.Scatter3d(
                        x=pts[:,0], y=pts[:,1], z=pts[:,2],
                        mode='markers',
                        marker=dict(
                            size=4, symbol='diamond-open',
                            line=dict(color=colors_solid.get(alg, 'black'), width=1.5),
                            opacity=0.6
                        ),
                        name=f'{alg}', legendgroup=alg, showlegend=False
                    ), row=1, col=1)

            # History
            if data["history"]["gens"]:
                h = data["history"]
                fig.add_trace(go.Scatter(
                    x=h["gens"], y=h["igd"], mode='lines+markers',
                    line=dict(color=colors_solid.get(alg, 'black')),
                    name=f'{alg} IGD', legend='legend2', legendgroup=alg
                ), row=1, col=2)
                fig.add_trace(go.Scatter(
                    x=h["gens"], y=h["hv_rel"], mode='lines+markers',
                    line=dict(color=colors_solid.get(alg, 'black'), dash='dash'),
                    name=f'{alg} HV%', legend='legend2', legendgroup=alg
                ), row=1, col=2, secondary_y=True)

            # CDF (Validation: Distance-to-GT)
            if data["cdf_dists"]:
                # SANITIZATION: Normalize raw distances by s_K for macroscopic view
                dists = np.array(data["cdf_dists"]) / s_fit
                y_cdf = np.arange(len(dists)) / float(len(dists))
                
                # Add threshold lines only once (for the first alg) to avoid clutter
                if alg == algs[0]:
                     # Utopia Wall (x=0) - Plot as Scatter to avoid Scatter3d/add_vline bug
                    fig.add_trace(go.Scatter(
                        x=[0, 0], y=[0, 1.05], 
                        mode='lines',
                        line=dict(color='gray', dash='dot', width=1),
                        name='Utopia (GT)',
                        showlegend=False,
                        hoverinfo='skip'
                    ), row=2, col=2)
                    
                    # 95% Population Target (y=0.95)
                    max_dist = np.max(dists) if len(dists) > 0 else 1.0
                    fig.add_trace(go.Scatter(
                        x=[0, max_dist * 1.5], y=[0.95, 0.95],
                        mode='lines', 
                        line=dict(color='gray', dash='dot', width=1),
                        name='95% Target',
                        showlegend=False,
                        hoverinfo='skip'
                    ), row=2, col=2)
                    
                    # Annotation for 95% line
                    fig.add_annotation(
                        x=max_dist*1.0, y=0.95,
                        text="95% Pop",
                        showarrow=False,
                        yshift=10,
                        font=dict(size=10, color="gray"),
                        row=2, col=2
                    )
                    
                    # Update X-Axis label to reflect sanitization
                    fig.update_xaxes(title_text="Normalized Distance (GD / s_K)", row=2, col=2)

                # Add main CDF curve
                fig.add_trace(go.Scatter(
                    x=dists, y=y_cdf, mode='lines',
                    line=dict(color=colors_solid.get(alg, 'black'), width=2),
                    name=f'{alg} CDF', legend='legend3', legendgroup=alg
                ), row=2, col=2)
                
                # Add per-algorithm 95% Intersection Drop-line (The "Biopsy" Marker)
                p95 = np.percentile(dists, 95)
                fig.add_trace(go.Scatter(
                    x=[p95, p95], y=[0, 0.95],
                    mode='lines',
                    line=dict(color=colors_solid.get(alg, 'black'), dash='dot', width=1),
                    name=f'{alg} 95%', showlegend=False,
                    legendgroup=alg,
                    hoverinfo='x'
                ), row=2, col=2)
            
            # --- Detailed Metrics Extraction ---
            igd_mean = stats.get('IGD_mean', 0.0)
            igd_std = stats.get('IGD_std', 0.0)
            gd_mean = stats.get('GD_mean', 0.0)
            gd_std = stats.get('GD_std', 0.0) 
            sp_mean = stats.get('SP_mean', 0.0)
            sp_std = stats.get('SP_std', 0.0)
            h_raw = stats.get('H_raw', 0.0)
            h_ratio = stats.get('H_ratio', 0.0)
            
            # Clinical (v0.9)
            igd_p_val = clinical.get('igd_p', {}).get('mean', 0)
            gd_p_val = clinical.get('gd_p', {}).get('mean', 0)

            # EMD Calculation
            emd_val = stats.get('EMD_mean', 0.0)
            if emd_val == 0.0 and "gt_points" in problem and data["final_front"]:
                try:
                    from scipy.stats import wasserstein_distance
                    gt_arr = np.array(problem["gt_points"])
                    front_arr = np.array(data["final_front"])
                    if gt_arr.shape[1] == front_arr.shape[1]:
                        dists = [wasserstein_distance(front_arr[:, i], gt_arr[:, i]) for i in range(gt_arr.shape[1])]
                        emd_val = np.mean(dists)
                except: pass

            # NEAR@1 and NEAR@2 (from clinical closeness)
            # Calculated from point_dists if available
            near1, near2 = 0.0, 0.0
            if data.get("point_dists"):
                u_pts = np.array(data["point_dists"]) / s_fit
                near1 = np.mean(u_pts <= 1.0) * 100
                near2 = np.mean(u_pts <= 2.0) * 100

            # Stabilization Heuristic (v0.9.1)
            # Find the first generation where H_rel reaches 99.9% of final value
            t_conv = "-"
            if data["history"]["gens"] and data["history"]["hv_rel"]:
                h_gen = data["history"]["gens"]
                h_hv = data["history"]["hv_rel"]
                final_hv = h_hv[-1]
                if final_hv > 0:
                    for i, v in enumerate(h_hv):
                        if v >= 0.999 * final_hv:
                            t_conv = str(h_gen[i])
                            break
                else:
                    t_conv = str(h_gen[-1])

            mop_metrics.append({
                "alg": alg,
                "igd": f"{igd_mean:.4f} &plusmn; {igd_std:.4f}",
                "igd_p": f"{igd_p_val:.4f}",
                "gd": f"{gd_mean:.4e} &plusmn; {gd_std:.2e}",
                "gd_p": f"{gd_p_val:.4e}",
                "sp": f"{sp_mean:.4e} &plusmn; {sp_std:.2e}",
                "emd": f"{emd_val:.4f}", 
                "h_raw": f"{h_raw:.4f}",
                "h_ratio": f"{h_ratio:.4f}",
                "h_rel": f"{stats.get('H_rel',0)*100:.3f}%",
                "near1": f"{near1:.1f}%",
                "near2": f"{near2:.1f}%",
                "time": f"{stats.get('Time_sec',0):.2f}",
                "clinical": clinical,
                "s_fit": s_fit, # Expose s_K scale
                "t_conv": t_conv
            })
            
        # Layout with multiple legends to fix disappearing plots
        fig.update_layout(
            height=850, margin=dict(l=0, r=0, t=60, b=0),
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.7)"),
            legend2=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.7)"),
            legend3=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.7)"),
            scene=dict(xaxis_title='f1', yaxis_title='f2', zaxis_title='f3', aspectmode='cube')
        )

        html_content.append(f"<div class='mop-section'><h2>{mop_name} Benchmark Analysis</h2>")
        
        # Numerical Table
        metrics_table = ["<table><tr><th>Algorithm</th><th>IGD (&plusmn; s)</th><th>IGD+</th><th>GD (&plusmn; s)</th><th>GD+</th><th>SP (&plusmn; s)</th><th>NEAR@1</th><th>NEAR@2</th><th>H_rel</th><th>Time(s)</th><th>Stabil.</th></tr>"]
        for m in mop_metrics:
            metrics_table.append(f"<tr><td style='font-weight: bold; color: {colors_solid.get(m['alg'], 'black')}'>{m['alg']}</td><td>{m['igd']}</td><td>{m['igd_p']}</td><td>{m['gd']}</td><td>{m['gd_p']}</td><td>{m['sp']}</td><td>{m['near1']}</td><td>{m['near2']}</td><td style='font-weight: 500'>{m['h_rel']}</td><td>{m['time']}</td><td>Gen {m['t_conv']}</td></tr>")
        metrics_table.append("</table>")
        html_content.append("".join(metrics_table))

        # Matrix Table
        matrix_table = [
            "<h3>Clinical Quality Matrix</h3>",
            "<table><colgroup><col style='width: 120px'><col style='width: 140px'><col style='width: 140px'><col style='width: 140px'><col style='width: 140px'><col style='width: 140px'><col style='width: 140px'><col style='width: auto'></colgroup>",
            "<thead><tr><th>Algorithm</th><th>DENOISE</th><th>CLOSENESS</th><th>COVERAGE</th><th>CONTINUITY</th><th>REGUL.</th><th>BALANCE</th><th>SUMMARY</th></tr></thead>"
        ]
        for m in mop_metrics:
            matrix_table.append(f"<tr><td style='font-weight: bold; color: {colors_solid.get(m['alg'], 'black')}'>{m['alg']}</td>")
            c = m["clinical"]
            s_fit = c.get("s_fit") # Correct nesting for s_fit

            for dim in ["denoise", "closeness", "cov", "gap", "reg", "bal"]:
                d = c.get(dim, {})
                q = d.get("q", 0)
                cls = "diag-optimal" if q >= 0.67 else ("diag-warning" if q >= 0.34 else "diag-failure")
                
                # Full Tip for Tooltip
                tip = f"Q: {q:.4f}&#013;Fair: {d.get('fair',0):.4f}&#013;Good: {d.get('anchor_good',0):.4f}&#013;Bad: {d.get('anchor_bad',0):.4f}"
                if dim == "denoise" and "s_fit" in m: tip += f"&#013;s_K: {m['s_fit']:.2e}"
                if dim == "closeness":
                    tip += f"&#013;sigma: {d.get('sigma',0):.2e}&#013;p95: {d.get('p95',0):.2f}"
                
                # Didactic Subtext (PDF-ready)
                f_val = d.get('fair', 0)
                g_val = d.get('anchor_good', 0)
                b_val = d.get('anchor_bad', 0)
                
                # Vertical Monospace Console (Neutral & Aligned)
                sub_style = "font-family: monospace; font-size: 0.65rem; color: #475569; line-height: 1.2; margin-top: 5px; text-align: left; display: inline-block;"
                
                # Base lines: f, g, b
                lines = [
                    f"f: {f_val:6.3f}",
                    f"g: {g_val:6.3f}",
                    f"b: {b_val:6.3f}"
                ]
                
                # Add s_K as the 4th line for Denoise and Closeness
                if dim in ["denoise", "closeness"] and s_fit is not None:
                    lines.append(f"s: {s_fit:9.2e}")
                
                sub_text = f"<div style='{sub_style}'>{'<br>'.join(lines)}</div>"
                
                # Q Label (No badges here, as requested)
                q_label = f"{q:.3f}"

                matrix_table.append(f"<td style='padding: 8px 4px; vertical-align: top;'><span class='diag-badge {cls}' title='{tip}'>{q_label}</span><br>{sub_text}</td>")
            
            summary_text = c.get("summary", "Standard Operational Performance")
            matrix_table.append(f"<td class='matrix-summary'>{summary_text}</td></tr>")
        matrix_table.append("</table>")
        
        html_content.append(fig.to_html(full_html=False, include_plotlyjs='cdn' if mop_name == mops[0] else False))
        html_content.append("".join(matrix_table))
        html_content.append("</div>")

    html_content.append("</body></html>")
    with open(OUTPUT_HTML, "w") as f: f.write("\n".join(html_content))
    print(f"Success! Report: {OUTPUT_HTML}")

if __name__ == "__main__":
    generate_visual_report()
