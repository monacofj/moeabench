# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
MoeaBench Calibration Visual Report Generator (v0.9.0)
=====================================================

Decoupled Renderer: Consumes pre-calculated audit data from JSON.
"""

import os
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Paths
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
AUDIT_JSON = os.path.join(PROJ_ROOT, "tests/calibration_audit_v0.9.json")
OUTPUT_HTML = os.path.join(PROJ_ROOT, "tests/CALIBRATION_v0.9.html")

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
        ".diag-badge { display: inline-block; padding: 0.25rem 0.5rem; border-radius: 6px; font-weight: 600; text-align: center; min-width: 45px; }",
        ".diag-optimal { background: #dcfce7; color: #166534; }",
        ".diag-warning { background: #fef9c3; color: #854d0e; }",
        ".diag-failure { background: #fee2e2; color: #991b1b; }",
        ".verdict-pass { background: #e0e7ff; color: #3730a3; }",
        ".verdict-fail { background: #ffedd5; color: #9a3412; }",
        "</style></head><body>",
        "<h1>MoeaBench v0.9.0 Technical Calibration Report</h1>",
        "<p>This report serves as the official scientific audit for <b>MoeaBench v0.9.0</b>. It implements the <i>Clinical Metrology</i> standard (ADR 0026).</p>",
        
        "<div class='intro-box'>",
        "<h2>1. Methodology & Experimental Context</h2>",
        "<p>The objective is to certify the framework's core algorithms against rigorous mathematical benchmarks (Ground Truth), using a scale-invariant quality assessment.</p>",
        "<ul>",
        "<li><b>Population:</b> Algorithms ran with <code>N=200</code>. Q-Scores are calculated on the <i>effective</i> non-dominated count ($K \\le 200$).</li>",
        "<li><b>Evolutionary Budget:</b> Fixed at <code>1000 generations</code> per run.</li>",
        "<li><b>Statistical Relevance:</b> Metrics derived from <b>30 independent runs</b> per algorithm/problem pair.</li>",
        "<li><b>Normalization:</b> Strict Theoretical Normalization [0, 1] using the Ground Truth's <i>Ideal</i> and <i>Nadir</i> points.</li>",
        "</ul>",

        "<h2>2. Clinical Metrology Guide</h2>",
        "<div style='background: #f8fafc; padding: 15px; border-left: 4px solid #3b82f6; margin-bottom: 20px;'>",
        "<b>Reading the Evidence:</b> This report uses a dual-layer validation framework.<br>",
        "<ul style='margin-bottom:0'>",
        "<li><b>Layer 1 - Clinical Matrix (The Verdict):</b> An engineering Q-Score (0.0 - 1.0).<br>",
        "<i>Pass ($Q \\ge 0.67$):</i> The algorithm is statistically indistinguishable from the theoretical limit.<br>",
        "<i>Fail ($Q < 0.34$):</i> The algorithm is performing closer to random noise than to the ideal.</li>",
        "<li><b>Layer 2 - Structural Evidence (The Biopsy):</b> The <b>Distance-to-GT CDF</b> graph (bottom right).<br>",
        "This graph reveals the 'physics' of the failure that the Q-Score summarizes. It plots the cumulative distribution of distances to the Ground Truth along the X-axis.",
        "<ul style='margin-top:0.5rem'>",
        "<li><b>Steep Left-Aligned Curve:</b> The ideal profile. High precision convergence where the entire population is uniformly close to the manifold.</li>",
        "<li><b>Long Tail (Right-skewed):</b> Indicates <b>Poor Regularity (REG)</b>. While most points may be close, a subset of the population is 'stuck' far away (outliers) or trapped in local optima.</li>",
        "<li><b>Rigid Shift (Offset):</b> Indicates <b>Good Geometry but Poor Fit (FIT)</b>. The curve has the correct vertical shape but is shifted to the right. The algorithm found the manifold's shape but stopped before reaching the true front.</li>",
        "<li><b>Discontinuous Plateaus:</b> Vertical gaps in the curve indicate <b>Coverage Gaps (GAP)</b>. The algorithm completely missed specific regions of the objective space.</li>",
        "</ul></li>",
        "</ul></div>",

        "<h3>3. Metric Glossary</h3>",
        "<ul>",
        "<li><b>FIT (Proximity):</b> How close the front is to the Optimal Manifold. High precision convergence.</li>",
        "<li><b>COV (Coverage):</b> The extent of the manifold covered (Volume).</li>",
        "<li><b>GAP (Continuity):</b> Absence of interruptions or holes in the Pareto approximation.</li>",
        "<li><b>REG (Regularity):</b> Uniformity of the distribution. Penalizes clustering and outliers.</li>",
        "<li><b>BAL (Balance):</b> Fairness of objective trade-offs (e.g., not favoring f1 over f2).</li>",
        "<li><b>IGD/EMD:</b> Legacy distance metrics provided for backward compatibility.</li>",
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
            if data["final_front"] and data["cdf_dists"]:
                front = np.array(data["final_front"])
                dists = np.array(data["cdf_dists"])
                
                # Fetch FIT baselines for per-point Q-Score calculation
                fit_data = clinical.get("fit", {})
                ideal = fit_data.get("ideal", 0.0)
                rand = fit_data.get("rand", 1.0)
                fair_agg = fit_data.get("fair", 0.0)
                
                # Recover s_gt (Resolution Factor) to align raw dists with normalized baselines
                # Rule: fair_agg = percentile(dists, 95) / s_gt
                d95 = np.percentile(dists, 95) if len(dists) > 0 else 0.0
                s_gt = d95 / fair_agg if fair_agg > 1e-12 else 1.0
                
                # Calculate Q-Score for each point (normalized)
                denom = rand - ideal
                if abs(denom) < 1e-12:
                    q_points = np.where(dists < 1e-6, 1.0, 0.0)
                else:
                    # Normalize dists by s_gt before computing Q
                    q_points = 1.0 - np.clip(((dists / s_gt) - ideal) / denom, 0.0, 1.0)
                
                mask_research = q_points >= 0.67
                mask_industry = (q_points >= 0.34) & (q_points < 0.67)
                mask_fail = q_points < 0.34
                
                # Trace 1: Research (Solid Circle)
                if np.any(mask_research):
                    pts = front[mask_research]
                    fig.add_trace(go.Scatter3d(
                        x=pts[:,0], y=pts[:,1], z=pts[:,2],
                        mode='markers', marker=dict(size=4, color=colors_solid.get(alg, 'black'), opacity=1.0),
                        name=f'{alg} (Research)', legendgroup=alg
                    ), row=1, col=1)

                # Trace 2: Industry (Hollow Circle)
                if np.any(mask_industry):
                    pts = front[mask_industry]
                    fig.add_trace(go.Scatter3d(
                        x=pts[:,0], y=pts[:,1], z=pts[:,2],
                        mode='markers',
                        marker=dict(
                            size=4, symbol='circle-open',
                            line=dict(color=colors_solid.get(alg, 'black'), width=2),
                            opacity=0.8
                        ),
                        name=f'{alg} (Industry)', legendgroup=alg, showlegend=False
                    ), row=1, col=1)

                # Trace 3: Fail (X)
                if np.any(mask_fail):
                    pts = front[mask_fail]
                    fig.add_trace(go.Scatter3d(
                        x=pts[:,0], y=pts[:,1], z=pts[:,2],
                        mode='markers',
                        marker=dict(
                            size=3, symbol='x',
                            color=colors_solid.get(alg, 'black'),
                            opacity=0.6
                        ),
                        name=f'{alg} (Fail)', legendgroup=alg, showlegend=False
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

            # CDF
            if data["cdf_dists"]:
                dists = np.array(data["cdf_dists"])
                y_cdf = np.arange(len(dists)) / float(len(dists))
                
                # Add threshold lines only once (for the first alg) to avoid clutter
                if alg == algs[0]:
                     # Ideal Wall (x=0) - Plot as Scatter to avoid Scatter3d/add_vline bug
                    fig.add_trace(go.Scatter(
                        x=[0, 0], y=[0, 1.05], 
                        mode='lines',
                        line=dict(color='gray', dash='dot', width=1),
                        name='Ideal (GT)',
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
                "h_rel": f"{stats.get('H_rel',0)*100:.2f}%",
                "time": f"{stats.get('Time_sec',0):.2f}",
                "clinical": clinical,
                "t_conv": str(data["history"]["gens"][-1]) if data["history"]["gens"] else "-"
            })
            
        # Layout with multiple legends to fix disappearing plots
        fig.update_layout(
            height=850, margin=dict(l=0, r=0, t=60, b=0),
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.7)"),
            legend2=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.7)"),
            legend3=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.7)"),
            scene=dict(xaxis_title='f1', yaxis_title='f2', zaxis_title='f3')
        )

        html_content.append(f"<div class='mop-section'><h2>{mop_name} Benchmark Analysis</h2>")
        
        # Numerical Table
        metrics_table = ["<table><tr><th>Algorithm</th><th>IGD (&plusmn; s)</th><th>IGD+</th><th>GD (&plusmn; s)</th><th>GD+</th><th>SP (&plusmn; s)</th><th>EMD</th><th>H_raw</th><th>H_ratio</th><th>H_rel</th><th>Time(s)</th><th>Stabil.</th></tr>"]
        for m in mop_metrics:
            metrics_table.append(f"<tr><td style='font-weight: bold; color: {colors_solid.get(m['alg'], 'black')}'>{m['alg']}</td><td>{m['igd']}</td><td>{m['igd_p']}</td><td>{m['gd']}</td><td>{m['gd_p']}</td><td>{m['sp']}</td><td>{m['emd']}</td><td>{m['h_raw']}</td><td>{m['h_ratio']}</td><td>{m['h_rel']}</td><td>{m['time']}</td><td>Gen {m['t_conv']}</td></tr>")
        metrics_table.append("</table>")
        html_content.append("".join(metrics_table))

        # Matrix Table
        matrix_table = [
            "<h3>Clinical Quality Matrix</h3>",
            "<table><colgroup><col style='width: 100px'><col style='width: 80px'><col style='width: 80px'><col style='width: 80px'><col style='width: 80px'><col style='width: 80px'><col style='width: auto'><col style='width: 120px'></colgroup>",
            "<thead><tr><th>Algorithm</th><th>FIT</th><th>COV</th><th>GAP</th><th>REG</th><th>BAL</th><th>SUMMARY</th><th>VERDICT</th></tr></thead>"
        ]
        for m in mop_metrics:
            matrix_table.append(f"<tr><td style='font-weight: bold; color: {colors_solid.get(m['alg'], 'black')}'>{m['alg']}</td>")
            c = m["clinical"]
            for dim in ["fit", "cov", "gap", "reg", "bal"]:
                d = c.get(dim, {})
                q = d.get("q", 0)
                cls = "diag-optimal" if q >= 0.67 else ("diag-warning" if q >= 0.34 else "diag-failure")
                tip = f"Q: {q:.2f}&#013;Fair: {d.get('fair',0):.4f}&#013;Ideal: {d.get('ideal',0):.4f}&#013;Rand: {d.get('rand',0):.4f}"
                matrix_table.append(f"<td><span class='diag-badge {cls}' title='{tip}'>{q:.2f}</span></td>")
            matrix_table.append(f"<td style='font-style: italic; color: #64748b'>{c.get('summary', '-')}</td>")
            v = c.get("verdict", "FAIL")
            v_cls = "verdict-pass" if v == "RESEARCH" else ("diag-warning" if v == "INDUSTRY" else "verdict-fail")
            matrix_table.append(f"<td><span class='diag-badge {v_cls}'>{v}</span></td></tr>")
        matrix_table.append("</table>")
        
        html_content.append(fig.to_html(full_html=False, include_plotlyjs='cdn' if mop_name == mops[0] else False))
        html_content.append("".join(matrix_table))
        html_content.append("</div>")

    html_content.append("</body></html>")
    with open(OUTPUT_HTML, "w") as f: f.write("\n".join(html_content))
    print(f"Success! Report: {OUTPUT_HTML}")

if __name__ == "__main__":
    generate_visual_report()
