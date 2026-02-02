# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
MoeaBench v0.7.6 Visual Calibration Generator
===============================================

This script processes the full calibration trace to create an interactive 
HTML report. It combines 3D Pareto front visualizations with dynamic 
convergence curves (IGD/HV) based on snapshots.

Key Features:
---------------------------
- 3D Overlays: Projection of algorithms onto the "Mathematical Truth" (Ground Truth).
- Convergence Timeplots: IGD and Evolutive HV graphs throughout generations.
- Plotly Interactivity: Zoom, rotation, and point inspection directly in the browser.
- Consistent Normalization: Uses the same Ideal/Nadir bounds as v0.7.6.

Output:
------
- tests/CALIBRATION_v0.7.6.html

Usage:
----
python tests/calibration/generate_visual_report.py
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure local MoeaBench is importable
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from MoeaBench.metrics.GEN_hypervolume import GEN_hypervolume
from MoeaBench.metrics.GEN_igd import GEN_igd

# Paths
DATA_DIR = os.path.join(PROJ_ROOT, "tests/calibration_data")
GT_DIR = os.path.join(PROJ_ROOT, "tests/ground_truth")
BASELINE_FILE = os.path.join(PROJ_ROOT, "tests/baselines_v0.7.6.csv")
OUTPUT_HTML = os.path.join(PROJ_ROOT, "tests/CALIBRATION_v0.7.6.html")

def generate_visual_report():
    if not os.path.exists(BASELINE_FILE):
        print("Baseline CSV not found. Run analysis first.")
        return

    df_base = pd.read_csv(BASELINE_FILE)
    mops = sorted(df_base['MOP'].unique())
    
    html_content = [
        "<html><head><title>MoeaBench v0.7.6 Calibration</title>",
        "<script type='text/x-mathjax-config'>MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\\\(','\\\\)']]}});</script>",
        "<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML'></script>",
        "<style>body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f4f7f9; line-height: 1.6; color: #333; }",
        "h1 { color: #1a2a3a; border-bottom: 3px solid #3498db; padding-bottom: 15px; margin-bottom: 30px; }",
        "h2 { color: #2c3e50; margin-top: 60px; border-left: 6px solid #3498db; padding-left: 15px; background: #ebf5fb; padding-top: 10px; padding-bottom: 10px; }",
        "h3 { color: #2980b9; margin-top: 30px; border-bottom: 1px solid #d4e6f1; padding-bottom: 5px; }",
        ".mop-section { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); margin-bottom: 50px; }",
        ".metrics-footer { font-size: 0.85em; color: #555; margin-top: 20px; font-family: 'Courier New', Courier, monospace; background: #fdfefe; padding: 15px; border: 1px dashed #bdc3c7; border-radius: 6px; }",
        ".intro-box { background: white; padding: 30px; border-radius: 12px; margin-bottom: 40px; border: 1px solid #e0e6ed; box-shadow: 0 2px 15px rgba(0,0,0,0.05); }",
        ".note-box { background: #fff9db; padding: 20px; border-radius: 8px; margin-top: 20px; border-left: 5px solid #f1c40f; font-size: 0.95em; }",
        "table { width: 100%; border-collapse: collapse; margin-top: 20px; margin-bottom: 30px; background: white; }",
        "th { background: #f2f4f6; color: #2c3e50; font-weight: 600; text-align: left; border: 1px solid #dee2e6; padding: 12px; }",
        "td { border: 1px solid #dee2e6; padding: 12px; }",
        "tr:nth-child(even) { background-color: #f9fbfd; }",
        "tr:hover { background-color: #f1f4f7; }",
        "code { background: #f0f2f5; padding: 2px 5px; border-radius: 4px; font-family: 'Consolas', monospace; font-size: 0.9em; }",
        "ul { padding-left: 20px; }",
        "li { margin-bottom: 8px; }",
        "</style></head><body>",
        "<h1>MoeaBench v0.7.6 Technical Calibration Report</h1>",
        
        "<div class='intro-box'>",
        "<h2>1. Methodology & Experimental Context</h2>",
        "<p>This report serves as the official scientific audit for <b>MoeaBench v0.7.6</b>. The objective is to validate and calibrate the numerical integrity and topological fidelity of the framework's core algorithms against established mathematical benchmarks (Ground Truth).</p>",
        
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
        "<li><b>EMD (Topological Error):</b> Earth Mover's Distance (Wasserstein metric). It quantifies the 'transport cost' required to map the algorithm's distribution onto the Ground Truth. <code>EMD < 0.1</code> represents a high-fidelity topological match.</li>",
        "<li><b>HV Raw (Absolute):</b> The hypervolume calculated with Reference Point 1.1. Can exceed 1.0 (e.g. 1.15) due to the reference boundary buffer.</li>",
        r"<li><b>HV Ratio (Coverage):</b> $HV_{raw} / RefBox$. Measures how much of the reference box volume ($1.1^3$) is covered. Strictly $\le 1.0$.</li>",
        r"<li><b>HV Rel (Convergence):</b> $HV_{raw} / HV_{GT}$. Measures convergence to the known optimum. Can slightly exceed 100% if the algorithm fills gaps in the discrete Ground Truth.</li>",
        "<li><b>T-conv (Stabilization):</b> The generation where the algorithm reaches a stable state (within 5% of its final IGD value).</li>",
        "<li><b>Time (s):</b> Average wall-clock execution time per run on the reference hardware.</li>",
        "</ul>",

        "<div class='note-box'>",
        "<strong>Scientific Note: The Discretization Effect & Negative HV Diff</strong><br>",
        "In cases of near-perfect convergence, you may observe an HV Rel exceeding 100%.<br>",
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

        # Layout: 1 row, 2 columns (3D Scatter, 2D Convergence)
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.6, 0.4],
            specs=[[{'type': 'scene'}, {'type': 'xy', 'secondary_y': True}]],
            subplot_titles=(f"Final Pareto Front (M=3)", "Convergence History (IGD & HV Rel)")
        )

        # 1. Add Ground Truth to 3D plot
        if F_opt is not None:
            print(f"    - Trace: Ground Truth Group (GT)")
            try:
                fig.add_trace(go.Scatter3d(
                    x=F_opt[:,0], y=F_opt[:,1], z=F_opt[:,2],
                    mode='markers',
                    marker=dict(
                        size=2.5, 
                        color='rgba(0, 0, 0, 0.25)', # Translucent Black Cloud
                        line=dict(width=0)
                    ),
                    name='Ground Truth',
                    legendgroup='GT',
                    showlegend=True
                ), row=1, col=1)
            except Exception as e:
                print(f"      ERROR adding GT trace: {e}")

        algs = sorted(mop_df['Algorithm'].unique())
        # Final Zen Mode: Clean organic look, 0.75 opacity, borderless
        colors_rgba = {
            'NSGA2': 'rgba(231, 76, 60, 0.75)',  # Red
            'NSGA3': 'rgba(52, 152, 219, 0.75)', # Blue
            'MOEAD': 'rgba(212, 172, 13, 0.75)'  # Deeper Yellow (Amber)
        }
        colors_solid = {
            'NSGA2': '#e74c3c', 
            'NSGA3': '#3498db', 
            'MOEAD': '#d4ac0d'  # Deeper Yellow
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
            hv_raw = 0.0
            hv_ratio = 0.0
            hv_rel_stat = 0.0
            time_avg = 0.0
            
            if not alg_stats.empty:
                row = alg_stats.iloc[0]
                igd_mean = row['IGD_mean']
                igd_std = row['IGD_std']
                # Updated keys from compute_baselines.py
                hv_raw = row['HV_raw'] 
                hv_ratio = row['HV_ratio']
                hv_rel_stat = row['HV_rel']
                time_avg = row['Time_sec']
            
            # Note: We re-calculate dynamic HV (rel) from the final snapshot (Gen 1000) for strictness
            hv_rel_final = 0.0

            # Final Front (from standard intensity)
            final_file = os.path.join(DATA_DIR, f"{mop_name}_{alg}_standard_run00.csv")
            F_obs = None
            if os.path.exists(final_file):
                print(f"    - Trace: {alg} 3D Final")
                try:
                    F_obs = pd.read_csv(final_file).values
                    # Trim to 3 obj if needed
                    if F_obs.shape[1] > 3: F_obs = F_obs[:, :3]
                    
                    fig.add_trace(go.Scatter3d(
                        x=F_obs[:,0], y=F_obs[:,1], z=F_obs[:,2],
                        mode='markers',
                        marker=dict(
                            size=3.0, 
                            color=colors_rgba.get(alg, 'rgba(0,0,0,0.75)'),
                            line=dict(width=0) # Final Zen: No border
                        ),
                        name=f'{alg} (Final)',
                        legendgroup=alg,
                        showlegend=True
                    ), row=1, col=1)
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
            if igd_vals:
                final_igd = igd_vals[-1]
                hv_rel_final = hv_rels[-1]
                
                # Topological Error (EMD) between final snapshot and GT
                if F_opt is not None and F_snap is not None:
                    import MoeaBench as mb
                    # Use internal topo_distribution with EMD method
                    # This calculates axis-wise Wasserstein distance (EMD)
                    res = mb.stats.topo_distribution(F_snap, F_opt, method='emd')
                    emd_val = np.mean(list(res.results.values()))
                
                for g_idx, g_val in enumerate(igd_vals):
                    if g_val <= 1.05 * final_igd:
                        t_conv = str(gens[g_idx])
                        break
            
            mop_metrics.append({
                "alg": alg,
                "igd": f"{igd_mean:.4e} &plusmn; {igd_std:.1e}",
                "emd": f"{emd_val:.4f}",
                "hv_raw": f"{hv_raw:.4f}",
                "hv_ratio": f"{hv_ratio:.4f}",
                "hv_rel": f"{hv_rel_stat * 100:.2f}%", 
                "time": f"{time_avg:.2f}",
                "t_conv": t_conv
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
                        name=f'{alg} IGD',
                        legendgroup=alg,
                        showlegend=False
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
                        name=f'{alg} HV %',
                        legendgroup=alg,
                        showlegend=False
                    ), row=1, col=2, secondary_y=True)
                except Exception as e:
                    print(f"      ERROR adding {alg} convergence traces: {e}")

        # Formatting
        fig.update_layout(
            height=600,
            template='plotly_white',
            scene=dict(
                xaxis_title='f1', yaxis_title='f2', zaxis_title='f3',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                aspectmode='cube' # Prevent squeezing for asymmetric ranges (e.g. DTLZ7)
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig.update_yaxes(title_text="IGD (Log Scale)", secondary_y=False, row=1, col=2, type="log")
        fig.update_yaxes(title_text="HV Rel % (Convergence)", secondary_y=True, row=1, col=2, range=[0, 115])
        fig.update_xaxes(title_text="Generations", row=1, col=2)

        # Convert to HTML
        div = fig.to_html(full_html=False, include_plotlyjs='cdn' if mop_name == mops[0] else False)
        
        html_content.append(f"<div class='mop-section'>")
        html_content.append(f"<h2>{mop_name} Benchmark Analysis</h2>")
        
        # Build metrics table
        table = ["<table>",
                 "<tr>",
                 "<th>Algorithm</th>",
                 "<th>IGD (Mean &plusmn; Std)</th>",
                 "<th>EMD (Topo Error)</th>",
                 "<th>HV Raw (Abs)</th>",
                 "<th>HV Ratio (Vol)</th>",
                 "<th>HV Rel (% GT)</th>",
                 "<th>Time (s)</th>",
                 "<th>Stabilization</th></tr>"]
        
        for m in mop_metrics:
            table.append(f"<tr><td style='font-weight: bold; color: {colors_solid.get(m['alg'], 'black')}'>{m['alg']}</td>")
            table.append(f"<td>{m['igd']}</td>")
            table.append(f"<td>{m['emd']}</td>")
            table.append(f"<td>{m['hv_raw']}</td>")
            table.append(f"<td>{m['hv_ratio']}</td>")
            table.append(f"<td>{m['hv_rel']}</td>")
            table.append(f"<td>{m['time']}</td>")
            table.append(f"<td>Gen {m['t_conv']}</td></tr>")
        table.append("</table>")
        
        html_content.append("".join(table))
        
        # Formatting bounds as readable tuples
        ideal_str = "(" + ", ".join([f"{v:.3f}" for v in ideal]) + ")"
        nadir_str = "(" + ", ".join([f"{v:.3f}" for v in nadir]) + ")"
        
        html_content.append(f"<div class='metrics-footer'><strong>Theoretical Reference:</strong><br>Ideal Point: {ideal_str}<br>Nadir Point: {nadir_str}<br>Sampled Reference HV: {hv_opt:.6f}</div>")
        html_content.append(div)
        html_content.append(f"</div>")

    html_content.append("</body></html>")
    
    with open(OUTPUT_HTML, "w") as f:
        f.write("\n".join(html_content))
    
    print(f"\nSuccess! Interactive report generated at: {OUTPUT_HTML}")

if __name__ == "__main__":
    generate_visual_report()
