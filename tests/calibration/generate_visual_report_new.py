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
        "</body></html>"
    ]
    # ... Omitted for brevity, using write_to_file to restore logic ...
