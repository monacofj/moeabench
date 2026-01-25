import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
import sys

# Ensure we can import MoeaBench
sys.path.append(os.path.abspath("."))

from MoeaBench.mops import DTLZ8
from MoeaBench.view import spaceplot

def update_dtlz8_fig():
    print(">>> Generating Updated DTLZ8 Audit Figure...")
    mop = DTLZ8(M=3)
    
    # load ground truth
    x_file = "MoeaBench/mops/data/DTLZ8_3_optimal_x.csv"
    X = pd.read_csv(x_file, header=None).values
    res = mop.evaluation(X)
    F_v075 = res['F']
    
    # Plotting logic compatible with benchmark_gallery
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(F_v075[:, 0], F_v075[:, 1], F_v075[:, 2], 
               c='blue', alpha=0.6, s=10, label='v0.7.5 Ground Truth')
    
    ax.set_title("DTLZ8 (M=3) Audit - v0.7.5 GT")
    ax.set_xlabel("F1")
    ax.set_ylabel("F2")
    ax.set_zlabel("F3")
    ax.legend()
    
    output_path = "misc/figs/DTLZ8_audit.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    update_dtlz8_fig()
