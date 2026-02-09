import numpy as np
import json
import os
from MoeaBench.mops import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7, DTLZ8, DTLZ9
from MoeaBench.mops import DPF1, DPF2, DPF3, DPF4, DPF5
from MoeaBench import metrics as mb_metrics

def calibrate():
    mops = [
        DTLZ1(), DTLZ2(), DTLZ3(), DTLZ4(), DTLZ5(), DTLZ6(), DTLZ7(), DTLZ8(), DTLZ9(),
        DPF1(), DPF2(), DPF3(), DPF4(), DPF5()
    ]
    
    n_sizes = [100, 200, 500]
    iterations = 50
    results = {}

    print(f"Starting Calibration of {len(mops)} MOPs...")

    for mop in mops:
        mop_name = mop.__class__.__name__
        print(f"  Calibrating {mop_name}...")
        
        results[mop_name] = {
            "diameter": 1.0,
            "floors": {}
        }
        
        # 1. Calculate diameter
        pf = mop.pf(n_points=2000)
        diff = np.max(pf, axis=0) - np.min(pf, axis=0)
        diameter = float(np.sqrt(np.sum(diff**2)))
        results[mop_name]["diameter"] = diameter
        
        # 2. Resampling for noise floors
        for n in n_sizes:
            igd_samples = []
            emd_samples = []
            
            for _ in range(iterations):
                # Sample N points from the reference front
                indices = np.random.choice(len(pf), size=n, replace=True)
                sample = pf[indices]
                
                # Calculate metrics relative to the full front
                igd_val = mb_metrics.igd(sample, ref=pf)
                emd_val = mb_metrics.emd(sample, ref=pf)
                
                igd_samples.append(float(igd_val))
                emd_samples.append(float(emd_val))
            
            # Use 95th percentile as the conservative noise floor
            results[mop_name]["floors"][str(n)] = {
                "nIGD_floor": np.percentile(igd_samples, 95) / diameter,
                "nEMD_floor": np.percentile(emd_samples, 95) / diameter
            }
            
    # Save results
    output_path = "MoeaBench/diagnostics/resources/baselines.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nSuccess! Baselines saved to {output_path}")

if __name__ == "__main__":
    calibrate()
