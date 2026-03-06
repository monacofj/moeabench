import json
import os
import numpy as np
import moeabench as mb
from moeabench.diagnostics import auditor

DATA_JSON_PATH = "tests/calibration_reference_audit_v0.13.1.json"
TARGETS_JSON_PATH = "tests/calibration_reference_targets_v0.13.1.json"

def update_targets():
    print(f"Loading Reference Data from {DATA_JSON_PATH}...")
    with open(DATA_JSON_PATH, "r") as f:
        ref_data = json.load(f)
    
    new_targets = {}
    problems = sorted(list(ref_data['problems'].keys()))
    algorithms = ['NSGA2', 'MOEAD', 'NSGA3']
    
    for prob in problems:
        p_data = ref_data['problems'][prob]
        gt_points = np.array(p_data['gt_points'])
        
        for alg in algorithms:
            if alg not in p_data['algorithms']:
                continue
                
            print(f"Auditing {prob} / {alg}...")
            final_front = np.array(p_data['algorithms'][alg]['final_front'])
            
            try:
                report = mb.diagnostics.audit(final_front, ground_truth=gt_points, problem=prob)
                
                # Extract metrics
                q_scores = {k: float(v.value) for k, v in report.q_audit_res.scores.items()}
                raw_metrics = {k: float(v.value) for k, v in report.fr_audit_res.metrics.items()}
                
                new_targets[f"{prob}/{alg}"] = {
                    "status": "OK",
                    "q_scores": q_scores,
                    "raw_metrics": raw_metrics
                }
            except Exception as e:
                print(f"  FAILED {prob}/{alg}: {str(e)}")
                new_targets[f"{prob}/{alg}"] = {
                    "status": "ERROR",
                    "error": str(e)
                }
                
    print(f"Saving new targets to {TARGETS_JSON_PATH}...")
    with open(TARGETS_JSON_PATH, "w") as f:
        json.dump(new_targets, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    update_targets()
