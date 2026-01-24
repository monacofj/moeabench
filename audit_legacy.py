
import os
import numpy as np
import pandas as pd
import MoeaBench as mb
from pymoo.indicators.igd import IGD

def calculate_igd(front, reference):
    """Calculate IGD using pymoo."""
    if front.shape[0] == 0: return 1.0
    metric = IGD(reference, zero_to_one=True)
    return metric.do(front)

def calculate_sos_error(F, mop_type):
    """Verify invariants (SOS=1.0 for spherical, Sum=0.5 for linear)."""
    if F.shape[0] == 0: return None
    if mop_type == "DTLZ1" or mop_type == "DPF1":
        sums = np.sum(F, axis=1)
        return np.mean(np.abs(sums - 0.5))
    else: # Sphericals and others (approx)
        sos = np.sum(F**2, axis=1)
        return np.mean(np.abs(sos - 1.0))

def run_current_audit(mop_name, M):
    """Run NSGA-III and return (F, X) using legacy parity parameters."""
    # Derived from conf.txt analysis for Parity of Instance
    if mop_name in ["DTLZ8", "DTLZ9"]:
        N_val = 10
        k = 0 
    elif mop_name.startswith("DPF"):
        k = 10
        D = M - 1
        N_val = D + k - 1
    else: # DTLZ1-7
        k = 10
        N_val = M + k - 1
    
    mop_cls = getattr(mb.mops, mop_name)
    if mop_name.startswith("DPF"):
        mop = mop_cls(M=M, D=(M-1), K=10, N=N_val)
    elif mop_name == "DTLZ9":
        mop = mop_cls(M=M, N=N_val)
    else:
        mop = mop_cls(M=M, N=N_val)
        
    moea = mb.moeas.NSGA3(seed=42)
    exp = mb.experiment(mop, moea)
    exp.run(pop_size=150, generations=300)
    run = exp.last_run
    return np.asarray(run.front()), np.asarray(run.set())

def main():
    mops = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7", "DTLZ8", "DTLZ9",
            "DPF1", "DPF2", "DPF3", "DPF4", "DPF5"]
    dimensions = [3, 5, 10]
    shadow_dir = "tests/shadow_data"
    os.makedirs(shadow_dir, exist_ok=True)
    
    results = []
    print("Starting Legacy Audit Confrontation (with Persistence)...")
    
    for mop_name in mops:
        print(f"Auditing {mop_name}...")
        for M in dimensions:
            legacy_dir = f"tests/audit_data/legacy_{mop_name}"
            prefix = "lg__" if "DTLZ" in mop_name else "lg_"
            legacy_moea_path = os.path.join(legacy_dir, f"{prefix}{mop_name}_{M}_front.csv")
            gt_path = f"tests/ground_truth/{mop_name}_{M}_optimal.csv"
            
            if not os.path.exists(legacy_moea_path) or not os.path.exists(gt_path):
                continue
                
            try:
                F_leg = pd.read_csv(legacy_moea_path, header=None).values
                F_gt = pd.read_csv(gt_path, header=None).values
            except Exception: continue
                
            # Run Current and Persist
            F_curr, X_curr = run_current_audit(mop_name, M)
            pd.DataFrame(F_curr).to_csv(os.path.join(shadow_dir, f"shadow_{mop_name}_{M}_F.csv"), index=False, header=False)
            pd.DataFrame(X_curr).to_csv(os.path.join(shadow_dir, f"shadow_{mop_name}_{M}_X.csv"), index=False, header=False)
            
            # Target Metric (Sum for linear, SOS for spherical)
            target_val = 0.5 if mop_name in ["DTLZ1", "DPF1"] else 1.0
            
            def get_val(F, name):
                if name in ["DTLZ1", "DPF1"]: return np.mean(np.sum(F, axis=1))
                return np.mean(np.sum(F**2, axis=1))

            val_leg = get_val(F_leg, mop_name)
            val_curr = get_val(F_curr, mop_name)
            
            results.append({
                "MOP": mop_name, "M": M,
                "IGD_Leg": calculate_igd(F_leg, F_gt),
                "IGD_Curr": calculate_igd(F_curr, F_gt),
                "Val_Leg": val_leg,
                "Val_Curr": val_curr,
                "Target": target_val,
                "Divergence_Leg %": abs(val_leg - target_val) / target_val * 100,
                "Divergence_Curr %": abs(val_curr - target_val) / target_val * 100
            })
            
    df = pd.DataFrame(results)
    with open("docs/legacy_report.md", "w") as f:
        f.write("# Legacy Audit Report: v0.6.x vs v0.7.5\n\n")
        f.write("Evaluation of dimensional divergence and implementation fixes.\n")
        f.write("Setup: NSGA-III | Pop 150 | Gen 300 | Ground Truth: v0.7.5 Analytical\n\n")
        f.write("## Comparison Table\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Data Persistence\n")
        f.write("- Shadow Data (v0.7.5) saved to `tests/shadow_data/` for all runs.\n")
        f.write("- Format: `shadow_{MOP}_{M}_F.csv` (Objectives) and `shadow_{MOP}_{M}_X.csv` (Variables).\n")

    print(f"Audit complete. Report saved to docs/legacy_report.md")

if __name__ == "__main__":
    main()
