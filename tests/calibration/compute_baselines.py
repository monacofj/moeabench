"""
MoeaBench Baseline Analysis Engine
==================================

Este script atua como a camada de inteligência analítica do framework de testes.
Ele processa os dados brutos gerados pelo `generate_baselines.py`, calculando as
métricas de desempenho (IGD, Hypervolume, KS) em relação à "Verdade" (Ground Truth).

Metodologia de Cálculo:
-----------------------
- O Hypervolume é calculado em um espaço normalizado [0, 1]^M.
- Os limites de normalização (Ideal/Nadir) são definidos pela união entre a frente
  ótima analítica e todas as frentes observadas, garantindo que nenhum ponto seja "cortado".
- O IGD é calculado individualmente para cada run para extração de média e desvio padrão.

Saída:
------
Os resultados consolidados são salvos em `tests/baselines_v0.7.6.csv`, que serve como
o oráculo para todos os testes de regressão do sistema.

Uso:
----
python tests/calibration/compute_baselines.py
"""
import os
import sys

# Ensure local MoeaBench is importable
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import numpy as np
import pandas as pd
import MoeaBench as mb
from MoeaBench.metrics.evaluator import normalize
from MoeaBench.metrics.GEN_hypervolume import GEN_hypervolume
import re

DATA_DIR = os.path.join(PROJ_ROOT, "tests/calibration_data")
STATS_FILE = os.path.join(DATA_DIR, "generation_stats.csv")
OUTPUT_FILE = os.path.join(PROJ_ROOT, "tests/baselines_v0.7.6.csv")

def compute_baselines():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} not found.")
        return

    # Load durations if available
    durations = {}
    if os.path.exists(STATS_FILE):
        df_stats = pd.read_csv(STATS_FILE)
        durations = dict(zip(df_stats['filename'], df_stats['duration']))

    # Group files by MOP, Algorithm, and Intensity
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and f != "generation_stats.csv"]
    groups = {}
    
    for f in files:
        # Pattern: MOP_ALG_INT_runXX.csv
        parts = f.replace(".csv", "").split("_")
        if len(parts) < 4: continue
        
        mop_name = parts[0]
        alg_name = parts[1]
        int_name = parts[2]
        
        key = (mop_name, alg_name, int_name)
        if key not in groups:
            groups[key] = []
        groups[key].append(f)

    results = []
    total_groups = len(groups)
    current_group = 0

    print(f"=== MoeaBench Baseline Analysis (Phase 1B-B) ===")
    print(f"Processing {total_groups} experiment groups...")
    print("-" * 70)

    for (mop_name, alg_name, int_name), group_files in groups.items():
        current_group += 1
        print(f"[{current_group}/{total_groups}] Analyzing {mop_name} | {alg_name} | {int_name} ... ", end="", flush=True)
        
        try:
            # Instantiate MOP and get Ground Truth
            try:
                mop = getattr(mb.mops, mop_name)(M=3)
            except TypeError:
                mop = getattr(mb.mops, mop_name)(M=3, D=2)
            
            exp_ref = mb.experiment(mop=mop)
            F_opt = exp_ref.optimal(n_points=2000).objs
            
            # Load all runs in the group first to determine bounds
            all_fronts = []
            group_durations = []
            for f in group_files:
                df_run = pd.read_csv(os.path.join(DATA_DIR, f))
                all_fronts.append(df_run.values)
                if f in durations:
                    group_durations.append(durations[f])
            
            # Determine normalization bounds (Ground Truth + All Observations)
            min_val, max_val = normalize([F_opt], all_fronts)
            
            # Theoretical max HV
            hv_opt_engine = GEN_hypervolume([F_opt], 3, min_val, max_val)
            hv_opt = float(hv_opt_engine.evaluate()[0])
            
            # 1. IGD Metrics
            # We need to compute IGD for each run against F_opt
            igd_values = []
            for F_obs in all_fronts:
                #mb.metrics.igd expects an experiment or front. 
                #Let's use a simpler way since we have the raw arrays.
                #Actually, mb.metrics.igd is a convenience wrapper.
                #We'll calculate it manually or use mb.metrics.evaluator
                from MoeaBench.metrics.GEN_igd import GEN_igd
                engine = GEN_igd([F_obs], F_opt)
                igd_values.append(engine.evaluate()[0])
            
            igd_mean = np.mean(igd_values)
            igd_std = np.std(igd_values)
            
            # 2. KS Metrics (Topology)
            # Combine all fronts for a high-fidelity distribution check
            combined_front = np.vstack(all_fronts)
            topo_match = mb.stats.topo_distribution(combined_front, F_opt)
            ks_p_val = np.mean(list(topo_match.p_values.values()))
            
            # 3. HV Metrics
            hv_engine = GEN_hypervolume(all_fronts, 3, min_val, max_val)
            hv_results = hv_engine.evaluate()
            hv_mean = np.mean(hv_results)
            hv_diff = hv_opt - hv_mean
            
            # 4. Meta
            avg_duration = np.mean(group_durations) if group_durations else 0
            
            # Detect Pop/Gen from filename or metadata (we know the defaults)
            pop = 52 if int_name == "light" else 200
            gen = 100 if int_name == "light" else 1000
            
            res_row = {
                "MOP": mop_name,
                "Algorithm": alg_name,
                "Intensity": int_name,
                "Pop": pop,
                "Gen": gen,
                "IGD_mean": igd_mean,
                "IGD_std": igd_std,
                "KS_p_val": ks_p_val,
                "HV_mean": hv_mean,
                "HV_opt": hv_opt,
                "HV_diff": hv_diff,
                "Ideal_1": min_val[0], "Ideal_2": min_val[1], "Ideal_3": min_val[2],
                "Nadir_1": max_val[0], "Nadir_2": max_val[1], "Nadir_3": max_val[2],
                "Time_sec": avg_duration
            }
            results.append(res_row)
            print("Done.")
            
        except Exception as e:
            print(f"FAILED: {str(e)}")
            continue

    if results:
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
        print("\n" + "="*70)
        print(f"Analysis Complete. Baselines saved to {OUTPUT_FILE}")
        print("="*70)

if __name__ == "__main__":
    compute_baselines()
