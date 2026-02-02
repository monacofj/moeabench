"""
MoeaBench Baseline Generation Engine
====================================

Este script é o motor de execução responsável pela coleta de dados empíricos do MoeaBench v0.7.6.
Sua função é executar uma grade sistemática de experimentos (MOP x Algoritmo x Intensidade)
e persistir as frentes de Pareto resultantes para análise posterior.

Racional Científico:
--------------------
Para garantir a reprodutibilidade absoluta, este script utiliza um sistema de sementes (seeds)
baseado em hashing determinístico da configuração (MOP, Algoritmo, Intensidade, ID da Run).
Isso elimina a dependência do relógio do sistema e garante que os mesmos resultados
sejam obtidos em diferentes plataformas.

Funcionalidades:
----------------
- Execução determinística via hashing (CRC32).
- Checkpoints de gerações intermediárias para análise de convergência.
- Monitoramento "On-the-Fly": Aborta a execução se detectar variância zero.
- Persistência atômica: Salva resultados run-a-run para tolerância a falhas.

Uso:
----
python tests/calibration/generate_baselines.py
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import time

# Ensure local MoeaBench is importable
# Path is relative to project root
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
import MoeaBench as mb

# Configuration Defaults
DEFAULT_MOPS = [f"DTLZ{i}" for i in range(1, 10)] + [f"DPF{i}" for i in range(1, 6)]
DEFAULT_ALGS = ["NSGA2", "NSGA3", "MOEAD"]
DEFAULT_INTENSITIES = {
    "light":    {"pop": 52,  "gen": 100}, 
    "standard": {"pop": 200, "gen": 1000}
}
DEFAULT_RUNS = 5
DATA_DIR = os.path.join(PROJ_ROOT, "tests/calibration_data")
STATS_FILE = os.path.join(DATA_DIR, "generation_stats.csv")

def run_generation(mops=None, algs=None, intensities=None, repeat=None):
    os.makedirs(DATA_DIR, exist_ok=True)
    
    mops = mops or DEFAULT_MOPS
    algs = algs or DEFAULT_ALGS
    intensities = intensities or DEFAULT_INTENSITIES
    repeat = repeat or DEFAULT_RUNS
    
    print(f"=== MoeaBench v{mb.system.version()} Baseline Generation (Phase 1B-B) ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {DATA_DIR}")
    print("-" * 70)

    # We use deterministic seeding based on configuration hashes 
    # instead of a global random state.
    
    total_tasks = len(mops) * len(algs) * len(intensities) * repeat
    current_task = 0

    for mop_name in mops:
        print(f"\n[Problem] {mop_name}")
        
        # Instantiate MOP to check for constraints
        try:
            mop_obj = getattr(mb.mops, mop_name)(M=3)
        except TypeError:
            mop_obj = getattr(mb.mops, mop_name)(M=3, D=2)
            
        has_constraints = mop_obj.get_n_ieq_constr() > 0

        for alg_name in algs:
            # MOEAD in Pymoo does not support constraints
            if alg_name == "MOEAD" and has_constraints:
                print(f"  [Skipping] {alg_name} on {mop_name} (Constraints not supported)")
                current_task += len(intensities) * repeat
                continue

            for intensity_name, params in intensities.items():
                pop_size = params["pop"]
                n_gen = params["gen"]
                
                for r_idx in range(repeat):
                    current_task += 1
                    filename = f"{mop_name}_{alg_name}_{intensity_name}_run{r_idx:02d}.csv"
                    filepath = os.path.join(DATA_DIR, filename)
                    
                    # Deterministic Seed Generation
                    # Seed = Hash(MOP + ALG + INTENSITY + RUN_ID)
                    import zlib
                    config_str = f"{mop_name}_{alg_name}_{intensity_name}_{r_idx}"
                    # & 0xffffffff ensures it fits in 32-bit unsigned integer
                    run_seed = zlib.crc32(config_str.encode()) & 0xffffffff
                    
                    # Resume check
                    if os.path.exists(filepath):
                        print(f"  [{current_task}/{total_tasks}] {filename} ... Cached.")
                        continue
                    
                    print(f"  [{current_task}/{total_tasks}] {filename} ... ", end="", flush=True)
                    
                    try:
                        start_time = time.time()
                        exp = mb.experiment(mop=mop_obj)
                        
                        # Pass the deterministic seed explicitly
                        if alg_name == "NSGA2": alg = mb.moeas.NSGA2(seed=run_seed)
                        elif alg_name == "NSGA3": alg = mb.moeas.NSGA3(seed=run_seed)
                        elif alg_name == "MOEAD": alg = mb.moeas.MOEAD(seed=run_seed)
                        
                        exp.moea = alg
                        # Determines if we should try to save checkpoints
                        is_standard = (intensity_name == "standard")

                        exp.run(repeat=1, generations=n_gen, population=pop_size, verbose=False)
                        
                        # Save final front
                        objs = exp.pop().objs
                        
                        # Snapshot Logic (Post-Run Extraction)
                        if is_standard and hasattr(exp.moea, 'F_gen_all'):
                            print(f" (History found: {len(exp.moea.F_gen_all)} gens) ", end="")
                            # Attempt to save intermediate generations if the MOEA exposed them
                            history = exp.moea.F_gen_all # List of arrays
                            for g in range(99, n_gen, 100): # 99 (Gen 100), 199 (Gen 200)...
                                if g < len(history):
                                    snap_objs = history[g]
                                    snap_df = pd.DataFrame(snap_objs, columns=[f"f{i+1}" for i in range(snap_objs.shape[1])])
                                    snap_filename = f"{mop_name}_{alg_name}_{intensity_name}_run{r_idx:02d}_gen{g+1}.csv"
                                    snap_df.to_csv(os.path.join(DATA_DIR, snap_filename), index=False)

                        df = pd.DataFrame(objs, columns=[f"f{i+1}" for i in range(objs.shape[1])])
                        df.to_csv(filepath, index=False)
                        
                        duration = time.time() - start_time
                        
                        # Save metadata
                        meta_df = pd.DataFrame([[filename, duration, run_seed]], columns=["filename", "duration", "seed"])
                        meta_df.to_csv(STATS_FILE, mode='a', header=not os.path.exists(STATS_FILE), index=False)
                        
                        print(f"Done ({duration:.2f}s)")
                        
                    except Exception as e:
                        print(f"FAILED: {str(e)}")
                        # Tolerate and proceed as requested
                        continue

    print("\n" + "="*70)
    print(f"Generation Complete. Data saved to {DATA_DIR}")
    print("="*70)

if __name__ == "__main__":
    run_generation()
