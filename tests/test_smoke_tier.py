"""
MoeaBench Smoke Tier Testing Suite (Regression)
===============================================

Este é o segundo nível da pirâmide de testes, focado em **convergência básica**.
O objetivo é garantir que mudanças no código não degradem o desempenho dos algoritmos
ou quebrem a reprodutibilidade numérica da v0.7.6.

Metodologia de Reprodutibilidade:
---------------------------------
Diferente de testes estocásticos comuns, este Smoke Test utiliza o motor de
**Hashing Determinístico** para rodar com a semente exata (`run00`) da calibração original.
Isso permite uma comparação bit-a-bit (ou quase exata) com o baseline oficial.

O que é testado:
----------------
- Convergência de 42 combinações (MOP x Algoritmo) em intensidade `light`.
- Comparação do IGD observado contra o `baseline_mean + 3-sigma` (com margem de segurança).

Execução:
---------
pytest tests/test_smoke_tier.py
"""
import pytest
import os
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.abspath("."))
import MoeaBench as mb
from MoeaBench.metrics.GEN_igd import GEN_igd

BASELINE_FILE = "tests/baselines_v0.7.6.csv"

def get_smoke_configs():
    """
    Returns a list of (MOP, Algorithm) combinations from the baseline file 
    that were calibrated with 'light' intensity.
    """
    if not os.path.exists(BASELINE_FILE):
        return []
    
    df = pd.read_csv(BASELINE_FILE)
    light_df = df[df['Intensity'] == 'light']
    
    configs = []
    for _, row in light_df.iterrows():
        configs.append((row['MOP'], row['Algorithm'], row['IGD_mean'], row['IGD_std'], row['Pop'], row['Gen']))
    return configs

# We skip this if baseline file is missing
configs = get_smoke_configs()

@pytest.mark.skipif(not os.path.exists(BASELINE_FILE), reason="Baseline CSV not found. Run calibration first.")
@pytest.mark.parametrize("mop_name, alg_name, base_igd_mean, base_igd_std, pop, gen", configs)
def test_smoke_convergence(mop_name, alg_name, base_igd_mean, base_igd_std, pop, gen):
    """
    Smoke test: Runs a single iteration of a MOEA and checks if IGD is within 3-sigma of the baseline.
    Threshold: IGD_obs < mean + 3*std
    """
    # 1. Instantiate MOP
    try:
        mop = getattr(mb.mops, mop_name)(M=3)
    except TypeError:
        mop = getattr(mb.mops, mop_name)(M=3, D=2)
    
    # 2. Get Ground Truth for IGD
    exp_ref = mb.experiment(mop=mop)
    F_opt = exp_ref.optimal(n_points=2000).objs
    
    # 3. Deterministic Seed Generation
    # To correspond exactly to Run 00 of the calibration (Standard of Truth)
    # Seed = Hash(MOP + ALG + INTENSITY + RUN_ID)
    intensity_name = "light"
    r_idx = 0 
    import zlib
    config_str = f"{mop_name}_{alg_name}_{intensity_name}_{r_idx}"
    run_seed = zlib.crc32(config_str.encode()) & 0xffffffff
    
    # 4. Run MOEA (single run)
    exp = mb.experiment(mop=mop)
    
    if alg_name == "NSGA2": alg = mb.moeas.NSGA2(seed=run_seed)
    elif alg_name == "NSGA3": alg = mb.moeas.NSGA3(seed=run_seed)
    elif alg_name == "MOEAD": alg = mb.moeas.MOEAD(seed=run_seed)
    else: pytest.skip(f"MOEA {alg_name} not yet supported in smoke test")
    
    exp.moea = alg
    exp.run(repeat=1, generations=int(gen), population=int(pop), verbose=False)
    
    F_obs = exp.pop().objs
    
    # 5. Compute IGD
    engine = GEN_igd([F_obs], F_opt)
    igd_obs = engine.evaluate()[0]
    
    # 5. Verify against baseline
    # Using 3-sigma as a threshold for a "smoke" failure (catastrophic loss of convergence)
    # We add a small epsilon to std to avoid division by zero or strict equality issues in deterministic cases
    threshold = base_igd_mean + 3 * (base_igd_std + 1e-6)
    
    assert igd_obs <= threshold, (
        f"Smoke test failed for {alg_name} @ {mop_name}.\n"
        f"Observed IGD: {igd_obs:.4f}\n"
        f"Baseline Mean: {base_igd_mean:.4f} (std: {base_igd_std:.4f})\n"
        f"Threshold: {threshold:.4f}"
    )
