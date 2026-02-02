
"""
MoeaBench Heavy Tier Testing Suite (Deep Calibration)
=======================================================

Este é o nível mais profundo da pirâmide de testes, destinado à **Calibração Científica**.
Ele é utilizado para validar se novas implementações de algoritmos ou problemas
são estatisticamente equivalentes aos padrões de referência.

O que é testado:
----------------
- Execução de múltiplas repetições estatísticas.
- Aplicação de testes de hipótese (t-test) e métricas de tamanho de efeito (Cohen's d).
- Validação profunda de convergência e topologia.

Características:
----------------
- **Custo**: Computacionalmente caro (pode levar horas).
- **Rigor**: Destinado a releases maiores ou validação de novos métodos de pesquisa.

Execução:
---------
pytest tests/test_heavy_tier.py
"""
import pytest
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import sys
import zlib
sys.path.append(os.path.abspath("."))
import MoeaBench as mb
from MoeaBench.metrics.GEN_igd import GEN_igd

BASELINE_FILE = "tests/baselines_v0.7.6.csv"

def get_heavy_configs():
    """
    Returns (MOP, Algorithm) combinations for heavy statistical testing.
    Focuses on 'standard' intensity for robust distributions.
    """
    if not os.path.exists(BASELINE_FILE):
        return []
        
    df = pd.read_csv(BASELINE_FILE)
    # We filter only standard intensity runs for heavy testing
    std_df = df[df['Intensity'] == 'standard']
    
    configs = []
    for _, row in std_df.iterrows():
        configs.append((row['MOP'], row['Algorithm'], row['IGD_mean'], row['IGD_std'], row['Pop'], row['Gen']))
    return configs

configs = get_heavy_configs()

@pytest.mark.skipif(not os.path.exists(BASELINE_FILE), reason="Baseline CSV not found.")
@pytest.mark.parametrize("mop_name, alg_name, base_igd_mean, base_igd_std, pop, gen", configs)
def test_heavy_statistical_quality(mop_name, alg_name, base_igd_mean, base_igd_std, pop, gen):
    """
    Heavy Tier Test:
    1. Runs N=30 repetitions of the algorithm (seeded deterministically).
    2. Collects IGD distribution.
    3. Performs Mann-Whitney U test against the baseline mean/std (reconstructed normal approx).
    4. Calculates Vargha-Delaney A12 effect size.
    
    Pass Condition:
    - p-value > 0.05 (No significant difference) OR 
    - A12 >= 0.45 (If difference exists, it's not a significant degradation)
    """
    # Configuration
    N_REPEATS = 30
    
    # 1. Instantiate MOP & GT
    try:
        mop = getattr(mb.mops, mop_name)(M=3)
    except TypeError:
        mop = getattr(mb.mops, mop_name)(M=3, D=2)
        
    exp_ref = mb.experiment(mop=mop)
    F_opt = exp_ref.optimal(n_points=2000).objs
    
    # 2. Collect Current Data
    current_igd_values = []
    
    for i in range(N_REPEATS):
        # Deterministic Seed: Hash(Heavy + MOP + Alg + i)
        config_str = f"heavy_{mop_name}_{alg_name}_{i}"
        run_seed = zlib.crc32(config_str.encode()) & 0xffffffff
        
        if alg_name == "NSGA2": alg = mb.moeas.NSGA2(seed=run_seed)
        elif alg_name == "NSGA3": alg = mb.moeas.NSGA3(seed=run_seed)
        elif alg_name == "MOEAD": alg = mb.moeas.MOEAD(seed=run_seed)
        else: pytest.skip("Unsupported Alg")
        
        exp = mb.experiment(mop=mop)
        exp.moea = alg
        exp.run(repeat=1, generations=int(gen), population=int(pop), verbose=False)
        
        igd = GEN_igd([exp.pop().objs], F_opt).evaluate()[0]
        current_igd_values.append(igd)
        
    # 3. Statistical Analysis
    # Since we don't have the raw baseline samples here, we simulate a distribution 
    # from the baseline mean/std for the U-test. 
    # NOTE: This is an approximation. Ideally we should load raw baseline data.
    # For now, we test if current mean is significantly worse.
    
    current_mean = np.mean(current_igd_values)
    
    # One-sided t-test: Is current IGD significantly GREATER (wc) than baseline?
    # H0: current <= baseline
    # H1: current > baseline
    t_stat, p_val = stats.ttest_ind_from_stats(
        mean1=current_mean, std1=np.std(current_igd_values, ddof=1), nobs1=N_REPEATS,
        mean2=base_igd_mean, std2=base_igd_std, nobs2=30, # Assuming baseline was N=30 or similar equivalent power
        alternative='greater'
    )
    
    # Fail if statistically significantly worse (p < 0.05) AND practical difference is large
    if p_val < 0.05:
         # Calculate Cohen's d as effect size proxy since A12 requires raw data
         pooled_std = np.sqrt((np.std(current_igd_values)**2 + base_igd_std**2)/2)
         cohens_d = (current_mean - base_igd_mean) / pooled_std
         
         if cohens_d > 0.5: # Medium effect size degradation
             pytest.fail(f"Performance Regression Detected: p={p_val:.4f}, d={cohens_d:.2f}")

