#!/usr/bin/env python3
import os
import sys
import pandas as pd

# 1. Garantir que a raiz do projeto esteja no PYTHONPATH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import MoeaBench as mb
from MoeaBench.core import SmartArray

def run_diagnostic():
    # Caminhos dos dados auditados
    leg_path = os.path.join(BASE_DIR, "tests/audit_data/legacy_DTLZ2/lg__DTLZ2_3_opt_front.csv")
    gt_path = os.path.join(BASE_DIR, "tests/ground_truth/DTLZ2_3_optimal.csv")
    
    print(f"--- MoeaBench Diagnostic: DTLZ2 (M=3) ---")
    
    try:
        F_leg = pd.read_csv(leg_path, header=None).values
        F_gt = pd.read_csv(gt_path, header=None).values
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return

    # 2. Dados Envelopados em SmartArray (Idiomático)
    gt_smart = SmartArray(F_gt, name="v0.7.5 Ground Truth", label="Analítico (Invariante SOS=1.0)")
    leg_smart = SmartArray(F_leg, name="Legado (MoeaBench/legacy)", label="Heurístico (Invariante SOS≈0.88)")

    # 3. Uso direto do topo_shape em modo Interativo
    # O Matplotlib 3D está quebrado neste ambiente (conflito User/System). 
    # Usamos o modo 'interactive' (Plotly) para garantir o 3D solicitado.
    print("\n[INFO] Forçando modo INTERATIVO (Plotly) para contornar falha do Matplotlib 3D.")
    print("Invocando mb.view.topo_shape...")
    mb.view.topo_shape(gt_smart, leg_smart, title="Confronto Geométrico: DTLZ2", mode='interactive')

if __name__ == "__main__":
    run_diagnostic()
