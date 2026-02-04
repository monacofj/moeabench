# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Ensure project root is in path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from MoeaBench import mb

def test_caste2():
    # Run NSGA3
    exp1 = mb.experiment()
    exp1.name = "NSGA3"
    exp1.mop = mb.mops.DTLZ2(M=3)
    exp1.moea = mb.moeas.NSGA3(population=100, generations=50)
    exp1.run(repeat=10)

    # Run SPEA2 as a competitor
    exp2 = mb.experiment()
    exp2.name = "SPEA2"
    exp2.mop = mb.mops.DTLZ2(M=3)
    exp2.moea = mb.moeas.SPEA2(population=100, generations=50)
    exp2.run(repeat=10)

    # Capture snapshots across all runs at G=10
    pop1 = exp1.pop(10)
    pop2 = exp2.pop(10)

    # 1. Generate Collective (default / GDP)
    output_path_coll = os.path.join(ROOT_DIR, "misc", "caste2_collective.png")
    print(f"Generating Collective (GDP) strat_caste plot -> {output_path_coll}")
    mb.view.strat_caste(pop1, pop2, mode='collective', title="Caste Hierarchy: Collective Quality (GDP)")
    plt.savefig(output_path_coll, bbox_inches='tight', dpi=150)
    plt.close('all')

    # 2. Generate Individual (Per Capita)
    output_path_ind = os.path.join(ROOT_DIR, "misc", "caste2_individual.png")
    print(f"Generating Individual (Per Capita) strat_caste plot -> {output_path_ind}")
    mb.view.strat_caste(pop1, pop2, mode='individual', title="Caste Hierarchy: Individual Merit (Per Capita)")
    plt.savefig(output_path_ind, bbox_inches='tight', dpi=150)
    plt.close('all')

    # 3. Generate Clean Collective (High-level summary)
    output_path_clean = os.path.join(ROOT_DIR, "misc", "caste2_clean.png")
    print(f"Generating Clean Collective strat_caste plot -> {output_path_clean}")
    mb.view.strat_caste(pop1, pop2, mode='collective', show_quartiles=False, 
                         title="Caste Hierarchy: Robustness Summary (Clean)")
    plt.savefig(output_path_clean, bbox_inches='tight', dpi=150)
    plt.close('all')

    print("Success! All plots saved.")

if __name__ == "__main__":
    test_caste2()
