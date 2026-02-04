# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import os
sys.path.append(os.getcwd())
from MoeaBench import mb
import numpy as np

def find_ranks_stable():
    print("Searching for Seed/Gen combo with ~5 ranks...")
    
    # Try different seeds
    for seed in range(1, 20):
        exp = mb.experiment()
        exp.mop = mb.mops.DTLZ2(M=3)
        # Use the same config as generate_images.py
        exp.moea = mb.moeas.NSGA3(population=100, generations=30, seed=seed) 
        exp.run(repeat=5)
        
        run = exp.last_run
        
        # Check early generations
        for g in range(1, 10):
            try:
                pop = run.pop(g)
                res = mb.stats.strata(pop)
                max_rank = res.max_rank
                
                # Check if it meets criteria (e.g. 4 to 6 ranks)
                if 4 <= max_rank <= 6:
                    print(f"MATCH FOUND! Seed: {seed}, Gen: {g} -> Max Rank: {max_rank}")
                    # Verify the distribution isn't trivial (e.g. 99% in rank 0)
                    counts = res.counts
                    print(f"  Distribution: {counts}")
                    return # Stop after finding first good one
            except:
                pass
    print("No suitable match found in search range.")

find_ranks_stable()
