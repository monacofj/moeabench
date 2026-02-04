# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import MoeaBench as mb
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.decomposition.pbi import PBI
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize

class MoeaBenchWrapper(Problem):
    def __init__(self, mop):
        super().__init__(n_var=mop.N, n_obj=mop.M, n_ieq_constr=0, xl=mop.xl, xu=mop.xu)
        self.mop = mop
    def _evaluate(self, X, out, *args, **kwargs):
        res = self.mop.evaluation(X)
        out["F"] = res["F"]

def audit_dpf3(theta_val):
    print(f"--- DPF3 audit with theta={theta_val} ---")
    mop = mb.mops.DPF3(M=3)
    pymoo_problem = MoeaBenchWrapper(mop)
    ref_dirs = get_reference_directions("energy", 3, 200, seed=1)
    
    algo = MOEAD(ref_dirs, n_neighbors=30, decomposition=PBI(theta=theta_val), seed=1)
    res = minimize(pymoo_problem, algo, ('n_gen', 1000), verbose=False)
    
    F = res.F
    if F is not None:
        sq_sum = np.sum(F**2, axis=1)
        print(f"Mean Sum Sq: {np.mean(sq_sum):.6f}")
        print(f"Max Sum Sq: {np.max(sq_sum):.6f}")
        print(f"Min Sum Sq: {np.min(sq_sum):.6f}")

if __name__ == "__main__":
    audit_dpf3(5.0)  # Current
    audit_dpf3(0.5)  # Proposed
    audit_dpf3(0.1)  # Radical
