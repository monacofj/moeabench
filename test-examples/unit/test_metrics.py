# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import numpy as np

# Ensure the library is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from MoeaBench import mb

def test_hv_normalization():
    """Verify Hypervolume normalization and reference points."""
    # Front at [0.5, 0.5]
    f1 = np.array([[0.5, 0.5]])
    # Reference at [1.0, 1.0]
    ref = np.array([[1.0, 1.0]])
    
    # Mode exact for 2D
    hv_res = mb.metrics.hypervolume(f1, ref=ref, mode='exact')
    
    # Normalize logic: global_max is [1.0, 1.0], global_min is [0.5, 0.5]
    # Actually, evaluator.normalize takes ref_exps and all_current_objs_list.
    # ref point is max_val * 1.1 usually in some systems, 
    # but MoeaBench uses internal normalization.
    
    assert hv_res.values.size == 1
    val = float(hv_res)
    assert val > 0
    # In MoeaBench, HV normalization adds a 10% margin to the bounding box.
    # [0.5, 0.5] in normalized box [0, 0] to [1.1, 1.1] gives 1.1 * 1.1 = 1.21
    assert np.allclose(val, 1.21, atol=1e-5)

def test_convergence_metrics():
    """Verify GD and IGD fallback to analytical optimal front."""
    exp = mb.experiment()
    exp.mop = mb.mops.DTLZ2(M=2)
    
    # Create artificial fronts
    # Optimal front is spherical: x^2 + y^2 = 1
    # Let's create a front that is exactly on the optimal surface
    t = np.linspace(0, np.pi/2, 10)
    ideal = np.column_stack([np.cos(t), np.sin(t)])
    
    # 1. GD (Generational Distance) - Avg distance from front to optimal
    res_gd = mb.metrics.gd(ideal, ref=ideal) # Should be 0
    assert np.allclose(float(res_gd), 0.0, atol=1e-5)
    
    # If we don't pass ref, it should use exp.optimal_front()
    # Note: internal _calc_metric will call exp.optimal_front()
    exp.moea = mb.moeas.NSGA2deap(population=20, generations=5)
    exp.run(repeat=1) # Minimal run to init histories
    res_igd = mb.metrics.igd(exp) 
    # res_igd is a MetricMatrix. To check values, we can look at the last generation
    last_igd = res_igd.gens(-1)
    assert np.all(last_igd >= 0)
    assert res_igd.metric_name == "IGD"
