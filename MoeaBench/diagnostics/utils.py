# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Internal utilities for the diagnostics package.
"""

import numpy as np
from typing import Any, Tuple, Optional

def _resolve_diagnostic_context(data: Any, ref: Any = None, s_k: Any = None, **kwargs) -> Tuple[np.ndarray, np.ndarray, float, str, int]:
    """
    Polymorphic resolver for diagnostic data.
    
    Returns:
        tuple: (front_array, ref_array, s_k_value, problem_name, k)
    """
    front = None
    gt = ref
    res = s_k
    problem_name = kwargs.get('problem', None)
    k_val = kwargs.get('k', None)

    # 1. Resolve Front from Data
    if hasattr(data, 'front') and callable(data.front):
        # Experiment or Run
        front = data.front()
    elif hasattr(data, 'objectives'):
        # Population
        front = data.objectives
    elif isinstance(data, np.ndarray):
        front = data
    else:
        try:
            front = np.asarray(data)
        except:
             raise TypeError(f"Unsupported data type for diagnostics: {type(data)}")

    # 2. Resolve Ground Truth (ref)
    if gt is None:
        if hasattr(data, 'pf') and callable(data.pf):
            gt = data.pf()
        elif hasattr(data, 'mop') and hasattr(data.mop, 'pf'):
            gt = data.mop.pf()
        elif hasattr(data, 'source'):
            src = data.source
            if hasattr(src, 'pf') and callable(src.pf):
                gt = src.pf()
            elif hasattr(src, 'mop') and hasattr(src.mop, 'pf'):
                gt = src.mop.pf()
        
        # 2.1 Sidecar/Cache Look-up (Plugin support)
        if gt is None and problem_name is not None:
             from .baselines import load_offline_baselines
             try:
                 bases = load_offline_baselines()
                 if "_gt_registry" in bases and problem_name in bases["_gt_registry"]:
                     gt = np.array(bases["_gt_registry"][problem_name])
             except:
                 pass

    # 3. Resolve Problem Name and K
    mop = None
    if hasattr(data, 'mop'):
        mop = data.mop
    elif hasattr(data, 'source') and hasattr(data.source, 'mop'):
        mop = data.source.mop

    if problem_name is None and mop is not None:
        problem_name = mop.__class__.__name__
    
    if k_val is None:
        if isinstance(front, np.ndarray) and front.ndim > 1:
            k_val = len(front)
        elif hasattr(data, 'pop'):
            # Experiment
            try:
                p = data.pop()
                k_val = len(p)
            except:
                pass
    
    # 4. Resolve Resolution Scale (s_k)
    if res is None:
        if mop is not None:
            res = getattr(mop, 's_fit', getattr(mop, 's_k', 1.0))
        else:
            res = 1.0

    return np.asarray(front), np.asarray(gt) if gt is not None else None, float(res), str(problem_name or "Unknown"), int(k_val or 0)
