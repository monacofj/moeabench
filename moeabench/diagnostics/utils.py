# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Internal utilities for the diagnostics package.
"""

import numpy as np
import os
from typing import Any, Tuple, Optional

def _resolve_diagnostic_context(data: Any, ref: Any = None, s_k: Any = None, **kwargs) -> dict:
    """
    Polymorphic resolver for diagnostic data.
    
    Returns:
        dict: {
            'P_final': np.ndarray,      # The evaluated population/front
            'P_initial': np.ndarray,    # The starting population (if available)
            'GT': np.ndarray,           # Ground Truth
            's_k': float,               # Resolution factor
            'problem': str,             # Problem name
            'k': int,                   # Number of points
            'is_container': bool        # True if data was a Run/Experiment
        }
    """
    p_final = None
    p_initial = kwargs.get('initial_data', None)
    gt = ref
    res = s_k
    problem_name = kwargs.get('problem', None)
    k_val = kwargs.get('k', None)
    is_container = False

    # 1. Resolve Populations from Data
    if hasattr(data, 'pop') and callable(data.pop):
        # Experiment or Run
        is_container = True
        p_final = data.pop(-1).objectives if not hasattr(data, 'front') else data.front(-1)
        try:
            p_initial = data.pop(0).objectives
        except:
            pass
    elif hasattr(data, 'objectives'):
        # Single Population object
        p_final = data.objectives
    elif isinstance(data, np.ndarray):
        p_final = data
    else:
        try:
            p_final = np.asarray(data)
        except:
             raise TypeError(f"Unsupported data type for diagnostics: {type(data)}")

    # 2. Resolve Problem Name and K
    mop = None
    if hasattr(data, 'mop'):
        mop = data.mop
    elif hasattr(data, 'source') and hasattr(data.source, 'mop'):
        mop = data.source.mop

    if problem_name is None and mop is not None:
        problem_name = mop.name if hasattr(mop, 'name') else mop.__class__.__name__
    
    if k_val is None:
        if isinstance(p_final, np.ndarray) and p_final.ndim > 1:
            k_val = len(p_final)
        elif is_container:
            try:
                p = data.pop()
                k_val = len(p)
            except:
                pass

    # 3. Resolve Ground Truth (ref)
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
        
        # 3.1 Sidecar/Cache Look-up
        if gt is None and problem_name is not None:
             from .baselines import load_offline_baselines
             try:
                 bases = load_offline_baselines()
                 if "_gt_registry" in bases and problem_name in bases["_gt_registry"]:
                     gt = np.array(bases["_gt_registry"][problem_name])
             except:
                 pass
        
    # 3.2 File Path Resolution
    if isinstance(gt, str) and os.path.exists(gt):
        try:
            if gt.endswith(".npy"):
                gt = np.load(gt)
            elif gt.endswith(".npz"):
                data_npz = np.load(gt)
                key = 'F' if 'F' in data_npz else data_npz.files[0]
                gt = data_npz[key]
            elif gt.endswith(".csv"):
                gt = np.genfromtxt(gt, delimiter=',', skip_header=1)
        except:
            pass
    
    # 4. Resolve Resolution Scale (s_k)
    if res is None:
        if mop is not None:
            res = getattr(mop, 's_fit', getattr(mop, 's_k', 1.0))
        else:
            res = 1.0

    return {
        'P_final': np.asarray(p_final),
        'P_initial': np.asarray(p_initial) if p_initial is not None else None,
        'GT': np.asarray(gt) if gt is not None else None,
        's_k': float(res),
        'problem': str(problem_name or "Unknown"),
        'k': int(k_val or 0),
        'is_container': is_container
    }
