# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Internal utilities for the diagnostics package.
"""

import numpy as np
import os
from typing import Any, Tuple, Optional

def _calibration_coordinates(gt: np.ndarray, problem: str):
    """Return (ideal, nadir) only when GT is the exact packaged normalized GT.

    The offline clinical anchors are calibrated on 'gt_norm' from the saved
    reference package, not on the raw objective values returned by an MOEA.
    Equality of the *whole* reference, rather than problem name alone, guards
    against silently normalizing explicitly supplied unrelated GT datasets.
    """
    if not problem or os.path.basename(problem) != problem:
        return None
    package = os.path.join(
        os.path.dirname(__file__), "resources", "references",
        problem, "calibration_package.npz",
    )
    if not os.path.isfile(package):
        return None
    with np.load(package, allow_pickle=False) as pkg:
        reference = np.asarray(pkg["gt_norm"], dtype=float)
        if not np.array_equal(np.asarray(gt), reference):
            return None
        ideal = np.asarray(pkg["ideal"], dtype=float)
        nadir = np.asarray(pkg["nadir"], dtype=float)
    if ideal.shape != (reference.shape[1],) or nadir.shape != ideal.shape:
        raise ValueError("Invalid calibration normalization bounds")
    delta = nadir - ideal
    if np.any(delta <= 0) or not np.all(np.isfinite(delta)):
        raise ValueError("Calibration normalization bounds are not positive")
    return ideal, nadir


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

    # 3. Resolve Ground Truth (ref).
    # With no explicit GT, prefer the canonical reference used to calibrate
    # Q-score baselines. Previously the resolver selected mop.pf() first and
    # only audit() silently replaced that reference later. That made public
    # individual calls and aggregated metrics disagree (ADR 0048).
    if gt is None and problem_name is not None:
        from .baselines import load_offline_baselines
        try:
            bases = load_offline_baselines()
            registered = bases.get("_gt_registry", {})
            m = p_final.shape[1] if isinstance(p_final, np.ndarray) and p_final.ndim == 2 else None
            key = f"{problem_name}__M{m}" if m is not None else problem_name
            reference = registered.get(key, registered.get(problem_name))
            if reference is not None:
                gt = np.asarray(reference)
        except Exception:
            pass

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
    
    # 3.3 Apply the same frozen min/max transform to the optimizer population
    # that was applied when the registered GT and Q baselines were generated.
    # Only MOP-bearing objects have raw objective coordinates to transform.
    # Arrays already prepared by audit() are not transformed twice.
    calibration_bounds = None
    if mop is not None and gt is not None:
        calibration_bounds = _calibration_coordinates(np.asarray(gt), problem_name)
        if calibration_bounds is not None:
            ideal, nadir = calibration_bounds
            span = nadir - ideal
            p_final = (np.asarray(p_final) - ideal) / span
            if p_initial is not None:
                p_initial = (np.asarray(p_initial) - ideal) / span

    # 4. Resolve Resolution Scale (s_k).
    # The resolution is a property of the *selected* GT and the effective
    # approximation size (ADR 0026 / dissertation Eq. 6.2), not an arbitrary
    # constant. In particular, the previous default 1.0 silently changed
    # the scientific meaning of Closeness and of its clinical baselines.
    #
    # Honor an explicitly supplied s_k. Otherwise compute a deterministic
    # FPS-based scale on the actual reference, using the same K policy as the
    # clinical baselines. Do not trust an unverified MOP-level cached scale
    # for a possibly different, user-supplied GT.
    if res is None:
        if gt is None or len(np.asarray(gt)) < 2:
            res = np.nan  # No meaningful geometric resolution exists.
        else:
            from .baselines import get_resolution_factor_k, snap_k
            m = p_final.shape[1] if isinstance(p_final, np.ndarray) and p_final.ndim == 2 else 0
            effective_k = snap_k(int(k_val or 0), problem=problem_name if m > 3 else None)
            res = get_resolution_factor_k(np.asarray(gt), effective_k, seed=0)

    return {
        'P_final': np.asarray(p_final),
        'P_initial': np.asarray(p_initial) if p_initial is not None else None,
        'GT': np.asarray(gt) if gt is not None else None,
        's_k': float(res),
        'normalization': (
            {'method': 'calibration_ideal_nadir', 'ideal': calibration_bounds[0],
             'nadir': calibration_bounds[1]}
            if calibration_bounds is not None else None
        ),
        'problem': str(problem_name or "Unknown"),
        'k': int(k_val or 0),
        'is_container': is_container
    }
