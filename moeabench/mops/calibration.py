# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Ground-Truth calibration for MOPs, independent of experimental FAIR scores.

Existing baselines_v4_ecdf sidecars can be read for their gt_reference, but
their FAIR/ECDF payload is deliberately neither computed nor interpreted.
"""

import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

GT_SCHEMA = "mop_gt_v1"


def _reference_array(raw, mop):
    gt = np.asarray(raw, dtype=float)
    if gt.ndim != 2 or gt.shape[1] != mop.M or not len(gt):
        raise ValueError(
            f"Ground Truth for {mop.name!r} must be a nonempty (N, {mop.M}) array."
        )
    if not np.all(np.isfinite(gt)):
        raise ValueError("Ground Truth must contain only finite objective values.")
    return gt


def _sidecar_path(mop, source_baseline):
    if source_baseline is not None:
        return Path(source_baseline)
    try:
        return Path(inspect.getfile(mop.__class__)).resolve().parent / (
            f"{mop.name}_M{mop.M}.json"
        )
    except (TypeError, OSError):
        return Path(f"{mop.name}_M{mop.M}.json")


def _load_sidecar(path, mop):
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or "gt_reference" not in data:
        raise ValueError(f"Calibration sidecar {path} contains no gt_reference.")
    if data.get("problem_id") != mop.name:
        raise ValueError(f"Calibration sidecar {path} belongs to a different MOP.")
    if int(data.get("mop_dimension", -1)) != mop.M:
        raise ValueError(f"Calibration sidecar {path} has a different objective count.")
    return _reference_array(data["gt_reference"], mop)


def calibrate_mop(
    mop,
    source_baseline=None,
    source_gt=None,
    source_search=None,
    force=False,
    *,
    n_points=None,
    **kwargs,
):
    """Calibrate and retain a MOP-owned objective-space Ground Truth.

    The historical public arguments and True=fresh/False=loaded return
    convention are preserved. Previously accepted ECDF-only options cannot
    be used to request FAIR baselines from the independent GT service.

    The reference is stored on this MOP instance and in a portable sidecar.
    The legacy baseline JSON format is accepted on load for GT migration.
    """
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise NotImplementedError(
            f"FAIR/ECDF calibration options are unavailable: {unsupported}. "
            "Ground-Truth calibration remains supported."
        )
    path = _sidecar_path(mop, source_baseline)
    if path.is_file() and not force:
        gt = _load_sidecar(path, mop)
        mop._gt_reference = np.array(gt, copy=True)
        mop._gt_source = str(path)
        return False

    if source_search is not None:
        from ..core.experiment import experiment
        exp = experiment(mop=mop, moea=source_search)
        exp.run()
        gt = exp.non_dominated().objectives
        provenance = "empirical_search"
    elif source_gt is not None:
        if isinstance(source_gt, (str, Path)):
            gt = np.loadtxt(source_gt, delimiter=",")
        else:
            gt = source_gt
        provenance = "external"
    else:
        if n_points is None:
            # ADR 0018: preserve the adopted DTLZ / DPF GT sample densities.
            n_points = 10000 if str(mop.name).upper().startswith("DPF") else 2000
        if not isinstance(n_points, int) or n_points < 1:
            raise ValueError("n_points must be a positive integer.")
        gt = mop.pf(n_points=n_points)
        provenance = "analytical"

    gt = _reference_array(gt, mop)
    # Write before updating in-memory state: a failed write must not claim
    # successful calibration. Do not overwrite the frozen FAIR payload unless
    # force=True was explicitly requested.
    data = {
        "schema": GT_SCHEMA,
        "problem_id": mop.name,
        "mop_dimension": mop.M,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gt_source": provenance,
        "gt_reference": gt.tolist(),
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, allow_nan=False)
    mop._gt_reference = np.array(gt, copy=True)
    mop._gt_source = str(path)
    return True


def calibrated_gt(mop):
    """Return a defensive copy of the calibrated MOP GT, or raise clearly."""
    gt = getattr(mop, "_gt_reference", None)
    if gt is None:
        raise ValueError(
            f"MOP {mop.name!r} has no calibrated Ground Truth. "
            "Call mop.calibrate() before using scale='abs'."
        )
    return _reference_array(gt, mop).copy()
