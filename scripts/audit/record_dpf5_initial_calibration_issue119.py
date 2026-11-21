# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Produce the inaugural corrected DPF5 (M=3,D=2) GT calibration candidate.

The JSON output is an explicit review artifact, never an automatic overwrite
of an existing fixture. Requires the pinned numerical environment.
"""
import hashlib
import json
import numpy as np
from moeabench.mops.DPF5 import DPF5
from scripts.audit.gt_dpf5_general_d_issue119 import reference


def produce(n=1000):
    mop=DPF5(M=3,D=2)
    ps=np.asarray(mop.ps(n_points=n),dtype=np.float64)
    got=np.asarray(mop.evaluation(ps)["F"],dtype=np.float64)
    repeat=np.asarray(mop.pf(n_points=n),dtype=np.float64)
    oracle=reference(ps,mop.M,mop.D)
    assert ps.shape==(n,mop.N)
    assert got.shape==(n,3)
    assert np.array_equal(got,repeat)
    assert len(np.unique(got,axis=0))==n
    assert np.isfinite(got).all()
    assert np.all((ps[:,0]<1/3)|(ps[:,0]>=1/3))
    assert np.any(ps[:,0]<1/3) and np.any(ps[:,0]>=1/3)
    assert np.array_equal(ps[:,mop.M-1],ps[:,0])
    assert np.array_equal(ps[:,mop.M:],np.full_like(ps[:,mop.M:],0.5))
    assert np.max(np.abs(got-oracle))<4e-14
    assert np.max(np.abs(np.linalg.norm(got,axis=1)-1))<4e-14
    return {
        "n_points":n,
        "gt_points":got.tolist(),
        "proof":"published DPF5 P/D, g=0 and norm=1",
        "generator":"DPF5(M=3,D=2).pf(n_points=1000)",
        "sha256_float64_le_C":hashlib.sha256(got.astype("<f8",copy=False).tobytes()).hexdigest(),
        "production_oracle_max_abs":float(np.max(np.abs(got-oracle))),
        "branch_P":int(np.sum(ps[:,0]<1/3)),
        "branch_D":int(np.sum(ps[:,0]>=1/3)),
    }


if __name__=="__main__":
    result=produce()
    print("FIRST_DPF5_GT_SUMMARY="+json.dumps({k:v for k,v in result.items() if k!="gt_points"},sort_keys=True))
    print("FIRST_DPF5_GT_JSON="+json.dumps(result,separators=(",",":"),allow_nan=False))
