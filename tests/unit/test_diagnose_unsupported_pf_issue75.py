# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The unsupported-PF diagnostic is evidence-only, never a baseline writer."""

import numpy as np
import pytest
from calibration.scripts import diagnose_unsupported_pf as diagnosis


def test_unsupported_pf_diagnostic_refuses_missing_calibration_package(tmp_path):
    with pytest.raises(FileNotFoundError):
        diagnosis.diagnose("FakeMOP",root=tmp_path)


def test_sampler_diagnosis_records_cardinality_and_freshness(monkeypatch,tmp_path):
    path=tmp_path/"moeabench/diagnostics/resources/references/Toy"
    path.mkdir(parents=True)
    x=np.linspace(0,1,41)
    gt=np.column_stack((x,1-x))
    np.savez(path/"calibration_package.npz",gt_norm=gt,ideal=[0,0],nadir=[1,1])

    class Toy:
        def pf(self,n_points):
            x=np.linspace(0,1,n_points)
            return np.column_stack((x,1-x))

    monkeypatch.setattr(diagnosis.mops,"Toy",Toy,raising=False)
    result=diagnosis.diagnose("Toy",root=tmp_path,fresh_points=10)
    assert result["status"]=="DIAGNOSTIC_ONLY_NO_MOP_CALIBRATION_CERTIFIED"
    assert result["sampling_probes"]["matched_requested_count"]["returned"]==41
    assert result["sampling_probes"]["fresh_extension"]["returned"]==51
    assert result["sampling_probes"]["fresh_extension"]["fresh_point_count"]>0
    assert result["archived_shape"]==[41,2]
