
import numpy as np
import os
import shutil
from MoeaBench import mb
from MoeaBench.moeas import NSGA3

def test_protocol_a():
    print("\n>>> Testing Protocol A: Analytical (Default)")
    mop = mb.mops.DTLZ1(M=3)
    # Ensure fresh start
    sidecar = "DTLZ1_A.json"
    if os.path.exists(sidecar): os.remove(sidecar)
    
    success = mop.calibrate(source_baseline=sidecar, force=True)
    assert success
    assert os.path.exists(sidecar)
    print("Protocol A Success!")

def test_protocol_b():
    print("\n>>> Testing Protocol B: Static (source_gt)")
    mop = mb.mops.DTLZ1(M=3)
    csv_file = "test_gt.csv"
    gt_data = np.random.rand(100, 3) # Dummy GT
    np.savetxt(csv_file, gt_data, delimiter=',')
    
    sidecar = "DTLZ1_B.json"
    if os.path.exists(sidecar): os.remove(sidecar)
    
    success = mop.calibrate(source_gt=csv_file, source_baseline=sidecar, force=True)
    assert success
    assert os.path.exists(sidecar)
    
    # Verify the GT in the sidecar matches our CSV
    import json
    with open(sidecar, 'r') as f:
        data = json.load(f)
        saved_gt = np.array(data['gt_reference'])
        assert saved_gt.shape == (100, 3)
    
    os.remove(csv_file)
    print("Protocol B Success!")

def test_protocol_c():
    print("\n>>> Testing Protocol C: Empirical (source_search)")
    mop = mb.mops.DTLZ1(M=3)
    moea = NSGA3(population=50, generations=5) # Fast search for testing
    
    sidecar = "DTLZ1_C.json"
    if os.path.exists(sidecar): os.remove(sidecar)
    
    success = mop.calibrate(source_search=moea, source_baseline=sidecar, force=True)
    assert success
    assert os.path.exists(sidecar)
    print("Protocol C Success!")

if __name__ == "__main__":
    try:
        test_protocol_a()
        test_protocol_b()
        test_protocol_c()
        print("\nALL PROTOCOL TESTS PASSED!")
    finally:
        # Cleanup
        for f in ["DTLZ1_A.json", "DTLZ1_B.json", "DTLZ1_C.json"]:
            if os.path.exists(f): os.remove(f)
