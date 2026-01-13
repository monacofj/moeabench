
from MoeaBench import mb
import numpy as np
import time

def main():
    print("--- Verifying Parallel Execution & Seed Determinism ---")
    
    # 1. Setup a standard experiment with a fixed seed
    mop = mb.mops.DTLZ2(M=3)
    moea_factory = lambda: mb.moeas.NSGA3(population=50, generations=20, seed=42)
    
    # Execute Serial
    print("\nRunning Serial (repeat=4)...")
    exp_serial = mb.experiment()
    exp_serial.name = "Serial"
    exp_serial.mop = mop
    exp_serial.moea = moea_factory()
    start_serial = time.time()
    exp_serial.run(repeat=4, workers=None)
    time_serial = time.time() - start_serial
    print(f"Serial Time: {time_serial:.2f}s")
    
    # Execute Parallel
    print("\nRunning Parallel (repeat=4, workers=2)...")
    exp_parallel = mb.experiment()
    exp_parallel.name = "Parallel"
    exp_parallel.mop = mop
    exp_parallel.moea = moea_factory()
    start_parallel = time.time()
    exp_parallel.run(repeat=4, workers=2)
    time_parallel = time.time() - start_parallel
    print(f"Parallel Time: {time_parallel:.2f}s")
    
    # 2. Check Seed Determinism
    print("\nVerifying Determinism (Serial vs Parallel):")
    for i in range(4):
        hv_s = mb.hv(exp_serial.runs[i].last_pop)
        hv_p = mb.hv(exp_parallel.runs[i].last_pop)
        print(f"  Run {i+1}: Serial HV={hv_s:.6f}, Parallel HV={hv_p:.6f}")
        assert np.isclose(hv_s, hv_p), f"Mismatch in Run {i+1}!"

    # 3. Check MetricMatrix stability
    mm_s = mb.hv(exp_serial).gens(-1)
    mm_p = mb.hv(exp_parallel).gens(-1)
    print(f"\nFinal Distribution (Serial):   {mm_s}")
    print(f"Final Distribution (Parallel): {mm_p}")
    assert np.allclose(mm_s, mm_p), "MetricMatrix distribution mismatch!"

    print("\nSUCCESS: Parallel execution is deterministic and faster!")

if __name__ == "__main__":
    main()
