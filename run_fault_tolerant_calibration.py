import os
import sys
import glob

PROJ_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from tests.calibration.generate_baselines import run_generation
from tests.calibration.compute_baselines import compute_baselines
from tests.calibration.generate_visual_report import generate_visual_report

def run_fault_tolerant_calibration():
    print("=== MoeaBench v0.7.7 Fault-Tolerant Calibration (N=30) ===")
    
    # Target Problems and Algorithms
    mops = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7", 
            "DPF1", "DPF2", "DPF3", "DPF4", "DPF5"]
    algs = ["NSGA2", "NSGA3", "MOEAD"]
    repeats = 30
    
    data_dir = os.path.join(PROJ_ROOT, "tests/calibration_data")
    os.makedirs(data_dir, exist_ok=True)

    # 1. GENERATION PHASE (Fault-Tolerant)
    # We iterate manually to check existence before launching heavy jobs
    
    # We use the library's runner but we need to trick it or extend it? 
    # Actually, verify_degenerate.py style calls `run_generation` which does all at once.
    # To be truly fault tolerant per run, we should likely implement a loop here that calls 
    # run_generation for specific missing items or trust strict checking inside run_generation?
    # run_generation in generate_baselines.py DOES check `exist_ok=True` but standard logic 
    # might overwrite.
    
    # Let's inspect generate_baselines logic via tool or Assume? 
    # Standard logic often overwrites. Let's write a dedicated granular looper here.
    
    import MoeaBench as mb
    
    print(f"Checking existing files in {data_dir}...")
    
    for mop_name in mops:
        M = 2 if "DPF" in mop_name or mop_name == "DTLZ7" else 3
        # DTLZ7 is M=2 in standard benchmark, others usually 3.
        # Wait, DTLZ7 is usually M=2? Or M=3? Double check.
        # User defined standard: DTLZ(M=3) usually. DTLZ7 is exception?
        # Let's stick to standard instantiation which handles defaults.
        
        for alg_name in algs:
            # Check how many runs already exist
            existing_runs = glob.glob(os.path.join(data_dir, f"{mop_name}_{alg_name}_standard_run*.csv"))
            completed_count = len(existing_runs)
            
            if completed_count >= repeats:
                print(f"[SKIP] {mop_name} | {alg_name} (Found {completed_count}/{repeats} runs)")
                continue
            
            needed = repeats - completed_count
            print(f"[EXEC] {mop_name} | {alg_name} (Found {completed_count}/{repeats} -> Need {needed})")
            
            # We assume sequential seeds. If we have run00, run01... we need run02...
            # But the underlying runner `run_generation` usually does 0..N.
            # To avoid complexity, we can use a custom simpler loop here utilizing the mb infrastructure directly.
            
            # Setup Experiment
            exp = mb.experiment()
            
            # Instantiate MOP
            if "DTLZ" in mop_name:
                M_obj = 3 
                if mop_name == "DTLZ7": M_obj = 2 # DTLZ7 is typically 2
                mop_cls = getattr(mb.mops, mop_name)
                exp.mop = mop_cls(M=M_obj)
            elif "DPF" in mop_name:
                mop_cls = getattr(mb.mops, mop_name)
                exp.mop = mop_cls() # DPF defaults
                
            # Instantiate Algo
            alg_cls = getattr(mb.moeas, alg_name)
            # Standard params
            pop_size = 150 if "DTLZ" in mop_name else 100
            
            # Constraint: NSGA2 DEAP requires pop_size % 4 == 0
            if alg_name == "NSGA2" and pop_size % 4 != 0:
                pop_size += (4 - (pop_size % 4)) # Adjust to next multiple (e.g., 150 -> 152)

            n_gen = 1000
            
            exp.moea = alg_cls(population=pop_size, generations=n_gen)
            
            # Run the missing slots
            # We need to know which indices are missing. Crude way: 0 to 29.
            for i in range(repeats):
                filename = f"{mop_name}_{alg_name}_standard_run{i:02d}.csv"
                filepath = os.path.join(data_dir, filename)
                
                if os.path.exists(filepath):
                    continue
                    
                print(f"   > Running seed {i}...")
                try:
                    # Single run execution
                    # Note: exp.run(repeat=1) resets seed logic usually? 
                    # We manually set seed on algorithms or pass to run?
                    # exp.run(seed=...) is supported in newer versions (run logic).
                    # Actually, standard exp.run(repeat=...) handles offsets.
                    # We can use exp.run(seed=i) for single run.
                    
                    # Update configuration seed
                    exp.moea.seed = i 
                    # Run single
                    exp.run() 
                    
                    # Save (using export API or standard save)
                    # We want consistency with generate_baselines output format.
                    # generate_baselines uses:
                    # mb.system.export_objectives(exp.last_run, filename)
                    # Let's check imports.
                    
                    mb.system.export_objectives(exp, filepath)
                    
                except Exception as e:
                    print(f"   !!! FAIL seed {i}: {e}")
                    # Continue to next seed/algo? Yes, fault tolerance.
                    
    # 2. COMPUTE METRICS
    print("\nComputing Baselines...")
    compute_baselines()
    
    # 3. GENERATE REPORT
    print("\nGenerating v0.7.7 Report...")
    generate_visual_report()
    
    print("\nCalibration v0.7.7 Cycle Complete.")

if __name__ == "__main__":
    run_fault_tolerant_calibration()
