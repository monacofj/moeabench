# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import importlib.util

def version() -> str:
    """Returns the current MoeaBench version."""
    return "0.10.8"

def check_dependencies():
    """Prints a report of installed optional dependencies."""
    deps = ["numpy", "matplotlib", "pandas", "plotly", "IPython", "joblib"]
    print(f"--- MoeaBench v{version()} Dependency Report ---")
    for dep in deps:
        spec = importlib.util.find_spec(dep)
        status = "INSTALLED" if spec is not None else "MISSING"
        print(f"  - {dep:<12}: {status}")
    
    # Check for moea engines
    engines = ["pymoo", "deap"]
    print("\n--- MOEA Engines ---")
    for eng in engines:
        spec = importlib.util.find_spec(eng)
        status = "AVAILABLE" if spec is not None else "UNAVAILABLE"
        print(f"  - {eng:<12}: {status}")

def export_objectives(data, filename=None):
    """
    Exports Pareto Front/objectives to a CSV file.
    
    Args:
        data: Experiment, Run, Population or objective array.
        filename (str): Output CSV path.
    """
    from .metrics.evaluator import _extract_data
    _, fronts, name, _ = _extract_data(data)
    
    if len(fronts) == 0:
         print("Warning: No objective data to export.")
         return
         
    import numpy as np
    import os
    
    # Merge if multiple fronts (Experiment case)
    final_objs = np.vstack(fronts)
    
    if filename is None:
        filename = f"{name}_objectives.csv".replace(" ", "_").lower()
    
    # Header
    M = final_objs.shape[1]
    header = ",".join([f"f{i+1}" for i in range(M)])
    
    try:
        import pandas as pd
        df = pd.DataFrame(final_objs, columns=[f"f{i+1}" for i in range(M)])
        df.to_csv(filename, index=False)
    except ImportError:
        np.savetxt(filename, final_objs, delimiter=",", header=header, comments='')
        
    print(f"Objectives exported to: {os.path.abspath(filename)}")

def export_variables(data, filename=None):
    """
    Exports Pareto Set/variables to a CSV file.
    """
    from .core.experiment import experiment
    from .core.run import Run, Population
    
    if isinstance(data, experiment):
        vars_list = [r.variables() for r in data]
        name = data.name
    elif isinstance(data, Run):
        vars_list = [data.variables()]
        name = data.name
    elif isinstance(data, Population):
        vars_list = [data.variables]
        name = data.label
    elif isinstance(data, np.ndarray):
        vars_list = [data]
        name = "Array"
    else:
        print("Error: export_variables requires Experiment, Run, Population or ndarray.")
        return

    import numpy as np
    import os
    
    final_vars = np.vstack([v for v in vars_list if v is not None])
    
    if filename is None:
        filename = f"{name}_variables.csv".replace(" ", "_").lower()
        
    N = final_vars.shape[1]
    header = ",".join([f"x{i+1}" for i in range(N)])

    try:
        import pandas as pd
        df = pd.DataFrame(final_vars, columns=[f"x{i+1}" for i in range(N)])
        df.to_csv(filename, index=False)
    except ImportError:
        np.savetxt(filename, final_vars, delimiter=",", header=header, comments='')
        
    print(f"Variables exported to: {os.path.abspath(filename)}")
