# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import importlib.util

def version() -> str:
    """Returns the current MoeaBench version."""
    return "0.10.6"

def check_dependencies():
    """Prints a report of installed optional dependencies."""
    deps = ["pymoo", "deap", "plotly", "pandas", "numpy", "scipy", "tqdm"]
    print("--- MoeaBench Dependency Check ---")
    for dep in deps:
        status = "Installed" if importlib.util.find_spec(dep) else "NOT FOUND"
        print(f"{dep:<15}: {status}")

def export_objectives(data, filename: str = None):
    """
    [mb.system.export_objectives] Exports objectives to a CSV file.
    
    Args:
        data: Experiment, Population, or array-like object.
        filename: Output filename. Defaults to <name>_objectives.csv.
    """
    objs, name = _resolve_data(data, mode='objectives')
    if not filename:
        prefix = name if name else "experiment"
        filename = f"{prefix}_objectives.csv"
    
    _save_to_csv(objs, filename, prefix='f')

def export_variables(data, filename: str = None):
    """
    [mb.system.export_variables] Exports decision variables to a CSV file.
    
    Args:
        data: Experiment, Population, or array-like object.
        filename: Output filename. Defaults to <name>_variables.csv.
    """
    vars, name = _resolve_data(data, mode='variables')
    if not filename:
        prefix = name if name else "experiment"
        filename = f"{prefix}_variables.csv"
        
    _save_to_csv(vars, filename, prefix='x')

def _resolve_data(data, mode='objectives'):
    """Internal helper to extract data and name from different MoeaBench objects."""
    import numpy as np
    
    name = getattr(data, 'name', None)
    if not name:
        name = getattr(data, 'label', None)
    
    # Empty strings should be treated as None for naming defaults
    if not name:
        name = None

    if mode == 'objectives':
        if hasattr(data, 'front'):
            return data.front(), name
        if hasattr(data, 'objs'):
            return data.objs, name
    else: # variables
        if hasattr(data, 'set'):
            return data.set(), name
        if hasattr(data, 'vars'):
            return data.vars, name
            
    return np.asarray(data), name

def _save_to_csv(data, filename, prefix='col'):
    """Internal helper to save data to CSV with optional headers."""
    import numpy as np
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        
    try:
        import pandas as pd
        cols = [f"{prefix}{i+1}" for i in range(data.shape[1])]
        df = pd.DataFrame(data, columns=cols)
        df.to_csv(filename, index=False)
        print(f"Data exported to {filename}")
    except ImportError:
        np.savetxt(filename, data, delimiter=",")
        print(f"Data exported to {filename} (using numpy, no headers)")
