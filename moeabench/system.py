# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import importlib.util as _importlib_util
from .core.base import emit_output as _emit_output

def version(show: bool = True) -> str:
    """Returns the current moeabench version and optionally prints it."""
    v = "0.14.0"
    if show:
        _emit_output(f"moeabench v{v}", markdown=f"**moeabench** `v{v}`")
    return v

def check_dependencies():
    """Prints a report of installed optional dependencies."""
    deps = ["numpy", "matplotlib", "pandas", "plotly", "IPython", "joblib"]
    lines = [f"moeabench v{version(show=False)} dependency report"]
    md_lines = [f"### moeabench `v{version(show=False)}` Dependency Report", ""]
    for dep in deps:
        spec = _importlib_util.find_spec(dep)
        status = "INSTALLED" if spec is not None else "MISSING"
        lines.append(f"- {dep:<12}: {status}")
        md_lines.append(f"- **{dep}**: `{status}`")
    
    # Check for moea engines
    engines = ["pymoo", "deap"]
    lines.append("")
    lines.append("moea engines")
    md_lines.extend(["", "#### MOEA Engines"])
    for eng in engines:
        spec = _importlib_util.find_spec(eng)
        status = "AVAILABLE" if spec is not None else "UNAVAILABLE"
        lines.append(f"- {eng:<12}: {status}")
        md_lines.append(f"- **{eng}**: `{status}`")
    _emit_output("\n".join(lines), markdown="\n".join(md_lines))

def info(show: bool = True) -> dict:
    """
    Returns environment metadata for scientific reproducibility and optionally displays it.
    """
    import sys
    import platform
    import numpy as np
    from datetime import datetime
    
    payload = {
        "moeabench_version": version(show=False),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat()
    }
    if show:
        lines = [
            "moeabench environment info",
            f"- moeabench : {payload['moeabench_version']}",
            f"- python    : {payload['python_version']}",
            f"- numpy     : {payload['numpy_version']}",
            f"- platform  : {payload['platform']}",
            f"- timestamp : {payload['timestamp']}",
        ]
        md_lines = [
            "### moeabench Environment Info",
            "",
            f"- **moeabench**: `{payload['moeabench_version']}`",
            f"- **python**: `{payload['python_version']}`",
            f"- **numpy**: `{payload['numpy_version']}`",
            f"- **platform**: `{payload['platform']}`",
            f"- **timestamp**: `{payload['timestamp']}`",
        ]
        _emit_output("\n".join(lines), markdown="\n".join(md_lines))
    return payload


def output(text: str, markdown: str | None = None) -> str:
    """Environment-aware output helper for non-report functions."""
    return _emit_output(text, markdown=markdown)

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
         _emit_output("Warning: no objective data to export.", markdown="> Warning: no objective data to export.")
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
        
    path = os.path.abspath(filename)
    _emit_output(f"Objectives exported to: {path}", markdown=f"Objectives exported to: `{path}`")

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
        _emit_output(
            "Error: export_variables requires Experiment, Run, Population or ndarray.",
            markdown="**Error**: `export_variables` requires `Experiment`, `Run`, `Population`, or `ndarray`."
        )
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
        
    path = os.path.abspath(filename)
    _emit_output(f"Variables exported to: {path}", markdown=f"Variables exported to: `{path}`")


__all__ = [
    "version",
    "check_dependencies",
    "info",
    "output",
    "export_objectives",
    "export_variables",
]
