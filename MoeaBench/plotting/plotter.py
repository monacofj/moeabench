# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .scatter3d import Scatter3D
from .scatter2d import Scatter2D
import numpy as np

def spaceplot(*args, objectives=None, mode='static', title=None, axis_labels=None):
    """
    Plots 3D scatter of objectives (Pareto Front).
    """
    processed_args = []
    names = []
    trace_modes = [] # Store if we want markers or lines
    
    # Defaults
    if title is None: title = "Pareto-optimal front"
    if axis_labels is None: axis_labels = "Objective"
    
    from ..stats.attainment import AttainmentSurface

    for i, arg in enumerate(args):
        val = arg
        name = None
        t_mode = 'markers'
        
        # Unwrap standard MoeaBench objects
        # 1. AttainmentSurface (special case)
        if isinstance(arg, AttainmentSurface):
             val = arg
             name = arg.name
             t_mode = 'lines+markers' # Show the boundary clearly
        # 2. Prioritize front() method if available (Experiment/Run)
        elif hasattr(arg, 'front') and callable(getattr(arg, 'front')):
             val = arg.front()
        # 3. Fallback to .objectives property (Population)
        elif hasattr(arg, 'objectives'):
             val = arg.objectives
        
        # Try to extract name metadata
        if not name:
            # 1. inner array .name (SmartArray)
            if hasattr(val, 'name') and val.name:
                 name = val.name
            # 2. original object .name
            elif hasattr(arg, 'name') and arg.name:
                 name = arg.name
            # 3. .label (Population)
            elif hasattr(arg, 'label') and arg.label:
                 name = arg.label

        # Fallback name
        if not name:
             name = f"Data {i+1}"
             
        # Extract Plot Metadata from first argument
        if i == 0:
            if hasattr(val, 'label') and val.label and title == "Pareto-optimal front":
                 title = val.label
            if hasattr(val, 'axis_label') and val.axis_label and axis_labels == "Objective":
                 axis_labels = val.axis_label
        
        # Convert to numpy if needed
        val = np.array(val)
        processed_args.append(val)
        names.append(name)
        trace_modes.append(t_mode)
        
    # Axis selection
    if objectives is None:
        # Auto-detect data dimension
        dims = [d.shape[1] for d in processed_args if len(d.shape) > 1]
        max_dim = max(dims) if dims else 2
        
        if max_dim < 3:
             objectives = [0, 1]
        else:
             objectives = [0, 1, 2]
    
    # Selection of Plotter based on dimensions
    if len(objectives) == 2:
        s = Scatter2D(names, processed_args, objectives, type=title, mode=mode, axis_label=axis_labels, trace_modes=trace_modes)
    else:
        # Ensure 3rd dimension exists for Scatter3D
        for k in range(len(processed_args)):
             d = processed_args[k]
             if d.shape[1] < 3:
                 # Pad with zeros to ensure at least 3 columns for 3D plotting
                 padding = np.zeros((d.shape[0], 3 - d.shape[1]))
                 processed_args[k] = np.column_stack([d, padding])

        # Ensure objectives has 3 elements
        while len(objectives) < 3:
             objectives.append(0)
             
        s = Scatter3D(names, processed_args, objectives, type=title, mode=mode, axis_label=axis_labels, trace_modes=trace_modes)
    
    s.show()

def timeplot(*args, **kwargs):
    """
    Plots metric matrices over time.
    Wrapper for metrics.plot_matrix.
    """
    from ..metrics.evaluator import plot_matrix
    plot_matrix(args, **kwargs)
