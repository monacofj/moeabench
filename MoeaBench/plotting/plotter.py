# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .scatter3d import Scatter3D
import numpy as np

def spaceplot(*args, objectives=None, mode='interactive', title=None, axis_labels=None):
    """
    Plots 3D scatter of objectives (Pareto Front).
    """
    processed_args = []
    names = []
    
    # Defaults
    if title is None: title = "Pareto-optimal front"
    if axis_labels is None: axis_labels = "Objective"
    
    for i, arg in enumerate(args):
        val = arg
        name = None
        
        # Unwrap standard MoeaBench objects
        # Prioritize front() method if available (Experiment/Run)
        if hasattr(arg, 'front') and callable(getattr(arg, 'front')):
             val = arg.front()
        # Fallback to .objectives property (Population)
        elif hasattr(arg, 'objectives'):
             val = arg.objectives
        
        # Try to extract name metadata
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
        
    # Axis selection
    if objectives is None:
        objectives = [0, 1, 2]
    # Ensure 3 axes
    if len(objectives) < 3:
         # Pad with 0? Or raise error?
         # Legacy used list generic filling.
         # For 2D data, we might want 2D plot?
         # Scatter3D supports 3D. 
         # If data is 2D, we can plot 2D on Z=0?
         # Or stick to 3 indices.
         while len(objectives) < 3:
             objectives.append(0)
             
    s = Scatter3D(names, processed_args, objectives, type=title, mode=mode, axis_label=axis_labels)
    s.show()

def timeplot(*args, **kwargs):
    """
    Plots metric matrices over time.
    Wrapper for metrics.plot_matrix.
    """
    from ..metrics.evaluator import plot_matrix
    plot_matrix(args, **kwargs)
