# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt
import warnings
from ..defaults import defaults
from ..plotting.scatter3d import Scatter3D
from ..plotting.scatter2d import Scatter2D
from ..stats.attainment import AttainmentSurface
from ..core.display import show_matplotlib

def topology(
    *args,
    objectives=None,
    mode='auto',
    title=None,
    axis_labels=None,
    labels=None,
    show=True,
    markers=False,
    show_gt=True,
    gt=None,
    gray_gt=True,
    **kwargs
):
    """
    [mb.view.topology] Topographic Shape Perspective.
    Visualizes the geometry of the solution set (scatter/surface).
    """
    from ..diagnostics.qscore import q_closeness_points
    
    # Environment detection for 'auto' mode
    if mode == 'auto':
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                mode = 'interactive'
            else:
                mode = 'static'
        except (ImportError, NameError):
            mode = 'static'
            
    processed_args = []
    names = []
    trace_modes = [] # Store if we want markers or lines
    
    # Defaults
    if title is None: title = "Solution Set Geometry"
    if axis_labels is None: axis_labels = "Objective"

    def _coerce_plot_array(value):
        """Normalize plot inputs to a numeric array with at least 2 dimensions."""
        arr = np.asarray(value)

        # Some metadata-carrying objects can collapse into a scalar object array.
        # If that happens, unwrap the payload once before normalizing shape.
        if arr.ndim == 0 and arr.dtype == object:
            inner = arr.item()
            if inner is not value:
                arr = np.asarray(inner)

        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            if arr.size == 0:
                return arr.reshape(0, 0)
            return arr.reshape(1, -1)
        return arr

    def _infer_gt_from_inputs(items):
        """Infer GT from experiment-like or sourced result inputs."""
        experiments = []
        for item in items:
            candidate = None
            if hasattr(item, "mop") and hasattr(item, "runs"):
                candidate = item
            elif hasattr(item, "source"):
                source = item.source
                if hasattr(source, "mop") and hasattr(source, "runs"):
                    candidate = source
                elif hasattr(source, "source") and hasattr(source.source, "mop") and hasattr(source.source, "runs"):
                    candidate = source.source
            if candidate is not None:
                experiments.append(candidate)

        if not experiments:
            warnings.warn(
                "view.topology(show_gt=True): no GT-bearing input found; GT could not be inferred.",
                RuntimeWarning,
            )
            return None

        gt_candidates = []
        mop_ids = []
        for exp in experiments:
            mop = getattr(exp, "mop", None)
            mop_id = getattr(mop, "name", mop.__class__.__name__ if mop is not None else "Unknown")
            mop_ids.append(mop_id)
            try:
                if hasattr(exp, "optimal_front"):
                    gt_candidates.append(np.asarray(exp.optimal_front()))
                elif mop is not None and hasattr(mop, "pf"):
                    gt_candidates.append(np.asarray(mop.pf()))
            except Exception:
                continue

        if len(set(mop_ids)) > 1:
            warnings.warn(
                f"view.topology(show_gt=True): experiments with different GT sources detected {sorted(set(mop_ids))}; using the first inferred GT.",
                RuntimeWarning,
            )

        if not gt_candidates:
            warnings.warn(
                "view.topology(show_gt=True): GT inference failed for provided experiments.",
                RuntimeWarning,
            )
            return None

        return gt_candidates[0]

    resolved_gt = None
    inferred_gt_idx = None
    if show_gt:
        if gt is not None:
            resolved_gt = np.asarray(gt)
        else:
            resolved_gt = _infer_gt_from_inputs(args)

    local_labels = list(labels) if labels is not None else None
    new_args = list(args)
    if show_gt and resolved_gt is not None:
        inferred_gt_idx = len(new_args)
        new_args.append(resolved_gt)
        if local_labels is not None:
            local_labels.append("True Front (GT)")

    for i, arg in enumerate(new_args):
        val = arg
        name = None
        t_mode = 'markers'
        
        # 1. Use explicit labels if provided
        if local_labels and i < len(local_labels):
            name = local_labels[i]
        elif inferred_gt_idx is not None and i == inferred_gt_idx:
            name = "True Front (GT)"

        # 2. Unwrap standard moeabench objects
        if isinstance(arg, AttainmentSurface):
             val = arg
             if not name: name = arg.name
             t_mode = 'lines'
        elif hasattr(arg, 'pf') and callable(getattr(arg, 'pf')) and not isinstance(arg, np.ndarray):
             # It's an experiment or problem, we want its front
             val = arg.front() if hasattr(arg, 'front') else arg.pf()
        elif hasattr(arg, 'front') and callable(getattr(arg, 'front')):
             val = arg.front()
        elif hasattr(arg, 'objectives'):
             val = arg.objectives
        
        if not name:
            # Smart Legend Naming: Standard Pattern 'Name (run, filter, gen)'
            exp_name = None
            run_idx = None
            filter_name = None
            gen_idx = None
            
            # 1. Extract Name & Run from metadata
            for obj in [val, arg]:
                if hasattr(obj, 'source'):
                    src_type = type(obj.source).__name__
                    if src_type == 'Run':
                        run_idx = getattr(obj.source, 'index', None)
                        if hasattr(obj.source, 'source'):
                            exp_name = getattr(obj.source.source, 'name', None)
                    elif src_type == 'experiment':
                        exp_name = getattr(obj.source, 'name', None)
                if exp_name: break
                
            if not exp_name:
                exp_name = getattr(arg, 'name', None) or getattr(val, 'name', None) or f"Data {i+1}"
                # Strip embedded "(run X)" if it leaked from Run.name
                import re
                exp_name = re.sub(r'\s*\(run\s*\d+\)', '', exp_name, flags=re.IGNORECASE)

            # 2. Extract Filter
            raw_label = getattr(val, 'label', None) or getattr(arg, 'label', None)
            if raw_label:
                 # run.py formats as "BaseLabel, Run X, Gen Y"
                 filter_name = raw_label.split(',')[0].strip()
                 if filter_name == "Superfront":
                     filter_name = None
                 
            # 3. Extract Generation
            gen_idx = getattr(val, 'gen', None)
            if gen_idx is None and raw_label and 'Gen ' in raw_label:
                try:
                    gen_part = raw_label.split('Gen ')[1].split(',')[0].strip()
                    gen_idx = int(gen_part)
                except ValueError: pass

            # 4. Assemble standard pattern
            pieces = []
            if run_idx is not None:
                pieces.append(f"run {run_idx}")
            if filter_name and filter_name != exp_name:
                pieces.append(filter_name)
            if gen_idx is not None and gen_idx >= 0:
                pieces.append(f"gen {gen_idx}")
                
            if pieces:
                name = f"{exp_name} ({', '.join(pieces)})"
            else:
                name = exp_name
             
        if i == 0:
            if hasattr(val, 'label') and val.label and title == "Solution Set Geometry" and val.label != "Superfront":
                 title = val.label
            if hasattr(val, 'axis_label') and val.axis_label and axis_labels == "Objective":
                 axis_labels = val.axis_label
        
        val = _coerce_plot_array(val)
        processed_args.append(val)
        names.append(name)
        
        # Override with explicit keyword traces if passed
        passed_traces = kwargs.get('trace_modes', None)
        if passed_traces and i < len(passed_traces) and passed_traces[i] is not None:
             trace_modes.append(passed_traces[i])
        else:
             trace_modes.append(t_mode)
        
    # Intercept custom marker_styles to avoid overwriting them
    marker_styles = kwargs.pop('marker_styles', [None] * len(processed_args))
    kwargs.pop('trace_modes', None) # We already parsed it into trace_modes
    
    # Reference for clinical markers: explicit gt (if shown) or explicit ref kwarg.
    auto_ref = resolved_gt if show_gt else kwargs.get('ref', None)

    if markers:
        # Clinical Quality Markers logic (Q-Closeness)
        for i, val in enumerate(processed_args):
            if trace_modes[i] != 'markers': continue
            
            # If the user already provided a specific symbol for this trace,
            # we don't overwrite it with semantic quality markers.
            if marker_styles[i] and 'symbol' in marker_styles[i]:
                continue
                
            try:
                # The original source object (carrying context like mop/problem)
                orig_obj = args[i]

                # Inject auto_ref if found and not explicitly overridden
                call_kwargs = kwargs.copy()
                if auto_ref is not None and 'ref' not in call_kwargs:
                    call_kwargs['ref'] = auto_ref
                # Baseline K must match calibration population size, not len(front).
                # For Experiment/Run objects, infer K from solver configuration when available.
                if 'k' not in call_kwargs:
                    pop_k = None
                    if hasattr(orig_obj, 'moea') and hasattr(orig_obj.moea, 'population'):
                        pop_k = getattr(orig_obj.moea, 'population', None)
                    elif hasattr(orig_obj, 'source') and hasattr(orig_obj.source, 'moea'):
                        pop_k = getattr(orig_obj.source.moea, 'population', None)
                    if isinstance(pop_k, int) and pop_k > 0:
                        call_kwargs['k'] = pop_k
                
                # Clinical Sensitivity Adjustment:
                # DTLZ1 global baseline is very large (~2.0), which makes 
                # points at 0.8 (visurally far) look like Solids.
                # We apply a didactic tightening (s_k=0.02) for better visual feedback.
                if 'problem' in call_kwargs:
                    pname = str(call_kwargs['problem']).upper()
                elif hasattr(orig_obj, 'mop'):
                    pname = str(orig_obj.mop.name).upper()
                else: pname = ""
                
                if "DTLZ1" in pname and 's_k' not in call_kwargs:
                    # DTLZ1 global baseline is very large (~2.0), making small errors look like Solids.
                    # We use a very strict 0.02 factor to match human visual perception in [0, 1] scale.
                    call_kwargs['s_k'] = 0.02 
                
                q_vals = q_closeness_points(orig_obj, **call_kwargs)
                
                # Fallback, if orig_obj failed (e.g. Population lacking context) but 'val' is a valid array
                if len(q_vals) == 0 and isinstance(val, (np.ndarray, list)):
                     try: q_vals = q_closeness_points(val, **call_kwargs)
                     except: pass
                
                if len(q_vals) > 0:
                    # Audit Report Parity:
                    # Q >= 0.5: Solid Circle
                    # 0.0 <= Q < 0.5: Hollow Circle
                    # Q < 0.0: Diamond Open
                    symbols = []
                    for q in q_vals:
                        if q >= 0.5: symbols.append('circle')
                        elif q >= 0.0: symbols.append('circle-open')
                        else: symbols.append('diamond-open')
                    
                    if marker_styles[i] is None:
                        marker_styles[i] = {}
                    marker_styles[i]['symbol'] = symbols
                    # High precision sizing (Solid=6, Hollow=10, Diamond=9)
                    marker_styles[i]['size'] = []
                    for q in q_vals:
                        if q >= 0.5: marker_styles[i]['size'].append(6)
                        elif q >= 0.0: marker_styles[i]['size'].append(10)
                        else: marker_styles[i]['size'].append(9) # Diamond=9 matches Circle=10 visual weight
            except Exception as e:
                # Fail gracefully if metrology metadata is missing
                # print(f"[moeabench] Warning: Could not calculate quality markers for trace {i}: {e}")
                pass
        
    if objectives is None:
        dims = [d.shape[1] for d in processed_args if len(d.shape) > 1]
        max_dim = max(dims) if dims else 2
        if max_dim < 3: objectives = [0, 1]
        else: objectives = [0, 1, 2]
    
    if len(objectives) == 2:
        s = Scatter2D(names, processed_args, objectives, type=title, mode=mode, axis_label=axis_labels, trace_modes=trace_modes, marker_styles=marker_styles, gray_gt=gray_gt, **kwargs)
    else:
        for k in range(len(processed_args)):
             d = processed_args[k]
             if d.shape[1] < 3:
                  # Pad with zeros to allow 3D plotting of 2D data
                  new_d = np.zeros((d.shape[0], 3))
                  new_d[:, :d.shape[1]] = d
                  processed_args[k] = new_d
        while len(objectives) < 3: objectives.append(0)
        s = Scatter3D(names, processed_args, objectives, type=title, mode=mode, axis_label=axis_labels, trace_modes=trace_modes, marker_styles=marker_styles, gray_gt=gray_gt, **kwargs)
    
    if show:
        s.show()
    return s

def density(*args, axes=None, layout='grid', alpha=None, threshold=None, space='objs', title=None, show=True, ax=None, **kwargs):
    """
    [mb.view.density] Spatial Distribution Perspective.
    Plots smooth Probability Density Estimates via Kernel Density Estimation (KDE)
    """
    from scipy.stats import gaussian_kde
    from ..stats.tests import topo_distribution as stats_topo_distribution, DistMatchResult
    
    # 1. Resolve Data and Labels
    buffers = []
    names = []
    match_res = None

    if len(args) == 1 and isinstance(args[0], DistMatchResult):
        match_res = args[0]
        buffers = [np.asarray(b) for b in match_res.buffers]
        names = list(match_res.names)
        space = match_res.space
        if axes is None:
            axes = list(match_res.axes)
    
    if not buffers:
        for arg in args:
            if hasattr(arg, 'runs') and hasattr(arg, 'pop'):
                data = arg.front(**kwargs) if space == 'objs' else arg.set(**kwargs)
            else:
                data = arg
            buffers.append(np.asarray(data))
            names.append(getattr(arg, 'name', 'unnamed'))

    n_dims = buffers[0].shape[1]
    if axes is None:
        axes = list(range(min(n_dims, 4)))
    
    alpha = alpha if alpha is not None else defaults.alpha
    threshold = threshold if threshold is not None else defaults.displacement_threshold

    # 2. Statistical Analysis
    if match_res is None:
        match_res = stats_topo_distribution(*args, space=space, axes=axes, method=kwargs.get('method', 'ks'), 
                                            alpha=alpha, threshold=threshold, **kwargs)
    
    # 3. Plotting Logic
    # 3. Plotting Logic
    if ax is not None:
        # Single axis injection mode
        if len(axes) != 1:
            raise ValueError("When 'ax' is provided, 'axes' must specify exactly one dimension.")
        axes_objs = [ax]
        layout = 'external'
    elif layout == 'grid':
        n_plots = len(axes)
        cols = 2 if n_plots > 1 else 1
        rows = (n_plots + 1) // 2
        fig, ax_grid = plt.subplots(rows, cols, figsize=(10, 4*rows), constrained_layout=True)
        axes_objs = np.atleast_1d(ax_grid).flatten()
        for j in range(len(axes), len(axes_objs)):
            axes_objs[j].axis('off')
    else:
        figures = []
        axes_objs = [None] * len(axes) 

    for i, ax_idx in enumerate(axes):
        if layout == 'independent':
            cur_fig, cur_ax = plt.subplots(figsize=(6, 4))
            figures.append(cur_fig)
        elif layout == 'external':
            cur_ax = axes_objs[i]
            cur_fig = cur_ax.figure
        else:
            cur_ax = axes_objs[i]
            cur_fig = fig
            
        axis_name = "f" if space == 'objs' else "x"
        cur_ax.set_xlabel(f"{axis_name}${ax_idx + 1}$")
        cur_ax.set_ylabel("Density (KDE)")
        
        ax_stats = match_res.results.get(ax_idx)
        p_val = getattr(ax_stats, 'p_value', 1.0) if ax_stats else 1.0
        d_stat = getattr(ax_stats, 'statistic', 0.0) if hasattr(ax_stats, 'statistic') else 0.0
        verdict = "Match" if p_val > alpha else "Mismatch"
        
        for b_idx, buffer in enumerate(buffers):
            sample = buffer[:, ax_idx]
            color = f"C{b_idx % 10}"
            sample = sample[np.isfinite(sample)]
            if len(np.unique(sample)) > 1:
                kde = gaussian_kde(sample)
                x_range = np.linspace(np.min(sample), np.max(sample), 100)
                y_vals = kde(x_range)
                lbl = f"{names[b_idx]}"
                if b_idx == 0: 
                     lbl += f" [{verdict}, p={p_val:.3f}]"
                cur_ax.plot(x_range, y_vals, color=color, label=lbl, lw=2)
                cur_ax.fill_between(x_range, y_vals, color=color, alpha=0.2)
            else:
                val = sample[0]
                cur_ax.axvline(val, color=color, label=names[b_idx], lw=2)
        
        cur_ax.set_title(f"Spatial Density: {axis_name}{ax_idx + 1}")
        cur_ax.legend(fontsize=8, frameon=True, facecolor='white', framealpha=0.9)
        cur_ax.grid(True, alpha=0.2, linestyle='--')
        
    if title: 
        if layout == 'external':
            pass # Title managed externally usually, or set specifically on ax
        else:
            fig.suptitle(title, fontsize=14)
            
    if show and layout != 'external':
        show_matplotlib(fig, auto_close=(layout != 'independent'))
    return figures if layout == 'independent' else (ax if layout == 'external' else fig)

def bands(*args, levels=[0.1, 0.5, 0.9], objectives=None, mode='auto', title=None, style='step', **kwargs):
    """
    [mb.view.bands] Search Corridor Perspective.
    Visualizes reliability bands using Empirical Attainment Functions (EAF).
    """
    from ..stats.attainment import attainment
    
    surfaces = []
    canonical_surfaces = len(args) > 0 and all(isinstance(arg, AttainmentSurface) for arg in args)
    
    if style == 'fill':
        if len(levels) >= 3:
            sorted_levels = sorted(levels)
            levels = [sorted_levels[0], sorted_levels[len(sorted_levels)//2], sorted_levels[-1]]
        elif len(levels) == 2:
            levels = [levels[0], sum(levels)/2.0, levels[1]]
        else:
            style = 'spline' # Fallback if not enough bands
            
    for arg in args:
        if isinstance(arg, AttainmentSurface):
            surfaces.append(arg)
        else:
            if style == 'fill':
                # Order matters for band fill logic in Scatter2D: median, low, high
                lv_low, lv_med, lv_high = levels[0], levels[1], levels[2]
                
                surf_med = attainment(arg, level=lv_med)
                # Keep names simple to avoid legend explosion
                surf_med.name = f"{getattr(arg, 'name', 'Exp')} Mediana"
                
                surf_low = attainment(arg, level=lv_low)
                surf_low.name = f"{getattr(arg, 'name', 'Exp')} ({lv_low*100:.0f}%)"
                
                surf_high = attainment(arg, level=lv_high)
                surf_high.name = f"{getattr(arg, 'name', 'Exp')} ({lv_high*100:.0f}%)"
                
                surfaces.extend([surf_med, surf_low, surf_high])
            else:
                for lv in levels:
                    surf = attainment(arg, level=lv)
                    surf.name = f"{getattr(arg, 'name', 'Exp')} ({lv*100:.0f}%)"
                    surfaces.append(surf)
    
    if title is None:
        title = "Search Corridor (Attainment Bands)"

    if canonical_surfaces and style == 'fill':
        kwargs['band_fill'] = True
        kwargs['line_shape'] = 'spline'
    elif canonical_surfaces and style in ['spline', 'linear']:
        kwargs['line_shape'] = style
    
    if style in ['spline', 'linear']:
        kwargs['line_shape'] = style
    elif style == 'fill':
        kwargs['band_fill'] = True
        kwargs['line_shape'] = 'spline'
    
    return topology(*surfaces, objectives=objectives, mode=mode, title=title, **kwargs)

def gap(exp1, exp2=None, level=0.5, objectives=None, mode='auto', title=None, style='step', **kwargs):
    """
    [mb.view.gap] Topologic Gap Perspective.
    Visualizes the spatial difference region between two algorithms.
    """
    from ..stats.attainment import attainment_gap as stats_attainment_gap

    if hasattr(exp1, 'surf1') and hasattr(exp1, 'surf2'):
        diff = exp1
        exp1 = diff.surf1
        exp2 = diff.surf2
        level = getattr(diff, 'level', level)
    else:
        diff = stats_attainment_gap(exp1, exp2, level=level)
    
    # To visualize the gap, we plot the two attainment surfaces that form it
    surf1 = diff.surf1
    surf2 = diff.surf2
    surf1.name = f"{getattr(surf1, 'name', getattr(exp1, 'name', 'Exp1'))} ({level*100:.0f}%)"
    surf2.name = f"{getattr(surf2, 'name', getattr(exp2, 'name', 'Exp2'))} ({level*100:.0f}%)"
    
    if title is None: 
        title = f"Topologic Gap ({level*100:.0f}% Attainment)"
        
    if style in ['spline', 'linear']:
        kwargs['line_shape'] = style
        
    # TODO: In the future, specialized plotting for the AttainmentDiff object 
    # to highlight the area/volume between them.
    return topology(surf1, surf2, objectives=objectives, mode=mode, title=title, **kwargs)
