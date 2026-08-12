# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import logging
from typing import Optional, Union, Any, List
from .GEN_hypervolume import GEN_hypervolume
from .GEN_mc_hypervolume import GEN_mc_hypervolume
from .GEN_igd import GEN_igd
from .GEN_gd import GEN_gd
from .GEN_gdplus import GEN_gdplus
from .GEN_igdplus import GEN_igdplus
from ..core.base import Reportable
from ..core.run import Population
from ..defaults import defaults
from ..core.display import show_matplotlib
import warnings
from ..progress import get_progress_bar

def _compute_bounds(fronts):
    """Compute ideal/nadir bounds from exactly the supplied fronts."""
    valid = [np.asarray(front) for front in fronts if front is not None and len(front) > 0]
    if not valid:
        raise ValueError("Hypervolume requires at least one non-empty front to define normalization bounds.")
    if any(front.ndim != 2 for front in valid):
        raise ValueError("Hypervolume bounds require two-dimensional fronts.")
    if len({front.shape[1] for front in valid}) != 1:
        raise ValueError("Hypervolume bounds require matching objective counts.")

    merged = np.vstack(valid)
    ideal = np.min(merged, axis=0)
    nadir = np.max(merged, axis=0)
    if np.any(nadir <= ideal):
        raise ValueError(
            "Hypervolume normalization bounds require non-zero range in every objective."
        )
    return ideal, nadir


def _extract_reference_fronts(ref):
    """Extract external-reference fronts without consulting evaluated data."""
    refs = ref if isinstance(ref, list) else [ref]
    fronts = []
    for item in refs:
        _, item_fronts, _, _ = _extract_data(item)
        fronts.extend(item_fronts)
    return fronts


def _non_dominated_union(fronts):
    """Return the globally non-dominated union of non-empty fronts."""
    valid = [np.asarray(front) for front in fronts if front is not None and len(front) > 0]
    if not valid:
        return None

    merged = np.vstack(valid)
    is_dominated = np.zeros(merged.shape[0], dtype=bool)
    for i, current in enumerate(merged):
        if np.any(np.all(merged <= current, axis=1) & np.any(merged < current, axis=1)):
            is_dominated[i] = True
    return merged[~is_dominated]


def _hypervolume_diagnostics(reference_fronts, evaluated_fronts, ideal, nadir, raw_mat):
    """Describe external-reference geometry without modifying Hypervolume inputs."""
    ref_all = np.vstack([np.asarray(front) for front in reference_fronts if len(front) > 0])
    ref_nd = _non_dominated_union(reference_fronts)
    nd_ideal = np.min(ref_nd, axis=0)
    nd_nadir = np.max(ref_nd, axis=0)

    range_all = nadir - ideal
    range_nd = nd_nadir - nd_ideal
    range_inflation = np.divide(
        range_all,
        range_nd,
        out=np.full(range_all.shape, np.inf, dtype=float),
        where=range_nd > 0,
    )
    bounds_match = (
        np.all(np.isclose(ideal, nd_ideal, rtol=1e-12, atol=1e-15))
        and np.all(np.isclose(nadir, nd_nadir, rtol=1e-12, atol=1e-15))
    )

    outside_nbox = []
    outside_bbox = []
    better_than_ideal = []
    for front in evaluated_fronts:
        front = np.asarray(front)
        if len(front) == 0:
            outside_nbox.append(np.nan)
            outside_bbox.append(np.nan)
            better_than_ideal.append(np.nan)
            continue
        normalized = (front - ideal) / range_all
        outside_nbox.append(np.mean(np.any(normalized > 1.0, axis=1)))
        outside_bbox.append(np.mean(np.any(normalized > 1.1, axis=1)))
        better_than_ideal.append(np.mean(np.any(normalized < 0.0, axis=1)))

    outside_nbox = np.asarray(outside_nbox, dtype=float)
    outside_bbox = np.asarray(outside_bbox, dtype=float)
    better_than_ideal = np.asarray(better_than_ideal, dtype=float)
    all_outside = outside_bbox == 1.0
    nominal_bbox_volume = float(1.1 ** len(ideal))
    final_raw = raw_mat[-1, :] if raw_mat.size else np.full(len(evaluated_fronts), np.nan)

    return {
        "nbox_ideal": ideal.copy(),
        "nbox_nadir": nadir.copy(),
        "bbox_reference_point": ideal + 1.1 * range_all,
        "nominal_bbox_volume": nominal_bbox_volume,
        "reference_points": len(ref_all),
        "reference_nd_points": len(ref_nd),
        "reference_dominated_fraction": 1.0 - len(ref_nd) / len(ref_all),
        "nd_ideal": nd_ideal,
        "nd_nadir": nd_nadir,
        "range_inflation": range_inflation,
        "max_range_inflation": float(np.max(range_inflation)),
        "bbox_expanded_by_dominated_points": not bounds_match,
        "outside_nbox_fraction": outside_nbox,
        "outside_bbox_fraction": outside_bbox,
        "better_than_ideal_fraction": better_than_ideal,
        "all_points_outside_bbox": all_outside,
        "raw_hv_fraction_of_nominal_bbox": final_raw / nominal_bbox_volume,
    }


class MetricMatrix(Reportable):
    """
    A matrix (Generations x Runs) of metric values.
    """
    def __init__(self, data, metric_name="Metric", source_name=None,
                 reference_context=None, diagnostics=None):
        self._data = np.array(data) 
        
        # Internal Storage Policy: (Generations, Runs)
        # We ensure it's always 2D
        if self._data.ndim == 1:
             # Case: Array of generations for a single run
             self._data = self._data.reshape(-1, 1)
             
        self.metric_name = metric_name
        self.source_name = source_name
        self.reference_context = reference_context
        self.diagnostics = {} if diagnostics is None else diagnostics
        
        # Determine if this metric is scaled 
        self.is_ratio = any(label in metric_name for label in ("(Ratio)", "(Rel)", "(Relative)"))
        self.is_raw = "(Raw)" in metric_name
        self.is_abs = "(Abs)" in metric_name

    def report(self, show: bool = True, **kwargs) -> str:
        """Narrative report of the metric performance and stability."""
        use_md = kwargs.get('markdown', self._is_notebook())
        data = self._data
        if data.size == 0:
            if use_md:
                content = f"### Metric Report: {self.metric_name}\n**Status**: No data available"
            else:
                content = f"--- Metric Report: {self.metric_name} ---\n  Status: No data available"
            return self._render_report(content, show, **kwargs)

        # Distribution at the last generation
        final_dist = data[-1, :]
        valid_final = final_dist[np.isfinite(final_dist)]
        
        if len(valid_final) == 0:
            if use_md:
                content = f"### Metric Report: {self.metric_name}\n**Status**: All values are NaN"
            else:
                content = f"--- Metric Report: {self.metric_name} ---\n  Status: All values are NaN"
            return self._render_report(content, show, **kwargs)

        mean = np.mean(valid_final)
        has_run_variability = len(valid_final) >= 2
        std = np.std(valid_final) if has_run_variability else None
        best = np.max(valid_final) # Assuming higher is better (Hypervolume)
        if any(m in self.metric_name.lower() for m in ['igd', 'gd', 'spacing']):
            best = np.min(valid_final)

        prec = defaults.precision
        source_info = f" ({self.source_name})" if self.source_name else ""

        def _diag_array(key):
            return np.array2string(
                np.asarray(self.diagnostics[key]), precision=prec, separator=", "
            )

        if not has_run_variability:
            std_display = "N/A"
            stability = "Undetermined (requires at least 2 valid runs)"
        else:
            cv = std / (abs(mean) + 1e-9)
            std_display = f"{std:.{prec}f}"
            if cv < defaults.cv_tolerance:
                stability = f"High (CV={cv:.{prec}f} < {defaults.cv_tolerance})"
            elif cv > defaults.cv_moderate:
                stability = f"Low (CV={cv:.{prec}f} > {defaults.cv_moderate})"
            else:
                stability = f"Moderate ({defaults.cv_tolerance} <= CV={cv:.{prec}f} <= {defaults.cv_moderate})"

        if use_md:
            lines = [
                f"### Metric Report: {self.metric_name}{source_info}",
            ]
            
            if "Hypervolume" in self.metric_name:
                if self.reference_context == "external":
                    lines.extend([
                        "> **Fixed Reference Context**: Normalization bounds come only from the external `ref`.",
                        "> Values are comparable with other Hypervolume results evaluated against the same reference.",
                        ""
                    ])
                elif self.reference_context == "self":
                    lines.extend([
                        "> **Self-Referenced Hypervolume**: Normalization bounds come from the evaluated experiment.",
                        "> Results from separately normalized experiments are not directly comparable.",
                        ""
                    ])
                if self.is_ratio:
                    lines.extend([
                        "> **Competitive Efficiency**: What percentage of the best observed performance did this algorithm achieve?",
                        "> Values are scaled by the maximum session volume ($1.0$ ceiling).",
                        ""
                    ])
                elif self.is_raw:
                    lines.extend([
                        "> **Raw Hypervolume**: Dominated volume within the selected normalization context.",
                        ""
                    ])
                elif self.is_abs:
                    lines.extend([
                        "> **Theoretical Optimality**: How close is this algorithm to mathematical perfection?",
                        "> Values are normalized by the pre-calculated Ground Truth of the problem ($1.0$ = Opt).",
                        ""
                    ])

            lines.extend([
                "#### Final Performance (Last Gen)",
                f"- **Mean**: {mean:.{prec}f}",
                f"- **StdDev**: {std_display}",
                f"- **Best**: {best:.{prec}f}",
                "",
                "#### Search Dynamics",
                f"- **Runs**: {data.shape[1]}",
                f"- **Generations**: {data.shape[0]}",
                f"- **Stability**: {stability}"
            ])
            if "Hypervolume" in self.metric_name and self.reference_context == "external" and self.diagnostics:
                lines.extend([
                    "",
                    "#### Reference Geometry",
                    f"- **N-box ideal**: {_diag_array('nbox_ideal')}",
                    f"- **N-box nadir**: {_diag_array('nbox_nadir')}",
                    f"- **B-box reference point**: {_diag_array('bbox_reference_point')}",
                    f"- **Reference points**: {self.diagnostics['reference_points']}",
                    f"- **Global ND reference points**: {self.diagnostics['reference_nd_points']}",
                    f"- **Dominated reference fraction**: {self.diagnostics['reference_dominated_fraction']:.{prec}f}",
                    f"- **Range inflation**: {_diag_array('range_inflation')}",
                    f"- **HV/BBox (final)**: {_diag_array('raw_hv_fraction_of_nominal_bbox')}",
                    f"- **Outside nbox (final)**: {_diag_array('outside_nbox_fraction')}",
                    f"- **Outside bbox (final)**: {_diag_array('outside_bbox_fraction')}",
                ])
                if self.diagnostics["bbox_expanded_by_dominated_points"]:
                    lines.append("- **Warning**: globally dominated reference points expand the bbox.")
                if np.any(self.diagnostics["all_points_outside_bbox"]):
                    lines.append("- **Warning**: at least one run is floor-saturated (all final-front points lie beyond the bbox).")
            content = "\n".join(lines)
        else:
            lines = [
                f"--- Metric Report: {self.metric_name}{source_info} ---"
            ]
            
            if "Hypervolume" in self.metric_name:
                if self.reference_context == "external":
                    lines.append("  Reference Context: Fixed external reference; comparable only within this context.")
                elif self.reference_context == "self":
                    lines.append("  Reference Context: Self-referenced; separately normalized experiments are not directly comparable.")
                if self.is_ratio:
                    lines.extend([
                        "  Question: What is the competitive efficiency relative to best session performance?"
                    ])
                elif self.is_raw:
                    lines.extend([
                        "  Question: What volume is dominated within this normalization context?"
                    ])
            
            lines.extend([
                f"  Final Performance (Last Gen):",
                f"    - Mean: {mean:.{prec}f}",
                f"    - StdDev: {std_display}",
                f"    - Best: {best:.{prec}f}",
                f"  Search Dynamics:",
                f"    - Runs: {data.shape[1]}",
                f"    - Generations: {data.shape[0]}",
                f"    - Stability: {stability}"
            ])
            if "Hypervolume" in self.metric_name and self.reference_context == "external" and self.diagnostics:
                lines.extend([
                    "  Reference Geometry:",
                    f"    - N-box ideal: {_diag_array('nbox_ideal')}",
                    f"    - N-box nadir: {_diag_array('nbox_nadir')}",
                    f"    - B-box reference point: {_diag_array('bbox_reference_point')}",
                    f"    - Reference points: {self.diagnostics['reference_points']}",
                    f"    - Global ND reference points: {self.diagnostics['reference_nd_points']}",
                    f"    - Dominated reference fraction: {self.diagnostics['reference_dominated_fraction']:.{prec}f}",
                    f"    - Range inflation: {_diag_array('range_inflation')}",
                    f"    - HV/BBox (final): {_diag_array('raw_hv_fraction_of_nominal_bbox')}",
                    f"    - Outside nbox (final): {_diag_array('outside_nbox_fraction')}",
                    f"    - Outside bbox (final): {_diag_array('outside_bbox_fraction')}",
                ])
                if self.diagnostics["bbox_expanded_by_dominated_points"]:
                    lines.append("    - Warning: globally dominated reference points expand the bbox.")
                if np.any(self.diagnostics["all_points_outside_bbox"]):
                    lines.append("    - Warning: at least one run is floor-saturated (all final-front points lie beyond the bbox).")
            content = "\n".join(lines)
        
        return self._render_report(content, show, **kwargs)

    def __getitem__(self, key: Union[int, slice]) -> 'MetricMatrix':
        """
        Selectors: Consistent with Experiment indexing (by Run).
        Selects columns (Runs) from the Generations x Runs matrix.
        """
        # _data is (G, R), key slices R (axis 1)
        if isinstance(key, int):
            # Preserve 2D shape (G, 1) to remain a MetricMatrix object
            new_data = self._data[:, key:key+1]
        else:
            new_data = self._data[:, key]
            
        return MetricMatrix(
            new_data,
            self.metric_name,
            self.source_name,
            reference_context=self.reference_context,
            diagnostics=self.diagnostics,
        )

    def __len__(self):
        """Returns the number of runs (consistent with Experiment)."""
        return self._data.shape[1]

    def __repr__(self):
        if self._data.size == 1:
            return f"{self._data.item():.6f}"
        return super().__repr__()

    def __float__(self):
        if self._data.size == 1:
            return float(self._data.item())
        raise TypeError(f"MetricMatrix ({self.metric_name}) contains {self._data.size} values and cannot be converted to a single float.")

    def __format__(self, format_spec):
        if self._data.size == 1:
            return format(float(self._data.item()), format_spec)
        return format(str(self), format_spec)

    def __array__(self):
        return self._data
        
    def run(self, i=-1):
        """
        Returns the metric trajectory (all generations) for a specific run.
        defaults to the last run (-1).
        """
        if self._data.ndim == 1:
            return self._data
        return self._data[:, i]

    def gen(self, n=-1):
        """
        Returns the metric distribution (all runs) for a specific generation.
        Defaults to the last generation (-1).
        """
        return self._data[n, :]

    # Legacy Aliases
    def runs(self, idx=-1): return self.run(idx)
    def gens(self, idx=-1): return self.gen(idx)
        
    @property
    def values(self):
        """Returns the raw numpy array (Generations x Runs)."""
        return self._data


    @property
    def last(self):
        """Shortcut for the mean value of the final generation."""
        return self.mean()

    def mean(self, n=-1):
        """Returns the mean value of the metric at generation n."""
        dist = self.gen(n)
        return float(np.mean(dist[np.isfinite(dist)]))

    def std(self, n=-1):
        """Returns the standard deviation of the metric at generation n."""
        dist = self.gen(n)
        return float(np.std(dist[np.isfinite(dist)]))

    def best(self, n=-1):
        """Returns the best value of the metric at generation n (handles min/max logic)."""
        dist = self.gen(n)
        valid = dist[np.isfinite(dist)]
        if not len(valid): return np.nan
        
        if any(m in self.metric_name.lower() for m in ['igd', 'gd', 'spacing']):
            return float(np.min(valid))
        return float(np.max(valid))


def _extract_data(data, gens: Optional[Union[int, slice]] = None):
    """
    Refines input into 
    (List[RunHistories], List[FinalFronts], SourceName, NumRuns)
    """
    from ..core.run import Run, Population
    from ..core.experiment import experiment

    # If gens is int, treat as slice[:gens]
    if gens is not None and isinstance(gens, int):
        if gens == -1:
            gens = slice(-1, None)
        else:
            gens = slice(gens)

    if isinstance(data, experiment):
        histories = [r.history('nd') for r in data]
        if gens is not None:
             histories = [h[gens] for h in histories]
        return histories, [r.front() for r in data], data.name, len(data)
    
    if isinstance(data, Run):
        h = data.history('nd')
        if gens is not None: h = h[gens]
        return [h], [data.front()], data.name, 1
        
    if isinstance(data, Population):
        return [[data.objectives]], [data.objectives], data.label, 1
        
    if isinstance(data, np.ndarray):
        # Treat as a single population
        return [[data]], [data], "Array", 1

    # Fallback for generic iterables
    try:
        histories = []
        fronts = []
        for item in data:
            if hasattr(item, 'history'): # Run-like
                h = item.history('nd')
                if gens is not None:
                    h = h[gens]
                    if isinstance(h, np.ndarray) and h.ndim == 1: h = [h] # single pop case
                histories.append(h)
                fronts.append(item.front())
            else:
                histories.append([item])
                fronts.append(item)
        
        # Adjust histories if not Run-like but gens provided for top-level list
        if gens is not None and not hasattr(data, 'history'):
            histories = histories[gens]
            fronts = fronts[gens]

        return histories, fronts, None, len(histories)
    except:
        raise TypeError(f"Unsupported data type for metric calculation: {type(data)}")

def _metric_progress_steps(histories):
    return sum(len(h) for h in histories if h is not None)


def _metric_progress(metric_name, enabled, histories, source_name=None):
    if not enabled:
        return None
    total = _metric_progress_steps(histories)
    if total <= 0:
        return None
    desc = f"Computing {metric_name}"
    if source_name:
        desc = f"{desc} ({source_name})"
    return get_progress_bar(total=total, desc=desc, leave=True, style="percent")


def hypervolume(exp, ref=None, mode='auto', scale='raw', n_samples=100000, gens=None, progress=True):
    """
    Calculates Hypervolume for an experiment, run, or population.
    Returns a MetricMatrix (G x R).

    Args:
        exp: Experiment, Run, or Population object.
        ref: External reference whose fronts exclusively define normalization bounds.
             When omitted, bounds are derived collectively from all runs in `exp`.
        mode (str): Algorithm to use: 'auto' (default), 'exact', or 'fast'.
        scale (str): Scaling perspective: 'raw' (default), 'rel', or 'abs'.
        n_samples (int): Number of Monte Carlo samples for 'fast'/'auto' mode.
        gens (int or slice): Limit calculation to specific generation(s).
        progress (bool): Whether to display metric-computation progress.

    Points outside externally supplied ideal/nadir bounds do not expand those bounds.
    They are passed through in the fixed normalized coordinate system: points beyond
    the normalized reference point contribute no dominated volume, while points better
    than the ideal may contribute volume outside the nominal unit cube.
    """
    has_external_ref = ref is not None
    ref_items = [] if ref is None else (ref if isinstance(ref, list) else [ref])
    
    # --- 0. MOP Validation & Meta-data ---
    # Check if we are mixing different problems
    scale = str(scale).lower()
    from ..diagnostics.baselines import load_offline_baselines
    
    mop_names = []
    for item in [exp] + ref_items:
        mop_obj = getattr(item, 'mop', None)
        if mop_obj is None and hasattr(item, 'source'):
            mop_obj = getattr(item.source, 'mop', None)
            
        if mop_obj is not None:
            mop_names.append(getattr(mop_obj, 'name', mop_obj.__class__.__name__))
        elif hasattr(item, 'evaluation') and hasattr(item, 'pf'):
            # It's likely a MOP object
            mop_names.append(getattr(item, 'name', item.__class__.__name__))
            
    if len(set(mop_names)) > 1:
        msg = f"Hypervolume: Mixed MOPs detected in session: {list(set(mop_names))}. " \
              f"Comparing different problems yields invalid geometric results."
        if scale == 'abs':
            raise ValueError(msg)
        else:
            warnings.warn(msg)
    
    # 1. Collect all data
    F_GENs, Fs, name, n_runs = _extract_data(exp, gens=gens)

    # 2. Select one unambiguous normalization context. Evaluated data never
    # participates in externally referenced bounds.
    bounds_fronts = _extract_reference_fronts(ref) if has_external_ref else Fs
    min_val, max_val = _compute_bounds(bounds_fronts)
    M = len(min_val)
    
    # 3. Calculate
    max_gens = max(len(h) for h in F_GENs) if F_GENs else 0
    mat = np.full((max_gens, n_runs), np.nan)
    
    pbar = _metric_progress("Hypervolume", progress, F_GENs, source_name=name)
    try:
        for r_idx, (f_gen, f_last) in enumerate(zip(F_GENs, Fs)):
            run_M = f_last.shape[1] if len(f_last) > 0 else 0
            if run_M == 0 and len(f_gen) > 0:
                 for f in f_gen:
                      if len(f) > 0:
                           run_M = f.shape[1]
                           break

            if run_M and run_M != M:
                raise ValueError(
                    f"Hypervolume evaluated data has {run_M} objectives but the normalization context has {M}."
                )

            if run_M > 0:
                # Smart Selection of Algorithm
                use_mc = False
                if mode == 'fast':
                    use_mc = True
                elif mode == 'auto' and M > 6:
                    use_mc = True
                    logging.info(f"Hypervolume: High-dimensional space (M={M}) detected. "
                                 f"Switching to Monte Carlo approximation (n={n_samples}).")
                elif mode == 'exact' and M > 8:
                    warnings.warn(f"Exact Hypervolume calculation for M={M} objectives has exponential complexity O(2^M) "
                                  f"and may be extremely slow. Consider using mode='auto' or mode='fast'.")
                
                if use_mc:
                    metric = GEN_mc_hypervolume(f_gen, M, min_val, max_val, n_samples=n_samples)
                else:
                    metric = GEN_hypervolume(f_gen, M, min_val, max_val)
                    
                values = metric.evaluate()
                
                # Fill matrix
                length = min(len(values), max_gens)
                mat[:length, r_idx] = values[:length]
            if pbar is not None:
                pbar.update_to(pbar.current_val + len(f_gen))
    finally:
        if pbar is not None:
            pbar.close()

    # Preserve the unscaled result for diagnostics. Geometry diagnostics are
    # observational and never feed back into bounds or Hypervolume values.
    raw_mat = mat.copy()
    diagnostics = {}
    if has_external_ref:
        diagnostics = _hypervolume_diagnostics(
            bounds_fronts, Fs, min_val, max_val, raw_mat
        )
        if diagnostics["bbox_expanded_by_dominated_points"]:
            warnings.warn(
                "Hypervolume reference bbox is expanded by globally dominated reference points. "
                "This may compress differences between competing fronts; the bbox is not modified. "
                "Inspect result.diagnostics for details.",
                UserWarning,
            )
        if np.any(diagnostics["all_points_outside_bbox"]):
            warnings.warn(
                "Hypervolume floor saturation: all final-front points of at least one run lie "
                "beyond the reference bbox and therefore cannot contribute to Hypervolume. "
                "The bbox is not modified.",
                UserWarning,
            )

    # --- 4. Dynamic Benchmarking ---
    # We normalize HVs post-hoc according to the requested scale.
    # --- 4. Scale Post-Processing ---
    scale = str(scale).lower()
    if scale == 'rel':
        if has_external_ref:
            # A) Explicit Reference (e.g., Competition Mode)
            ref_hvs = []
            for r in ref_items:
                _, r_fs, _, _ = _extract_data(r)
                for f in r_fs:
                    if len(f) > 0:
                        if mode == 'fast' or (mode == 'auto' and M > 6):
                            m = GEN_mc_hypervolume([f], M, min_val, max_val, n_samples=n_samples)
                        else:
                            m = GEN_hypervolume([f], M, min_val, max_val)
                        ref_hvs.append(float(m.evaluate()[0]))
            
            ref_hv_val = np.max(ref_hvs) if ref_hvs else 0
        else:
            # B) Implicit Self-Reference (Best run in current data)
            ref_hv_val = np.nanmax(mat[-1, :]) if mat.size > 0 else 0

        if ref_hv_val > 0:
            mat /= ref_hv_val
            
        final_name = "Hypervolume (Relative)"
    
    elif scale == 'abs':
        # Retrieve Ground Truth from Calibration Registry
        mop_obj = getattr(exp, 'mop', None)
        if mop_obj is None and hasattr(exp, 'source'):
             mop_obj = getattr(exp.source, 'mop', None)
             
        if mop_obj is None:
             raise ValueError("Hypervolume 'abs' scale requires an experiment with an associated MOP.")
             
        mop_id = getattr(mop_obj, 'name', mop_obj.__class__.__name__)
        absolute_ok = False
        try:
            bases = load_offline_baselines()
            gt_registry = bases.get("_gt_registry", {})
            dim_key = f"{mop_id}__M{M}"
            gt_raw = gt_registry.get(dim_key, gt_registry.get(mop_id))
            if gt_raw is None:
                warnings.warn(
                    f"Hypervolume abs unavailable: '{mop_id}' is not calibrated for M={M}. "
                    "Falling back to raw scale.",
                    UserWarning
                )
            else:
                gt = np.array(gt_raw)
                if gt.ndim != 2 or gt.shape[1] != M:
                    warnings.warn(
                        f"Hypervolume abs unavailable: incompatible GT for '{mop_id}' (shape={gt.shape}, M={M}). "
                        "Falling back to raw scale.",
                        UserWarning
                    )
                else:
                    # Calculate reference HV (1.0 ceiling) using GT.
                    if mode == 'fast' or (mode == 'auto' and M > 6):
                        m = GEN_mc_hypervolume([gt], M, min_val, max_val, n_samples=n_samples)
                    else:
                        m = GEN_hypervolume([gt], M, min_val, max_val)

                    gt_hv = float(m.evaluate()[0])
                    if gt_hv > 0:
                        mat /= gt_hv
                        absolute_ok = True
                    else:
                        warnings.warn(
                            f"Hypervolume abs unavailable: non-positive GT HV for '{mop_id}'. "
                            "Falling back to raw scale.",
                            UserWarning
                        )
        except Exception as e:
            warnings.warn(
                f"Hypervolume abs failed with '{type(e).__name__}: {e}'. "
                "Falling back to raw scale.",
                UserWarning
            )

        final_name = "Hypervolume (Absolute)" if absolute_ok else "Hypervolume (Raw)"

    elif scale == 'raw':
        final_name = "Hypervolume (Raw)"
    else:
        raise ValueError(f"Unknown scale parameter: {scale}. Use 'raw', 'rel', or 'abs'.")

    reference_context = "external" if has_external_ref else "self"
    return MetricMatrix(
        mat,
        final_name,
        source_name=name,
        reference_context=reference_context,
        diagnostics=diagnostics,
    )

def get_reference_front(ref_exps, current_fronts):
    """
    Construct a reference Pareto front from external references only when supplied.
    """
    if ref_exps is None:
        ref_exps = []
    elif not isinstance(ref_exps, list):
        ref_exps = [ref_exps]

    all_fronts = []
    
    # Add external references
    for ref in ref_exps:
         _, fronts, _, _ = _extract_data(ref)
         all_fronts.extend(fronts)
    
    # If no refs provided, usage strategy:
    if not all_fronts and not ref_exps:
        all_fronts.extend(current_fronts)
        
    if not all_fronts:
        return None

    return _non_dominated_union(all_fronts)

def _calc_metric(exp, ref, MetricClass, name, gens=None, progress=True):
    if ref is None: ref = []
    if not isinstance(ref, list): ref = [ref]
    
    F_GENs, Fs, source_name, n_runs = _extract_data(exp, gens=gens)

    # Helper for GD/IGD reference front
    ref_front = get_reference_front(ref, Fs)
    
    max_gens = max(len(h) for h in F_GENs) if F_GENs else 0
    mat = np.full((max_gens, n_runs), np.nan)
    
    pbar = _metric_progress(name, progress, F_GENs, source_name=source_name)
    try:
        for r_idx, (f_gen, f_last) in enumerate(zip(F_GENs, Fs)):
            if ref_front is None:
                values = np.full(len(f_gen), np.nan)
            else:
                metric = MetricClass(f_gen, ref_front)
                values = metric.evaluate()
            
            length = min(len(values), max_gens)
            mat[:length, r_idx] = values[:length]
            if pbar is not None:
                pbar.update_to(pbar.current_val + len(f_gen))
    finally:
        if pbar is not None:
            pbar.close()
        
    return MetricMatrix(mat, name, source_name=source_name)

def gd(exp, ref=None, gens=None, progress=True):
    """Generational Distance; `ref` is the external reference front."""
    if ref is None:
        try:
            if hasattr(exp, 'optimal_front'):
                ref = exp.optimal_front()
            elif hasattr(exp, 'source') and hasattr(exp.source, 'optimal_front'):
                ref = exp.source.optimal_front()
        except (AttributeError, NotImplementedError):
            logging.warning(f"GD: Reference front not provided and MOP does not implement 'ps()'. Falling back to found front.")
            pass
    
    # Check if single population (ndarray)
    if isinstance(exp, np.ndarray):
        if ref is None: return MetricMatrix(np.array([np.nan]))
        from .GEN_gd import GEN_gd
        # GEN_gd expects (Hist, Ref)
        metric = GEN_gd([exp], ref)
        return MetricMatrix(metric.evaluate(), "GD")

    from .GEN_gd import GEN_gd
    return _calc_metric(exp, ref, GEN_gd, "GD", gens=gens, progress=progress)

def gdplus(exp, ref=None, gens=None, progress=True):
    """GD+; `ref` is the external reference front."""
    if ref is None:
        try:
            if hasattr(exp, 'optimal_front'):
                ref = exp.optimal_front()
            elif hasattr(exp, 'source') and hasattr(exp.source, 'optimal_front'):
                ref = exp.source.optimal_front()
        except (AttributeError, NotImplementedError):
            logging.warning(f"GD+: Reference front not provided and MOP does not implement 'ps()'. Falling back to found front.")
            pass
    
    if isinstance(exp, np.ndarray):
        if ref is None: return MetricMatrix(np.array([np.nan]))
        from .GEN_gdplus import GEN_gdplus
        metric = GEN_gdplus([exp], ref)
        return MetricMatrix(metric.evaluate(), "GD+")

    from .GEN_gdplus import GEN_gdplus
    return _calc_metric(exp, ref, GEN_gdplus, "GD+", gens=gens, progress=progress)

def igd(exp, ref=None, gens=None, progress=True):
    """Inverted Generational Distance; `ref` is the external reference front."""
    if ref is None:
        try:
            if hasattr(exp, 'optimal_front'):
                ref = exp.optimal_front()
            elif hasattr(exp, 'source') and hasattr(exp.source, 'optimal_front'):
                ref = exp.source.optimal_front()
        except (AttributeError, NotImplementedError):
            logging.warning(f"IGD: Reference front not provided and MOP does not implement 'ps()'. Falling back to found front.")
            pass
    
    if isinstance(exp, np.ndarray):
        if ref is None: return MetricMatrix(np.array([np.nan]))
        from .GEN_igd import GEN_igd
        metric = GEN_igd([exp], ref)
        return MetricMatrix(metric.evaluate(), "IGD")

    from .GEN_igd import GEN_igd
    return _calc_metric(exp, ref, GEN_igd, "IGD", gens=gens, progress=progress)

def emd(exp, ref=None, gens=None, progress=True):
    """
    Computes Earth Mover's Distance between a population and external `ref`.
    For multivariate data, this implementation uses the average 1D Wasserstein distance 
    per objective as a fast and robust distributional shift proxy.
    """
    if ref is None:
        try:
            if hasattr(exp, 'optimal_front'):
                ref = exp.optimal_front()
            elif hasattr(exp, 'source') and hasattr(exp.source, 'optimal_front'):
                ref = exp.source.optimal_front()
        except:
            pass
    
    from scipy.stats import wasserstein_distance
    
    def _calc_emd_pair(pts, r_pts):
        if pts is None or r_pts is None or len(pts) == 0 or len(r_pts) == 0:
            return np.nan
        M = pts.shape[1]
        w_dists = []
        for m in range(M):
            d = wasserstein_distance(pts[:, m], r_pts[:, m])
            w_dists.append(d)
        
        # DEBUG
        # if np.mean(w_dists) > 0.01:
        #    print(f"DEBUG EMD: M={M}, Dists={w_dists}")
        #    print(f"PTS Shape: {pts.shape}, REF Shape: {r_pts.shape}")
        return np.mean(w_dists)

    F_GENs, Fs, source_name, n_runs = _extract_data(exp, gens=gens)
    ref_front = get_reference_front(ref, Fs)
    
    if ref_front is None:
        return MetricMatrix(np.full((1, n_runs), np.nan), "EMD")

    max_gens = max(len(h) for h in F_GENs) if F_GENs else 0
    mat = np.full((max_gens, n_runs), np.nan)
    
    pbar = _metric_progress("EMD", progress, F_GENs, source_name=source_name)
    try:
        for r_idx, f_gen in enumerate(F_GENs):
            values = []
            for g_pop in f_gen:
                values.append(_calc_emd_pair(g_pop, ref_front))
            
            length = min(len(values), max_gens)
            mat[:length, r_idx] = values[:length]
            if pbar is not None:
                pbar.update_to(pbar.current_val + len(f_gen))
    finally:
        if pbar is not None:
            pbar.close()
        
    return MetricMatrix(mat, "EMD", source_name=source_name)

def igdplus(exp, ref=None, gens=None, progress=True):
    """IGD+; `ref` is the external reference front."""
    if ref is None:
        try:
            if hasattr(exp, 'optimal_front'):
                ref = exp.optimal_front()
            elif hasattr(exp, 'source') and hasattr(exp.source, 'optimal_front'):
                ref = exp.source.optimal_front()
        except (AttributeError, NotImplementedError):
            logging.warning(f"IGD+: Reference front not provided and MOP does not implement 'ps()'. Falling back to found front.")
            pass
    
    if isinstance(exp, np.ndarray):
        if ref is None: return MetricMatrix(np.array([np.nan]))
        from .GEN_igdplus import GEN_igdplus
        metric = GEN_igdplus([exp], ref)
        return MetricMatrix(metric.evaluate(), "IGD+")

    from .GEN_igdplus import GEN_igdplus
    return _calc_metric(exp, ref, GEN_igdplus, "IGD+", gens=gens, progress=progress)

def plot_matrix(metric_matrices, mode='auto', show_bounds=False, title=None, **kwargs):
    """
    Plots a list of MetricMatrix objects.
    mode: 'auto' (detects environment), 'interactive' (Plotly) or 'static' (Matplotlib)
    """
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
            
    if not isinstance(metric_matrices, (list, tuple)):
        metric_matrices = [metric_matrices]

    # Handle nested tuples/lists from wrappers like timeplot(*args)
    if len(metric_matrices) == 1 and isinstance(metric_matrices[0], (list, tuple)):
        metric_matrices = metric_matrices[0]

    # Determine common name
    names = sorted(list(set(m.metric_name for m in metric_matrices)))
    if len(names) == 1:
        plot_name = names[0]
    else:
        plot_name = ", ".join(names)

    final_title = title if title else f"{plot_name} over Time"

    if mode == 'static':
        import matplotlib.pyplot as plt
        
        ax = kwargs.get('ax', None)
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        else:
            fig = ax.get_figure()
        
        lstyles = kwargs.get('linestyles', ['-', '--', ':', '-.'])
        if not isinstance(lstyles, (list, tuple)): lstyles = [lstyles]

        labels = kwargs.get('labels', [])
        for i, mat in enumerate(metric_matrices):
             data = mat.values
             
             if i < len(labels):
                 label = labels[i]
             else:
                 # Standard Legend Logic: Name (G: XX, R: YY)
                 name = mat.source_name if mat.source_name else mat.metric_name
                 G, R = data.shape
                 meta = []
                 
                 # Rule: If G=1 (snapshot), specify G. If history (G>1), omit G.
                 if G == 1:
                     meta.append(f"G: 1")
                 
                 # Rule: If R=1 (single run), specify R. If multiple (aggregated), R is implied 
                 # unless it's a specific subset (not detectable here yet, so we omit if R > 1).
                 if R == 1:
                     meta.append(f"R: 1")
                     
                 suffix = f" ({', '.join(meta)})" if meta else ""
                 label = f"{name}{suffix}"
             
             ls = lstyles[i % len(lstyles)]
             
             if data.shape[1] > 1:
                mean = np.nanmean(data, axis=1)
                std = np.nanstd(data, axis=1)
                gens = np.arange(1, len(mean) + 1)
                v_min = np.nanmin(data, axis=1)
                v_max = np.nanmax(data, axis=1)
                
                ax.plot(gens, mean, label=label, linestyle=ls)
                ax.fill_between(gens, np.maximum(0, mean-std), mean+std, alpha=0.2)
                
                if show_bounds:
                    ax.plot(gens, v_min, '--', color=ax.get_lines()[-1].get_color(), alpha=0.5, linewidth=1)
                    ax.plot(gens, v_max, '--', color=ax.get_lines()[-1].get_color(), alpha=0.5, linewidth=1)
             else:
                ax.plot(np.arange(1, len(data)+1), data[:, 0], label=label, linestyle=ls)
        
        ax.set_title(final_title)
        ax.set_xlabel("Generation")
        ax.set_ylabel(plot_name)
        ax.legend()
        if kwargs.get('show', True) and kwargs.get('ax') is None:
            show_matplotlib(fig, auto_close=True)
        
        return fig, ax
        
    else:
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        for mat in metric_matrices:
            data = mat.values
            label = f"{mat.metric_name} ({mat.source_name})" if mat.source_name else mat.metric_name
            
            if data.shape[1] > 1:
                mean = np.nanmean(data, axis=1)
                std = np.nanstd(data, axis=1)
                gens = np.arange(1, len(mean) + 1)
                v_min = np.nanmin(data, axis=1)
                v_max = np.nanmax(data, axis=1)
                
                fig.add_trace(go.Scatter(
                    x=gens, y=mean,
                    mode='lines',
                    name=label,
                    line=dict(width=3),
                    hovertemplate=f"{label}<br>Gen: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>"
                ))
                
                lower_bound = np.maximum(0, mean - std)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([gens, gens[::-1]]),
                    y=np.concatenate([mean + std, lower_bound[::-1]]),
                    fill='toself',
                    fillcolor='rgba(100, 100, 100, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f'{label} StdDev'
                ))
                
                if show_bounds:
                    fig.add_trace(go.Scatter(
                        x=gens, y=v_min,
                        mode='lines',
                        line=dict(dash='dash', width=1),
                        name=f'{label} Min',
                        showlegend=False,
                        opacity=0.5,
                        hovertemplate=f"{label} Min<br>Gen: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>"
                    ))
                    fig.add_trace(go.Scatter(
                        x=gens, y=v_max,
                        mode='lines',
                        line=dict(dash='dash', width=1),
                        name=f'{label} Max',
                        showlegend=False,
                        opacity=0.5,
                        hovertemplate=f"{label} Max<br>Gen: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>"
                    ))
                
            else:
                fig.add_trace(go.Scatter(
                    x=np.arange(1, len(data)+1),
                    y=data[:, 0],
                    mode='lines',
                    name=label,
                    hovertemplate=f"{label}<br>Gen: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>"
                ))
                
        fig.update_layout(
            title=final_title, 
            xaxis_title="Generation", 
            yaxis_title=plot_name,
            hovermode='closest'
        )
        if kwargs.get('show', True):
            fig.show()

def front_ratio(exp, mode='run', gens=None):
    """
    Calculates the proportion of non-dominated individuals (Front Ratio)
    relative to the total population size at each generation.
    Returns a MetricMatrix (G x R) or (G x 1) if mode='consensus'.

    Args:
        exp: Experiment, Run, or Population object.
        mode (str): 'run' (per-run ratio) or 'consensus' (superfront density).
        gens (int or slice): Limit calculation to specific generation(s).
    """
    # 1. Data extraction logic
    from ..core.run import Run, Population
    from ..core.experiment import experiment
    from ..stats.stratification import _layer
    
    if isinstance(exp, experiment):
        histories = [r.history('f') for r in exp._runs]
        name = exp.name
    elif isinstance(exp, Run):
        histories = [exp.history('f')]
        name = exp.name
    elif isinstance(exp, Population):
        histories = [[exp.objectives]]
        name = exp.label
    elif isinstance(exp, np.ndarray):
        histories = [[exp]]
        name = "Array"
    else:
        from .evaluator import _extract_data
        histories, _, name, _ = _extract_data(exp, gens=gens)

    # 2. Slice generations if requested
    if gens is not None and not isinstance(exp, Population):
        g_slice = slice(-1, None) if gens == -1 else (gens if isinstance(gens, slice) else slice(gens))
        histories = [h[g_slice] for h in histories]

    n_runs = len(histories)
    max_gens = max(len(h) for h in histories) if histories else 0
    mode = str(mode).lower()
    
    if mode == 'consensus':
        # AGGREGATE MODE: (G x 1) matrix
        mat = np.full((max_gens, 1), np.nan)
        for g_idx in range(max_gens):
            pops_at_g = []
            for r_h in histories:
                if g_idx < len(r_h) and r_h[g_idx] is not None:
                    pops_at_g.append(r_h[g_idx])
            
            if not pops_at_g:
                continue
                
            combined_objs = np.vstack(pops_at_g)
            try:
                # Calculate non-dominance ratio on the combined cloud
                s_res = _layer(Population(combined_objs))
                n_nd = np.sum(s_res.rank_array == 1)
                mat[g_idx, 0] = n_nd / len(combined_objs)
            except Exception:
                mat[g_idx, 0] = 1.0 # Fallback
                
        label = "Front Ratio (Consensus)"
    else:
        # PER-RUN MODE: (G x R) matrix
        mat = np.full((max_gens, n_runs), np.nan)
        for r_idx, h_run in enumerate(histories):
            for g_idx, pop_data in enumerate(h_run):
                if pop_data is None or len(pop_data) == 0:
                    mat[g_idx, r_idx] = 0.0
                    continue
                
                try:
                    active_pop = Population(pop_data) if isinstance(pop_data, np.ndarray) else pop_data
                    s_res = _layer(active_pop)
                    n_nd = np.sum(s_res.rank_array == 1)
                    n_tot = len(pop_data)
                    mat[g_idx, r_idx] = n_nd / n_tot
                except Exception:
                    mat[g_idx, r_idx] = 1.0
        
        label = "Front Ratio (Run)"

    return MetricMatrix(mat, metric_name=label, source_name=name)
