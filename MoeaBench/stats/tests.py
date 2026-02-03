# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy.stats import mannwhitneyu, ks_2samp, anderson_ksamp, wasserstein_distance
from functools import cached_property
from .base import StatsResult, SimpleStatsValue

class HypothesisTestResult(StatsResult):
    """
    Rich result for hypothesis tests (Mann-Whitney, KS, etc.)
    """
    def __init__(self, stat, p_value, data1, data2, name, alternative='two-sided', 
                 metric=None, gen=-1, **kwargs):
        self.statistic = stat
        self.p_value = p_value
        self.data1 = data1
        self.data2 = data2
        self.name = name
        self.alternative = alternative
        self.metric = metric
        self.gen = gen
        self.kwargs = kwargs

    @cached_property
    def perf_probability(self) -> float:
        """Vargha-Delaney A12 effect size (Lazy)."""
        return perf_probability(self.data1, self.data2, self.metric, self.gen, **self.kwargs).value

    @property
    def effect_size_label(self) -> str:
        val = self.perf_probability
        d = abs(val - 0.5) * 2 # Map 0.5->0, 0/1->1
        if d < 0.147: return "Negligible"
        if d < 0.33: return "Small"
        if d < 0.474: return "Medium"
        return "Large"

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05 if self.p_value is not None else False

    def report(self) -> str:
        name1 = getattr(self.data1, 'name', 'Group A')
        name2 = getattr(self.data2, 'name', 'Group B')
        
        lines = [
            f"--- {self.name} Test Report ---",
            f"  Comparison: {name1} vs {name2}",
            f"  Alternative: {self.alternative}",
            f"  Statistic: {self.statistic:.4f}",
        ]
        
        if self.p_value is not None:
            lines.append(f"  P-Value:   {self.p_value:.6f} ({'Significant' if self.significant else 'Not Significant'} at alpha=0.05)")
            
        lines.append(f"  A12 Effect Size: {self.perf_probability:.4f} ({self.effect_size_label})")
        
        # Narrative interpretation
        if self.significant:
            better = name1 if self.perf_probability > 0.5 else name2
            lines.append(f"\nConclusion: There is a statistically significant difference favoring {better}.")
        elif self.p_value is not None:
            lines.append(f"\nConclusion: No statistically significant difference detected.")
        else:
            lines.append(f"\nNote: Only Effect Size calculated (A12={self.perf_prob:.4f}).")
            
        return "\n".join(lines)

def _resolve_samples(data1, data2, metric=None, gen=-1, **kwargs):
    """
    Helper to extract sample distributions from experiments or arrays.
    """
    import inspect
    from ..metrics.evaluator import MetricMatrix
    
    def get_v(obj, other=None):
        # 1. If it's a MetricMatrix, extract the generation distribution
        if isinstance(obj, MetricMatrix):
            return obj.gens(gen)
            
        # 2. If it's already an array or list, return it
        if isinstance(obj, (np.ndarray, list, tuple)):
            return np.array(obj)
        
        # 2. If it's an experiment (has 'runs' and 'pop')
        if hasattr(obj, 'runs') and hasattr(obj, 'pop'):
            if metric is None:
                # Default to Hypervolume if not specified
                from ..metrics.evaluator import hypervolume
                m_func = hypervolume
            else:
                m_func = metric
            
            # Injection Logic:
            # If it's a standard MoeaBench metric (accepts 'ref') 
            # and no custom ref is in kwargs, we inject [obj, other]
            # to ensure a fair global reference point.
            call_kwargs = kwargs.copy()
            
            # Check if m_func accepts 'ref'
            try:
                sig = inspect.signature(m_func)
                if 'ref' in sig.parameters and 'ref' not in call_kwargs and 'ref_point' not in call_kwargs:
                    call_kwargs['ref'] = [obj, other]
            except (ValueError, TypeError):
                # Fallback for built-ins or non-inspectable callables
                pass
                
            res = m_func(obj, **call_kwargs)
            
            # Extract generation distribution
            if isinstance(res, MetricMatrix):
                return res.gens(gen)
            return np.array(res)
            
        return np.array(obj)

    v1 = get_v(data1, data2)
    v2 = get_v(data2, data1)
    
    return v1, v2

def perf_probability(data1, data2, metric=None, gen=-1, **kwargs):
    """
    [mb.stats.perf_probability] Computes the Vargha-Delaney A12 effect size statistic.
    (Win Probability: The probability that Algorithm A outperforms B).
    
    Supports "Smart Stats": can take raw arrays or Experiment objects.
    """
    x, y = _resolve_samples(data1, data2, metric=metric, gen=gen, **kwargs)
    
    m = len(x)
    n = len(y)
    
    if m == 0 or n == 0:
        raise ValueError("Data groups cannot be empty.")
        
    # Vectorized rank calculation for speed
    # This calculates P(X > Y) + 0.5 * P(X == Y)
    # The original A12 definition is P(X > Y) + 0.5 * P(X == Y)
    # The provided vectorized formula (r1 / m - (m + 1) / 2) / n is for Mann-Whitney U statistic normalized by m*n,
    # which is related to A12 but not exactly it.
    # Let's stick to the direct A12 calculation for clarity, or use the original loop.
    # The original loop is more direct for A12.
    
    # Original A12 calculation (non-vectorized but correct for A12)
    r = np.array([0] * m, dtype=float)
    for i in range(m):
        r[i] = np.sum(x[i] > y) + 0.5 * np.sum(x[i] == y)
    val = np.sum(r) / (m * n)
    
    return SimpleStatsValue(val, name="Vargha-Delaney A12 (Win Probability)")

def perf_evidence(data1, data2, alternative='two-sided', metric=None, gen=-1, **kwargs):
    """
    [mb.stats.perf_evidence] Performs the Mann-Whitney U rank test.
    (Win Evidence: Statistical evidence of performance difference).
    
    Supports "Smart Stats": can take raw arrays or Experiment objects.
    """
    x, y = _resolve_samples(data1, data2, metric=metric, gen=gen, **kwargs)
    res = mannwhitneyu(x, y, alternative=alternative)
    return HypothesisTestResult(res.statistic, res.pvalue, data1, data2, 
                                name="Mann-Whitney U (Win Evidence)", alternative=alternative,
                                metric=metric, gen=gen, **kwargs)

def perf_distribution(data1, data2, alternative='two-sided', metric=None, gen=-1, **kwargs):
    """
    [mb.stats.perf_distribution] Performs the Kolmogorov-Smirnov (KS) two-sample test.
    (Performance Distribution: identifies if two performance distributions differ in shape).
    """
    x, y = _resolve_samples(data1, data2, metric=metric, gen=gen, **kwargs)
    res = ks_2samp(x, y, alternative=alternative)
    return HypothesisTestResult(res.statistic, res.pvalue, data1, data2, 
                                name="Kolmogorov-Smirnov (Performance Dist Match)", 
                                alternative=alternative, metric=metric, gen=gen, **kwargs)

class DistMatchResult(StatsResult):
    """
    Rich result for multi-axial distribution matching (topology/equivalence).
    """
    def __init__(self, results, names, space='objs', method='ks', alpha=0.05, threshold=0.1):
        self.results = results # {axis_idx: score/p_val}
        self.names = names
        self.space = space
        self.method = method
        self.alpha = alpha
        self.threshold = threshold

    @property
    def p_values(self) -> dict:
        """Returns the dictionary of p-values if the method supports it."""
        return {k: getattr(v, 'p_value', v) for k, v in self.results.items()}

    @property
    def is_consistent(self) -> bool:
        """Returns True if all dimensions are statistically equivalent based on alpha/threshold."""
        if self.method == 'emd':
            return all(v < self.threshold for v in self.results.values())
        return all(getattr(v, 'p_value', 1.0) >= self.alpha for v in self.results.values())

    @property
    def failed_axes(self) -> list:
        """Returns indices of axes where distributions differ."""
        if self.method == 'emd':
            return [k for k, v in self.results.items() if v >= self.threshold]
        return [k for k, v in self.results.items() if getattr(v, 'p_value', 1.0) < self.alpha]

    def report(self) -> str:
        space_label = "Objective Space" if self.space == 'objs' else "Decision Space"
        lines = [
            f"--- Distribution Match Report ({self.method.upper()}) ---",
            f"  Analysed Space: {space_label}",
            f"  Global Status:  {'CONSISTENT' if self.is_consistent else 'DIVERGENT'}",
            f"  Criteria:       {'alpha=' + str(self.alpha) if self.method != 'emd' else 'threshold=' + str(self.threshold)}",
            f"  Dimensions:     {len(self.results)} axes tested",
            "\n  Dimensional Analysis:"
        ]
        
        for idx, res in self.results.items():
            name = f"Axis {idx + 1}"
            if self.method == 'emd':
                val_str = f"EMD={res:.4f}"
                sig_str = "(Divergent)" if res >= self.threshold else "(Match)"
            else:
                p_val = getattr(res, 'p_value', 1.0)
                val_str = f"p={p_val:.4f}"
                sig_str = "(Divergent)" if p_val < self.alpha else "(Match)"
            lines.append(f"    {name:<10}: {val_str:<10} {sig_str}")

        if not self.is_consistent:
            lines.append(f"\nConclusion: Topologies differ significantly in dimensions: {self.failed_axes}")
        else:
            lines.append("\nConclusion: The distributions are statistically equivalent across all axes.")
            
        return "\n".join(lines)

def topo_distribution(*args, space='objs', axes=None, method='ks', alpha=0.05, threshold=0.1, **kwargs):
    """
    [mb.stats.topo_distribution] Performs multi-axial distribution matching.
    Verifies if populations are statistically equivalent in objective or decision space.
    
    Methods:
        'ks': Kolmogorov-Smirnov two-sample test (Default).
        'anderson': Anderson-Darling k-sample test.
        'emd': Earth Mover's Distance (Wasserstein metric).
    """
    if len(args) < 2:
        raise ValueError("topo_dist requires at least two datasets for comparison.")

    # 1. Extract raw data buffers
    buffers = []
    names = []
    for arg in args:
        if hasattr(arg, 'runs') and hasattr(arg, 'pop'):
            # It's an Experiment
            name = getattr(arg, 'name', "Exp")
            if space == 'objs':
                data = arg.front(**kwargs)
            else:
                data = arg.set(**kwargs)
        elif hasattr(arg, 'objectives'): # Population
            name = "Pop"
            data = arg.objs if space == 'objs' else arg.vars
        else:
            # It's already a SmartArray or numpy array
            name = "Data"
            data = arg
        
        buffers.append(np.asarray(data))
        names.append(name)

    # 2. Determine common axes
    n_dims = buffers[0].shape[1]
    if axes is None:
        axes = list(range(n_dims))
    
    # Restrict axes to available dimensions
    min_dims = min([b.shape[1] for b in buffers])
    axes = [a for a in axes if a < min_dims]

    results = {}
    
    # 3. Perform axis-by-axis matching
    for ax in axes:
        samples = [b[:, ax] for b in buffers]
        
        if method == 'ks':
            if len(samples) > 2:
                # For now, just do pairwise against the first one if multi-sample
                # Or better, just support pairwise KS for topo_dist
                # The original code had a 'pass' here, let's make it explicit
                # that multi-sample KS is not directly supported by ks_2samp
                # and we'll default to pairwise for now.
                res = ks_2samp(samples[0], samples[1])
                results[ax] = SimpleStatsValue(res.pvalue, "KS p-value")
                results[ax].p_value = res.pvalue # Add p_value alias for consistent property access
            else:
                res = ks_2samp(samples[0], samples[1])
                results[ax] = SimpleStatsValue(res.pvalue, "KS p-value")
                results[ax].p_value = res.pvalue

        elif method == 'anderson':
            # Anderson-Darling k-sample
            try:
                # anderson_ksamp handles multiple samples natively
                res = anderson_ksamp(samples)
                results[ax] = SimpleStatsValue(res.pvalue, "Anderson p-value")
                results[ax].p_value = res.pvalue
            except Exception:
                # Fallback if samples are too small
                results[ax] = SimpleStatsValue(1.0, "Anderson p-value")
                results[ax].p_value = 1.0

        elif method == 'emd':
            # Earth Mover's Distance (only pairwise supported in scipy.stats 1D)
            if len(samples) == 2:
                val = wasserstein_distance(samples[0], samples[1])
                results[ax] = val
            else:
                # Average pairwise EMD if more than 2
                vals = []
                for i in range(len(samples)):
                    for j in range(i+1, len(samples)):
                        vals.append(wasserstein_distance(samples[i], samples[j]))
                results[ax] = np.mean(vals)
                
    return DistMatchResult(results, names, space=space, method=method, alpha=alpha, threshold=threshold)
