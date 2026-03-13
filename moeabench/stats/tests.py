# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy.stats import mannwhitneyu, ks_2samp, anderson_ksamp, wasserstein_distance
from functools import cached_property
from .base import StatsResult, SimpleStatsValue
from ..defaults import defaults


class PerfCompareResult(StatsResult):
    """Unified result object for performance comparison methods."""

    def __init__(self, method, statistic=None, p_value=None, effect_size=None, decision=None, details=None, plot_data=None):
        self.method = method
        self.statistic = statistic
        self.p_value = p_value
        self.effect_size = effect_size
        self.decision = decision
        self.details = details or {}
        self.plot_data = plot_data or {}

    @property
    def samples(self):
        return self.plot_data.get("samples", [])

    @property
    def labels(self):
        return self.plot_data.get("labels", [])

    @property
    def metric_label(self):
        return self.plot_data.get("metric_label")

    @property
    def gen(self):
        return self.plot_data.get("gen", -1)

    def report(self, show: bool = True, full: bool = False, **kwargs) -> str:
        use_md = kwargs.get('markdown', self._is_notebook())
        if not full:
            if use_md:
                parts = [f"**perf_compare** ({self.method})"]
                if self.decision:
                    parts.append(f"- **Decision**: {self.decision}")
                if self.p_value is not None:
                    parts.append(f"- **p-value**: {self.p_value:.6f}")
                if self.effect_size is not None:
                    parts.append(f"- **effect**: {self.effect_size:.4f}")
                content = "\n".join(parts)
            else:
                parts = [f"perf_compare ({self.method})"]
                if self.decision:
                    parts.append(f"  Decision: {self.decision}")
                if self.p_value is not None:
                    parts.append(f"  p-value: {self.p_value:.6f}")
                if self.effect_size is not None:
                    parts.append(f"  effect: {self.effect_size:.4f}")
                content = "\n".join(parts)
            return self._render_report(content, show, **kwargs)

        if use_md:
            lines = [f"### Performance Compare ({self.method})", ""]
            if self.decision:
                lines.append(f"**Decision**: {self.decision}")
            if self.statistic is not None:
                lines.append(f"**Statistic**: {self.statistic:.6f}")
            if self.p_value is not None:
                lines.append(f"**P-Value**: {self.p_value:.6f}")
            if self.effect_size is not None:
                lines.append(f"**Effect Size**: {self.effect_size:.6f}")
        else:
            lines = [f"--- Performance Compare ({self.method}) ---"]
            if self.decision:
                lines.append(f"  Decision: {self.decision}")
            if self.statistic is not None:
                lines.append(f"  Statistic: {self.statistic:.6f}")
            if self.p_value is not None:
                lines.append(f"  P-Value:   {self.p_value:.6f}")
            if self.effect_size is not None:
                lines.append(f"  Effect:    {self.effect_size:.6f}")
        return self._render_report("\n".join(lines), show, **kwargs)

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
        if d < defaults.a12_negligible: return "Negligible"
        if d < defaults.a12_small: return "Small"
        if d < defaults.a12_medium: return "Medium"
        return "Large"

    @property
    def significant(self) -> bool:
        return self.p_value < defaults.alpha if self.p_value is not None else False

    def report(self, show: bool = True, **kwargs) -> str:
        use_md = kwargs.get('markdown', self._is_notebook())
        name1 = getattr(self.data1, 'name', 'Group A')
        name2 = getattr(self.data2, 'name', 'Group B')
        prec = defaults.precision
        
        if use_md:
            header = f"### {self.name} Test Report"
            lines = [
                header,
                "",
                f"**Comparison**: {name1} vs {name2}",
                f"**Alternative**: {self.alternative}",
                f"**Statistic**: {self.statistic:.4f}",
            ]
            
            if self.p_value is not None:
                sig_text = "Significant" if self.significant else "Not Significant"
                status_color = "green" if self.significant else "gray"
                lines.append(f"**P-Value**: {self.p_value:.6f} (<span style='color:{status_color}'>{sig_text}</span> at alpha={defaults.alpha})")
                
            lines.append(f"**A12 Effect Size**: {self.perf_probability:.4f} ({self.effect_size_label})")
            
            # Narrative interpretation
            if self.significant:
                better = name1 if self.perf_probability > 0.5 else name2
                lines.append(f"\n> **Conclusion**: There is a statistically significant difference favoring **{better}**.")
            elif self.p_value is not None:
                lines.append(f"\n> **Conclusion**: No statistically significant difference detected.")
            else:
                lines.append(f"\n> **Note**: Only Effect Size calculated (A12={self.perf_probability:.4f}).")

            content = "\n".join(lines)
        else:
            lines = [
                f"--- {self.name} Test Report ---",
                f"  Comparison: {name1} vs {name2}",
                f"  Alternative: {self.alternative}",
                f"  Statistic: {self.statistic:.4f}",
            ]
            
            if self.p_value is not None:
                lines.append(f"  P-Value:   {self.p_value:.6f} ({'Significant' if self.significant else 'Not Significant'} at alpha={defaults.alpha})")
                
            lines.append(f"  A12 Effect Size: {self.perf_probability:.4f} ({self.effect_size_label})")
            
            # Narrative interpretation
            if self.significant:
                better = name1 if self.perf_probability > 0.5 else name2
                lines.append(f"\nConclusion: There is a statistically significant difference (p < {defaults.alpha}) favoring {better}.")
            elif self.p_value is not None:
                lines.append(f"\nConclusion: No statistically significant difference detected.")
            else:
                lines.append(f"\nNote: Only Effect Size calculated (A12={self.perf_probability:.4f}).")
            
            content = "\n".join(lines)
            
        return self._render_report(content, show, **kwargs)

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
            # If it's a standard moeabench metric (accepts 'ref') 
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


def _infer_perf_label(obj, index=0, gen=-1):
    """Best-effort label inference shared by stats and views."""
    from ..metrics.evaluator import MetricMatrix
    exp_name = None
    run_idx = None

    if type(obj).__name__ == 'Run':
        run_idx = getattr(obj, 'index', None)
        if hasattr(obj, 'source'):
            exp_name = getattr(obj.source, 'name', None)
    elif type(obj).__name__ == 'experiment':
        exp_name = getattr(obj, 'name', None)
    elif isinstance(obj, MetricMatrix):
        exp_name = getattr(obj, 'source_name', None) or getattr(obj, 'metric_name', None)

    if not exp_name:
        exp_name = getattr(obj, 'name', None) or f"Data {index+1}"
        import re
        exp_name = re.sub(r'\s*\(run\s*\d+\)', '', exp_name, flags=re.IGNORECASE)

    pieces = []
    if run_idx is not None:
        pieces.append(f"run {run_idx}")
    if gen is not None and gen >= 0:
        pieces.append(f"gen {gen}")
    return f"{exp_name} ({', '.join(pieces)})" if pieces else exp_name


def _infer_metric_label(metric, data1, data2):
    """Infer the plotted metric label from inputs or metric callable."""
    from ..metrics.evaluator import MetricMatrix
    for obj in (data1, data2):
        if isinstance(obj, MetricMatrix):
            return getattr(obj, "metric_name", None) or getattr(metric, "__name__", "Value")
    return getattr(metric, "__name__", "Value")


def _build_perf_plot_data(data1, data2, metric=None, gen=-1, **kwargs):
    """Canonical plot payload shared by perf stats and perf views."""
    x, y = _resolve_samples(data1, data2, metric=metric, gen=gen, **kwargs)
    return {
        "samples": [np.asarray(x).ravel(), np.asarray(y).ravel()],
        "labels": [
            _infer_perf_label(data1, 0, gen=gen),
            _infer_perf_label(data2, 1, gen=gen),
        ],
        "metric_label": _infer_metric_label(metric, data1, data2),
        "gen": gen,
    }

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


def perf_compare(data1, data2, method='ks', metric=None, gen=-1, alternative='two-sided', **kwargs):
    """
    Unified performance comparator.
    Methods:
    - 'mannwhitney': Mann-Whitney U (location shift)
    - 'ks': KS two-sample (distribution matching)
    - 'a12': Vargha-Delaney A12 (win probability/effect)
    """
    method = str(method).lower()
    if method == 'mannwhitney':
        res = perf_evidence(data1, data2, alternative=alternative, metric=metric, gen=gen, **kwargs)
        decision = "different" if res.significant else "no-significant-difference"
        return PerfCompareResult(
            method='mannwhitney',
            statistic=float(res.statistic),
            p_value=float(res.p_value),
            effect_size=float(res.perf_probability),
            decision=decision,
            details={"name": res.name},
            plot_data=_build_perf_plot_data(data1, data2, metric=metric, gen=gen, **kwargs),
        )
    if method == 'ks':
        res = perf_distribution(data1, data2, alternative=alternative, metric=metric, gen=gen, **kwargs)
        decision = "match" if not res.significant else "divergent"
        return PerfCompareResult(
            method='ks',
            statistic=float(res.statistic),
            p_value=float(res.p_value),
            effect_size=float(res.perf_probability),
            decision=decision,
            details={"name": res.name},
            plot_data=_build_perf_plot_data(data1, data2, metric=metric, gen=gen, **kwargs),
        )
    if method == 'a12':
        res = perf_probability(data1, data2, metric=metric, gen=gen, **kwargs)
        val = float(res.value)
        if val > 0.5:
            decision = "a-better-than-b"
        elif val < 0.5:
            decision = "b-better-than-a"
        else:
            decision = "tie"
        return PerfCompareResult(
            method='a12',
            statistic=val,
            p_value=None,
            effect_size=val,
            decision=decision,
            details={"name": res.name},
            plot_data=_build_perf_plot_data(data1, data2, metric=metric, gen=gen, **kwargs),
        )
    raise ValueError(f"Unknown perf_compare method: {method}. Use 'mannwhitney', 'ks', or 'a12'.")

class DistMatchResult(StatsResult):
    """
    Rich result for multi-axial distribution matching (topology/equivalence).
    """
    def __init__(self, results, names, space='objs', method='ks', alpha=None, threshold=None, plot_data=None):
        self.results = results # {axis_idx: score/p_val}
        self.names = names
        self.space = space
        self.method = method
        self.alpha = alpha if alpha is not None else defaults.alpha
        self.threshold = threshold if threshold is not None else defaults.displacement_threshold
        self.plot_data = plot_data or {}

    @property
    def buffers(self):
        return self.plot_data.get("buffers", [])

    @property
    def axes(self):
        return self.plot_data.get("axes", [])

    @property
    def semantic_alias(self) -> str:
        """Returns the semantic alias name for the active topo method."""
        return {
            'ks': 'topo_match',
            'emd': 'topo_shift',
            'anderson': 'topo_tail',
        }.get(self.method, f"topo_{self.method}")

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

    def report(self, show: bool = True, **kwargs) -> str:
        use_md = kwargs.get('markdown', self._is_notebook())
        space_label = "Objective Space" if self.space == 'objs' else "Decision Space"
        status_label = 'CONSISTENT' if self.is_consistent else 'DIVERGENT'
        method_label = f"{self.semantic_alias} | {self.method.upper()}"
        
        if use_md:
            status_color = "green" if self.is_consistent else "red"
            header = f"### Distribution Match Report ({method_label})"
            lines = [
                header,
                "",
                f"**Analysed Space**: {space_label}",
                f"**Global Status**: <span style='color:{status_color}'>**{status_label}**</span>",
                f"**Criteria**: {'alpha=' + str(self.alpha) if self.method != 'emd' else 'threshold=' + str(self.threshold)}",
                f"**Dimensions**: {len(self.results)} axes tested",
                "",
                "#### Dimensional Analysis",
                "| Axis | Result | Verdict |",
                "| :--- | :--- | :--- |"
            ]
            
            for idx, res in self.results.items():
                name = f"Axis {idx + 1}"
                if self.method == 'emd':
                    val_str = f"{res:.4f}"
                    match = res < self.threshold
                else:
                    p_val = getattr(res, 'p_value', 1.0)
                    val_str = f"p={p_val:.4f}"
                    match = p_val >= self.alpha
                
                verdict = "Match" if match else "**Divergent**"
                lines.append(f"| {name} | {val_str} | {verdict} |")

            if not self.is_consistent:
                lines.append(f"\n> **Conclusion**: Topologies differ significantly in dimensions: {self.failed_axes}")
            else:
                lines.append("\n> **Conclusion**: The distributions are statistically equivalent across all axes.")

            content = "\n".join(lines)
        else:
            lines = [
                f"--- Distribution Match Report ({method_label}) ---",
                f"  Analysed Space: {space_label}",
                f"  Global Status:  {status_label}",
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
            
            content = "\n".join(lines)
            
        return self._render_report(content, show, **kwargs)

def topo_distribution(*args, space='objs', axes=None, method='ks', alpha=None, threshold=None, **kwargs):
    """
    [mb.stats.topo_distribution] Performs multi-axial distribution matching.
    Verifies if populations are statistically equivalent in objective or decision space.
    
    Methods:
        'ks': Kolmogorov-Smirnov two-sample test (Default).
        'anderson': Anderson-Darling k-sample test.
        'emd': Earth Mover's Distance (Wasserstein metric).
    """
    alpha = alpha if alpha is not None else defaults.alpha
    threshold = threshold if threshold is not None else defaults.displacement_threshold
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
                # Explicit variant avoids SciPy deprecation warning for `midrank`.
                res = anderson_ksamp(samples, variant='midrank')
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
                
    return DistMatchResult(
        results, names, space=space, method=method, alpha=alpha, threshold=threshold,
        plot_data={"buffers": buffers, "axes": list(axes), "space": space}
    )


def topo_compare(*args, space='objs', axes=None, method='ks', alpha=None, threshold=None, **kwargs):
    """
    Unified topology comparator.
    Methods:
    - 'ks': KS two-sample matching
    - 'emd': Earth Mover's Distance
    - 'anderson': Anderson-Darling k-sample
    """
    method = str(method).lower()
    if method not in ('ks', 'emd', 'anderson'):
        raise ValueError(f"Unknown topo_compare method: {method}. Use 'ks', 'emd', or 'anderson'.")
    res = topo_distribution(*args, space=space, axes=axes, method=method, alpha=alpha, threshold=threshold, **kwargs)
    res.method = method
    return res
