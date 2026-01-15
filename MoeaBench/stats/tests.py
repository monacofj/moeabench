# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy.stats import mannwhitneyu, ks_2samp
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
    def a12(self) -> float:
        """Vargha-Delaney A12 effect size (Lazy)."""
        return a12(self.data1, self.data2, self.metric, self.gen, **self.kwargs).value

    @property
    def effect_size_label(self) -> str:
        val = self.a12
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
            
        lines.append(f"  A12 Effect Size: {self.a12:.4f} ({self.effect_size_label})")
        
        # Narrative interpretation
        if self.significant:
            better = name1 if self.a12 > 0.5 else name2
            lines.append(f"\nConclusion: There is a statistically significant difference favoring {better}.")
        elif self.p_value is not None:
            lines.append(f"\nConclusion: No statistically significant difference detected.")
        else:
            lines.append(f"\nNote: Only Effect Size calculated (A12={self.a12:.4f}).")
            
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

def a12(data1, data2, metric=None, gen=-1, **kwargs):
    """
    Computes the Vargha-Delaney A12 effect size statistic.
    
    Supports "Smart Stats": can take raw arrays or Experiment objects.
    If Experiments are passed, the specified 'metric' (default mb.hv) 
    is calculated for both with a shared reference point.
    
    Args:
        data1: First group of samples or Experiment.
        data2: Second group of samples or Experiment.
        metric (callable): Metric function to use if experiments are passed.
        gen (int): Generation index to extract from MetricMatrix (default -1).
        **kwargs: Arguments passed to the metric function.
        
    Returns:
        float: The A12 statistic.
    """
    v1, v2 = _resolve_samples(data1, data2, metric, gen, **kwargs)
    
    m = len(v1)
    n = len(v2)
    
    if m == 0 or n == 0:
        raise ValueError("Data groups cannot be empty.")
        
    r = np.array([0] * m, dtype=float)
    
    for i in range(m):
        r[i] = np.sum(v1[i] > v2) + 0.5 * np.sum(v1[i] == v2)
        
    val = np.sum(r) / (m * n)
    return SimpleStatsValue(val, "A12 Effect Size")

def mann_whitney(data1, data2, alternative='two-sided', metric=None, gen=-1, **kwargs):
    """
    Performs the Mann-Whitney U rank test on two independent samples.
    
    Supports "Smart Stats": can take raw arrays or Experiment objects.
    If Experiments are passed, the specified 'metric' (default mb.hv) 
    is calculated for both with a shared reference point.
    
    Args:
        data1: First group of samples or Experiment.
        data2: Second group of samples or Experiment.
        alternative (str): 'two-sided', 'less', or 'greater'.
        metric (callable): Metric function to use if experiments are passed.
        gen (int): Generation index to extract from MetricMatrix (default -1).
        **kwargs: Arguments passed to the metric function.
                           
    Returns:
        MannwhitneyuResult: Object containing 'statistic' and 'pvalue'.
    """
    v1, v2 = _resolve_samples(data1, data2, metric, gen, **kwargs)
    res = mannwhitneyu(v1, v2, alternative=alternative)
    return HypothesisTestResult(res.statistic, res.pvalue, data1, data2, 
                                "Mann-Whitney U", alternative, metric, gen, **kwargs)

def ks_test(data1, data2, alternative='two-sided', metric=None, gen=-1, **kwargs):
    """
    Performs the Kolmogorov-Smirnov two-sample test.
    
    The KS test identifies if two distributions are different in shape, 
    detecting differences in variance, bimodality, or stability 
    that a rank-sum test might miss.
    
    Supports "Smart Stats": can take raw arrays or Experiment objects.
    
    Args:
        data1: First group of samples or Experiment.
        data2: Second group of samples or Experiment.
        alternative (str): 'two-sided', 'less', or 'greater'.
        metric (callable): Metric function to use if experiments are passed.
        gen (int): Generation index to extract from MetricMatrix (default -1).
        **kwargs: Arguments passed to the metric function.
                           
    Returns:
        KstestResult: Object containing 'statistic' and 'pvalue'.
    """
    v1, v2 = _resolve_samples(data1, data2, metric, gen, **kwargs)
    res = ks_2samp(v1, v2, alternative=alternative)
    return HypothesisTestResult(res.statistic, res.pvalue, data1, data2, 
                                "Kolmogorov-Smirnov", alternative, metric, gen, **kwargs)
