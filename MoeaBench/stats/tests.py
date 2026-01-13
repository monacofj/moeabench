# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy.stats import mannwhitneyu

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
        
    return np.sum(r) / (m * n)

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
    return mannwhitneyu(v1, v2, alternative=alternative)

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
    from scipy.stats import ks_2samp
    v1, v2 = _resolve_samples(data1, data2, metric, gen, **kwargs)
    return ks_2samp(v1, v2, alternative=alternative)
