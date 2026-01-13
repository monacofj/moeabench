# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy.stats import mannwhitneyu

def a12(data1, data2):
    """
    Computes the Vargha-Delaney A12 effect size statistic.
    
    A12 measures the probability that a random draw from group 1 
    is larger than a random draw from group 2.
    
    Interpretation:
        0.50: Groups are equivalent.
        <0.50: Group 1 < Group 2.
        >0.50: Group 1 > Group 2.
        
    Magnitudes (Vargha & Delaney, 2000):
        Small:  > 0.56 (or < 0.44)
        Medium: > 0.64 (or < 0.36)
        Large:  > 0.71 (or < 0.29)
        
    Args:
        data1 (array-like): First group of samples.
        data2 (array-like): Second group of samples.
        
    Returns:
        float: The A12 statistic (0.0 to 1.0).
    """
    x = np.array(data1)
    y = np.array(data2)
    
    m = len(x)
    n = len(y)
    
    if m == 0 or n == 0:
        raise ValueError("Data groups cannot be empty.")
        
    r = np.array([0] * m, dtype=float)
    
    for i in range(m):
        r[i] = np.sum(x[i] > y) + 0.5 * np.sum(x[i] == y)
        
    return np.sum(r) / (m * n)

def mann_whitney(data1, data2, alternative='two-sided'):
    """
    Wrapper for scipy.stats.mannwhitneyu.
    
    Performs the Mann-Whitney U rank test on two independent samples.
    
    Args:
        data1 (array-like): First group of samples.
        data2 (array-like): Second group of samples.
        alternative (str): Defines the alternative hypothesis. 
                           'two-sided', 'less', or 'greater'.
                           
    Returns:
        MannwhitneyuResult: Object containing 'statistic' and 'pvalue'.
    """
    return mannwhitneyu(data1, data2, alternative=alternative)
