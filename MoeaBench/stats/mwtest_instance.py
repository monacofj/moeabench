# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from scipy.stats import mannwhitneyu
from .allowed_stats import allowed_stats
import numpy as np
import logging


class mwtest_instance(allowed_stats):
    
    def __init__(self, args, alternative_metric):
        self.args = args
        self.statistic = None
        self.pvalue = None
        self.alternative_metric = alternative_metric


    def allowed(self):
        valid = [True if isinstance(arr,np.ndarray) and  arr.ndim == 1 else False for arr in self.args]
        if False in valid:
            raise ValueError("only one-dimensional arrays are allowed.")    
        if valid is not None and len(self.args) != 2:
            raise ValueError("only two arrays are allowed for the metric calculation.")


    def __call__(self):      
        try:
            self.allowed()
            stat, value = mannwhitneyu(self.args[0],self.args[1], alternative=self.alternative_metric)
            self.statistic = float(stat)
            self.pvalue = float(value)
        except Exception as e:
            print(e)  
          
        
