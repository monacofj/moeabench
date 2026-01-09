
"""     
        - Benchmarks:       
            MoeaBench uses several evolutionary algorithms from the pymoo library.
               ...
               - NSGA-III:
                      - sinxtase:
                      experiment.moea = moeabench.moeas.NSGA_III(args) 
                      - [NSGA-III](https://moeabench-rgb.github.io/MoeaBench/algorithms/NSGA3/) information about the genetic algorithm
               ... 
               - U-NSGA-III:
                      - sinxtase:
                      experiment.moea = moeabench.moeas.U_NSGA_III(args)
                      - [U-NSGA-3](https://moeabench-rgb.github.io/MoeaBench/algorithms/UNSGA3/) information about the genetic algorithm
                ...
               - SPEA-II:
                      - sinxtase:
                      experiment.moea = moeabench.moeas.SPEA_II(args) 
                      - [SPEA-II](https://moeabench-rgb.github.io/MoeaBench/algorithms/SPEA2/) information about the genetic algorithm
               ... 
               - MOEA/D:
                      - sinxtase:
                      experiment.moea = moeabench.moeas.MOEAD(args) 
                      - [MOEA/D](https://moeabench-rgb.github.io/MoeaBench/algorithms/MOEAD/) information about the genetic algorithm
                 ... 
               - RVEA:
                      - sinxtase:
                      experiment.moea = moeabench.moeas.RVEA(args) 
                      - [RVEA](https://moeabench-rgb.github.io/MoeaBench/algorithms/RVEA/) information about the genetic algorithm
                 ...       
               - my_new_moea:      
                      - sinxtase:
                      experiment.benchmark = moeabench.moeas.my_new_moea(args)       
                      - [my_new_moea](https://moeabench-rgb.github.io/MoeaBench/implement_moea/memory/memory/) information about the method, 
                 ...          
"""

import os, importlib
import MoeaBench.moeas.my_new_moea as my_moea


_dir = os.path.dirname(__file__)

for root, dirs , files in os.walk(_dir):
    for fl in files:
        if fl.endswith(".py") and fl not in ("__init__.py",):
            path  = os.path.relpath(os.path.join(root,fl),_dir)
            name_module = path.replace(os.sep,".")[:-3]
            module = importlib.import_module(f'{__name__}.{name_module}')
            cls_name = fl[:-3]
            globals()[cls_name] = getattr(module, cls_name)

my_module_cache = importlib.import_module("MoeaBench.CACHE")
globals()['CACHE'] = my_module_cache.CACHE
my_module_cache = importlib.import_module("MoeaBench.CACHE_bk_user")
globals()['CACHE_bk_user'] = my_module_cache.CACHE_bk_user
globals()['register_moea'] = my_moea.register_moea
