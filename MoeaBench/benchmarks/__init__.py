"""     
        - Benchmarks:       
            MoeaBench has implementations of several benchmark problems. 
            Click on the link for the respective benchmark problem of each experiment to 
            obtain more information about the problem.
               ...
               - DTLZ1:
                      - sinxtase:
                      experiment.benchmark = moeabench.benchmarks.DTLZ1(args) 
                      - [DTLZ1](https://moeabench-rgb.github.io/MoeaBench/problems/DTLZ/DTLZ1/) detailed information about the problem
               ... 
               - DTLZ2:
                      - sinxtase:
                      experiment.benchmark = moeabench.benchmarks.DTLZ2(args) 
                      - [DTLZ2](https://moeabench-rgb.github.io/MoeaBench/problems/DTLZ/DTLZ2/) detailed information about the problem
                ...
               - DTLZ3:
                      - sinxtase:
                      experiment.benchmark = moeabench.benchmarks.DTLZ3(args) 
                      - [DTLZ3](https://moeabench-rgb.github.io/MoeaBench/problems/DTLZ/DTLZ3/) detailed information about the problem
               ... 
               - DTLZ4:
                      - sinxtase:
                      experiment.benchmark = moeabench.benchmarks.DTLZ4(args) 
                      - [DTLZ4](https://moeabench-rgb.github.io/MoeaBench/problems/DTLZ/DTLZ4/) detailed information about the problem
                 ... 
               - DTLZ5:
                      - sinxtase:
                      experiment.benchmark = moeabench.benchmarks.DTLZ5(args) 
                      - [DTLZ5](https://moeabench-rgb.github.io/MoeaBench/problems/DTLZ/DTLZ5/) detailed information about the problem
                ... 
               - DTLZ6:
                      - sinxtase:
                      experiment.benchmark = moeabench.benchmarks.DTLZ6(args) 
                      - [DTLZ6](https://moeabench-rgb.github.io/MoeaBench/problems/DTLZ/DTLZ6/) detailed information about the problem
                ...
                - DTLZ7:
                      - sinxtase:
                      experiment.benchmark = moeabench.benchmarks.DTLZ7(args) 
                      - [DTLZ7](https://moeabench-rgb.github.io/MoeaBench/problems/DTLZ/DTLZ7/) detailed information about the problem
                 ...
                - DTLZ8:
                      - sinxtase:
                      experiment.benchmark = moeabench.benchmarks.DTLZ8(args) 
                      - [DTLZ8](https://moeabench-rgb.github.io/MoeaBench/problems/DTLZ/DTLZ8/) detailed information about the problem
                 ...
                - DTLZ9:
                      - sinxtase:
                      experiment.benchmark = moeabench.benchmarks.DTLZ9(args) 
                      - [DTLZ9](https://moeabench-rgb.github.io/MoeaBench/problems/DTLZ/DTLZ9/) detailed information about the problem
                 ...
                - DPF1:
                      - sinxtase:
                      experiment.benchmark = moeabench.benchmarks.DPF1(args) 
                      - [DPF1](https://moeabench-rgb.github.io/MoeaBench/problems/DPF/DPF1/) detailed information about the problem
                 ...
                - DPF2:
                      - sinxtase:
                      experiment.benchmark = moeabench.benchmarks.DPF2(args) 
                      - [DPF2](https://moeabench-rgb.github.io/MoeaBench/problems/DPF/DPF2/) detailed information about the problem
                 ...
                - DPF3:
                      - sinxtase:
                      experiment.benchmark = moeabench.benchmarks.DPF3(args) 
                      - [DPF3](https://moeabench-rgb.github.io/MoeaBench/problems/DPF/DPF3/) detailed information about the problem
                 ...
                - DPF4:
                      - sinxtase:
                      experiment.benchmark = moeabench.benchmarks.DPF4(args) 
                      - [DPF4](https://moeabench-rgb.github.io/MoeaBench/problems/DPF/DPF4/) detailed information about the problem
                 ...
                - DPF5:
                      - sinxtase:
                      experiment.benchmark = moeabench.benchmarks.DPF5(args) 
                      - [DPF5](https://moeabench-rgb.github.io/MoeaBench/problems/DPF/DPF5/) detailed information about the problem
                 ...
                - my_new_benchmark:      
                      - sinxtase:
                      experiment.benchmark = moeabench.benchmarks.my_new_benchmark()        
                      - [my_new_benchmark](https://moeabench-rgb.github.io/MoeaBench/implement_benchmark/memory/memory/) information about the method 
                                            
"""
import os, importlib
import MoeaBench.benchmarks.my_new_benchmark as m_bk

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
globals()['register_benchmark'] = m_bk.register_benchmark







    