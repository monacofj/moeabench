from .analyse_obj import analyse_obj
from .analyse_surface_obj import analyse_surface_obj
from .I_UserMoeaBench import I_UserMoeaBench
import importlib
from .experiment import experiment
from MoeaBench.stats.stats import stats
from MoeaBench.hypervolume.hypervolume import hypervolume
from MoeaBench.gd.gd import gd
from MoeaBench.gdplus.gdplus import gdplus
from MoeaBench.igd.igd import igd
from MoeaBench.igdplus.igdplus import igdplus
import MoeaBench as mb


class MoeaBench(I_UserMoeaBench):

    @property
    def stats(self):
        return stats(self.result_population.result_population)
    

    @property
    def hypervolume(self):
        return hypervolume(self.result_population.result_population, self.analyse_metric_gen.analyse_metric_gen)
    

    @property
    def gd(self):
        return gd(self.result_population.result_population, self.analyse_metric_gen.analyse_metric_gen)
    

    @property
    def gdplus(self):
        return gdplus(self.result_population.result_population, self.analyse_metric_gen.analyse_metric_gen)
    

    @property
    def igd(self):
        return igd(self.result_population.result_population, self.analyse_metric_gen.analyse_metric_gen)
    

    @property
    def igdplus(self):
        return igdplus(self.result_population.result_population, self.analyse_metric_gen.analyse_metric_gen)
  
    
    def experiment(self):
        return experiment(self)
    

    def __getattr__(self,name):
        if name.startswith("__") and name.endswith("__"):
            return super().__getattribute__(name)
        if name.startswith("_") and name in ["_benchmark","_moea","_result","_pof"]:
           raise AttributeError(name)
        try:
            return importlib.import_module(f"MoeaBench.{name}")
        except ModuleNotFoundError:
            raise AttributeError(name)
    

    def surfaceplot(self, *args, objectives = None):
        objectives = [] if objectives is None else objectives
        """
        - **3D graph of the Pareto boundary surface:**
        Click on the links for more
        ...
                - **Informations:**
                      - sinxtase:
                      moeabench.pareto_surface(exp.problem, experiment2_result, experiment.pof...)  
                      - [pareto_surface](https://moeabench-rgb.github.io/MoeaBench/analysis/objectives/plot/pareto_surface/) information about the method, accepted variable types, examples and more...
                      - [Exception](https://moeabench-rgb.github.io/MoeaBench/analysis/objectives/plot/exceptions/) information on possible error types
       
        """       
        try:
            analyse_surface_obj.IPL_plot_3D(args, objectives)   
        except Exception as e:
            print(e)   
        

    def spaceplot(self, *args, objectives = None):
        objectives = [] if objectives is None else objectives
        """
        - **3D graph for Pareto front:**
        Click on the links for more
        ...
                - **Informations:**
                      - sinxtase:
                      moeabench.pareto(args)  
                      - [pareto](https://moeabench-rgb.github.io/MoeaBench/analysis/objectives/plot/pareto/) information about the method, accepted variable types, examples and more...   
                      - [Exception](https://moeabench-rgb.github.io/MoeaBench/analysis/objectives/plot/exceptions/) information on possible error types

        """
      

        try:     
            analyse_obj.IPL_plot_3D(args, objectives)     
        except Exception as e:
            print(e)
        

    def add_benchmark(self,problem):
        """
        - **Integrates a user benchmark problem implementation in MoeaBench:**
        Click on the links for more
        ...
                - **Informations:**
                      - sinxtase:
                      experiment.add_benchmark(module)  
                      - [add_benchmark](https://moeabench-rgb.github.io/MoeaBench/implement_benchmark/integration/integration/) information about the method 
                     
        """
        import MoeaBench.benchmarks as bk
        setattr(bk,problem.__name__,problem)


    def add_moea(self,moea):
        """
        - **integrates a user genetic algorithm implementation into MoeaBench:**
        Click on the links for more
        ...
                - **Informations:**
                      - sinxtase:
                      experiment.add_moea(module)  
                      - [add_moea](https://moeabench-rgb.github.io/MoeaBench/implement_moea/integration/integration/) information about the method 
                     
        """
        import MoeaBench.moeas as algotithm
        setattr(algotithm,moea.__name__,moea)
    


    






    





 

    

    
    


 
        



    
    

    
