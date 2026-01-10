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
        """       
        try:
            analyse_surface_obj.IPL_plot_3D(args, objectives)   
        except Exception as e:
            print(e)   
        

    def timeplot(self, *args, objectives=None, mode='interactive'):
        """
        Plots metrics over time (generations).
        Accepts MetricMatrix objects (from mb.metrics.*) or legacy inputs.
        """
        try:
            # Check if args are MetricMatrix (new API)
            is_new_metric = any(hasattr(a, 'runs') and hasattr(a, 'gens') for a in args)
            
            if is_new_metric:
                 from MoeaBench.metrics import plot_matrix
                 plot_matrix(args, mode=mode)
            else:
                 print("Warning: use mb.metrics.* for timeplot support.")
                 # Fallback logic could go here if we identified legacy metric objects
                 
        except Exception as e:
            print(e)

    def spaceplot(self, *args, objectives = None, mode='interactive'):
        objectives = [] if objectives is None else objectives
        """
        - **3D graph for Pareto front:**
        """
        try:     
             new_args = []
             for arg in args:
                 if hasattr(arg, 'objectives'): # Population / JoinedPopulation
                      new_args.append(arg.objectives)
                 elif hasattr(arg, 'front'): # Run / Experiment? No, front() is method.
                      new_args.append(arg)
                 else:
                      new_args.append(arg)
                      
             analyse_obj.IPL_plot_3D(tuple(new_args), objectives, mode=mode)     
        except Exception as e:
            print(e)
        

    def add_benchmark(self,problem):
        """
        - **Integrates a user benchmark problem implementation in MoeaBench:**
        """
        import MoeaBench.benchmarks as bk
        setattr(bk,problem.__name__,problem)


    def add_moea(self,moea):
        """
        - **integrates a user genetic algorithm implementation into MoeaBench:**
        """
        import MoeaBench.moeas as algotithm
        setattr(algotithm,moea.__name__,moea)
    
