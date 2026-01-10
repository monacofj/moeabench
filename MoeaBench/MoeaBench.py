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
             
             # Smart Title / Axis Label Deduction
             # We look at the first argument to guess the plot type (title)
             # And axis labels.
             
             plot_title = "Pareto-optimal front"
             axis_label = "Objective"
             
             # Inspect variable names from caller
             import inspect
             frame = inspect.currentframe().f_back
             callers_vars = frame.f_locals
             
             inferred_names = []
             
             for i, arg in enumerate(args):
                 # unwrapping Population
                 if hasattr(arg, 'objectives'): 
                      val = arg.objectives
                 elif hasattr(arg, 'front'): # Method?
                      # If user passed exp.front(), it's already an array/SmartArray 
                      # handled by checking arg directly
                      val = arg
                 else:
                      val = arg

                 new_args.append(val)
                 
                 # Deduce metadata if SmartArray
                 if hasattr(val, 'label') and val.label:
                     if i == 0: plot_title = val.label
                 if hasattr(val, 'axis_label') and val.axis_label:
                     if i == 0: axis_label = val.axis_label
                     
                 # Deduce variable name
                 # This is tricky because `exp.front()` doesn't map to a single var name easily.
                 # But if user passed `my_data`, we can find it.
                 # Heuristics: search for arg in callers_vars values
                 found_name = None
                 
                 # Prioritize explicit name in SmartArray, 
                 # UNLESS it's the generic default "experiment"
                 if hasattr(val, 'name') and val.name and val.name != "experiment":
                      found_name = val.name
                 else:
                     # 1. Check if arg represents an Experiment with a name
                     # If user passed exp.front(), determining that 'exp' was the source is hard.
                     # But maybe the SmartArray or experiment object has a name?
                     
                     # If user passed 'exp', we likely extracted .objectives from it above.
                     # But we iterate 'args'.
                     
                     # Try to find the object in local vars
                     # This is O(N_locals) per arg.
                     # Also check .source if available
                     target_obj = arg
                     if hasattr(arg, 'source'):
                         target_obj = arg.source
                         
                     for var_name, var_val in callers_vars.items():
                          if var_val is target_obj:
                              found_name = var_name
                              break
                 
                 if found_name:
                     inferred_names.append(found_name)
                 else:
                     inferred_names.append(None) # Will default later
                     
             # print(f"DEBUG: inferred_names={inferred_names}") # Debugging
             
             # Pass metadata to IPL_plot_3D
             # It expects (args, objectives, mode)
             # We might need to extend it or set globals/attributes on the class?
             # analyse_obj (line 8) calls extract_pareto_result(args).
             # extracting names happens there.
             
             # To avoid changing signature of IPL_plot_3D too much affecting existing code,
             # we can pass naming info attached to args or via a side channel?
             # args is a tuple.
             
             # Better: wrapper objects around args that carry the name if not present?
             # analyse_pareto.py line 62 checks: name = f'{i.name}' if hasattr(i,'_name') else False
             
             # So if we attach `_name` to the arrays (SmartArrays), it will pick it up!
             for i, arg in enumerate(new_args):
                  if inferred_names[i] and not hasattr(arg, '_name'):
                      # SmartArray/ndarray allows setting attributes usually
                      try:
                          final_name = inferred_names[i]
                          if hasattr(arg, 'label') and arg.label and arg.label not in str(final_name):
                              final_name = f"{final_name} ({arg.label})"
                          arg._name = final_name
                          # print(f"DEBUG: Set _name={arg._name}")
                      except:
                          pass # standard array might fail without subclass
                  
                  # Also helpful: if SmartArray has .label, maybe usage for legend? 
                  # analyse_pareto uses i.name for legend.
             
             # How to pass plot_title and axis_labels?
             # analyse_obj.IPL_plot_3D -> analyse_obj(...) -> plot_3D(...)
             # plot_3D init takes 'type' (which is title) and 'axis' (indexes).
             # It doesn't seem to take custom axis string labels (it uses "Objective {i}").
             
             # usage: plot_3D_obj = analyse_obj(benk, data, axis, mode=mode)
             # analyse_obj inherits from plot_3D.
             # We can pass `type=plot_title` kwarg if we update IPL_plot_3D signature?
             # Or pass it via `mode` hack? No.
             
             analyse_obj.IPL_plot_3D(tuple(new_args), objectives, mode=mode, title=plot_title, axis_label=axis_label)     
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
        

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
    
