from MoeaBench.I_metric import I_metric 


class igd(I_metric):
    """
        - **array with IGD in generations:**
        Click on the links for more
        ...
                - **Informations:**
                      - sinxtase:
                      experiment.IGD(args)  
                      - [IGD](https://moeabench-rgb.github.io/MoeaBench/analysis/metrics/data/IGD/) information about the method, examples and more...   
                      - [Exception](https://moeabench-rgb.github.io/MoeaBench/analysis/metrics/data/exceptions/) information on possible error types

        """
    def __init__(self, cls_result_population, cls_analyse_metric_gen ):
        self.result_population = cls_result_population()
        self.analyse_metric_gen = cls_analyse_metric_gen()
      

    def __call__(self, *args, generation = None):
        try:
            self.result_population.allowed_gen(generation)
            gen = [-2,-1] if generation is None else [generation-1, generation] 
            objectives = [1,2,3] 
            evaluate, GD_gen, bench = self.analyse_metric_gen.IPL_IGD(args, gen, objectives = objectives)
            return float(GD_gen[0][0])
        except Exception as e:
             print(e)
           

    def trace(self, *args, objectives = None):
        objectives = [] if objectives is None else objectives
        try:
            generations = []
            objectives = [1,2,3] if len(objectives) == 0 else objectives
            evaluate, GD_gen, bench = self.analyse_metric_gen.IPL_IGD(args, generations, objectives = objectives)
            return GD_gen[0]
        except Exception as e:
            print(e)
              
    
    def timeplot(self, *args, generations = None, objectives = None):
        generations = [] if generations is None else generations
        objectives = [] if objectives is None else objectives
        """
         - **2D graph for IGD:**
         Click on the links for more
         ...
                - **Informations:**
                      - sinxtase:
                      moeabench.plot_IGD(args) 
                      - [plot_IGD](https://moeabench-rgb.github.io/MoeaBench/analysis/metrics/plot/plot_IGD/) information about the method, accepted variable types, examples and more...   
                      - [Exception](https://moeabench-rgb.github.io/MoeaBench/analysis/metrics/plot/exceptions/) information on possible error types

         """
        try:
            objectives = [1,2,3] if len(objectives) == 0 else objectives
            self.analyse_metric_gen.IPL_plot_IGD(args,generations, objectives = objectives)
        except Exception as e:
            print(e)