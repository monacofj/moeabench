from MoeaBench.I_metric import I_metric 


class hypervolume(I_metric):
    """
        - **array with hypervolume in generations:**
        Click on the links for more
        ...
                - **Informations:**
                      - sinxtase:
                      experiment.hypervolume(args)  
                      - [hypervolume](https://moeabench-rgb.github.io/MoeaBench/analysis/metrics/data/hypervolume/) information about the method, examples and more...   
                      - [Exception](https://moeabench-rgb.github.io/MoeaBench/analysis/metrics/data/exceptions/) information on possible error types

        """
    def __init__(self, cls_result_population, cls_analyse_metric_gen ):
        self.result_population = cls_result_population()
        self.analyse_metric_gen = cls_analyse_metric_gen()
      

    def __call__(self, *args, generation = None, reference = None):
        reference = [] if reference is None else reference
        try:
            self.result_population.allowed_gen(generation)
            gen = [-2,-1] if generation is None else [generation-1, generation] 
            objectives = [1,2,3] 
            evaluate, hypervolume_gen, bench = self.analyse_metric_gen.IPL_hypervolume(args, gen, objectives = objectives, reference = reference)
            return float(hypervolume_gen[0][0])
        except Exception as e:
             print(e)
           

    def trace(self, *args, objectives = None, reference = None):
        objectives = [] if objectives is None else objectives
        reference = [] if reference is None else reference
        try:
            generations = []
            objectives = [1,2,3] if len(objectives) == 0 else objectives
            evaluate, hypervolume_gen, bench = self.analyse_metric_gen.IPL_hypervolume(args, generations, objectives = objectives, reference = reference)
            return hypervolume_gen[0]
        except Exception as e:
            print(e)
              
    
    def timeplot(self, *args, generations = None, objectives = None, reference = None):
        generations = [] if generations is None else generations
        objectives = [] if objectives is None else objectives
        reference = [] if reference is None else reference
        """
        - **2D graph for hypervolume:**
        Click on the links for more
        ...
                - **Informations:**
                      - sinxtase:
                      moeabench.plot_hypervolume(args) 
                      - [plot_hypervolume](https://moeabench-rgb.github.io/MoeaBench/analysis/metrics/plot/plot_hypervolume/) information about the method, accepted variable types, examples and more...   
                      - [Exception](https://moeabench-rgb.github.io/MoeaBench/analysis/metrics/plot/exceptions/) information on possible error types

        """
        try:
            objectives = [1,2,3] if len(objectives) == 0 else objectives
            self.analyse_metric_gen.IPL_plot_Hypervolume(args,generations, objectives = objectives, reference = reference)
        except Exception as e:
            print(e)
