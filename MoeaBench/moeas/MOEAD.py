from .moea_algorithm import moea_algorithm


class MOEAD:
    """
        - genetic algorithm:
        Click on the links for more
        ...
                - MOEA/D:
                      - sinxtase:
                      experiment.moea = moeabench.moeas.MOEAD(args)  
                      - [general](https://moeabench-rgb.github.io/MoeaBench/algorithms/MOEAD/) references and more...
                      - ([arguments](https://moeabench-rgb.github.io/MoeaBench/algorithms/arguments/)) custom and default settings problem
                      - [configurations](https://moeabench-rgb.github.io/MoeaBench/algorithms/configuration/) algorithm configuration adopted by MoeaBench
        
        """

    def __init__(self,population = 150, generations = 300, seed = 0):
        self._population=population
        self._generations=generations
        self.seed = seed
        self.result = None


    def __call__(self, problem, default = None, stop = None, seed = 0):
        self.problem = problem
        moea = moea_algorithm()
        algoritm = moea.get_MOEA(self.__class__.__name__)
        class_algoritm = getattr(algoritm[0],algoritm[1].name)
        instance = class_algoritm(problem,self._population,self._generations, seed, stop)
        result = moea.get_CACHE()
        result.get_DATA_conf().set_DATA_MOEA(instance,problem.benchmark)
        self.result = result
        return result 
    
    
    @property
    def generations(self):
        return self._generations
    

    @generations.setter
    def generations(self,value):
        self._generations = value   
        if hasattr(self,"problem"):
           self.result.edit_DATA_conf().get_DATA_MOEA().generations=value


    @property
    def population(self):
        return self._population
    

    @population.setter
    def population(self,value):
        self._population = value   
        if hasattr(self,"problem"):
           self.result.edit_DATA_conf().get_DATA_MOEA().population=value