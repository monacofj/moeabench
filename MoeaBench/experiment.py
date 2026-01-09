from .RUN import RUN
from .RUN_user import RUN_user
from .result_set import result_set
from .moea_round import moea_round
from .save import save
from .loader import loader
from .I_UserExperiment import I_UserExperiment
import inspect
import numpy as np


class experiment(I_UserExperiment):

    def __init__(self, imports):
        self.pof=None
        self.result=None
        self.imports = imports
        self.result_set=result_set()
        self.hist_M_user = []


    @property
    def variables(self):    
        return self.imports.variables.variables(self.imports.result_var.result_var, self.result, self._rounds)
    

    @property
    def objectives(self):    
        return self.imports.objectives.objectives(self.imports.result_obj.result_obj, self.result, self._rounds)
    

    @property
    def front(self):    
        return self.imports.front.front(self.imports.result_front.result_front, self.result, self._rounds)
    

    @property
    def set(self):    
        return self.imports.set.set(self.imports.result_set.result_set, self.result, self._rounds)

     
    @property
    def name(self):
        return self._name 
    

    @name.setter
    def name(self, value):
        self._name = value


    @property
    def rounds(self):
        return self._rounds
    

    @rounds.setter
    def rounds(self, value):
        if not hasattr(self,'_rounds'):
            self._rounds = []
        self._rounds.extend(value)


    @property
    def stop(self):
        return self._stop
    

    @stop.setter
    def stop(self, value):
        self._stop = value


    @property
    def optimal(self):
        return self.imports.optimal.optimal(self)
    
    
    @property
    def dominated(self):
        return self.imports.dominated.dominated(self.imports.result_dominated.result_dominated, self.result, self._rounds)


    @optimal.setter
    def optimal(self, value):
        self._optimal = value


    @property
    def moea(self):
        return self._moea
    

    @moea.setter
    def moea(self,value):  
        stop = self.stop if hasattr(self,'_stop') else None
        self.result = value(self, self.imports.moeas, stop) if callable(value) else value
        self._moea = value


    @property
    def benchmark(self):
        return self._benchmark
      

    @benchmark.setter
    def benchmark(self,value):
        self._benchmark=value(self.imports.benchmarks) if callable(value) else value
        self.pof=self._benchmark 
    

    def load(self,file):
        """
        - **Loads a user experiment into MoeaBench:**
        Click on the links for more
        ...
                - Informations:
                      - sinxtase:
                      experiment.load(nameFile)  
                      - [load](https://moeabench-rgb.github.io/MoeaBench/experiments/load_experiment/load_experiment/) information about the method, 
                     
        """
        try:
            loader.IPL_loader(self,file)    
            if isinstance(self.result,tuple):
                self.result = self.result[0]  
        except Exception as e:
            print(e)


    def save(self, file):
        """
        - **save the user's experiment in a zip file:**
        Click on the links for more
        ...
                - **Informations:**
                      - sinxtase:
                      experiment.save(nameFile)  
                      - [save](https://moeabench-rgb.github.io/MoeaBench/experiments/save_experiment/save_experiment/) information about the method, 
                     
        """
        algoritm = None
        if (hasattr(self,'_moea')):
            moea_found = self.imports.moeas.moea_algorithm()
            algoritm = moea_found.get_MOEA(self.moea.__class__.__name__)
        try:
            if isinstance(algoritm,tuple) and inspect.isclass(algoritm[0]):
                raise ValueError("experiments using the methods @mb.benchmarks.register_benchmark() and @mb.moeas.register_moea() cannot be saved.")
            save.IPL_save(self,file)
        except Exception as e:
            print(e)
   

    def run_moea(self, seed):

        if isinstance(self.result,tuple):
            name_moea = self.result[2]
        else:
            name_moea = self.result.edit_DATA_conf().get_DATA_MOEA().__class__.__name__       
        try:
            moea_found = self.imports.moeas.moea_algorithm()
            algoritm = moea_found.get_MOEA(self.moea.__class__.__name__)
            execute = RUN() if not isinstance(algoritm, bool ) and not inspect.isclass(algoritm[0]) else RUN_user()
            
            if isinstance(execute, RUN_user):
                self.hist_M_user.append(self.benchmark.M)
                self.result = self.moea(self.benchmark, self.imports.moeas) if not len(set(self.hist_M_user)) == 1 else self.result
                         
            elif isinstance(execute, RUN):
                stop = self.stop if hasattr(self,'_stop') else None
                self.result = self.moea(self, None, stop, seed) 
            
            self.result_moea = self.result[0] if isinstance(self.result,tuple) else self.result
            try:
                name_benchmark = self.benchmark.__class__.__name__.split("_")[1]
            except Exception as e:
                name_benchmark = self.benchmark.__class__.__name__
                
            return execute.MOEA_execute(self.result_moea,name_moea,name_benchmark)
        except Exception as e:
            print(e)

    
    def run(self, repeat = 0):
        """
        - **run the genetic algorithm:**
        Click on the links for more
        ...
               - **Informations:**
                      - sinxtase:
                      experiment.run()   
                      - [run()](https://moeabench-rgb.github.io/MoeaBench/experiments/combinations/combinations/#moeabench-run-the-experiment) Information about the method and return variables.

        """
        try:
            generator = np.random.default_rng()
            if not isinstance(repeat,int):
                raise TypeError('Only integers are allowed as parameters for the run() method.')
            self.result_moea = self.result[0] if isinstance(self.result,tuple) else self.result
            execution = repeat-1 if repeat > 0 else 0
            cont = 0
            for exe in range(0,execution):
                cont += 1
                self.run_moea(generator)
                self.rounds = [moea_round(b, f'round {cont}') for i in self.result_moea.get_elements() for b in i if hasattr(b,'get_F_GEN')]           
            cont += 1  
            seed_moea = generator if self.moea.seed == 0 else self.moea.seed
            self.run_moea(seed_moea)
            
            self.rounds = [moea_round(b, f'round {cont}') for i in self.result_moea.get_elements() for b in i if hasattr(b,'get_F_GEN')]
        except Exception as e:
            print(e)


           
           