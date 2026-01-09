from .problems import problems


class DTLZ2:
     """
        - benchmark problem:
        Click on the links for more
        ...
                - DTLZ2:
                      - sinxtase:
                      experiment.benchmark = moeabench.benchmarks.DTLZ2(args) 
                      - [general](https://moeabench-rgb.github.io/MoeaBench/problems/DTLZ/DTLZ2/) POF sampling, results obtained in tests 
                      with genetic algorithms, references and more... 
                      - [implementation](https://moeabench-rgb.github.io/MoeaBench/problems/DTLZ/DTLZ2/DTLZ2/) detailed implementation information
                      - ([arguments](https://moeabench-rgb.github.io/MoeaBench/problems/DTLZ/arguments/)) custom and default settings problem
                      - [Exception](https://moeabench-rgb.github.io/MoeaBench/problems/DTLZ/exceptions/) information on possible error types
        
     """
     
     def __init__(self, M = 3, K = 5, P = 700):
          self.M = M
          self.K = K
          self.P = P


     def __call__(self, default = None):
 
        try:
            problem = problems()
            bk = problem.get_problem(self.__class__.__name__)
            class_bk =  getattr(bk[0],bk[1].name)
            instance = class_bk(self.M, self.K, self.P, problem.get_CACHE())
            instance.P_validate(self.P)
            instance.set_BENCH_conf()
            instance.POFsamples()
            return instance
        except Exception as e:
            print(e)

 
            


            