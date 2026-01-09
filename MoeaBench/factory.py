from .MoeaBench import MoeaBench


class _MoeaBenchWrapper:
    """
        - Description:   
          MoeaBench is a framework for experimentation, analysis, and 
          development of benchmark problems for validating the performance     
          of genetic algorithms.
             
          - how to start
            Click the link for a step-by-step guide:
            https://moeabench-rgb.github.io/MoeaBench/step_by_step/
    """

    
    def __getattr__(self, name):
        inst = MoeaBench()
        return getattr(inst, name)


    def __call__(self, *args, **kwargs):
        return MoeaBench(*args,  **kwargs)

    
    def help(self):
        print(self.__class__.__doc__)

mb = _MoeaBenchWrapper()
