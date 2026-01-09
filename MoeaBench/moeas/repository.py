


class repository:
        
        def __init__(self, algorithm):
            self.algorithm = algorithm

        
        def __call__(self):
          moea = self.algorithm.get_CACHE()
          moea.get_DATA_conf().set_DATA_MOEA(self.algorithm,self.algorithm.get_problem()) 
          return (moea,self.algorithm.__class__,self.algorithm.__class__.__name__)