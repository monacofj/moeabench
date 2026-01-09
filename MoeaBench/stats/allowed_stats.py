import numpy as np

class allowed_stats:

    def allowed(self,args):
        results = [True if isinstance(exp,np.ndarray) or hasattr(exp,'result')  else False for exp in args]
        if False in results:
            raise ValueError("only array or experiment data types are allowed.")    
        if results is not None and len(results) != 2:
            raise ValueError("Only two arrays are allowed for the metric calculation.")
       