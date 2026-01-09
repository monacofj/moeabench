from .plot_3D import plot_3D
import numpy as np


class analyse_obj(plot_3D):
           
    @staticmethod
    def IPL_plot_3D(args, objectives):
        benk, data = analyse_obj.extract_pareto_result(args)
        axis =  [i for i in range(0,3)]    if len(objectives) == 0 else [i-1 if i > 0 else 0 for i in objectives] 
        analyse_obj.allowed_obj(objectives, data, benk)
        plot_3D_obj =  analyse_obj(benk, data,axis)
        plot_3D_obj.configure()

      
        
            
    

      
    
