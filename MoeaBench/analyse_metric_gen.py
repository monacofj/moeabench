from .plot_gen import plot_gen
import numpy as np


class analyse_metric_gen(plot_gen):
    
    @staticmethod
    def DATA(args,generation, objective, experiments, bench):
        gen_f_test = [b[0].get_F_gen_non_dominate() for i in args for b in i.result.get_elements()] 
        gen_f_max = max([len(gen)  for gen in gen_f_test])
        generations = [0,gen_f_max] if isinstance(generation, (list)) and len(generation) == 0 else generation
        analyse_metric_gen.allowed_gen(generations)
        analyse_metric_gen.allowed_gen_max(gen_f_max,generations[1])
        objectives = [1,2,3] if isinstance(objective, (list)) and  len(objective) == 0 else objective  
        analyse_metric_gen.allowed_obj(objectives)
        analyse_metric_gen.allowed_obj_equal(bench,bench[0],experiments,objectives)
        gen_f_valid = [b[0].get_F_gen_non_dominate()[generations[0]:generations[1]] for i in args for b in i.result.get_elements()]
        slicing = [[i-1,i]  for i in objectives]
        F_gen = []
        for i in range(len(gen_f_valid)):
            vet_aux = []
            for z in range(len(gen_f_valid[i])):
                vet_aux.append(analyse_metric_gen.slicing_arr(slicing,gen_f_valid[i][z]))
            F_gen.append(vet_aux)           
        F = [b[0].get_arr_DATA() for i in args for b in i.result.get_elements()]
        F_slice = [np.hstack( [b[:,i:j]  for i,j in slicing]) for b in F ]        
        evaluate = [np.arange(generations[0],generations[1]) for _ in range(len(gen_f_valid))]
        return evaluate,F_gen,F_slice 
      

    @staticmethod
    def IPL_hypervolume(args, generations, objectives, reference= []):    
        bench, data = analyse_metric_gen.extract_pareto_result(args)
        evaluate,F_GEN,F = analyse_metric_gen.DATA(args, generations , objectives, bench, data)
        min_nondominate = []
        max_nondominate = []
        if not isinstance(reference,list):
            raise TypeError("Only arrays are allowed in 'references'")
        
        if len(reference) == 0:
            for exp in args:
                reference.append(exp)

        if len(reference) > 0:  
            min_nondominate, max_nondominate = analyse_metric_gen.normalize(reference,F)
            min_slice = [float(min_nondominate[i-1]) for i in objectives] if min_nondominate[0] is not None else min_nondominate
            max_slice = [float(max_nondominate[i-1]) for i in objectives] if max_nondominate[0] is not None else max_nondominate


        hv_gen = analyse_metric_gen.set_hypervolume(F_GEN,F, min_slice, max_slice)
        hypervolume_gen = [hv.evaluate() for hv in hv_gen]
        return evaluate,hypervolume_gen,bench
    

    @staticmethod
    def IPL_plot_Hypervolume(args, generations, objectives, reference= []):   
        evaluate,hypervolume_gen,bench = analyse_metric_gen.IPL_hypervolume(args, generations, objectives, reference)
        plot_g = analyse_metric_gen([evaluate,hypervolume_gen],bench,metric = ['Hypervolume','Generations'])
        plot_g.configure()
             

    @staticmethod
    def IPL_GD(args, generations, objectives):
        bench, data = analyse_metric_gen.extract_pareto_result(args)
        evaluate,F_GEN,F = analyse_metric_gen.DATA(args, generations , objectives, bench, data)
        metric = analyse_metric_gen.set_GD(F_GEN,F)
        mtc_evaluate = [hv.evaluate() for hv in metric]
        return evaluate,mtc_evaluate,bench


    @staticmethod
    def IPL_plot_GD(args, generations, objectives):
        evaluate,GD__gen,bench = analyse_metric_gen.IPL_GD(args, generations, objectives)  
        plot_g = analyse_metric_gen([evaluate,GD__gen],bench,metric = ['GD','Generations'])
        plot_g.configure()
       
    
    @staticmethod
    def IPL_GDplus(args, generations, objectives):
        bench, data = analyse_metric_gen.extract_pareto_result(args)
        evaluate,F_GEN,F = analyse_metric_gen.DATA(args, generations , objectives, bench, data)
        metric = analyse_metric_gen.set_GDplus(F_GEN,F)
        mtc_evaluate = [hv.evaluate() for hv in metric]
        return evaluate,mtc_evaluate,bench


    @staticmethod
    def IPL_plot_GDplus(args, generations, objectives):
        evaluate,GD__gen,bench = analyse_metric_gen.IPL_GDplus(args, generations, objectives)  
        plot_g = analyse_metric_gen([evaluate,GD__gen],bench,metric = ['GD plus','Generations'])
        plot_g.configure()
    

    @staticmethod
    def IPL_IGD(args, generations, objectives):
        bench, data = analyse_metric_gen.extract_pareto_result(args)
        evaluate,F_GEN,F = analyse_metric_gen.DATA(args, generations , objectives, bench, data)
        metric = analyse_metric_gen.set_IGD(F_GEN,F)
        mtc_evaluate = [hv.evaluate() for hv in metric]
        return evaluate,mtc_evaluate,bench


    @staticmethod
    def IPL_plot_IGD(args, generations, objectives):
        evaluate,GD__gen,bench = analyse_metric_gen.IPL_IGD(args, generations, objectives)  
        plot_g = analyse_metric_gen([evaluate,GD__gen],bench,metric = ['IGD','Generations'])
        plot_g.configure()
    

    @staticmethod
    def IPL_IGDplus(args, generations, objectives):
        bench, data = analyse_metric_gen.extract_pareto_result(args)
        evaluate,F_GEN,F = analyse_metric_gen.DATA(args, generations , objectives, bench, data)
        metric = analyse_metric_gen.set_IGD_plus(F_GEN,F)
        mtc_evaluate = [hv.evaluate() for hv in metric]
        return evaluate,mtc_evaluate,bench

    
    @staticmethod
    def IPL_plot_IGDplus(args, generations, objectives):
        evaluate,GD__gen,bench = analyse_metric_gen.IPL_IGDplus(args, generations, objectives)  
        plot_g = analyse_metric_gen([evaluate,GD__gen],bench,metric = ['IGD plus','Generations'])
        plot_g.configure()


