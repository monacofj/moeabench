from .file import file
from joblib import dump
import numpy as np
import zipfile
from io import BytesIO, StringIO


class save(file):   
    
    @staticmethod
    def verify(obj):
        pof =  True if hasattr(obj,'get_CACHE') and len(obj.get_CACHE().get_elements()) > 0 else False
        result = True if hasattr(obj,'get_elements') and len(obj.get_elements()) > 0  else False
        return True if pof or result else False
       
    
    @staticmethod
    def IPL_save(obj, folder):
        result_exists = save.verify(obj.result)
        if result_exists is True:
            result_moea = obj.result[0] if isinstance(obj.result,tuple) else obj.result
            NonDominate = result_moea.get_elements()[0][0].get_arr_DATA()
            Dominate = result_moea.get_elements()[0][0].get_F_GEN()[-1]
            result =  NonDominate if NonDominate.shape[0] > 1 else Dominate
            solutions =  f'non-dominated solutions of the Pareto front' if NonDominate.shape[0] > 1 else f'Only Pareto-dominated solutions were found.'        
            data = result_moea.get_elements()[0][0]

        pof_exists = save.verify(obj.pof)
        if pof_exists is True:
            pof =  obj.pof.get_CACHE().get_elements()[0][0].get_arr_DATA()  
            bench_pof = obj.pof.get_CACHE().get_elements()[0][1]
      
        path_z = save.DATA(folder)
        if path_z.exists():
            raise FileExistsError("file already exists")
        dt_MoeaBench = []

        if result_exists is True:
            dt_MoeaBench.append(f'{data.get_description()} Evolucionary algorithm data:\n')
            dt_MoeaBench.append(f'generations: {data.get_generations()}')
            dt_MoeaBench.append(f'population: {data.get_population()}')
            dt_MoeaBench.append(f'{solutions}: {result.shape[0]}')

        if pof_exists is True:
            dt_MoeaBench.append(f'\n{bench_pof.get_BENCH()} problem test benchmark data:\n')
            dt_MoeaBench.append(f'objectives: {bench_pof.get_M()}')
            dt_MoeaBench.append(f'decision variabels: {bench_pof.get_Nvar()}')
            if bench_pof.get_K() > 0:
                dt_MoeaBench.append(f'size vector K: {bench_pof.get_K()}')
            if bench_pof.get_D() > 0:
                dt_MoeaBench.append(f'essencial objectves D: {bench_pof.get_D()}')
            dt_MoeaBench.append(f'simulated POF solutions: {pof.shape[0]}')

        if pof_exists is True or result_exists is True:
            dt_MoeaBench.append(f'\nThe zip file contains the following:\n')
        if pof_exists is True:
            dt_MoeaBench.append(f'pof.csv file contains sample simulations of Pareto-optimal front solutions')
        if result_exists is True:
            dt_MoeaBench.append(f'result.csv file contains results of solutions of the evolucionary algorithm related to a problem')
        if pof_exists is True or result_exists is True:
            dt_MoeaBench.append(f"the Movebench.joblib file contains the experiment object, which provides data for use with all of MoveBench's analysis tools.")
          
        
        with zipfile.ZipFile(path_z, 'w') as zf:
            
            zf.writestr('problem.txt',"\n".join(dt_MoeaBench))
            

            if pof_exists is True:
                header_result = ",".join([f'objective {i}' for i in range(1, bench_pof.get_M()+1)])
                mem_csv_pof =  StringIO()
                np.savetxt(mem_csv_pof,pof, delimiter=",", fmt="%.16f", header=header_result, comments='')
                zf.writestr('pof.csv',mem_csv_pof.getvalue())
                            
            if result_exists is True:   
                mem_csv_result =  StringIO()
                np.savetxt(mem_csv_result,result, delimiter=",", fmt="%.16f", header=header_result, comments='')
                zf.writestr('result.csv',mem_csv_result.getvalue())

            mem_obj =  BytesIO()
            dump(obj,mem_obj )
            mem_obj.seek(0)
            zf.writestr('Moeabench.joblib',mem_obj.read())
        



   

        
 
