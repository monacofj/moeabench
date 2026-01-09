from .kernel_benchmark.K_DTLZ4 import K_DTLZ4


class P_DTLZ4(K_DTLZ4):
    
    def __init__(self, M, K, P, CACHE, **kwargs):
        self.CACHE=CACHE
        self.M = M
        self.K = K
        self.P = P
        super().__init__(CACHE = CACHE, **kwargs)


    def get_CACHE(self):
        return self.CACHE
      
        
    def set_BENCH_conf(self): 
        self.set_Penalty_param(1.2)
        self.get_CACHE().set_BENCH_CI(self.M,0,40,self.P,self.K,1,4) 
        self.get_CACHE().get_BENCH_CI().set_Nvar()
        self.set_Point()
        self.set_POF(0.5)
        self.set_Pareto(1)
        

    def POFsamples(self):
        try:
            if self.K_validate(self.get_CACHE().get_BENCH_CI().get_K()) == True and self.M_validate(self.get_CACHE().get_BENCH_CI().get_M()) == True:
                F, X = self.minimize()
                for key,value in F.items():
                    self.get_CACHE().DATA_store(key,0,0,value,[0],[0],self,self.__class__.__name__.split("_")[1],[0],[0],[0],[0])  
        except Exception as e:
            print(e)


    @property
    def M(self):
         return self._M 
     

    @M.setter
    def M(self, value):
         self._M = value
         if hasattr(self,"_H_DTLZ__arr_ENUM"):
             self.set_BENCH_conf()
             self.POFsamples()
         

    @property
    def K(self):
         return self._K
        
     

    @K.setter
    def K(self, value):
         self._K = value
         if hasattr(self,"_H_DTLZ__arr_ENUM"):
             self.set_BENCH_conf()
             self.POFsamples()


    @property
    def P(self):
         return self._P 
     

    @P.setter
    def P(self, value):
         self._P = value
         if hasattr(self,"_H_DTLZ__arr_ENUM"):
             self.set_BENCH_conf()
             self.POFsamples()