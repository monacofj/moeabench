from  .kernel_benchmark.K_DPF1 import K_DPF1


class P_DPF1(K_DPF1):

    def __init__(self, M, K, D, P, CACHE, **kwargs):
        self.CACHE=CACHE
        self.M = M
        self.K = K
        self.D = D 
        self.P = P
        super().__init__(CACHE = CACHE, **kwargs)
        

    def get_CACHE(self):
        return self.CACHE
    
        
    def set_BENCH_conf(self): 
        self.set_Penalty_param(0.65)
        self.get_CACHE().set_BENCH_CI(self.M,self.D,11,self.P,self.K,1,10) 
        self.get_CACHE().get_BENCH_CI().set_Nvar()
        self.set_Point()
        self.set_POF(0.5)
        self.set_Pareto(0.5)
         
        
    def POFsamples(self):
        try:
            if self.K_validate(self.get_CACHE().get_BENCH_CI().get_K()) == True and self.MN_validate(self.get_CACHE().get_BENCH_CI().get_K(),self.get_CACHE().get_BENCH_CI().get_M(),self.get_CACHE().get_BENCH_CI().get_D()) == True and self.MN1_validate(self.get_CACHE().get_BENCH_CI().get_M(),self.get_CACHE().get_BENCH_CI().get_D()) == True:
                F , X = self.minimize()
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
         if hasattr(self,"_H_DPF__arr_ENUM"):
             self.set_BENCH_conf()
             self.POFsamples()
         

    @property
    def K(self):
         return self._K
         

    @K.setter
    def K(self, value):
         self._K = value
         if hasattr(self,"_H_DPF__arr_ENUM"):
             self.set_BENCH_conf()
             self.POFsamples()

    @property
    def D(self):
         return self._D 
     

    @D.setter
    def D(self, value):
         self._D = value
         if hasattr(self,"_H_DPF__arr_ENUM"):
             self.set_BENCH_conf()
             self.POFsamples()


    @property
    def P(self):
         return self._P 
     

    @P.setter
    def P(self, value):
         self._P = value
         if hasattr(self,"_H_DPF__arr_ENUM"):
             self.set_BENCH_conf()
             self.POFsamples()

