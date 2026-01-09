from .kernel_benchmark.K_DTLZ9 import K_DTLZ9


class P_DTLZ9(K_DTLZ9):

    def __init__(self, M, N, P, CACHE, **kwargs):
        self.CACHE=CACHE
        self.N = N
        self.M = M
        self.P = P
        super().__init__(CACHE=CACHE, **kwargs)
    

    def get_CACHE(self):
        return self.CACHE


    def set_BENCH_conf(self):   
        self.get_CACHE().set_BENCH_CI(self.M,0,90,self.P,0,self.M-1,9) 
        self.get_CACHE().get_BENCH_CI().set_Nvar(self.N)
        self.set_Point()


    def POFsamples(self):
        try:
            if  self.N_validate(self.get_CACHE().get_BENCH_CI().get_Nvar()) == True and self.M_validate(self.get_CACHE().get_BENCH_CI().get_M()) == True:
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
         if hasattr(self,"_H_DTLZ__arr_ENUM"):
             self.set_BENCH_conf()
             self.POFsamples()
         

    @property
    def N(self):
         return self._N
        
     
    @N.setter
    def N(self, value):
         self._N = value
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
