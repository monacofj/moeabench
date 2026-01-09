from .I_BENCH_conf import I_BENCH_conf

class BENCH_conf(I_BENCH_conf):

    def set_user(self,name_benchmark = None,M = 0, N=0,n_ieq_constr=0,P=0,K=0,D=0):
        self.__BENCH = name_benchmark 
        self.__M = M
        self.__Nvar=N
        self.__n_ieq_constr=n_ieq_constr
        self.__P=P
        self.__K=K
        self.__D=D
        self.__BENCH_Nvar=1


    def set(self,M=0,D=0,BENCH=0,P=0,K=0,n_ieq_constr=0,BENCH_Nvar=1):
        self.__M = M
        self.__D = D
        self.__BENCH=BENCH
        self.__P=P
        self.__K=K
        self.__n_ieq_constr=n_ieq_constr
        self.__BENCH_Nvar=BENCH_Nvar
     
    
    def get_M(self):
        return self.__M
         

    def set_M(self,M):
        self.__M=M


    def set_K(self,K):
        self.__K=K
         

    def get_K(self):
        return self.__K
    

    def get_Nvar(self):
        return self.__Nvar
    

    def set_Nvar(self,Nvar=None):
        if Nvar != None:
            self.__Nvar=Nvar
            return
        
        N_Bench = self.get_BENCH_Nvar()
        if N_Bench <=7:
            self.__Nvar = self.get_K()+self.get_M()-1
        elif N_Bench > 7 and N_Bench <= 9:
            self.__Nvar = self.get_K()
        elif N_Bench >= 10:
            self.__Nvar = self.get_D()+self.get_K()-1
        

    def get_D(self):
        return self.__D
    

    def set_D(self,D):
        self.__D=D
     

    def get_BENCH(self):
        return self.__BENCH
    
    
    def set_BENCH(self,BENCH):
        self.__BENCH = BENCH
       

    def get_P(self):
        return self.__P
    

    def get_n_ieq_constr(self):
        return self.__n_ieq_constr
    

    def get_BENCH_Nvar(self):
        return self.__BENCH_Nvar
    
        
    def set_FILE(self,file):
        self.__file = file
    

    def get_FILE(self):
        return self.__file

    
    
    

    

    

    
    
    
    
 
  

  
