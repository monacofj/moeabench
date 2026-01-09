class allowed_benchmark:    
        
    def K_validate(self,K_N):
        if not K_N > 0:
            raise ValueError("this value of 'k' is not valid, it must be greater than 0")
        return True
    

    def P_validate(self,P):
        if not P > 0:
            raise ValueError("this value of 'P' is not valid, it must be greater than 0")
        return True
    

    def M_validate(self,M):
        if not  M > 2:
            raise ValueError("this value of 'M' is not valid, it must be greater or equal than 2" )
        return True
    

    def N_validate(self,N):
        if not N >= 5:
            raise ValueError("this value of 'N' is not valid, it must be greater or equal than 5" )
        return True
    
    
    def MN_validate(self,K_N,M,D):
        if not M <= D+K_N-1:
            raise ValueError("this value of 'M' is not valid, it must be lass or equal than N")
        return True
    
    
    def MN1_validate(self,M,D):
        if not M > D > 1:
            raise ValueError("The value of 'D' must be greater than 1 and less than 'M'")
        return True
    

    

    
    

    
    
    
    


       


   
    

