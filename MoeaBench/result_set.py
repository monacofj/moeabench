from .result_population import result_population


class result_set(result_population):
      
    def IPL_set(self, result, generation):
        return self.DATA([dt.get_X_gen_non_dominate() 
                          for data in result.get_elements() 
                          for dt in data 
                          if hasattr(dt,"get_X_gen_non_dominate")][0],generation)