# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .result_population import result_population


class result_obj(result_population):
      
    def IPL_objectives(self, result, generation):
        return self.DATA([dt.get_F_GEN() 
                          for data in result.get_elements() 
                          for dt in data 
                          if hasattr(dt,"get_F_GEN")][0],generation)
        
    

