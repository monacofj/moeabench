# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .result_population import result_population


class result_var(result_population):
  
    def IPL_variables(self, result, generation):
        return self.DATA([dt.get_X_GEN() 
                          for data in result.get_elements() 
                          for dt in data 
                          if hasattr(dt,"get_X_GEN")][0],generation)
      

       

    