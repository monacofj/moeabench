# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .runner import runner

class RUN(runner):    

    def MOEA_execute(self,result, name_moea = None, name_benchmark=None): 
            result.edit_DATA_conf().get_DATA_MOEA().exec()
          
          
            
        
            

    


        





    