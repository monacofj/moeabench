# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .runner import runner


class RUN_user(runner):

     def MOEA_execute(self,result,name_moea,name_benchmark):
          problem = result.edit_DATA_conf().get_problem()
          data = result.edit_DATA_conf().get_DATA_MOEA().evaluation()
          result.DATA_store(name_moea,
                            result.edit_DATA_conf().get_DATA_MOEA().generations,
                            result.edit_DATA_conf().get_DATA_MOEA().population,
                            data[2],
                            data[0],
                            data[1],
                            problem,
                            name_benchmark,
                            data[3],
                            data[4],
                            data[5],
                            data[6])
      
    
         
           

        