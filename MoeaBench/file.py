# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .IPL_MoeaBench import IPL_MoeaBench
from pathlib import Path


class file(IPL_MoeaBench):

    @staticmethod
    def DATA(folder, path_folder = 'analysis', extension='zip'):
        base = Path.cwd()
        dir_z = base  / path_folder
        dir_z.mkdir(parents=True, exist_ok = True)
        return dir_z / f'{folder}.{extension}'
    



    

                
                
                
                
                
            
       
        
        
       
     
            
        

   



