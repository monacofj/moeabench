# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later




class repository:
        
        def __init__(self, algorithm):
            self.algorithm = algorithm

        
        def __call__(self):
          moea = self.algorithm.get_CACHE()
          moea.get_DATA_conf().set_DATA_MOEA(self.algorithm,self.algorithm.get_problem()) 
          return (moea,self.algorithm.__class__,self.algorithm.__class__.__name__)