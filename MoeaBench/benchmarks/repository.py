# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

class repository:
        
        def __init__(self, bench):
             self.bench = bench
             
        
        def __call__(self):
          samples = self.bench.benchmark.POFsamples()
          self.bench.get_CACHE().DATA_store(self.bench.__class__.__name__,self.bench.get_type(),self.bench.get_M(),self.bench.get_N(),self.bench.get_n_ieq_constr(),samples,self.bench.get_P() ,self.bench.get_K())
          return self.bench
        
         
  

        

    