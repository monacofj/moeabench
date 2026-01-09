from .DATA_arr import DATA_arr
from .BENCH_conf import BENCH_conf
from .DATA_conf import DATA_conf
from .DATA_arr import DATA_arr
from .I_memory import I_memory


class memory(DATA_arr,I_memory):
        
    def __init__(self,**kwargs):
          self.data_conf=None
          super().__init__(list_g=[],**kwargs)
          
  
    def get_DATA_conf(self):
          self.data_conf=DATA_conf()
          return self.data_conf
  

    def edit_DATA_conf(self):
          return self.data_conf
       

    def get_BENCH_conf(self):
          return BENCH_conf()