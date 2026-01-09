from typing import TypeVar, List, Generic
from .I_DATA_arr import I_DATA_arr


T = TypeVar('T')

class DATA_arr(Generic[T],I_DATA_arr):
    
    def __init__(self, list_g,**kwargs):
       self.elements: List[T] = list_g
       super().__init__(**kwargs)


    def get_elements(self):
        return self.elements
    

    def clear(self):
        self.elements.clear()
         

    def add_T(self, element: List[T]):
        self.elements.append(element)


   


    

    
                     




