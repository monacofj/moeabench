# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .file import file
from joblib import load
import zipfile
from io import BytesIO
import os
import sys


class loader(file):
     
    def IPL_loader(self, folder):
      
        dir = os.path.dirname(__file__)
       
        if dir not in sys.path:
            sys.path.append(dir)

        path_z = loader.DATA(folder)
        if not path_z.exists():
            raise FileExistsError("folder not found")
       
        obj = None
        with zipfile.ZipFile(path_z, 'r') as zf:
            bytes = zf.read('Moeabench.joblib')
            obj = load(  BytesIO(bytes))

        self.__dict__.update(obj.__dict__)


       
      
 

        