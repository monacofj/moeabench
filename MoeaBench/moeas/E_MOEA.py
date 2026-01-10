# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from enum import Enum

class E_MOEA(Enum):
  NSGA3 = 0
  SPEA2 = 1
  U_NSGA3 = 2
  MOEAD = 3
  RVEA = 4
  my_new_moea = 5
  