#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
MoeaBench Version Script
------------------------
Prints the current version of the library.
"""

from MoeaBench import mb

if __name__ == "__main__":
    print(mb.system.version())
