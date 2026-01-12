# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

M_register = {}

class my_new_benchmark:
    """
    Template for implementing a new benchmark.
    """
    def __init__(self):
        # Placeholder for user initialization
        pass
         
    def __call__(self, default=None):
        """Standard MoeaBench instantiation entry point."""
        return self

def register_benchmark():
    def decorator(cls):
        try:
            name = cls.__name__
            M_register[name] = cls
        except Exception as e:
                print(e)
        return cls
    return decorator

def get_benchmark():
    return next(iter(M_register.values())) if len(M_register.values()) > 0 else None