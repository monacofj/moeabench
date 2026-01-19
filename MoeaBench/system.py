# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import importlib.util

def version() -> str:
    """Returns the current MoeaBench version."""
    return "0.6.1"

def check_dependencies():
    """Prints a report of installed optional dependencies."""
    deps = ["pymoo", "deap", "plotly", "pandas", "numpy", "scipy", "tqdm"]
    print("--- MoeaBench Dependency Check ---")
    for dep in deps:
        status = "Installed" if importlib.util.find_spec(dep) else "NOT FOUND"
        print(f"{dep:<15}: {status}")
