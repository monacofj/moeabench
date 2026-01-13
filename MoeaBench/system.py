# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import importlib.util

def cpus(safe: bool = False) -> int:
    """
    Returns the number of logical CPU cores.
    If safe=True, returns cpus - 1 to preserve system responsiveness.
    """
    count = os.cpu_count() or 1
    if safe and count > 1:
        return count - 1
    return count

def version() -> str:
    """Returns the current MoeaBench version."""
    return "0.1.0"

def check_dependencies():
    """Prints a report of installed optional dependencies."""
    deps = ["pymoo", "deap", "plotly", "pandas", "numpy", "scipy", "tqdm"]
    print("--- MoeaBench Dependency Check ---")
    for dep in deps:
        status = "Installed" if importlib.util.find_spec(dep) else "NOT FOUND"
        print(f"{dep:<15}: {status}")

def memory() -> dict:
    """Returns basic system memory information (Linux/Darwin)."""
    info = {"total_gb": None, "available_gb": None}
    try:
        if sys.platform == "linux":
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if "MemTotal" in line:
                        info["total_gb"] = round(int(line.split()[1]) / (1024*1024), 2)
                    if "MemAvailable" in line:
                        info["available_gb"] = round(int(line.split()[1]) / (1024*1024), 2)
        elif sys.platform == "darwin":
            # Very basic fallback for macOS
            import subprocess
            res = subprocess.check_output(['sysctl', 'hw.memsize']).decode().split()[1]
            info["total_gb"] = round(int(res) / (1024**3), 2)
    except Exception:
        pass
    return info
