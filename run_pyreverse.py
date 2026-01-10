# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import subprocess

format = "png"
target_folder = "."

cmd = [
    "pyreverse",
    "-o", format,
    "-p", "EVO_NEW",
    "-a","10",
    "C:\\MoeaBench\\MoeaBench"
]

print("Executando:", " ".join(cmd))
subprocess.run(cmd, check=True)