# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from setuptools import setup, find_packages


setup (

    packages=find_packages(),
    install_requires = open("MoeaBench/requirements.txt").read().splitlines(),


)