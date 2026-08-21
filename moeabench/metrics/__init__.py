# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .evaluator import (
    hypervolume,
    ordinal_hypervolume,
    igd, 
    gd,
    gdplus,
    igdplus,
    emd,
    MetricMatrix,
    front_ratio
)

# Aliases for convenience
hv = hypervolume
ohv = ordinal_hypervolume

__all__ = [
    "hypervolume",
    "hv",
    "ordinal_hypervolume",
    "ohv",
    "igd",
    "gd",
    "gdplus",
    "igdplus",
    "emd",
    "MetricMatrix",
    "front_ratio",
]

for _name in (
    "evaluator",
    "GEN_hypervolume",
    "GEN_mc_hypervolume",
    "GEN_ordinal_hypervolume",
    "GEN_mc_ordinal_hypervolume",
    "GEN_igd",
    "GEN_gd",
    "GEN_igdplus",
    "GEN_gdplus",
):
    globals().pop(_name, None)
