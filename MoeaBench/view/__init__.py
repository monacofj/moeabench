# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .plotters import spaceplot, timeplot, rankplot, casteplot, tierplot
from .style import apply_style

# Initialize the MoeaBench visual identity (Ocean Palette)
apply_style()

__all__ = ["spaceplot", "timeplot", "rankplot", "casteplot", "tierplot"]
