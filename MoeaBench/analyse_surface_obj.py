# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .plot_surface_3D import plot_surface_3D

class analyse_surface_obj(plot_surface_3D):

    def IPL_plot_3D(args, objectives):
        benk, data = analyse_surface_obj.extract_pareto_result(args)
        axis =  [i for i in range(0,3)]    if len(objectives) == 0 else [i-1 if i > 0 else 0 for i in objectives] 
        analyse_surface_obj.allowed_obj(objectives, data, benk)
        plot_surface_3D_obj =  analyse_surface_obj(benk, data, axis)
        plot_surface_3D_obj.configure()