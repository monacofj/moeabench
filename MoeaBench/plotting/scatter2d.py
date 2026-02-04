# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go
import numpy as np
from ..defaults import defaults

try:
    import google.colab
    from google.colab import output
    output.enable_custom_widget_manager()
    import plotly.io as pio
    pio.renderers.default = "colab"
except ImportError:
    pass

class Scatter2D:
    def __init__(self, names, data_arrays, axis, type='pareto-optimal front', mode='interactive', axis_label='Objective', trace_modes=None):
        """
        names: list of names for legend
        data_arrays: list of numpy arrays (Nx2 or more)
        axis: list of 2 indices [x, y]
        """
        self.vet_pts = data_arrays
        self.experiments = names
        self.axis = axis
        self.type = type
        self.mode = mode
        self.axis_label = axis_label
        self.trace_modes = trace_modes if trace_modes else ['markers'] * len(names)

    def show(self):
        # Honor global backend override
        mode = self.mode
        if defaults.backend == 'matplotlib':
            mode = 'static'
        elif defaults.backend == 'plotly':
            mode = 'interactive'

        if mode == 'static':
            self.configure_static()
        else:
            self.configure_interactive()

    def configure_static(self):
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=defaults.figsize)
        
        prop_cycle = plt.rcParams['axes.prop_cycle']
        cycle_colors = prop_cycle.by_key()['color']
        
        for i in range(len(self.vet_pts)):
            ax_data = self.vet_pts[i][:, self.axis[0]]
            ay_data = self.vet_pts[i][:, self.axis[1]]
            
            msk = ~(np.isnan(ax_data) | np.isnan(ay_data))
            if np.any(msk):
                current_color = cycle_colors[i % len(cycle_colors)]
                label = f'{self.experiments[i]}'
                t_mode = self.trace_modes[i]
                
                # Sort for cleaner line plotting (staircase effect often used in EAF)
                indices = np.argsort(ax_data[msk])
                x_sorted = ax_data[msk][indices]
                y_sorted = ay_data[msk][indices]

                if 'lines' in t_mode:
                    # For topo_attain/Pareto, draw_style='steps-post' is often better
                    ax.plot(x_sorted, y_sorted, label=label, color=current_color, drawstyle='steps-post')
                
                if 'markers' in t_mode:
                    ax.scatter(ax_data[msk], ay_data[msk], label=label if 'lines' not in t_mode else None, color=current_color)
        
        ax.set_xlabel(f"{self.axis_label} {self.axis[0]+1}")
        ax.set_ylabel(f"{self.axis_label} {self.axis[1]+1}")
        ax.set_title(f"2D Chart for {self.type}")
        ax.grid(True, linestyle='--', alpha=0.7) # Grid density is standard
        ax.legend()
        
        if defaults.save_format:
            filename = f"mb_plot_{self.type.replace(' ', '_')}.{defaults.save_format}"
            plt.savefig(filename, dpi=defaults.dpi, bbox_inches='tight')
            # print(f"[MoeaBench] Plot saved as {filename}")

        plt.show()

    def configure_interactive(self):
        self.figure = go.Figure()
        for i in range(len(self.vet_pts)):
            ax = self.vet_pts[i][:, self.axis[0]]
            ay = self.vet_pts[i][:, self.axis[1]]
            msk = ~(np.isnan(ax) | np.isnan(ay))
            
            if np.any(msk):
                indices = np.argsort(ax[msk])
                x_sorted = ax[msk][indices]
                y_sorted = ay[msk][indices]

                # Translate trace mode for Plotly
                p_mode = self.trace_modes[i]
                line_shape = 'hv' if 'lines' in p_mode else None # hv = horizontal-vertical steps

                self.figure.add_trace(go.Scatter(
                    x=x_sorted if 'lines' in p_mode else ax[msk],
                    y=y_sorted if 'lines' in p_mode else ay[msk],
                    mode=p_mode,
                    line=dict(shape=line_shape) if line_shape else None,
                    marker=dict(size=6 if 'lines' in p_mode else 8),
                    name=f'{self.experiments[i]}',
                    showlegend=True,
                    hovertemplate=(f"{self.experiments[i]}<br>"
                                   f"{self.axis_label} {self.axis[0]+1}: %{{x}}<br>"
                                   f"{self.axis_label} {self.axis[1]+1}: %{{y}}<br><extra></extra>"),
                ))

        self.figure.update_layout(
            xaxis=dict(title=f"{self.axis_label} {self.axis[0]+1}", showgrid=True, gridcolor="LightGray"),
            yaxis=dict(title=f"{self.axis_label} {self.axis[1]+1}", showgrid=True, gridcolor="LightGray"),
            width=defaults.plot_width,
            height=defaults.plot_height,
            title=dict(
                text=f'2D Chart for {self.type}',
                x=0.5,
                xanchor='center',
            ),
            legend=dict(
                x=1.05,
                y=0.5,
                xanchor='left',
                yanchor='middle'
            ),
            hovermode='closest',
            template=defaults.theme if defaults.theme != 'moeabench' else 'moeabench'
        )
        self.figure.show()
