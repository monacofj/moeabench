# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-FileCopyrightText: 2025 Silva F. F. <fernandoferreira.silva42@usp.br>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go
import numpy as np
from ..defaults import defaults
from ..view.style import MOEABENCH_PALETTE

try:
    import google.colab
    from google.colab import output
    output.enable_custom_widget_manager()
    import plotly.io as pio
    pio.renderers.default = "colab"
except ImportError:
    pass

class Scatter2D:
    def __init__(self, names, data_arrays, axis, type='pareto-optimal front', mode='interactive', axis_label='Objective', trace_modes=None, **kwargs):
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
        self.ax = kwargs.get('ax', None)
        self.show_plot = kwargs.get('show', True)
        self.marker_styles = kwargs.get('marker_styles', [None] * len(names))
        self.figure = go.Figure()
        self._build()

    def _build(self):
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

    def show(self):
        if not self.show_plot: return
        mode = self.mode
        if defaults.backend == 'matplotlib': mode = 'static'
        elif defaults.backend == 'plotly': mode = 'interactive'
        
        if mode == 'static':
            import matplotlib.pyplot as plt
            plt.show()
        else:
            self.figure.show()

    def configure_static(self):
        import matplotlib.pyplot as plt
        
        if self.ax is None:
            fig, ax = plt.subplots(figsize=defaults.figsize)
        else:
            ax = self.ax
            fig = ax.get_figure()
        
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
                    style = self.marker_styles[i].copy() if self.marker_styles[i] is not None else {}
                    
                    # Retrieve explicit static color if set, else fallback to prop_cycle
                    opt_color = style.get('color', current_color)
                    
                    if 'symbol' in style and isinstance(style['symbol'], (list, np.ndarray)):
                        symbols = np.array(style['symbol'])
                        sizes = np.array(style['size']) if 'size' in style else np.full(len(ax_data), 20)
                        
                        # Note: 2D drawing requires sorting logic to be independent of scatter 
                        # but markers themselves don't strictly need sorting except for Z-order overlap.
                        for symbol_type in ['circle', 'circle-open', 'diamond-open']:
                            sub_msk = (symbols == symbol_type) & msk
                            if np.any(sub_msk):
                                plt_marker = 'o'
                                plt_fc = opt_color
                                plt_ec = opt_color
                                plt_size = sizes[sub_msk]
                                
                                if symbol_type == 'circle-open':
                                    plt_marker = 'o'
                                    plt_fc = 'none'
                                elif symbol_type == 'diamond-open':
                                    plt_marker = 'D'
                                    plt_fc = 'none'
                                elif symbol_type == 'circle':
                                    # Solid markers: no border
                                    plt_ec = 'none'
                                
                                # Scale semantic sizes for Matplotlib (Plotly 6 ~ MPL 24)
                                plt_size = np.array(sizes[sub_msk]) * 4 if symbol_type == 'circle' else np.array(sizes[sub_msk]) * 3.5
                                
                                ax.scatter(ax_data[sub_msk], ay_data[sub_msk], 
                                           label=label if symbol_type == 'circle' and 'lines' not in t_mode else None, 
                                           facecolors=plt_fc, edgecolors=plt_ec,
                                           marker=plt_marker, s=plt_size,
                                           linewidths=1.5 if symbol_type != 'circle' else 0)
                    else:
                        custom_marker = style.get('symbol', 'o')
                        # Sync with topo_shape solid markers (Plotly 6 -> Matplotlib 24)
                        custom_size = style.get('size', 24)
                        if custom_marker == 'circle': custom_marker = 'o'
                        
                        kwa = {'label': label if 'lines' not in t_mode else None, 
                               'color': opt_color, 'marker': custom_marker}
                        if custom_size is not None:
                            kwa['s'] = custom_size
                            
                        ax.scatter(ax_data[msk], ay_data[msk], **kwa)
        
        ax.set_xlabel(f"{self.axis_label} {self.axis[0]+1}")
        ax.set_ylabel(f"{self.axis_label} {self.axis[1]+1}")
        ax.set_title(f"2D Chart for {self.type}")
        ax.grid(True, linestyle='--', alpha=0.7) # Grid density is standard
        ax.legend()
        
        if defaults.save_format:
            filename = f"mb_plot_{self.type.replace(' ', '_')}.{defaults.save_format}"
            plt.savefig(filename, dpi=defaults.dpi, bbox_inches='tight')
            # print(f"[MoeaBench] Plot saved as {filename}")
        
        if self.show_plot and self.ax is None:
            plt.show()

    def configure_interactive(self):
        if not hasattr(self, 'figure'):
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

                # Custom markers support
                style = self.marker_styles[i].copy() if self.marker_styles[i] is not None else {}
                
                # Plotly trace-splitting requires explicit color management
                opt_color = style.get('color', MOEABENCH_PALETTE[i % len(MOEABENCH_PALETTE)])
                
                # Check for individualized quality markers (list of symbols)
                if 'symbol' in style and isinstance(style['symbol'], (list, np.ndarray)):
                    symbols = np.array(style['symbol'])
                    sizes = np.array(style['size']) if 'size' in style else np.full(len(ax), 6)
                    
                    for symbol_type in ['circle', 'circle-open', 'diamond-open', 'cross-open']:
                        sub_msk = (symbols == symbol_type) & msk
                        if np.any(sub_msk):
                            sub_marker = dict(size=sizes[sub_msk], color=opt_color)
                            # Style based on audit report aesthetics
                            if symbol_type == 'circle-open':
                                 sub_marker.update(dict(symbol='circle-open', line=dict(width=2.0, color=opt_color)))
                            elif symbol_type == 'diamond-open':
                                 sub_marker.update(dict(symbol='diamond-open', line=dict(width=2.0, color=opt_color)))
                            else:
                                 # Solid markers: no border
                                 sub_marker.update(dict(symbol='circle', opacity=1.0, line=dict(width=0)))
                            
                            self.figure.add_trace(go.Scatter(
                                x=ax[sub_msk], y=ay[sub_msk],
                                mode=self.trace_modes[i],
                                marker=sub_marker,
                                name=f'{self.experiments[i]}',
                                legendgroup=f'{self.experiments[i]}',
                                showlegend=(symbol_type == 'circle'), # Only show one in legend
                                hovertemplate=(f"{self.experiments[i]}<br>"
                                               f"{self.axis_label} {self.axis[0]+1}: %{{x}}<br>"
                                               f"{self.axis_label} {self.axis[1]+1}: %{{y}}<br><extra></extra>"),
                            ))
                else:
                    # Static/Standard Marker
                    # Sync with topo_shape solid markers (size 6)
                    marker_config = dict(size=6, line=dict(width=0))
                    marker_config.update(style)
                    if 'color' not in marker_config:
                         marker_config['color'] = MOEABENCH_PALETTE[i % len(MOEABENCH_PALETTE)]

                    self.figure.add_trace(go.Scatter(
                        x=x_sorted if 'lines' in p_mode else ax[msk],
                        y=y_sorted if 'lines' in p_mode else ay[msk],
                        mode=p_mode,
                        line=dict(shape=line_shape) if line_shape else None,
                        marker=marker_config,
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
