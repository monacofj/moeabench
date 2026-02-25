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

# Remove legacy inheritance
# from .analyse_pareto import analyse_pareto

try:
    import google.colab
    from google.colab import output
    output.enable_custom_widget_manager()
    import plotly.io as pio
    pio.renderers.default = "colab"
except ImportError:
    pass


class Scatter3D:
     def __init__(self, names, data_arrays, axis, type = 'pareto-optimal front', mode='interactive', axis_label='Objective', trace_modes=None, **kwargs):
         """
         names: list of names for legend
         data_arrays: list of numpy arrays (Nx3 or more)
         axis: list of 3 indices [x, y, z]
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
         import matplotlib.cm as cm
         
         if self.ax is None:
             fig = plt.figure(figsize=defaults.figsize)
             try:
                 ax = fig.add_subplot(111, projection='3d')
             except (ValueError, KeyError) as e:
                 plt.close(fig)
                 print("\n[WARNING] MoeaBench: Matplotlib '3d' projection not available. Falling back to 2D visualization (f1, f2).")
                 try:
                     from .scatter2d import Scatter2D
                     s2d = Scatter2D(self.experiments, self.vet_pts, self.axis[:2], 
                                    type=self.type, mode=self.mode, 
                                    axis_label=self.axis_label, trace_modes=self.trace_modes)
                     s2d.show()
                     return
                 except Exception as e2:
                     raise RuntimeError(
                         "MoeaBench Error: Matplotlib 3D projection '3d' is not available and 2D fallback failed. "
                         "This often happens due to broken Matplotlib installations."
                     ) from e2
         else:
             ax = self.ax
             fig = ax.get_figure()
         
         # Use standard property cycle for distinct categorical colors
         prop_cycle = plt.rcParams['axes.prop_cycle']
         cycle_colors = prop_cycle.by_key()['color']
         
         for i in range(0, len(self.vet_pts)):
            ax_data = self.vet_pts[i][:,self.axis[0]]
            ay_data = self.vet_pts[i][:,self.axis[1]]
            az_data = self.vet_pts[i][:,self.axis[2]]
            
            msk = ~(np.isnan(ax_data) | np.isnan(ay_data) | np.isnan(az_data))
            if np.any(msk):
                # Pick color from cycle
                current_color = cycle_colors[i % len(cycle_colors)]
                label = f'{self.experiments[i]}'
                
                t_mode = self.trace_modes[i]
                style = self.marker_styles[i].copy() if self.marker_styles[i] is not None else {}
                
                # Retrieve explicit static color if set, else fallback to prop_cycle
                opt_color = style.get('color', current_color)
                
                if 'markers' in t_mode:
                     if 'symbol' in style and isinstance(style['symbol'], (list, np.ndarray)):
                         symbols = np.array(style['symbol'])
                         sizes = np.array(style['size']) if 'size' in style else np.full(len(ax_data), 20)
                         
                         for symbol_type in ['circle', 'circle-open', 'diamond-open']:
                             sub_msk = (symbols == symbol_type) & msk
                             if np.any(sub_msk):
                                 plt_marker = 'o'
                                 plt_fc = opt_color
                                 plt_ec = opt_color
                                 # Scale semantic sizes for Matplotlib (Plotly 6 ~ MPL 20)
                                 plt_size = np.array(sizes[sub_msk]) * 4 if symbol_type == 'circle' else np.array(sizes[sub_msk]) * 3
                                 
                                 if symbol_type == 'circle-open':
                                     plt_marker = 'o'
                                     plt_fc = 'none'
                                 elif symbol_type == 'diamond-open':
                                     plt_marker = 'D'
                                     plt_fc = 'none'
                                 elif symbol_type == 'circle':
                                     # Solid markers: no border
                                     plt_ec = 'none'
                                 
                                 ax.scatter(ax_data[sub_msk], ay_data[sub_msk], az_data[sub_msk], 
                                            label=label if symbol_type == 'circle' else None, 
                                            facecolors=plt_fc, edgecolors=plt_ec,
                                            marker=plt_marker, s=plt_size,
                                            linewidths=1.5 if symbol_type != 'circle' else 0)
                     else:
                         custom_marker = style.get('symbol', 'o')
                         custom_size = style.get('size', 20)
                         if custom_marker == 'circle': custom_marker = 'o'
                         
                         kwa = {'label': label, 'color': opt_color, 'marker': custom_marker}
                         if custom_size is not None:
                             kwa['s'] = custom_size
                             
                         ax.scatter(ax_data[msk], ay_data[msk], az_data[msk], **kwa)
                
                if 'lines' in t_mode:
                     ax.plot(ax_data[msk], ay_data[msk], az_data[msk], 
                             label=label if 'markers' not in t_mode else None, 
                             color=opt_color)
        
         ax.set_xlabel(f"{self.axis_label} {self.axis[0]+1}")
         ax.set_ylabel(f"{self.axis_label} {self.axis[1]+1}")
         ax.set_zlabel(f"{self.axis_label} {self.axis[2]+1}")
         ax.set_title(f"3D Chart for {self.type}")
         ax.legend()
         
         if defaults.save_format:
             filename = f"mb_plot_{self.type.replace(' ', '_')}.{defaults.save_format}"
             plt.savefig(filename, dpi=defaults.dpi, bbox_inches='tight')

         if self.show_plot and self.ax is None:
             plt.show()

     def configure_interactive(self):
         if not hasattr(self, 'figure'):
             self.figure=go.Figure()
         for i in range(len(self.vet_pts)):
                ax_data = self.vet_pts[i][:,self.axis[0]]
                ay_data = self.vet_pts[i][:,self.axis[1]]
                az_data = self.vet_pts[i][:,self.axis[2]]
                msk = ~(np.isnan(ax_data) | np.isnan(ay_data) | np.isnan(az_data))
                
                if np.any(msk):
                    style = self.marker_styles[i].copy() if self.marker_styles[i] is not None else {}
                    
                    # Plotly trace-splitting requires explicit color management
                    opt_color = style.get('color', MOEABENCH_PALETTE[i % len(MOEABENCH_PALETTE)])
                    
                    # Plotly Scatter3d does NOT support arrays for 'symbol' or 'opacity'
                    if 'symbol' in style and isinstance(style['symbol'], (list, np.ndarray)):
                         symbols = np.array(style['symbol'])
                         sizes = np.array(style['size']) if 'size' in style else np.full(len(ax_data), 6)
                         
                         for symbol_type in ['circle', 'circle-open', 'diamond-open']:
                              sub_msk = (symbols == symbol_type) & msk
                              if np.any(sub_msk):
                                   sub_marker = dict(size=sizes[sub_msk], color=opt_color)
                                   if symbol_type == 'circle-open':
                                        sub_marker.update(dict(symbol='circle-open', line=dict(width=2.0, color=opt_color)))
                                   elif symbol_type == 'diamond-open':
                                        sub_marker.update(dict(symbol='diamond-open', line=dict(width=2.0, color=opt_color)))
                                   else:
                                        # Solid markers: no border, sync size 6
                                        sub_marker.update(dict(symbol='circle', opacity=1.0, line=dict(width=0)))
                                   
                                   self.figure.add_trace(go.Scatter3d(
                                       x=ax_data[sub_msk], y=ay_data[sub_msk], z=az_data[sub_msk],
                                       mode=self.trace_modes[i],
                                       marker=sub_marker,
                                       name=f'{self.experiments[i]}',
                                       legendgroup=f'{self.experiments[i]}',
                                       showlegend=(symbol_type == 'circle'), # Only show one in legend
                                       hovertemplate = (f"{self.experiments[i]}<br>"
                                                        f"{self.axis_label} {self.axis[0]+1}: %{{x}}<br>"
                                                        f"{self.axis_label} {self.axis[1]+1}: %{{y}}<br>"
                                                        f"{self.axis_label} {self.axis[2]+1}: %{{z}}<br><extra></extra>"),
                                   ))
                    else:
                        # Static/Standard Marker
                        # Sync with topo_shape solid markers (size 6)
                        marker_config = dict(size=6, line=dict(width=0))
                        marker_config.update(style)
                        if 'color' not in marker_config:
                             marker_config['color'] = opt_color

                        self.figure.add_trace(go.Scatter3d(
                            x=ax_data[msk], y=ay_data[msk], z=az_data[msk],
                            mode=self.trace_modes[i],
                            marker=marker_config,
                            name=f'{self.experiments[i]}',                       
                            showlegend=True,
                            hovertemplate = (f"{self.experiments[i]}<br>"
                                             f"{self.axis_label} {self.axis[0]+1}: %{{x}}<br>"
                                             f"{self.axis_label} {self.axis[1]+1}: %{{y}}<br>"
                                             f"{self.axis_label} {self.axis[2]+1}: %{{z}}<br><extra></extra>"),
                        ))

         self.figure.update_layout(
                template="moeabench",
                scene = dict(
                    xaxis=dict(title=f"{self.axis_label} {self.axis[0]+1}", showbackground=True, backgroundcolor="aliceblue", showgrid=True, gridcolor="#C3BDBD"),
                    yaxis=dict(title=f"{self.axis_label} {self.axis[1]+1}", showbackground=True, backgroundcolor="aliceblue", showgrid=True, gridcolor="#C3BDBD"),
                    zaxis=dict(title=f"{self.axis_label} {self.axis[2]+1}", showbackground=True, backgroundcolor="aliceblue", showgrid=True, gridcolor="#C3BDBD"),
                    aspectmode='manual',
                    aspectratio=dict(x=1,y=1,z=1)
                 ),
                 
                 width=defaults.plot_width,
                 height=defaults.plot_height,
                 margin=dict(l=0,r=0,b=0,t=0),
                 title=dict(
                     text=f'3D Chart for {self.type}',
                     x=0.5,
                     xanchor='center',
                     y=0.9,
                     yanchor='bottom',
                     pad=dict(t=0),
                     font=dict(size=16,weight='bold')
                 ),
                 legend=dict(
                    x=1.05,
                    y=0.5,
                    xanchor='left',
                    yanchor='middle'
                ),
                hovermode='closest'
            )