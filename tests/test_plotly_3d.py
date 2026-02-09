import plotly.graph_objects as go
import numpy as np

fig = go.Figure(data=[go.Scatter3d(
    x=[1, 2, 3], y=[1, 2, 3], z=[1, 2, 3],
    mode='markers',
    marker=dict(
        size=10,
        color='rgba(0,0,0,0)',
        line=dict(color='red', width=5),
        symbol='circle'
    )
)])

try:
    fig.show()
    print("Success: Scatter3d accepted color='rgba(0,0,0,0)' and line.")
except Exception as e:
    print(f"Error: {e}")

# Check symbols
symbols = ['circle', 'circle-open', 'cross', 'x', 'diamond', 'diamond-open']
for s in symbols:
    try:
        go.Scatter3d(x=[0], y=[0], z=[0], marker=dict(symbol=s))
        print(f"Symbol '{s}' is valid.")
    except Exception as e:
        print(f"Symbol '{s}' is INVALID: {e}")
