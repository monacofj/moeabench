import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=[1, 2, 3], y=[1, 2, 3], z=[1, 2, 3],
    mode='markers',
    marker=dict(symbol='circle-open', size=10, line=dict(width=2))
))
fig.write_html('test_empty.html')
