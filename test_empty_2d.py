import json
with open("test_visual_markers.html", "r") as f:
    html = f.read()

# Try to extract the Plotly config block to see if it's 'scatter3d'
import re
match = re.search(r'"type":"(scatter3d|scatter)"', html)
if match:
    print(f"Trace type found: {match.group(1)}")
else:
    print("No trace type found")
    
traces = re.findall(r'"type":"[^"]+"', html)
print(set(traces))
