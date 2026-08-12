from __future__ import annotations
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from brain import SphereBrain


def build_html(
    brain: SphereBrain,
    output_path: str | Path,
    highlighted_edges=None,
    highlighted_nodes=None,
    title="Sphere Brain",
) -> None:
    highlighted_edges = {tuple(sorted(e)) for e in (highlighted_edges or [])}
    highlighted_nodes = set(highlighted_nodes or [])

    upper = np.triu_indices(brain.node_count, k=1)
    mask = brain.adjacency[upper]
    edges = list(zip(upper[0][mask], upper[1][mask]))
    edges.sort(key=lambda e: brain.weights[e[0], e[1]], reverse=True)
    edges = edges[:700]

    nx, ny, nz = [], [], []
    hx, hy, hz = [], [], []

    for a, b in edges:
        target = (hx, hy, hz) if tuple(sorted((int(a), int(b)))) in highlighted_edges else (nx, ny, nz)
        target[0].extend([brain.positions[a,0], brain.positions[b,0], None])
        target[1].extend([brain.positions[a,1], brain.positions[b,1], None])
        target[2].extend([brain.positions[a,2], brain.positions[b,2], None])

    u = np.linspace(0, 2*np.pi, 36)
    v = np.linspace(0, np.pi, 36)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=sx, y=sy, z=sz, opacity=0.05, showscale=False, hoverinfo="skip",
        colorscale=[[0, "#dbeafe"], [1, "#dbeafe"]]
    ))
    fig.add_trace(go.Scatter3d(
        x=nx, y=ny, z=nz, mode="lines",
        line={"width": 1, "color": "rgba(80,100,130,0.16)"},
        hoverinfo="skip", name="経路"
    ))
    if hx:
        fig.add_trace(go.Scatter3d(
            x=hx, y=hy, z=hz, mode="lines",
            line={"width": 5, "color": "rgba(238,96,52,0.9)"},
            hoverinfo="skip", name="活性経路"
        ))

    colors = [1 if i in highlighted_nodes else 0 for i in range(brain.node_count)]
    sizes = [7 if i in highlighted_nodes else 4 for i in range(brain.node_count)]

    fig.add_trace(go.Scatter3d(
        x=brain.positions[:,0], y=brain.positions[:,1], z=brain.positions[:,2],
        mode="markers",
        marker={
            "size": sizes,
            "color": colors,
            "colorscale": [[0, "#355c7d"], [1, "#ff6b35"]],
            "showscale": False,
            "opacity": 0.92,
        },
        text=[f"ノード {i}<br>使用回数 {brain.node_usage[i]}" for i in range(brain.node_count)],
        hoverinfo="text",
        name="ノード",
    ))

    fig.update_layout(
        title=title,
        scene={
            "xaxis": {"visible": False, "range": [-1.1,1.1]},
            "yaxis": {"visible": False, "range": [-1.1,1.1]},
            "zaxis": {"visible": False, "range": [-1.1,1.1]},
            "aspectmode": "cube",
        },
        margin={"l":0,"r":0,"t":45,"b":0},
    )
    fig.write_html(output_path, include_plotlyjs=True, auto_open=False)
