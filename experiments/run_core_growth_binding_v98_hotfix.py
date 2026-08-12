from __future__ import annotations

import threading
import webbrowser

import numpy as np
from waitress import serve

import run_core_growth_binding_v98 as v98


def trace_signature(brain, spec):
    result = v98.v83.episode_experience(brain, spec, learn=False)
    direct = v98.episode_direct_nodes(brain, spec)
    nodes: set[int] = set()
    edges: set[tuple[int, int]] = set()
    activation = np.zeros(brain.node_count, dtype=float)

    for stage in ("subject", "place_relation", "context", "action_relation", "action"):
        trace = result[stage]
        nodes.update(int(x) for x in trace.activated_nodes if int(x) not in direct)
        edges.update(
            tuple(sorted((int(a), int(b))))
            for a, b in trace.traversed_edges
            if int(a) not in direct and int(b) not in direct
        )
        arr = np.asarray(trace.final_activation, dtype=float)
        activation = np.maximum(activation, arr)

    for node in direct:
        activation[int(node)] = 0.0
    return {"nodes": nodes, "edges": edges, "activation": activation}


def entity_signature(brain, world):
    parts = [trace_signature(brain, spec) for spec in world.episodes]
    nodes: set[int] = set()
    edges: set[tuple[int, int]] = set()
    activation = np.zeros(brain.node_count, dtype=float)
    for part in parts:
        nodes.update(part["nodes"])
        edges.update(part["edges"])
        activation = np.maximum(activation, part["activation"])
    return {"nodes": nodes, "edges": edges, "activation": activation}


# Patch only the two observer helpers that referenced the wrong attribute name.
# Experiment data, training, thresholds, seeds, clustering, and verdict logic remain v98 unchanged.
v98.trace_signature = trace_signature
v98.entity_signature = entity_signature


def open_browser():
    webbrowser.open(f"http://{v98.HOST}:{v98.PORT}")


if __name__ == "__main__":
    print(f"Core Growth Binding v98 (hotfix): http://{v98.HOST}:{v98.PORT}")
    print("Hotfix scope: observer attribute n_nodes -> node_count only; experiment logic unchanged.")
    threading.Timer(0.8, open_browser).start()
    serve(v98.app, host=v98.HOST, port=v98.PORT)
