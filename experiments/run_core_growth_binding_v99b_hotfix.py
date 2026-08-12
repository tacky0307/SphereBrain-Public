from __future__ import annotations

import threading
import webbrowser

from waitress import serve

import run_core_growth_binding_v99b as v99b


_original_run_mode = v99b.v99.run_mode


def run_mode_without_brain(seed: int, mode: str) -> dict:
    """Preserve v99 run logic but remove the non-JSON SphereBrain object from its result."""
    result = _original_run_mode(seed, mode)
    return {k: v for k, v in result.items() if k != "brain"}


# v99B uses v99.run_mode only for matched_novel trials. Removing `brain` here
# changes output serialization only; experiment logic and measurements are unchanged.
v99b.v99.run_mode = run_mode_without_brain


def open_browser():
    webbrowser.open(f"http://{v99b.HOST}:{v99b.PORT}")


if __name__ == "__main__":
    print(f"Core Growth Binding v99B (hotfix): http://{v99b.HOST}:{v99b.PORT}")
    print("Hotfix scope: remove SphereBrain object from matched trial output before JSON serialization only.")
    threading.Timer(0.8, open_browser).start()
    serve(v99b.app, host=v99b.HOST, port=v99b.PORT)
