from __future__ import annotations

import sys
import threading
import webbrowser
from pathlib import Path

from waitress import serve

# This script is executed directly from the experiments directory, so import
# the sibling v1 module directly instead of requiring experiments as a package.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_structural_grid_puzzle_v1 as v1


# Historical puzzle notation uses E as the end point.
v1.ROLE[v1.GOAL] = "E"

_original_candidates = v1.PuzzleSession.candidates
_original_move = v1.PuzzleSession.move
_original_auto_step = v1.PuzzleSession.auto_step


def candidates_until_end(self: v1.PuzzleSession) -> list[dict]:
    """Return no candidates after E has been reached."""
    if self.current == v1.GOAL:
        return []
    return _original_candidates(self)


def move_until_end(self: v1.PuzzleSession, target: int) -> dict:
    """Freeze the session after E has been reached."""
    if self.current == v1.GOAL:
        raise ValueError("Eに到達したため、パズルは終了しています。")
    return _original_move(self, target)


def auto_step_until_end(self: v1.PuzzleSession, mode: str) -> dict:
    """Prevent automatic movement after E has been reached."""
    if self.current == v1.GOAL:
        raise ValueError("Eに到達したため、パズルは終了しています。")
    return _original_auto_step(self, mode)


v1.PuzzleSession.candidates = candidates_until_end
v1.PuzzleSession.move = move_until_end
v1.PuzzleSession.auto_step = auto_step_until_end

# The session was created while importing v1. Re-observe it with the patched rules.
v1.session.last = v1.session.observe()

# Disable both automatic-step buttons when finished and show E explicitly.
v1.PAGE = v1.PAGE.replace(
    "function render(s){state=s;",
    "function render(s){state=s;document.querySelectorAll('.controls button:not(.secondary)').forEach(b=>b.disabled=s.finished);",
).replace(
    "`${s.turn}手でゴールしました。`",
    "`${s.turn}手でEに到達しました。ここで停止します。`",
)


def open_browser() -> None:
    webbrowser.open(f"http://{v1.HOST}:{v1.PORT}")


if __name__ == "__main__":
    v1.OUT.mkdir(parents=True, exist_ok=True)
    threading.Timer(1.0, open_browser).start()
    print(f"Structural Grid Puzzle v2: http://{v1.HOST}:{v1.PORT}")
    print("Stops at E / learning OFF / noise OFF / brain.json saveなし")
    serve(v1.app, host=v1.HOST, port=v1.PORT)
