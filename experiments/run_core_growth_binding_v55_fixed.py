from __future__ import annotations

import threading
import webbrowser

from waitress import serve

import run_core_growth_binding_v44 as v44
import run_core_growth_binding_v50 as v50
import run_core_growth_binding_v55 as v55


def fixed_event_row(position: str, condition: str) -> dict[str, float]:
    """Build one v55 event-derived context row using the v44 condition-row contract.

    v44.make_scaled_report() returns the report itself, while v50.named_rows()
    expects rows shaped like v44.condition_runs(): {event_formed, report, ...}.
    """
    e, p = v55.condition_map()[condition]
    report = v44.make_scaled_report(position, e, p)
    wrapped = {
        "condition": condition,
        "echo_scale": e,
        "position_scale": p,
        "event_formed": bool(report.get("event_formed")),
        "report": report,
    }
    named = v50.named_rows([wrapped])
    if len(named) != 1:
        raise RuntimeError(f"Expected Contact Event for {position}/{condition}")
    return named[0]


# Patch only the broken adapter. All v55 experiment logic remains unchanged.
v55.event_row = fixed_event_row


def open_browser() -> None:
    webbrowser.open(f"http://{v55.HOST}:{v55.PORT}")


if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    print(f"Core Growth Binding v55: http://{v55.HOST}:{v55.PORT}")
    print("Repeated Mixed Experience Stability / fixed v44->v50 row adapter / no Core changes")
    serve(v55.app, host=v55.HOST, port=v55.PORT)
