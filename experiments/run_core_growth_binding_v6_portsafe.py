from __future__ import annotations

import socket
import threading
import webbrowser

from waitress import serve

import run_core_growth_binding_v6 as v6


def choose_port(host: str, start: int = 5040, end: int = 5060) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No available local port found in {start}-{end}.")


def open_browser(host: str, port: int) -> None:
    webbrowser.open(f"http://{host}:{port}")


if __name__ == "__main__":
    port = choose_port(v6.HOST)
    threading.Timer(1.0, open_browser, args=(v6.HOST, port)).start()
    print(f"Core Growth Binding v6: http://{v6.HOST}:{port}")
    if port != v6.PORT:
        print(f"Port {v6.PORT} was unavailable, so port {port} was selected automatically.")
    print("Existing-edge recall path comparison / no teacher / no reward")
    serve(v6.app, host=v6.HOST, port=port)
