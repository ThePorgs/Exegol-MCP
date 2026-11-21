"""Serve an ASGI app with a real uvicorn server on 127.0.0.1, in a background thread, for the HTTP legs.

The socket is bound before uvicorn starts (port 0, so the OS picks a free port) and handed to it, so there is no
window in which another process could take the port. uvicorn runs its own event loop in the thread, like the real
`exegol-mcp` process does, so a blocked loop on the server side cannot be hidden by the test's loop.
"""

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import uvicorn
from starlette.types import ASGIApp

from exegol_mcp.bootstrap.cli import HTTP_HOST
from exegol_mcp.bootstrap.transports import UVICORN_LOG_CONFIG

STARTUP_TIMEOUT = 15.0


def free_port() -> int:
    """A currently free TCP port on 127.0.0.1 (for subprocess launches that bind it themselves)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HTTP_HOST, 0))
        return int(sock.getsockname()[1])


@contextmanager
def serve_in_thread(app: ASGIApp) -> Iterator[int]:
    """Serve `app` on 127.0.0.1:<free port> with uvicorn in a daemon thread; yield the port; stop it on exit."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HTTP_HOST, 0))
    port = int(sock.getsockname()[1])
    config = uvicorn.Config(app, host=HTTP_HOST, port=port, log_config=UVICORN_LOG_CONFIG, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("uvicorn did not start")
            time.sleep(0.02)
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=STARTUP_TIMEOUT)
        sock.close()
