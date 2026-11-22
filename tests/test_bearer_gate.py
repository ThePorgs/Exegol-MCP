"""BearerGate: v1 HTTP auth parity (400 on missing/wrong token, valid token passes, no 500). Docker-free."""

import json
from typing import Any

import httpx
import pytest
from starlette.types import Receive, Scope, Send

from exegol_mcp.bootstrap.security import AUTH_ERROR_MESSAGE, BearerGate

SECRET = "s3cret-token_value"


class RecordingApp:
    """Minimal ASGI app standing in for the MCP SDK app: answers 200 and records that it was reached."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.calls.append(scope["type"])
        if scope["type"] != "http":
            return
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"ok"})


def _client(app: RecordingApp) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=BearerGate(app, SECRET)), base_url="http://127.0.0.1")


async def _raw_request(gate: BearerGate, headers: list[tuple[bytes, bytes]]) -> tuple[int, dict[str, Any]]:
    """Drive the gate as a bare ASGI app with arbitrary header bytes (httpx would refuse non-ASCII values)."""
    scope: Scope = {"type": "http", "method": "POST", "path": "/mcp", "headers": headers, "query_string": b""}
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await gate(scope, receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, json.loads(body) if status == 400 else {}


async def test_missing_header_is_rejected_with_400() -> None:
    app = RecordingApp()
    async with _client(app) as client:
        response = await client.post("/mcp")
    assert response.status_code == 400
    assert response.json() == {"error": AUTH_ERROR_MESSAGE}
    assert app.calls == []


@pytest.mark.parametrize(
    "authorization",
    [
        "Bearer wrong-token",
        f"Basic {SECRET}",
        f"Bearer {SECRET} extra",
        "Bearer",
        SECRET,
        f"Bearer {SECRET}x",
    ],
)
async def test_wrong_credentials_are_rejected_with_400(authorization: str) -> None:
    app = RecordingApp()
    async with _client(app) as client:
        response = await client.post("/mcp", headers={"Authorization": authorization})
    assert response.status_code == 400
    assert response.json() == {"error": AUTH_ERROR_MESSAGE}
    assert app.calls == []


@pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER"])
async def test_valid_token_passes_through(scheme: str) -> None:
    app = RecordingApp()
    async with _client(app) as client:
        response = await client.post("/mcp", headers={"Authorization": f"{scheme} {SECRET}"})
    assert response.status_code == 200
    assert response.text == "ok"
    assert app.calls == ["http"]


@pytest.mark.parametrize(
    "value",
    [
        "Bearer café".encode(),  # non-ASCII token
        b"Bearer \xff\xfe\x00",  # raw bytes that are not valid UTF-8
        "Béarer tok".encode(),  # non-ASCII scheme
        b"Bearer",  # malformed: one part, no token
        b"Basic \xff\xfe",  # malformed: non-bearer scheme with raw bytes
    ],
)
async def test_sec05_malformed_or_non_ascii_bearer_returns_400_never_500(value: bytes) -> None:
    """SEC-05/D-14: a malformed or non-ASCII Authorization header is a generic 400, never a 500.

    BearerGate works on raw header bytes (no decode), so a non-UTF-8 or malformed value fails the compare like
    any wrong token instead of raising on the way to it. The wrapped app is never reached. bootstrap/security.py
    is already correct here and is not modified (D-14) — this test locks the contract.
    """
    app = RecordingApp()
    status, body = await _raw_request(BearerGate(app, SECRET), [(b"authorization", value)])
    assert status == 400  # a 400 (never a 500): the malformed/non-ASCII token is rejected, not an error
    assert body == {"error": AUTH_ERROR_MESSAGE}
    assert app.calls == []  # the wrapped app was never reached


async def test_lifespan_passes_through_without_auth() -> None:
    app = RecordingApp()

    async def receive() -> dict[str, Any]:
        return {"type": "lifespan.startup"}

    async def send(message: dict[str, Any]) -> None:
        pass

    await BearerGate(app, SECRET)({"type": "lifespan"}, receive, send)
    assert app.calls == ["lifespan"]


async def test_unauthenticated_websocket_is_closed_before_reaching_the_app() -> None:
    app = RecordingApp()
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "websocket.connect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await BearerGate(app, SECRET)({"type": "websocket", "path": "/mcp", "headers": []}, receive, send)
    assert sent == [{"type": "websocket.close", "code": 1008}]
    assert app.calls == []


def test_secret_is_injected_and_must_not_be_empty() -> None:
    with pytest.raises(ValueError):
        BearerGate(RecordingApp(), "")
