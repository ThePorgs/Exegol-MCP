"""HTTP bearer authentication as a pure-ASGI wrapper around the MCP SDK app (COMP-01 parity).

v1 rejected a missing or wrong bearer token with HTTP 400 and a generic JSON error, deliberately not 401/403: a
client that sees 401/403 may start an OAuth discovery dance against a server that has none. BearerGate keeps that
contract while wrapping the SDK's own streamable HTTP app directly, so the SDK's localhost DNS-rebinding
protection stays in place (no re-mount inside a custom Starlette app).
"""

import hmac

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

AUTH_ERROR_MESSAGE = "Invalid bearer auth credentials"


class BearerGate:
    """Let a request through only when it carries `Authorization: Bearer <secret>`.

    The secret is injected by the caller, which loads it once at startup; the gate never reads secret storage.
    """

    def __init__(self, app: ASGIApp, secret: str) -> None:
        if not secret:
            raise ValueError("BearerGate needs a non-empty secret")
        self.app = app
        self._secret = secret.encode("utf-8")

    def _is_authorized(self, scope: Scope) -> bool:
        # Work on the raw header bytes: a non-ASCII value then fails the comparison like any wrong token
        # instead of raising while it is decoded or compared (which would surface as a 500).
        value = next((v for k, v in scope.get("headers", []) if k.lower() == b"authorization"), None)
        if value is None:
            return False
        parts = value.split()
        if len(parts) != 2 or parts[0].lower() != b"bearer":
            return False
        # Constant-time comparison to mitigate timing attacks
        return hmac.compare_digest(parts[1], self._secret)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket") or self._is_authorized(scope):
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            # The MCP app serves no websocket; refuse the handshake rather than let it reach the app unchecked.
            await send({"type": "websocket.close", "code": 1008})
            return
        # Always 400 with one generic message: do not reveal which check failed, and do not invite OAuth (v1 parity).
        response = JSONResponse({"error": AUTH_ERROR_MESSAGE}, status_code=400)
        await response(scope, receive, send)
