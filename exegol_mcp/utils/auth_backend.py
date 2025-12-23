import hmac
from starlette.authentication import AuthenticationBackend, AuthenticationError, AuthCredentials, SimpleUser
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from exegol_mcp.utils.secrets_manager import SecretManager


class BearerAuthBackend(AuthenticationBackend):
    def __init__(self, expected_secret: str | None = None):
        # Delay loading to avoid generating a secret unless bearer mode is used
        if expected_secret is None:
            secret, _, _ = SecretManager.load_bearer_secret()
            self.__secret = secret
        else:
            self.__secret = expected_secret

    async def authenticate(self, conn):
        if "Authorization" not in conn.headers:
            return

        auth = conn.headers["Authorization"]
        try:
            scheme, credentials = auth.split()
            if scheme.lower() != 'bearer':
                return
        except (ValueError, UnicodeDecodeError) as exc:
            raise AuthenticationError('Invalid bearer auth credentials')

        # Constant-time comparison to mitigate timing attacks
        if not hmac.compare_digest(credentials, self.__secret):
            raise AuthenticationError('Invalid bearer auth credentials')

        return AuthCredentials(["authenticated"]), SimpleUser("user")

def on_auth_error(request: Request, exc: Exception) -> Response:
    if isinstance(exc, AuthenticationError):
        # Always return 400 error code (client might interpret 401 / 403 as redirect to oauth)
        return JSONResponse({"error": str(exc)}, status_code=400)
    else:
        return AuthenticationMiddleware.default_on_error(request, exc)
