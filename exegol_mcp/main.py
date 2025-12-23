import asyncio
import contextlib
import logging
import sys

import uvicorn
from exegol.utils.ExeLog import ExeLog
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.routing import Mount

from exegol_mcp.assets import *
from exegol_mcp.mcp_app import mcp_host, mcp_port

__version__ = "0.0.1a1"

from exegol_mcp.utils.auth_backend import BearerAuthBackend, on_auth_error
from exegol_mcp.utils.cli import parse_args
from exegol_mcp.utils.exegol_utils import check_exegol_readiness
from exegol_mcp.utils.secrets_manager import SecretManager

from exegol_mcp.utils.mocking.patching import ExegolPatching


@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    async with mcp_server.session_manager.run():
        yield

def display_template(args):
    print("MCP client template:")
    print("========================================================")
    if args.auth == "bearer":
        # Only generate/read the secret in bearer mode
        secret, created, path = SecretManager.load_bearer_secret()
        if path:
            logging.debug(f"Bearer secret stored at '{path}'")
        secret_hint = secret if args.print_secret or created else "<redacted> (use --print-secret to reveal)"

        print(f""""exegol-mcp": {{
        "type": "http",
        "url": "http://{mcp_host}:{mcp_port}/mcp",
        "headers": {{
            "Authorization": "Bearer {secret_hint}"
        }}
    }}""")
        print("========================================================")
        if created:
            print("Use '--print-secret' to display the secret again if needed later.")
        if args.print_secret:
            # Do not run MCP server with this option
            sys.exit(0)
    elif args.auth == "none":
        print(f""""exegol-mcp": {{
        "type": "http",
        "url": "http://{mcp_host}:{mcp_port}/mcp"
    }}""")
        print("========================================================")
        logging.warning("Authentication is DISABLED (mode: none). Use only in trusted environments.")
    else:
        raise NotImplementedError(f"Unknown authentication mode: {args.auth}")

def main(argv: list[str] | None = None):
    args = parse_args(argv)
    if args.verbose > 0:
        logging.getLogger().setLevel(logging.DEBUG)
        ExeLog.setVerbosity(args.verbose)

    logging.info(f"Starting exegol-mcp v{__version__} !")
    ExegolPatching.patch_exegol_sdk()
    asyncio.run(check_exegol_readiness())
    # Display client template and secret guidance
    display_template(args)

    if args.auth == "none":
        mcp_server.run(transport="streamable-http")
    else:
        app = Starlette(
            routes=[
                Mount("/", app=mcp_server.streamable_http_app()),
            ],
            lifespan=lifespan,
            middleware=(
                [Middleware(AuthenticationMiddleware, backend=BearerAuthBackend(), on_error=on_auth_error)]
                if args.auth == "bearer" else []
            )
        )
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
