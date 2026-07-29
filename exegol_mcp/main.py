import asyncio
import contextlib
import logging
import sys

import uvicorn
from exegol.utils.ExeLog import ExeLog
from rich.console import Console
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.routing import Mount

from exegol_mcp.assets import *

__version__ = "1.0.1"

from exegol_mcp.utils.auth_backend import BearerAuthBackend, on_auth_error
from exegol_mcp.utils.cli import parse_args
from exegol_mcp.utils.exegol_utils import check_exegol_readiness
from exegol_mcp.utils.secrets_manager import SecretManager

from exegol_mcp.utils.mocking.patching import ExegolPatching


@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    async with mcp_server.session_manager.run():
        yield


def display_header(args):
    logging.info(f"Starting exegol-mcp v{__version__} !")
    print("MCP client template:")
    print("========================================================")
    if args.type == "stdio":
        print(f""""exegol-mcp": {{
    "type": "stdio",
    "command": "{sys.executable}",
    "args": ["{__file__}", "--type", "stdio"]
}}""")
        print("========================================================")
    elif args.auth == "bearer":
        # Only generate/read the secret in bearer mode
        secret, created, path = SecretManager.load_bearer_secret()
        if path:
            logging.debug(f"Bearer secret stored at '{path}'")
        secret_hint = secret if args.print_config or created else "<redacted> (use --print-config to reveal)"

        print(f""""exegol-mcp": {{
    "type": "http",
    "url": "http://127.0.0.1:{args.port}/mcp",
    "headers": {{
        "Authorization": "Bearer {secret_hint}"
    }}
}}""")
        print("========================================================")
        if created:
            print("Use '--print-config' to display the secret again if needed later.")
    elif args.auth == "none":
        print(f""""exegol-mcp": {{
    "type": "http",
    "url": "http://127.0.0.1:{args.port}/mcp"
}}""")
        print("========================================================")
        logging.warning("Authentication is DISABLED (mode: none). Use only in trusted environments.")
    else:
        raise NotImplementedError(f"Unknown authentication mode: {args.auth}")
    if args.print_config:
        # Do not run MCP server with this option
        sys.exit(0)


def start_http(args):
    middlewares = []
    if args.auth == "bearer":
        middlewares.append(Middleware(AuthenticationMiddleware, backend=BearerAuthBackend(), on_error=on_auth_error))

    app = Starlette(
        routes=[
            Mount("/", app=mcp_server.streamable_http_app()),
        ],
        lifespan=lifespan,
        middleware=middlewares
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    ExegolPatching.patch_exegol_sdk()
    asyncio.run(check_exegol_readiness())

    if args.type == "http":
        if args.verbose > 0:
            logging.getLogger().setLevel(logging.DEBUG)
            ExeLog.setVerbosity(args.verbose)
        # Display client template and secret guidance
        display_header(args)

        # Start HTTP MCP server
        start_http(args)
    elif args.type == "stdio":
        if args.print_config:
            display_header(args)
        # Disable logging to avoid interfering with the MCP protocol
        logging.getLogger().setLevel(100)
        ExeLog.console = Console(quiet=True)
        ExeLog.setVerbosity(0, quiet=True)

        # Start stdio MCP server
        mcp_server.run()
    else:
        raise NotImplementedError(f"Unknown server type: {args.type}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
