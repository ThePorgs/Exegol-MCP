from exegol.utils.ExeLog import logger
from exegol.utils.MetaSingleton import MetaSingleton
from exegol.console.cli.ParametersManager import ParametersManager

from exegol_mcp.assets import *
from exegol_mcp.mcp_app import mcp_host, mcp_port
from exegol_mcp.utils.ParametersMocked import ParametersMocked

__version__ = "0.0.1a1"

def patch_critical_loggers():
    """Update exegol logger to raise exceptions instead of exit."""
    logger.setCriticalMethod("raise")

def main():
    print(f"Starting exegol-mcp v{__version__} !")
    patch_critical_loggers()
    MetaSingleton.mock(ParametersManager, ParametersMocked())
    # TODO
    #  - authentification (server / client)
    #  - add tools for orchestrator
    #  - add tools for container (✅ DONE: start_container, stop_container, execute_command_in_container)

    print("MCP client template:")
    print("========================================================")
    print(f""""exegol-mcp": {{
    "type": "http",
    "url": "http://{mcp_host}:{mcp_port}/mcp"
}}""")
    print("========================================================")
    mcp_server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
