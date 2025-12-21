from exegol_mcp.assets import *
from exegol_mcp.mcp_app import mcp_host, mcp_port

__version__ = "0.0.1a1"

from exegol_mcp.utils.mocking.patching import ExegolPatching

def main():
    print(f"Starting exegol-mcp v{__version__} !")
    ExegolPatching.patch_exegol_sdk()
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
