from src.assets import *
from src.mcp_app import mcp_host, mcp_port

__version__ = "0.0.1a1"


def main():
    print(f"Starting exegol-mcp v{__version__} !")
    # TODO
    #  - authentification (server / client)
    #  - add tools for orchestrator
    #  - add tools for container

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
