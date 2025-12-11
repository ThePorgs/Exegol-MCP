from src.assets import *

__version__ = "0.0.1a1"



def main():
    print(f"Starting exegol-mcp v{__version__} !")
    # TODO
    #  - authentification (server / client)
    #  - add tools for orchestrator
    #  - add tools for container

    mcp_server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
