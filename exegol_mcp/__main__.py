"""`python -m exegol_mcp` entry point: same as the exegol-mcp console script."""

import sys

from exegol_mcp.main import main

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
