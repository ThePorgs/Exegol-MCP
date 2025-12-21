from mcp.server.fastmcp import FastMCP
from mcp.types import Icon

mcp_host = "127.0.0.1"
mcp_port = 8000
mcp_server = FastMCP(
    "Exegol",
    host=mcp_host,
    port=mcp_port,
    stateless_http=False,
    json_response=True,
    website_url="https://exegol.com",
    icons=[Icon(
        src="https://docs.exegol.com/images/Exegol_Symbol_DarkVersion.svg",
        mimeType="image/svg+xml"
    )]
)
