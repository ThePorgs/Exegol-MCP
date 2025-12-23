from mcp.server.fastmcp import FastMCP
from mcp.types import Icon

mcp_server = FastMCP(
    "Exegol",
    stateless_http=False,
    json_response=True,
    website_url="https://exegol.com",
    icons=[Icon(
        src="https://docs.exegol.com/images/Exegol_Symbol_DarkVersion.svg",
        mimeType="image/svg+xml"
    )]
)
