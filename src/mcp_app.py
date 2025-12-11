from mcp.server.fastmcp import FastMCP

mcp_host = "127.0.0.1"
mcp_port = 8000
mcp_server = FastMCP(
    "Exegol",
    host=mcp_host,
    port=mcp_port,
    stateless_http=True,
    json_response=True
)
