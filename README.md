> **📌 This repository hosts code for Exegol MCP, a sub-repository of the Exegol project.
> If you were looking for Exegol, go to [the main repo](https://github.com/ThePorgs/Exegol)**
___

![banner](https://docs.exegol.com/external/banner.png)

# Exegol MCP

Exegol MCP is a Model Context Protocol (MCP) server that lets AI clients and agents drive Exegol over stdio or HTTP: list, create, start and remove Exegol containers, manage images, run offensive security tools and read or write files inside the containers. The agent works in an isolated Exegol container, not on the host.

Get started on [docs.exegol.com/mcp/getting-started](https://docs.exegol.com/mcp/getting-started).

More information on [docs.exegol.com/mcp](https://docs.exegol.com/mcp/).

## Command line

- `exegol-mcp` starts the server (streamable HTTP on `127.0.0.1:2187/mcp`, bearer-authenticated) and prints one line with its address. Its first start also prints the client configuration entry, bearer token included; later starts point at `exegol-mcp --print-config`. `-v` adds the versions, the Docker state and the config directories.
- `exegol-mcp --type stdio` is the command an MCP client launches. It prints nothing but warnings.
- `exegol-mcp --print-config` prints the client configuration entry again (it contains the bearer token).
- `exegol-mcp version` prints the Exegol MCP and Exegol library versions, and the Exegol wrapper's when it differs.
- `exegol-mcp version -vvv` adds the debug information to include when reporting a problem, and never prints the token.

Every option and the `config.yml` file are described on [docs.exegol.com/mcp/configuration](https://docs.exegol.com/mcp/configuration).
