from mcp.server.fastmcp import Context


def is_elicitation_form_supported(ctx: Context):
    """Check if the MCP client supports elicitation form."""
    capabilities = ctx.session.client_params.capabilities
    return capabilities.elicitation is not None and capabilities.elicitation.form is not None

def is_elicitation_url_supported(ctx: Context):
    """Check if the MCP client supports elicitation url."""
    capabilities = ctx.session.client_params.capabilities
    return capabilities.elicitation is not None and capabilities.elicitation.url is not None

def is_roots_supported(ctx: Context):
    """Check if the MCP client supports roots."""
    capabilities = ctx.session.client_params.capabilities
    return capabilities.roots is not None

def is_sampling_supported(ctx: Context):
    """Check if the MCP client supports sampling."""
    capabilities = ctx.session.client_params.capabilities
    return capabilities.sampling is not None

def is_tasks_supported(ctx: Context):
    """Check if the MCP client supports tasks."""
    capabilities = ctx.session.client_params.capabilities
    return capabilities.tasks is not None

# Context doesn't provider support for the following feature:
# - Resources
# - Prompts
# - Discovery
# - Instructions
