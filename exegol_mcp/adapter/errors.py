"""Exegol SDK guard errors, kept SDK-free so tools/ can map them without importing the SDK (D-19)."""


class UnknownParameterError(AttributeError):
    """The Exegol SDK read a CLI parameter that exegol-mcp does not set"""


class HeadlessPromptError(RuntimeError):
    """The Exegol SDK tried to prompt a human during a headless run"""


class PatchPointError(RuntimeError):
    """An Exegol SDK internal that exegol-mcp relies on is missing or was renamed"""


class ContainerNameConflictError(Exception):
    """Docker refused exegol-mcp's own create because the container name is taken (HTTP 409, WR-03).

    Raised instead of docker-py's APIError while create_conflict_guard() is bound, so Exegol's createContainer
    rollback, which would remove the other container of that name, never runs. It subclasses neither APIError nor
    RuntimeError, so no Exegol or exegol-mcp handler of those catches it by accident.
    """
