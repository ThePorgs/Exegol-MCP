"""server_info service (D-13): versions, Docker status, compatibility state and update notice.

The data comes from a ServerInfoSource. The bootstrap (plan 05) provides the full source (wrapper version, Docker
status, config dirs, update check) through the server lifespan. Until a lifespan context implements the source,
DefaultServerInfoSource reports what package metadata alone can tell.
"""

import importlib.metadata
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from exegol_mcp import __version__
from exegol_mcp.adapter.offload import run_blocking
from exegol_mcp.models.server_info import CompatibilityState, ServerInfo


@dataclass(frozen=True)
class VersionReport:
    """Raw facts a ServerInfoSource collects; server_info() derives the compatibility state from them"""

    exegol_mcp_version: str
    exegol_library_version: str | None
    exegol_wrapper_version: str | None
    docker_status: str
    config_dirs: list[str] = field(default_factory=list)
    update_notice: str | None = None


@runtime_checkable
class ServerInfoSource(Protocol):
    """Provides the facts server_info reports."""

    async def version_report(self) -> VersionReport:
        """Return the current versions, Docker status, config dirs and update notice."""
        ...


def _library_version() -> str | None:
    try:
        return importlib.metadata.version("exegol")
    except importlib.metadata.PackageNotFoundError:
        return None


class DefaultServerInfoSource:
    """Package-metadata-only source, used when the lifespan context provides no ServerInfoSource."""

    async def version_report(self) -> VersionReport:
        """Report the exegol-mcp and bundled Exegol library versions; the rest is not checked here."""
        return VersionReport(
            exegol_mcp_version=__version__,
            # Reading distribution metadata touches the filesystem, so it runs off the event loop.
            exegol_library_version=await run_blocking(_library_version),
            exegol_wrapper_version=None,
            docker_status="not checked",
        )


def source_from(lifespan_context: object) -> ServerInfoSource:
    """Return the lifespan context when it is a ServerInfoSource, else the package-metadata fallback."""
    if isinstance(lifespan_context, ServerInfoSource):
        return lifespan_context
    return DefaultServerInfoSource()


def _normalize(version: str) -> str:
    return version.strip().lower().removeprefix("v")


def compatibility_state(library_version: str | None, wrapper_version: str | None) -> CompatibilityState:
    """Compare the bundled Exegol library with the host wrapper (D-07/D-12: a mismatch is information only).

    Returns:
        "unknown" when either version is not known, "match" when they are equal, "mismatch" otherwise.
    """
    if not library_version or not wrapper_version:
        return "unknown"
    return "match" if _normalize(library_version) == _normalize(wrapper_version) else "mismatch"


async def server_info(source: ServerInfoSource) -> ServerInfo:
    """Build the ServerInfo the server_info tool returns.

    Args:
        source: Where the versions, Docker status, config dirs and update notice come from.

    Returns:
        The server information, with the compatibility state derived from the library and wrapper versions.
    """
    report = await source.version_report()
    return ServerInfo(
        exegol_mcp_version=report.exegol_mcp_version,
        exegol_library_version=report.exegol_library_version,
        exegol_wrapper_version=report.exegol_wrapper_version,
        docker_status=report.docker_status,
        config_dirs=list(report.config_dirs),
        compatibility_state=compatibility_state(report.exegol_library_version, report.exegol_wrapper_version),
        update_notice=report.update_notice,
    )
