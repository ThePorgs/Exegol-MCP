from exegol.console.TUI import ExegolTUI
from exegol.console.cli.ParametersManager import ParametersManager
from exegol.utils.ExeLog import logger
from exegol.utils.MetaSingleton import MetaSingleton

from exegol_mcp.utils.mocking.ParametersMocked import ParametersMocked
from exegol_mcp.utils.mocking.TUI_MCP import TUI_MCP


class ExegolPatching:

    @classmethod
    def patch_exegol_sdk(cls):
        cls.__patch_critical_loggers()
        cls.__patch_parameters()
        cls.__patch_TUI()

    @classmethod
    def __patch_critical_loggers(cls):
        """Update exegol logger to raise exceptions instead of exit."""
        logger.setCriticalMethod("raise")

    @classmethod
    def __patch_parameters(cls):
        """Patch ParametersManager() to not use argparse"""
        MetaSingleton.mock(ParametersManager, ParametersMocked())

    @classmethod
    def __patch_TUI(cls):
        """Patch ExegolTUI() to use MCP Context instead of rich"""
        ExegolTUI.downloadDockerLayer = TUI_MCP.downloadDockerLayer
