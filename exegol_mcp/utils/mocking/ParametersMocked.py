import logging
from typing import Dict, Any

from exegol.config.EnvInfo import EnvInfo
from exegol.utils.MetaSingleton import MetaSingleton


class ParametersMocked(metaclass=MetaSingleton):

    def __init__(self) -> None:
        self.quiet = False
        self.verbose = 1
        self.arch = EnvInfo.arch
        self.offline_mode = False
        self.accept_eula = False
        self.update_fs_perms = False

        self.__dynamic_parameters: Dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any | None:
        """
        Return None when an attribute is not defined
        """
        if name in self.__dynamic_parameters:
            return self.__dynamic_parameters[name]
        logging.warning(f"Try to get parameter '{name}' which is not defined in the mocked parameters.")
        return None

    def set_parameter(self, name: str, value: Any) -> None:
        """Set a parameter dynamically"""
        self.__dynamic_parameters[name] = value

    def reset_all_parameters(self) -> None:
        """Reset all parameters to None"""
        self.__dynamic_parameters.clear()
