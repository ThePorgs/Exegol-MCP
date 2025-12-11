from typing import List, Optional, TypedDict

from pydantic import BaseModel


class ContainerInfo(TypedDict):
    """Container information from Exegol SDK"""
    name: str
    image_name: str
    image_version: str
    status: str
    network_driver: Optional[str]
    network_name: Optional[str]

class ExecutionResult(BaseModel):
    exit_code: int
    output: str
