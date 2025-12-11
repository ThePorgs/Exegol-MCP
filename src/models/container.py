from typing import List, Optional, TypedDict

from pydantic import BaseModel, Field


class ContainerInfo(BaseModel):
    """Container information from Exegol SDK"""
    name: str
    creation_date: str
    image_name: str
    image_version: str
    status: str
    network_driver: Optional[str]
    network_name: Optional[str]
    features: List[str]
    devices: List[str]
    vpn: Optional[str]
    env: List[str]
    is_privileged: bool
    capabilities: List[str]
    comment: Optional[str]

class ExecutionResult(BaseModel):
    exit_code: int = Field(description="Exit code of the command")
    output: str = Field(description="Command output (stdout + stderr)")
