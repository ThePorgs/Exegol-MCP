from pydantic import BaseModel


class ImageInfo(BaseModel):
    """Exegol image information"""
    name: str
    version: str
    lastest_version: str
    is_installed: bool
    is_built_locally: bool
    is_up_to_date: bool
    build_date: str
    license: str
