from typing import Literal

from pydantic import BaseModel, Field

# Scope of list_images: only the images installed on this host, or every image Exegol knows about
# (installed plus available for download).
ImageScope = Literal["installed", "all"]


class ImageInfo(BaseModel):
    """Exegol image information"""

    name: str = Field(description="Exegol image name, the value download_image expects as image_name")
    version: str = Field(description="Installed version of the image, empty when the image is not installed")
    latest_version: str = Field(description="Latest version available upstream for this image")
    is_installed: bool = Field(description="True when the image is installed on this host")
    is_built_locally: bool = Field(description="True when the image was built locally rather than pulled")
    is_up_to_date: bool = Field(description="True when the installed version matches the latest available version")
    build_date: str = Field(description="Build date of the installed image, empty when not installed")
    license: str = Field(description="License tier required to use the image (for example free or pro)")
