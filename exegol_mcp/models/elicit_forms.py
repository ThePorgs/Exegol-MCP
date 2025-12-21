from pydantic import BaseModel, Field


class UserConfirmation(BaseModel):
    """Schema for user confirmation form"""

    confirmation: bool = Field(description="Do you want to execute the operation?", default=False)
