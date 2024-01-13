from pydantic import BaseModel, ConfigDict


class ConfigBaseModel(BaseModel):
    """Base model for all invoke training configuration models."""

    # Configure to raise if extra fields are passed in.
    model_config = ConfigDict(extra="forbid")
