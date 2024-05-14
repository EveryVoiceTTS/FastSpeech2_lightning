"""
This file is for types that are reused, but potentionally expensive to load.
Types requiring pydantic belong here.
Enums and other light-weight types should be defined in type_definitions.py instead.

Be careful not to cause this file to be imported when the CLI is launched, so that
everyvoice -h and command line completion can remain fast.
"""

from pydantic import BaseModel, ConfigDict


class InferenceControl(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pitch: float = 1.0
    energy: float = 1.0
    duration: float = 1.0


class StatsInfo(BaseModel):
    min: float
    max: float
    std: float
    mean: float
    norm_min: float
    norm_max: float


class Stats(BaseModel):
    pitch: StatsInfo
    energy: StatsInfo
