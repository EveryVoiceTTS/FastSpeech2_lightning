from typing import Union

from pydantic import BaseModel
from torch import Tensor


class InferenceControl(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    pitch: Union[float, Tensor] = 1.0
    energy: Union[float, Tensor] = 1.0
    duration: Union[float, Tensor] = 1.0


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
