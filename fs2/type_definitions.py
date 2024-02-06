from enum import Enum

from pydantic import BaseModel, ConfigDict

# Redining LookupTable to minimize import cost.
LookupTable = dict[str, int]


class SynthesizeOutputFormats(str, Enum):
    """Valid output formats for synthesize"""

    wav = "wav"
    npy = "npy"
    pt = "pt"


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
