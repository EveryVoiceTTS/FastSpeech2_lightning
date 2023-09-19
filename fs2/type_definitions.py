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
