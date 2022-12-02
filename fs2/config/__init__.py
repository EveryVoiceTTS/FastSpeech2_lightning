from enum import Enum
from pathlib import Path
from typing import Union

from pydantic import FilePath
from smts.config.preprocessing_config import PreprocessingConfig
from smts.config.shared_types import (
    BaseTrainingConfig,
    ConfigModel,
    NoamOptimizer,
    PartialConfigModel,
)
from smts.config.text_config import TextConfig
from smts.utils import load_config_from_json_or_yaml_path, return_configs_from_dir


class TransformerConfig(ConfigModel):
    layers: int
    heads: int
    hidden_dim: int
    feedforward_dim: int
    conv_filter_size: int
    conv_kernel_size: int
    dropout: float
    depthwise: bool
    conformer: bool


class FastSpeech2Variances(ConfigModel):
    energy: bool
    duration: bool
    pitch: bool


class VarianceLevelEnum(str, Enum):
    phone = "phone"
    frame = "frame"


class VarianceTypeEnum(str, Enum):
    energy = "energy"
    duration = "duration"
    pitch = "pitch"


class VarianceTransformEnum(str, Enum):
    log = "log"
    cwt = "cwt"
    none = "none"


class VarianceLossEnum(str, Enum):
    mse = "mse"
    mae = "mae"
    l1 = "l1"
    soft_dtw = "soft_dtw"


class VariancePredictorBase(ConfigModel):
    transform: VarianceTransformEnum
    loss: VarianceLossEnum
    n_layers: int
    loss_weights: float
    kernel_size: int
    dropout: float
    hidden_dim: int
    n_bins: int
    depthwise: bool


class VariancePredictorConfig(VariancePredictorBase):
    level: VarianceLevelEnum


class VariancePredictors(ConfigModel):
    energy: VariancePredictorConfig
    duration: VariancePredictorBase
    pitch: VariancePredictorConfig


# TODO: maybe flatten this to just variance_adaptor: VariancePredictors
class VarianceAdaptorConfig(ConfigModel):
    variance_predictors: VariancePredictors


class MultiSpeakerConfig(ConfigModel):
    embedding_type: Enum(  # type: ignore
        "EmbeddingType", {v: v for v in ["id", "dvector", "none"]}  # noqa: F821
    )
    every_layer: bool
    dvector_gmm: bool


class FastSpeech2ModelConfig(ConfigModel):
    encoder: TransformerConfig
    decoder: TransformerConfig
    variance_adaptor: VarianceAdaptorConfig
    learn_alignment: bool
    max_length: int
    mel_loss: Enum(  # type: ignore
        "variance_loss", {k: k for k in ["mse", "l1", "softw_dtw"]}  # noqa: F821
    )
    mel_loss_weight: float
    phonological_feats_size: int
    use_phonological_feats: bool
    use_postnet: bool
    multilingual: bool
    multispeaker: MultiSpeakerConfig


class FastSpeech2FreezeLayersConfig(ConfigModel):
    all_layers: bool
    encoder: bool
    decoder: bool
    postnet: bool
    variance: FastSpeech2Variances


class EarlyStoppingConfig(ConfigModel):
    metric: Enum(  # type: ignore
        "EarlyStoppingMetric", {v: v for v in ["none", "mae", "js"]}  # noqa: F821
    )
    patience: int


class TFConfig(ConfigModel):
    ratio: float
    linear_schedule: bool
    linear_schedule_start: int
    linear_schedule_end: int
    linear_schedule_end_ratio: float


class FastSpeech2TrainingConfig(BaseTrainingConfig):
    use_weighted_sampler: bool
    optimizer: NoamOptimizer
    freeze_layers: FastSpeech2FreezeLayersConfig
    early_stopping: EarlyStoppingConfig
    tf: TFConfig
    vocoder_path: Union[FilePath, None]


class FastSpeech2Config(PartialConfigModel):
    model: FastSpeech2ModelConfig
    training: FastSpeech2TrainingConfig
    preprocessing: PreprocessingConfig
    text: TextConfig

    @staticmethod
    def load_config_from_path(path: Path) -> "FastSpeech2Config":
        """Load a config from a path"""
        config = load_config_from_json_or_yaml_path(path)
        return FastSpeech2Config(**config)


CONFIG_DIR = Path(__file__).parent
CONFIGS = return_configs_from_dir(CONFIG_DIR)
