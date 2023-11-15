from enum import Enum
from pathlib import Path
from typing import Optional, Union

from everyvoice.config.preprocessing_config import PreprocessingConfig
from everyvoice.config.shared_types import (
    BaseTrainingConfig,
    ConfigModel,
    NoamOptimizer,
    PartialLoadConfig,
    init_context,
)
from everyvoice.config.text_config import TextConfig
from everyvoice.config.utils import load_partials
from everyvoice.utils import load_config_from_json_or_yaml_path
from pydantic import Field, FilePath, ValidationInfo, model_validator


class TransformerConfig(ConfigModel):
    layers: int = 4
    heads: int = 2
    hidden_dim: int = 256
    feedforward_dim: int = 1024
    conv_filter_size: int = 1024
    conv_kernel_size: int = 9
    dropout: float = 0.2
    depthwise: bool = True
    conformer: bool = True


class FastSpeech2Variances(ConfigModel):
    energy: bool = False
    duration: bool = False
    pitch: bool = False


class VarianceLevelEnum(str, Enum):
    phone = "phone"
    frame = "frame"


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
    transform: VarianceTransformEnum = VarianceTransformEnum.none
    loss: VarianceLossEnum = VarianceLossEnum.mse
    n_layers: int = 5
    loss_weights: float = 5e-2
    kernel_size: int = 3
    dropout: float = 0.5
    hidden_dim: int = 256
    n_bins: int = 256
    depthwise: bool = True


class VariancePredictorConfig(VariancePredictorBase):
    level: VarianceLevelEnum = VarianceLevelEnum.phone


class VariancePredictors(ConfigModel):
    energy: VariancePredictorConfig = Field(default_factory=VariancePredictorConfig)
    duration: VariancePredictorBase = Field(default_factory=VariancePredictorBase)
    pitch: VariancePredictorConfig = Field(default_factory=VariancePredictorConfig)


# TODO: maybe flatten this to just variance_adaptor: VariancePredictors
class VarianceAdaptorConfig(ConfigModel):
    variance_predictors: VariancePredictors = Field(default_factory=VariancePredictors)


class FastSpeech2ModelConfig(ConfigModel):
    encoder: TransformerConfig = Field(default_factory=TransformerConfig)
    decoder: TransformerConfig = Field(default_factory=TransformerConfig)
    variance_adaptor: VarianceAdaptorConfig = Field(
        default_factory=VarianceAdaptorConfig
    )
    learn_alignment: bool = True
    max_length: int = 1000
    mel_loss: VarianceLossEnum = VarianceLossEnum.mse
    mel_loss_weight: float = 5e-1
    phonological_feats_size: int = 38
    use_phonological_feats: bool = False
    use_postnet: bool = True
    multilingual: bool = False
    multispeaker: bool = False


class FastSpeech2FreezeLayersConfig(ConfigModel):
    all_layers: bool = False
    encoder: bool = False
    decoder: bool = False
    postnet: bool = False
    variance: FastSpeech2Variances = Field(default_factory=FastSpeech2Variances)


class EarlyStoppingMetricEnum(str, Enum):
    none = "none"
    mae = "mae"
    js = "js"


class EarlyStoppingConfig(ConfigModel):
    metric: EarlyStoppingMetricEnum = EarlyStoppingMetricEnum.none
    patience: int = 4


class TFConfig(ConfigModel):
    ratio: float = 1.0
    linear_schedule: bool = False
    linear_schedule_start: int = 0
    linear_schedule_end: int = 20
    linear_schedule_end_ratio: float = 0.0


class FastSpeech2TrainingConfig(BaseTrainingConfig):
    use_weighted_sampler: bool = False
    optimizer: NoamOptimizer = Field(default_factory=NoamOptimizer)
    freeze_layers: FastSpeech2FreezeLayersConfig = Field(
        default_factory=FastSpeech2FreezeLayersConfig
    )
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    tf: TFConfig = Field(default_factory=TFConfig)
    vocoder_path: Union[FilePath, None] = None


class FastSpeech2Config(PartialLoadConfig):
    model: FastSpeech2ModelConfig = Field(default_factory=FastSpeech2ModelConfig)
    path_to_model_config_file: Optional[FilePath] = None

    training: FastSpeech2TrainingConfig = Field(
        default_factory=FastSpeech2TrainingConfig
    )
    path_to_training_config_file: Optional[FilePath] = None

    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    path_to_preprocessing_config_file: Optional[FilePath] = None

    text: TextConfig = Field(default_factory=TextConfig)
    path_to_text_config_file: Optional[FilePath] = None

    @model_validator(mode="before")  # type: ignore
    def load_partials(self, info: ValidationInfo):
        config_path = (
            info.context.get("config_path", None) if info.context is not None else None
        )
        return load_partials(
            self,
            ("model", "training", "preprocessing", "text"),
            config_path=config_path,
        )

    @model_validator(mode="after")
    def probable_out_of_memory_check(self, info: ValidationInfo):
        """
        Guesstimate the possibility of the configuration trigerring and out of GPU memory.
        """
        import torch

        from pudb import set_trace

        set_trace()
        if torch.cuda.is_available():
            max_wav_length = 11
            estimate = (
                self.training.batch_size
                * max_wav_length
                / (
                    self.preprocessing.audio.fft_hop_frames
                    / self.preprocessing.audio.input_sampling_rate
                )
            )
            estimate *= 4  # Bytes per float
            global_free_memory, total_GPU_memory = torch.cuda.mem_get_info()
            assert estimate < global_free_memory

    @staticmethod
    def load_config_from_path(path: Path) -> "FastSpeech2Config":
        """Load a config from a path"""
        config = load_config_from_json_or_yaml_path(path)
        with init_context({"config_path": path}):
            config = FastSpeech2Config(**config)
        return config
