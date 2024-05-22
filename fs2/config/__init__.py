from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Dict, Optional, Union

from annotated_types import Ge
from everyvoice.config.preprocessing_config import PreprocessingConfig
from everyvoice.config.shared_types import (
    BaseModelWithContact,
    BaseTrainingConfig,
    ConfigModel,
    NoamOptimizer,
    init_context,
)
from everyvoice.config.text_config import TextConfig
from everyvoice.config.type_definitions import TargetTrainingTextRepresentationLevel
from everyvoice.config.utils import load_partials
from everyvoice.utils import load_config_from_json_or_yaml_path
from pydantic import (
    Field,
    FilePath,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)


class ConformerConfig(ConfigModel):
    layers: int = Field(4, description="The number of layers in the Conformer.")
    heads: int = Field(
        2, description="The number of heads in the multi-headed attention modules."
    )
    input_dim: int = Field(
        256,
        description="The number of hidden dimensions in the input. The input_dim value declared in the encoder and decoder modules must match the input_dim value declared in each variance predictor module.",
    )
    feedforward_dim: int = Field(
        1024, description="The number of dimensions in the feedforward layers."
    )
    conv_kernel_size: int = Field(
        9,
        description="The size of the kernel in each convoluational layer of the Conformer.",
    )
    dropout: float = Field(0.2, description="The amount of dropout to apply.")


class FastSpeech2Variances(ConfigModel):
    energy: bool = False
    duration: bool = False
    pitch: bool = False


class VarianceLevelEnum(str, Enum):
    phone = "phone"
    frame = "frame"


class VarianceLossEnum(str, Enum):
    mse = "mse"
    mae = "mae"


class VariancePredictorBase(ConfigModel):
    loss: VarianceLossEnum = Field(
        VarianceLossEnum.mse,
        description="The loss function to use when calculate variance loss. Either 'mse' or 'mae'.",
    )
    n_layers: int = Field(
        5, description="The number of layers in the variance predictor module."
    )
    kernel_size: int = Field(
        3,
        description="The kernel size of each convolutional layer in the variance predictor module.",
    )
    dropout: float = Field(0.5, description="The amount of dropout to apply.")
    input_dim: int = Field(
        256,
        description="The number of hidden dimensions in the input. This must match the input_dim value declared in the encoder and decoder modules.",
    )
    n_bins: int = Field(
        256, description="The number of bins to use in the variance predictor module."
    )
    depthwise: bool = Field(
        True, description="Whether to use depthwise separable convolutions."
    )

    @field_serializer("loss")
    def convert_loss_enum(self, loss: VarianceLossEnum):
        return loss.value


class VariancePredictorConfig(VariancePredictorBase):
    level: VarianceLevelEnum = Field(
        VarianceLevelEnum.phone,
        description="The level for the variance predictor to use. 'frame' will make predictions at the frame level. 'phone' will average predictions across all frames in each phone.",
    )

    @field_serializer("level")
    def convert_level_enum(self, level: VarianceLevelEnum):
        return level.value


class VariancePredictors(ConfigModel):
    energy: VariancePredictorConfig = Field(
        default_factory=VariancePredictorConfig,
        description="The variance predictor for energy",
    )
    duration: VariancePredictorBase = Field(
        default_factory=VariancePredictorBase,
        description="The variance predictor for duration",
    )
    pitch: VariancePredictorConfig = Field(
        default_factory=VariancePredictorConfig,
        description="The variance predictor for pitch",
    )


class FastSpeech2ModelConfig(ConfigModel):
    encoder: ConformerConfig = Field(
        default_factory=ConformerConfig,
        description="The configuration of the encoder module.",
    )
    decoder: ConformerConfig = Field(
        default_factory=ConformerConfig,
        description="The configuration of the decoder module.",
    )
    variance_predictors: VariancePredictors = Field(
        default_factory=VariancePredictors,
        description="Configuration for energy, duration, and pitch variance predictors.",
    )
    target_text_representation_level: TargetTrainingTextRepresentationLevel = (
        TargetTrainingTextRepresentationLevel.characters
    )
    learn_alignment: bool = Field(
        True,
        description="Whether to jointly learn alignments using monotonic alignment search module (See Badlani et. al. 2021: https://arxiv.org/abs/2108.10447). If set to False, you will have to provide text/audio alignments separately before training a text-to-spec (feature prediction) model.",
    )
    max_length: int = Field(
        1000, description="The maximum length (i.e. number of symbols) for text inputs."
    )
    mel_loss: VarianceLossEnum = Field(
        VarianceLossEnum.mse,
        description="The loss function to use when calculating Mel spectrogram loss.",
    )
    use_postnet: bool = Field(True, description="Whether to use a postnet module.")
    multilingual: bool = Field(
        False,
        description="Whether to train a multilingual model. For this to work, your filelist must contain a column/field for 'language' with values for each utterance.",
    )
    multispeaker: bool = Field(
        False,
        description="Whether to train a multispeaker model. For this to work, your filelist must contain a column/field for 'speaker' with values for each utterance.",
    )

    @field_serializer("mel_loss")
    def convert_mel_loss_enum(self, mel_loss: VarianceLossEnum):
        return mel_loss.value

    @field_serializer("target_text_representation_level")
    def convert_training_enum(
        self, target_text_representation_level: TargetTrainingTextRepresentationLevel
    ):
        return target_text_representation_level.value


class EarlyStoppingMetricEnum(str, Enum):
    none = "none"
    mae = "mae"
    js = "js"


class EarlyStoppingConfig(ConfigModel):
    metric: EarlyStoppingMetricEnum = EarlyStoppingMetricEnum.none
    patience: int = 4

    @field_serializer("metric")
    def convert_metric_enum(self, metric: EarlyStoppingMetricEnum):
        return metric.value


class FastSpeech2TrainingConfig(BaseTrainingConfig):
    use_weighted_sampler: bool = Field(
        False,
        description="Whether to use a sampler which oversamples from the minority language or speaker class for balanced training.",
    )
    optimizer: NoamOptimizer = Field(
        NoamOptimizer(
            betas=(0.9, 0.999), learning_rate=1e-3, warmup_steps=1000, weight_decay=1e-6
        ),
        description="The optimizer to use during training.",
    )
    # TODO: Implement early stopping
    # early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    vocoder_path: Union[FilePath, None] = None
    mel_loss_weight: float = Field(
        1.0, description="Multiply the spec loss by this weight"
    )
    postnet_loss_weight: float = Field(
        1.0, description="Multiply the postnet loss by this weight"
    )
    pitch_loss_weight: float = Field(
        0.1, description="Multiply the pitch loss by this weight"
    )
    energy_loss_weight: float = Field(
        0.1, description="Multiply the energy loss by this weight"
    )
    duration_loss_weight: float = Field(
        0.1, description="Multiply the duration loss by this weight"
    )
    attn_ctc_loss_weight: float = Field(
        0.1, description="Multiply the Attention CTC loss by this weight"
    )
    attn_bin_loss_weight: float = Field(
        0.1, description="Multiply the Attention Binarization loss by this weight"
    )
    attn_bin_loss_warmup_epochs: Annotated[int, Ge(1)] = Field(
        100,
        description="Scale the Attention Binarization loss by (current_epoch / attn_bin_loss_warmup_epochs) until the number of epochs defined by attn_bin_loss_warmup_epochs is reached.",
    )

    # NOTE: We must use a different name for this function than the one in the
    # base class or else it overrides the base class and we lose validation.
    @field_validator("vocoder_path", mode="before")
    @classmethod
    def relative_to_absolute_vocoder_path(
        cls, value: Any, info: ValidationInfo
    ) -> Path:
        if isinstance(value, (str, Path)):
            return cls.path_relative_to_absolute(Path(value), info)
        return value


class FastSpeech2Config(BaseModelWithContact):
    model: FastSpeech2ModelConfig = Field(
        default_factory=FastSpeech2ModelConfig,
        description="The model configuration settings.",
    )
    path_to_model_config_file: Optional[FilePath] = Field(
        None, description="The path of a model configuration file."
    )

    training: FastSpeech2TrainingConfig = Field(
        default_factory=FastSpeech2TrainingConfig,
        description="The training configuration hyperparameters.",
    )
    path_to_training_config_file: Optional[FilePath] = Field(
        None, description="The path of a training configuration file."
    )

    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig,
        description="The preprocessing configuration, including information about audio settings.",
    )
    path_to_preprocessing_config_file: Optional[FilePath] = Field(
        None, description="The path of a preprocessing configuration file."
    )

    text: TextConfig = Field(
        default_factory=TextConfig, description="The text configuration."
    )
    path_to_text_config_file: Optional[FilePath] = Field(
        None, description="The path of a text configuration file."
    )

    @model_validator(mode="before")  # type: ignore
    def load_partials(self: Dict[Any, Any], info: ValidationInfo):
        config_path = (
            info.context.get("config_path", None) if info.context is not None else None
        )
        return load_partials(
            self,
            ("model", "training", "preprocessing", "text"),
            config_path=config_path,
        )

    @staticmethod
    def load_config_from_path(path: Path) -> "FastSpeech2Config":
        """Load a config from a path"""
        config = load_config_from_json_or_yaml_path(path)
        with init_context({"config_path": path}):
            config = FastSpeech2Config(**config)
        return config

    # INPUT_TODO: initialize text with union of symbols from dataset
