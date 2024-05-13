from typing import Tuple

import numpy as np
import torch
from loguru import logger

from .config import FastSpeech2Config


# TODO: This should go under everyvoice/model/vocoder/.
#       It would define a common interface for all vocoder.
#       Each specific vocoder should implement SynthesizerBase.
class SynthesizerBase:
    """
    A common interface between the generator_universal and Everyvoice's vocoder.
    """

    def __init__(self, vocoder) -> None:
        self.vocoder = vocoder

    def __call__(self, input_: torch.Tensor) -> Tuple[np.ndarray, int]:
        raise NotImplementedError


# TODO: We should have a less generic name for the EveryVoice synthesizer.
class Synthesizer(SynthesizerBase):
    """
    A synthesizer that uses EveryVoice models.
    """

    def __init__(self, vocoder) -> None:
        super().__init__(vocoder)

    def __call__(self, input_: torch.Tensor) -> Tuple[np.ndarray, int]:
        """
        Generate wavs using Everyvoice model.
        """
        from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import (
            synthesize_data,
        )

        wavs, sr = synthesize_data(input_, self.vocoder)

        return wavs, sr


def get_synthesizer(
    config: FastSpeech2Config,
    device: torch.device,
) -> Synthesizer:
    """Given a config with a vocoder_path, this factory function loads the
    proper checkpoint for an EveryVoice vocoder.

    Return:
        Synthesizer
    """
    if config.training.vocoder_path is None:
        import sys

        logger.error(
            "No vocoder was provided, please specify "
            "--vocoder-path /path/to/vocoder on the command line."
        )
        sys.exit(1)
    else:
        vocoder = torch.load(config.training.vocoder_path, map_location=device)
        return Synthesizer(vocoder)
