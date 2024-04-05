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

    def __call__(self, inputs: torch.Tensor) -> Tuple[np.ndarray, int]:
        raise NotImplementedError


# TODO: This should be implemented/moved under
#       everyvoice/model/vocoder/original_hifigan_helper/
class SynthesizerUniversal(SynthesizerBase):
    """
    A synthesizer that uses the generator_universal.
    """

    def __init__(self, vocoder, config) -> None:
        super().__init__(vocoder)
        # TODO: If we don't need all of config but simply output_sampling_rate,
        # may be we should only store that.
        self.config = config

    def __call__(self, inputs: torch.Tensor) -> Tuple[np.ndarray, int]:
        """
        Generate wavs using the generator_universal model.
        """
        from everyvoice.model.vocoder.original_hifigan_helper import vocoder_infer

        wavs = vocoder_infer(inputs, self.vocoder)
        sr = self.config.preprocessing.audio.output_sampling_rate

        return wavs, sr


# TODO: We should have a less generic name for the EveryVoice synthesizer.
# TODO: This should be implemented under
#       everyvoice/model/vocoder/HiFiGAN_iSTFT_lightning/hfgl/ as it is
#       specific to hfgl.
class Synthesizer(SynthesizerBase):
    """
    A synthesizer that uses EveryVoice models.
    """

    def __init__(self, vocoder) -> None:
        super().__init__(vocoder)

    def __call__(self, inputs: torch.Tensor) -> Tuple[np.ndarray, int]:
        """
        Generate wavs using Everyvoice model.
        """
        from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import (
            synthesize_data,
        )

        wavs, sr = synthesize_data(inputs, self.vocoder)

        return wavs, sr


def get_synthesizer(
    config: FastSpeech2Config,
    device: torch.device,
) -> SynthesizerBase:
    if config.training.vocoder_path is None:
        import sys

        logger.error(
            "No vocoder was provided, please specify "
            "--vocoder-path /path/to/vocoder on the command line."
        )
        sys.exit(1)
    else:
        vocoder = torch.load(config.training.vocoder_path, map_location=device)
        if "generator" in vocoder.keys():
            # Necessary when passing --filelist
            from everyvoice.model.vocoder.original_hifigan_helper import get_vocoder

            vocoder = get_vocoder(config.training.vocoder_path, device=device)
            return SynthesizerUniversal(vocoder, config)

        return Synthesizer(vocoder)
