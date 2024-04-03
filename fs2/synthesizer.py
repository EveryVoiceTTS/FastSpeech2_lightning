from typing import Tuple

import numpy as np
import torch
from loguru import logger

from .config import FastSpeech2Config


class SynthesizerBase:
    """
    A common interface between the generator_universal and Everyvoice's vocoder.
    """

    def __init__(self, vocoder) -> None:
        self.vocoder = vocoder

    def __call__(self, inputs) -> Tuple[np.ndarray, int]:
        raise NotImplementedError


class SynthesizerUniversal(SynthesizerBase):
    """
    A synthesizer that uses the generator_universal.
    """

    def __init__(self, vocoder, config) -> None:
        super().__init__(vocoder)
        # TODO: if we don't need all of config but simply output_sampling_rate,
        # may be we should only store that.
        self.config = config

    def __call__(self, inputs) -> Tuple[np.ndarray, int]:
        """
        Generate wavs using the generator_universal model.
        """
        from everyvoice.model.vocoder.original_hifigan_helper import vocoder_infer

        wavs = vocoder_infer(inputs, self.vocoder)
        sr = self.config.preprocessing.audio.output_sampling_rate

        return wavs, sr


class Synthesizer(SynthesizerBase):
    """
    A synthesizer that uses EveryVoice models.
    """

    def __init__(self, vocoder) -> None:
        super().__init__(vocoder)

    def __call__(self, inputs) -> Tuple[np.ndarray, int]:
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
