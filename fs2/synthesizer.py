from typing import Tuple

import numpy as np
import torch
from loguru import logger

from .config import FastSpeech2Config


class Synthesizer:
    def __init__(
        self,
        config: FastSpeech2Config,
        device: torch.device,
    ):
        self.device = device
        self.config = config

        if self.config.training.vocoder_path is None:
            import sys

            logger.error(
                "No vocoder was provided, please specify "
                "--vocoder-path /path/to/vocoder on the command line."
            )
            sys.exit(1)
        else:
            self.vocoder = torch.load(
                self.config.training.vocoder_path, map_location=self.device
            )
            if "generator" in self.vocoder.keys():
                # Necessary when passing --filelist
                from everyvoice.model.vocoder.original_hifigan_helper import get_vocoder

                self.vocoder = get_vocoder(
                    self.config.training.vocoder_path, device=self.device
                )
                self.synthesize = self._infer_generator_universal
            else:
                self.synthesize = self._infer_everyvoice

    def _infer_generator_universal(self, inputs) -> Tuple[np.ndarray, int]:
        """
        Generate wavs using the generator_universal model.
        """
        from everyvoice.model.vocoder.original_hifigan_helper import vocoder_infer

        wavs = vocoder_infer(inputs, self.vocoder)
        sr = self.config.preprocessing.audio.output_sampling_rate
        return wavs, sr

    def _infer_everyvoice(self, inputs) -> Tuple[np.ndarray, int]:
        """
        Generate wavs using Everyvoice model.
        """
        from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import (
            synthesize_data,
        )

        wavs, sr = synthesize_data(inputs, self.vocoder)
        return wavs, sr

    def __call__(self, inputs) -> Tuple[np.ndarray, int]:
        return self.synthesize(inputs)
