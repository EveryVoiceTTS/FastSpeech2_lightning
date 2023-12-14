import os
from pathlib import Path
from typing import Sequence

import torch
from loguru import logger
from pytorch_lightning.callbacks import Callback

from .config import FastSpeech2Config
from .synthesis_outputs import SynthesisOutputs


class PredictionWritingCallback(Callback):
    """
    Given text-to-spec, this callback does spec-to-wav.
    """

    def __init__(
        self,
        output_types: Sequence[SynthesisOutputs],
        output_dir: Path,
        config: FastSpeech2Config,
        output_key: str,
        device: torch.device,
    ):
        self.output_key = output_key
        self.device = device
        self.save_dir = output_dir
        self.config = config
        self.sep = "--"
        self.output_types: Sequence[SynthesisOutputs] = output_types
        logger.info(f"Saving output to {self.save_dir / 'synthesized_spec'}")
        if "pt" in self.output_types:
            (self.save_dir / "synthesized_spec").mkdir(parents=True, exist_ok=True)
        if "npy" in self.output_types:
            (self.save_dir / "original_hifigan_spec").mkdir(parents=True, exist_ok=True)
        if "wav" in self.output_types:
            (self.save_dir / "wav").mkdir(parents=True, exist_ok=True)

    def on_predict_batch_end(
        self, _trainer, _pl_module, outputs, batch, _batch_idx, _dataloader_idx=0
    ):
        if "wav" in self.output_types:
            if (
                os.path.basename(self.config.training.vocoder_path)
                == "generator_universal.pth.tar"
            ):
                from everyvoice.model.vocoder.original_hifigan_helper import (
                    get_vocoder,
                    vocoder_infer,
                )

                logger.info(f"Loading Vocoder from {self.config.training.vocoder_path}")
                ckpt = get_vocoder(
                    self.config.training.vocoder_path, device=self.device
                )
                logger.info("Generating waveform...")
                wavs = vocoder_infer(
                    outputs[self.output_key],
                    ckpt,
                )
                sr = self.config.preprocessing.audio.output_sampling_rate
                # Necessary when passing --filelist
                sampling_rate_change = (
                    self.config.preprocessing.audio.output_sampling_rate
                    // self.config.preprocessing.audio.input_sampling_rate
                )
                output_hop_size = (
                    sampling_rate_change * self.config.preprocessing.audio.fft_hop_size
                )
            else:
                from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import (
                    HiFiGANConfig,
                )
                from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import (
                    synthesize_data,
                )

                ckpt = torch.load(self.config.training.vocoder_path)
                vocoder_config: HiFiGANConfig = ckpt["config"]  # type: ignore
                sampling_rate_change = (
                    vocoder_config.preprocessing.audio.output_sampling_rate
                    // vocoder_config.preprocessing.audio.input_sampling_rate
                )
                output_hop_size = (
                    sampling_rate_change
                    * vocoder_config.preprocessing.audio.fft_hop_size
                )
                wavs, sr = synthesize_data(outputs[self.output_key], ckpt)
                # synthesize 16 bit audio
                if wavs.dtype != "int16":
                    wavs = wavs * self.config.preprocessing.audio.max_wav_value
                    wavs = wavs.astype("int16")

        if "npy" in self.output_types:
            import numpy as np

            specs = outputs[self.output_key].transpose(1, 2).cpu().numpy()

        for b in range(batch["text"].size(0)):
            basename = batch["basename"][b]
            speaker = batch["speaker"][b]
            language = batch["language"][b]
            unmasked_len = outputs["tgt_lens"][
                b
            ]  # the vocoder output includes padding so we have to remove that
            if "pt" in self.output_types:
                torch.save(
                    outputs[self.output_key][b][:unmasked_len].transpose(0, 1).cpu(),
                    self.save_dir
                    / "synthesized_spec"
                    / self.sep.join(
                        [
                            basename,
                            speaker,
                            language,
                            f"spec-pred-{self.config.preprocessing.audio.input_sampling_rate}-{self.config.preprocessing.audio.spec_type}.pt",
                        ]
                    ),
                )
            if "wav" in self.output_types:
                from scipy.io.wavfile import write

                write(
                    self.save_dir
                    / "wav"
                    / self.sep.join([basename, speaker, language, "pred.wav"]),
                    sr,
                    wavs[b][: (unmasked_len * output_hop_size)],
                )
            if "npy" in self.output_types:
                np.save(
                    self.save_dir
                    / "original_hifigan_spec"
                    / self.sep.join([basename, speaker, language, "pred.npy"]),
                    specs[b][:, :unmasked_len].squeeze(),
                )
