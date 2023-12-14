import os
from pathlib import Path
from typing import List, Sequence

import torch
from loguru import logger
from pytorch_lightning.callbacks import Callback

from .config import FastSpeech2Config
from .synthesis_outputs import SynthesisOutputs


def get_synthesis_output_callbacks(
    output_type: Sequence[SynthesisOutputs],
    output_dir: Path,
    config: FastSpeech2Config,
    output_key: str,
    device: torch.device,
):
    """
    Given a list of desired output file format, create the proper callbacks
    that will generate those files.
    """
    callbacks: List[Callback] = []
    if SynthesisOutputs.npy in output_type:
        callbacks.append(
            PredictionWritingNpyCallback(
                output_dir=output_dir,
                output_key=output_key,
            )
        )
    if SynthesisOutputs.pt in output_type:
        callbacks.append(
            PredictionWritingPtCallback(
                output_dir=output_dir,
                config=config,
                output_key=output_key,
            )
        )
    if SynthesisOutputs.wav in output_type:
        callbacks.append(
            PredictionWritingWavCallback(
                output_dir=output_dir,
                config=config,
                output_key=output_key,
                device=device,
            )
        )

    return callbacks


class PredictionWritingNpyCallback(Callback):
    """
    Given text-to-spec, this callback does spec-to-wav and writes numpy files.
    """

    def __init__(
        self,
        output_dir: Path,
        output_key: str,
    ):
        self.output_key = output_key
        self.save_dir = output_dir
        self.sep = "--"
        logger.info(f"Saving numpy output to {self.save_dir / 'synthesized_spec'}")
        (self.save_dir / "original_hifigan_spec").mkdir(parents=True, exist_ok=True)

    def on_predict_batch_end(
        self, _trainer, _pl_module, outputs, batch, _batch_idx, _dataloader_idx=0
    ):
        import numpy as np

        specs = outputs[self.output_key].transpose(1, 2).cpu().numpy()

        for b in range(batch["text"].size(0)):
            basename = batch["basename"][b]
            speaker = batch["speaker"][b]
            language = batch["language"][b]
            unmasked_len = outputs["tgt_lens"][
                b
            ]  # the vocoder output includes padding so we have to remove that
            np.save(
                self.save_dir
                / "original_hifigan_spec"
                / self.sep.join([basename, speaker, language, "pred.npy"]),
                specs[b][:, :unmasked_len].squeeze(),
            )


class PredictionWritingPtCallback(Callback):
    """
    Given text-to-spec, this callback does spec-to-wav and writes pytorch files.
    """

    def __init__(
        self,
        output_dir: Path,
        config: FastSpeech2Config,
        output_key: str,
    ):
        self.output_key = output_key
        self.save_dir = output_dir
        self.config = config
        self.sep = "--"
        logger.info(f"Saving pytorch output to {self.save_dir / 'synthesized_spec'}")
        (self.save_dir / "synthesized_spec").mkdir(parents=True, exist_ok=True)

    def on_predict_batch_end(
        self, _trainer, _pl_module, outputs, batch, _batch_idx, _dataloader_idx=0
    ):
        for b in range(batch["text"].size(0)):
            basename = batch["basename"][b]
            speaker = batch["speaker"][b]
            language = batch["language"][b]
            unmasked_len = outputs["tgt_lens"][
                b
            ]  # the vocoder output includes padding so we have to remove that
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


class PredictionWritingWavCallback(Callback):
    """
    Given text-to-spec, this callback does spec-to-wav and writes wav files.
    """

    def __init__(
        self,
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
        logger.info(f"Saving wav output to {self.save_dir / 'synthesized_spec'}")
        (self.save_dir / "wav").mkdir(parents=True, exist_ok=True)

    def on_predict_batch_end(
        self, _trainer, _pl_module, outputs, batch, _batch_idx, _dataloader_idx=0
    ):
        from scipy.io.wavfile import write

        if (
            os.path.basename(self.config.training.vocoder_path)
            == "generator_universal.pth.tar"
        ):
            from everyvoice.model.vocoder.original_hifigan_helper import (
                get_vocoder,
                vocoder_infer,
            )

            logger.info(f"Loading Vocoder from {self.config.training.vocoder_path}")
            vocoder = get_vocoder(self.config.training.vocoder_path, device=self.device)
            logger.info("Generating waveform...")
            wavs = vocoder_infer(
                outputs[self.output_key],
                vocoder,
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

            vocoder = torch.load(self.config.training.vocoder_path)
            vocoder_config: HiFiGANConfig = vocoder["config"]  # type: ignore
            sampling_rate_change = (
                vocoder_config.preprocessing.audio.output_sampling_rate
                // vocoder_config.preprocessing.audio.input_sampling_rate
            )
            output_hop_size = (
                sampling_rate_change * vocoder_config.preprocessing.audio.fft_hop_size
            )
            wavs, sr = synthesize_data(outputs[self.output_key], vocoder)
            # synthesize 16 bit audio
            if wavs.dtype != "int16":
                wavs = wavs * self.config.preprocessing.audio.max_wav_value
                wavs = wavs.astype("int16")

        for b in range(batch["text"].size(0)):
            basename = batch["basename"][b]
            speaker = batch["speaker"][b]
            language = batch["language"][b]
            unmasked_len = outputs["tgt_lens"][
                b
            ]  # the vocoder output includes padding so we have to remove that

            write(
                self.save_dir
                / "wav"
                / self.sep.join([basename, speaker, language, "pred.wav"]),
                sr,
                wavs[b][: (unmasked_len * output_hop_size)],
            )
