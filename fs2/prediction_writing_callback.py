from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from loguru import logger
from pytorch_lightning.callbacks import Callback

from .config import FastSpeech2Config
from .type_definitions import SynthesizeOutputFormats


def get_synthesis_output_callbacks(
    output_type: Sequence[SynthesizeOutputFormats],
    output_dir: Path,
    config: FastSpeech2Config,
    output_key: str,
    device: torch.device,
):
    """
    Given a list of desired output file formats, return the proper callbacks
    that will generate those files.
    """
    callbacks: list[Callback] = []
    if SynthesizeOutputFormats.npy in output_type:
        callbacks.append(
            PredictionWritingNpyCallback(
                output_dir=output_dir,
                output_key=output_key,
            )
        )
    if SynthesizeOutputFormats.pt in output_type:
        callbacks.append(
            PredictionWritingPtCallback(
                output_dir=output_dir,
                config=config,
                output_key=output_key,
            )
        )
    if SynthesizeOutputFormats.wav in output_type:
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
    This callback runs inference on a provided text-to-spec model and writes the output to numpy files in the format required (B, K, T) for fine-tuning a hifi-gan model using the author's repository (i.e. not EveryVoice): https://github.com/jik876/hifi-gan
    """

    def __init__(
        self,
        output_dir: Path,
        output_key: str,
    ):
        self.output_key = output_key
        self.save_dir = output_dir / "original_hifigan_spec"
        self.sep = "--"
        logger.info(f"Saving numpy output to {self.save_dir}")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _get_filename(self, basename: str, speaker: str, language: str) -> Path:
        return self.save_dir / self.sep.join([basename, speaker, language, "pred.npy"])

    def on_predict_batch_end(  # pyright: ignore [reportIncompatibleMethodOverride]
        self, _trainer, _pl_module, outputs, batch, _batch_idx, _dataloader_idx=0
    ):
        import numpy as np

        specs = outputs[self.output_key].transpose(1, 2).cpu().numpy()

        for basename, speaker, language, spec, unmasked_len in zip(
            batch["basename"],
            batch["speaker"],
            batch["language"],
            specs,
            outputs["tgt_lens"],
        ):
            np.save(
                self._get_filename(
                    basename=basename,
                    speaker=speaker,
                    language=language,
                ),
                spec[:, :unmasked_len].squeeze(),
            )


class PredictionWritingPtCallback(Callback):
    """
    This callback runs inference on a provided text-to-spec model and saves the resulting Mel spectrograms to disk as pytorch files. These can be used to fine-tune an EveryVoice spec-to-wav model.
    """

    def __init__(
        self,
        output_dir: Path,
        config: FastSpeech2Config,
        output_key: str,
    ):
        self.output_key = output_key
        self.save_dir = output_dir / "synthesized_spec"
        self.config = config
        self.sep = "--"
        logger.info(f"Saving pytorch output to {self.save_dir}")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _get_filename(self, basename: str, speaker: str, language: str) -> Path:
        return self.save_dir / self.sep.join(
            [
                basename,
                speaker,
                language,
                f"spec-pred-{self.config.preprocessing.audio.input_sampling_rate}-{self.config.preprocessing.audio.spec_type}.pt",
            ]
        )

    def on_predict_batch_end(  # pyright: ignore [reportIncompatibleMethodOverride]
        self, _trainer, _pl_module, outputs, batch, _batch_idx, _dataloader_idx=0
    ):
        for basename, speaker, language, data, unmasked_len in zip(
            batch["basename"],
            batch["speaker"],
            batch["language"],
            outputs[self.output_key],
            outputs["tgt_lens"],
        ):
            torch.save(
                data[:unmasked_len].cpu(),
                self._get_filename(
                    basename=basename,
                    speaker=speaker,
                    language=language,
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
        self.save_dir = output_dir / "wav"
        self.config = config
        self.sep = "--"
        logger.info(f"Saving wav output to {self.save_dir}")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading Vocoder from {self.config.training.vocoder_path}")
        if self.config.training.vocoder_path is None:
            import sys

            logger.error(
                "Sorry, no vocoder was provided, please add it to model.config.training.vocoder_path"
                " or as --vocoder-path /path/to/vocoder in the command line"
            )
            sys.exit(1)
        else:
            vocoder = torch.load(
                self.config.training.vocoder_path, map_location=torch.device("cpu")
            )
            if "generator" in vocoder.keys():
                # Necessary when passing --filelist
                from everyvoice.model.vocoder.original_hifigan_helper import get_vocoder

                self.vocoder = get_vocoder(
                    self.config.training.vocoder_path, device=self.device
                )
                sampling_rate_change = (
                    self.config.preprocessing.audio.output_sampling_rate
                    // self.config.preprocessing.audio.input_sampling_rate
                )
                self.output_hop_size = (
                    sampling_rate_change * self.config.preprocessing.audio.fft_hop_size
                )
                self.synthesize = self._infer_generator_universal
            else:
                from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import (
                    HiFiGANConfig,
                )

                self.vocoder = torch.load(self.config.training.vocoder_path)
                vocoder_config: dict | HiFiGANConfig = self.vocoder["hyper_parameters"][
                    "config"
                ]
                if isinstance(vocoder_config, dict):
                    vocoder_config = HiFiGANConfig(**vocoder_config)
                sampling_rate_change = (
                    vocoder_config.preprocessing.audio.output_sampling_rate
                    // vocoder_config.preprocessing.audio.input_sampling_rate
                )
                self.output_hop_size = (
                    sampling_rate_change
                    * vocoder_config.preprocessing.audio.fft_hop_size
                )
                self.synthesize = self._infer_everyvoice

    def _infer_generator_universal(self, outputs) -> Tuple[np.ndarray, int]:
        """
        Generate wavs using the generator_universal model.
        """
        from everyvoice.model.vocoder.original_hifigan_helper import vocoder_infer

        wavs = vocoder_infer(
            outputs[self.output_key],
            self.vocoder,
        )
        sr = self.config.preprocessing.audio.output_sampling_rate
        return wavs, sr

    def _infer_everyvoice(self, outputs) -> Tuple[np.ndarray, int]:
        """
        Generate wavs using Everyvoice model.
        """
        from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import (
            synthesize_data,
        )

        wavs, sr = synthesize_data(outputs[self.output_key], self.vocoder)
        return wavs, sr

    def _get_filename(self, basename: str, speaker: str, language: str) -> Path:
        return self.save_dir / self.sep.join([basename, speaker, language, "pred.wav"])

    def on_predict_batch_end(  # pyright: ignore [reportIncompatibleMethodOverride]
        self,
        _trainer,
        _pl_module,
        outputs,
        batch,
        _batch_idx,
        _dataloader_idx=0,
    ):
        from scipy.io.wavfile import write

        logger.trace("Generating waveform...")

        wavs, sr = self.synthesize(outputs)

        # wavs: [B (batch_size), T (samples)]
        assert (
            wavs.ndim == 2
        ), f"The generated audio contained more than 2 dimensions. First dimension should be B(atch) and the second dimension should be T(ime) in samples. Got {wavs.shape} instead."
        assert wavs.shape[0] == outputs["output"].size(
            0
        ), f"You provided {outputs['output'].size(0)} utterances, but {wavs.shape[0]} audio files were synthesized instead."

        # synthesize 16 bit audio
        # we don't do this higher up in the inference methods
        # because tensorboard logs require audio data as floats
        if (wavs >= -1.0).all() & (wavs <= 1.0).all():
            wavs = wavs * self.config.preprocessing.audio.max_wav_value
            wavs = wavs.astype("int16")

        for basename, speaker, language, wav, unmasked_len in zip(
            batch["basename"],
            batch["speaker"],
            batch["language"],
            wavs,
            outputs["tgt_lens"],
        ):
            write(
                self._get_filename(
                    basename=basename,
                    speaker=speaker,
                    language=language,
                ),
                sr,
                # the vocoder output includes padding so we have to remove that
                wav[: (unmasked_len * self.output_hop_size)],
            )
