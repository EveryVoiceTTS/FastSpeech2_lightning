import hashlib
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np
import torch
from everyvoice.utils import slugify
from loguru import logger
from pytorch_lightning.callbacks import Callback

from .config import FastSpeech2Config
from .type_definitions import SynthesizeOutputFormats

BASENAME_MAX_LENGTH = 20


def truncate_basename(basename: str) -> str:
    """
    Shortens basename to BASENAME_MAX_LENGTH and uses the rest of basename to generate a sha1.
    This is done to make sure the file name stays short but that two utterances
    starting with the same prefix doesn't get ovverridden.
    """
    basename_cleaned = slugify(basename)
    if len(basename_cleaned) <= BASENAME_MAX_LENGTH:
        return basename_cleaned

    m = hashlib.sha1()
    m.update(bytes(basename, encoding="UTF-8"))
    return basename_cleaned[:BASENAME_MAX_LENGTH] + "-" + m.hexdigest()[:8]


def get_synthesis_output_callbacks(
    output_type: Sequence[SynthesizeOutputFormats],
    output_dir: Path,
    config: FastSpeech2Config,
    output_key: str,
    device: torch.device,
    global_step: int,
):
    """
    Given a list of desired output file formats, return the proper callbacks
    that will generate those files.
    """
    callbacks: list[Callback] = []
    if SynthesizeOutputFormats.npy in output_type:
        callbacks.append(
            PredictionWritingNpyCallback(
                global_step=global_step,
                output_dir=output_dir,
                output_key=output_key,
            )
        )
    if SynthesizeOutputFormats.pt in output_type:
        callbacks.append(
            PredictionWritingPtCallback(
                config=config,
                global_step=global_step,
                output_dir=output_dir,
                output_key=output_key,
            )
        )
    if SynthesizeOutputFormats.wav in output_type:
        callbacks.append(
            PredictionWritingWavCallback(
                config=config,
                device=device,
                global_step=global_step,
                output_dir=output_dir,
                output_key=output_key,
            )
        )

    return callbacks


class PredictionWritingCallbackBase(Callback):
    def __init__(
        self,
        file_extension: str,
        global_step: int,
        save_dir: Path,
    ) -> None:
        super().__init__()
        self.file_extension = file_extension
        self.global_step = f"ckpt={global_step}"
        self.save_dir = save_dir
        self.sep = "--"

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _get_filename(self, basename: str, speaker: str, language: str) -> Path:
        return self.save_dir / self.sep.join(
            [
                truncate_basename(basename),
                speaker,
                language,
                self.global_step,
                self.file_extension,
            ]
        )


class PredictionWritingNpyCallback(PredictionWritingCallbackBase):
    """
    This callback runs inference on a provided text-to-spec model and writes the output to numpy files in the format required (B, K, T) for fine-tuning a hifi-gan model using the author's repository (i.e. not EveryVoice): https://github.com/jik876/hifi-gan
    """

    def __init__(
        self,
        global_step: int,
        output_dir: Path,
        output_key: str,
    ):
        super().__init__(
            file_extension="pred.npy",
            global_step=global_step,
            save_dir=output_dir / "original_hifigan_spec",
        )

        self.output_key = output_key
        logger.info(f"Saving numpy output to {self.save_dir}")

    def on_predict_batch_end(  # pyright: ignore [reportIncompatibleMethodOverride]
        self,
        _trainer,
        _pl_module,
        outputs: dict[str, torch.Tensor | None],
        batch: dict[str, Any],
        _batch_idx: int,
        _dataloader_idx: int = 0,
    ):
        import numpy as np

        assert self.output_key in outputs and outputs[self.output_key] is not None
        specs = outputs[self.output_key].transpose(1, 2).cpu().numpy()  # type: ignore

        assert "tgt_lens" in outputs and outputs["tgt_lens"] is not None
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


class PredictionWritingPtCallback(PredictionWritingCallbackBase):
    """
    This callback runs inference on a provided text-to-spec model and saves the resulting Mel spectrograms to disk as pytorch files. These can be used to fine-tune an EveryVoice spec-to-wav model.
    """

    def __init__(
        self,
        config: FastSpeech2Config,
        global_step: int,
        output_dir: Path,
        output_key: str,
    ):
        super().__init__(
            global_step=global_step,
            file_extension=f"spec-pred-{config.preprocessing.audio.input_sampling_rate}-{config.preprocessing.audio.spec_type}.pt",
            save_dir=output_dir / "synthesized_spec",
        )

        self.output_key = output_key
        self.config = config
        logger.info(f"Saving pytorch output to {self.save_dir}")

    def on_predict_batch_end(  # pyright: ignore [reportIncompatibleMethodOverride]
        self,
        _trainer,
        _pl_module,
        outputs: dict[str, torch.Tensor | None],
        batch: dict[str, Any],
        _batch_idx: int,
        _dataloader_idx: int = 0,
    ):
        assert self.output_key in outputs and outputs[self.output_key] is not None
        assert "tgt_lens" in outputs and outputs["tgt_lens"] is not None
        for basename, speaker, language, data, unmasked_len in zip(
            batch["basename"],
            batch["speaker"],
            batch["language"],
            outputs[self.output_key],  # type: ignore
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


class PredictionWritingWavCallback(PredictionWritingCallbackBase):
    """
    Given text-to-spec, this callback does spec-to-wav and writes wav files.
    """

    def __init__(
        self,
        output_dir: Path,
        config: FastSpeech2Config,
        output_key: str,
        device: torch.device,
        global_step: int,
    ):
        super().__init__(
            file_extension="pred.wav",
            global_step=global_step,
            save_dir=output_dir / "wav",
        )

        self.output_key = output_key
        self.device = device
        self.config = config
        logger.info(f"Saving wav output to {self.save_dir}")

        logger.info(f"Loading Vocoder from {self.config.training.vocoder_path}")
        if self.config.training.vocoder_path is None:
            import sys

            logger.error(
                "No vocoder was provided, please specify "
                "--vocoder-path /path/to/vocoder on the command line."
            )
            sys.exit(1)
        else:
            self.vocoder = torch.load(
                self.config.training.vocoder_path,
                map_location=self.device,
            )
            if "generator" in self.vocoder.keys():
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

                self.file_extension = self.sep.join(
                    ("v=universal", self.file_extension)
                )
            else:
                from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import (
                    HiFiGANConfig,
                )

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

                vocoder_global_step = self.vocoder.get("global_step", 0)
                self.file_extension = self.sep.join(
                    (f"v_ckpt={vocoder_global_step}", self.file_extension)
                )

    def _infer_generator_universal(self, outputs) -> Tuple[np.ndarray, int]:
        """
        Generate wavs using the generator_universal model.
        """
        from everyvoice.model.vocoder.original_hifigan_helper import vocoder_infer

        wavs = vocoder_infer(
            outputs,
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

        wavs, sr = synthesize_data(outputs, self.vocoder)
        return wavs, sr

    def on_predict_batch_end(  # pyright: ignore [reportIncompatibleMethodOverride]
        self,
        _trainer,
        _pl_module,
        outputs: dict[str, torch.Tensor | None],
        batch: dict[str, Any],
        _batch_idx: int,
        _dataloader_idx: int = 0,
    ):
        from scipy.io.wavfile import write

        logger.trace("Generating waveform...")

        sr: int
        wavs: np.ndarray
        wavs, sr = self.synthesize(outputs[self.output_key])

        # wavs: [B (batch_size), T (samples)]
        assert (
            wavs.ndim == 2
        ), f"The generated audio contained more than 2 dimensions. First dimension should be B(atch) and the second dimension should be T(ime) in samples. Got {wavs.shape} instead."
        assert "output" in outputs and outputs["output"] is not None
        assert wavs.shape[0] == outputs["output"].size(
            0
        ), f"You provided {outputs['output'].size(0)} utterances, but {wavs.shape[0]} audio files were synthesized instead."

        # synthesize 16 bit audio
        # we don't do this higher up in the inference methods
        # because tensorboard logs require audio data as floats
        if (wavs >= -1.0).all() & (wavs <= 1.0).all():
            wavs = wavs * self.config.preprocessing.audio.max_wav_value
            wavs = wavs.astype("int16")

        assert "tgt_lens" in outputs and outputs["tgt_lens"] is not None
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
