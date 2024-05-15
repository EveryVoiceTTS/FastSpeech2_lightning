from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from loguru import logger
from pytorch_lightning.callbacks import Callback

from .config import FastSpeech2Config
from .synthesizer import Synthesizer, get_synthesizer
from .type_definitions import SynthesizeOutputFormats


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
    if SynthesizeOutputFormats.spec in output_type:
        callbacks.append(
            PredictionWritingSpecCallback(
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
        path = self.save_dir / self.sep.join(
            [
                basename,
                speaker,
                language,
                self.global_step,
                self.file_extension,
            ]
        )
        path.parent.mkdir(
            parents=True, exist_ok=True
        )  # synthesizing spec allows nested outputs
        return path


class PredictionWritingSpecCallback(PredictionWritingCallbackBase):
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

    def _get_filename(self, basename: str, speaker: str, language: str) -> Path:
        # the spec should not have the global step printed because it is used to fine-tune
        # and the dataloader does not expect a global step in the filename
        path = self.save_dir / self.sep.join(
            [
                basename,
                speaker,
                language,
                self.file_extension,
            ]
        )
        path.parent.mkdir(
            parents=True, exist_ok=True
        )  # synthesizing spec allows nested outputs
        return path

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
            self.synthesizer = get_synthesizer(self.config, self.device)
            vocoder_config = self.config
            # [Example converted to pattern matching](https://stackoverflow.com/a/67524642)
            # [Class Patterns](https://peps.python.org/pep-0634/#class-patterns)
            # [PEP 636 – Structural Pattern Matching: Tutorial](https://peps.python.org/pep-0636)
            match self.synthesizer:
                case Synthesizer():
                    from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import (
                        HiFiGANConfig,
                    )

                    vocoder_config_tmp: dict | HiFiGANConfig = self.synthesizer.vocoder[
                        "hyper_parameters"
                    ]["config"]
                    if isinstance(vocoder_config_tmp, dict):
                        vocoder_config = HiFiGANConfig(**vocoder_config_tmp)
                    vocoder_global_step = self.synthesizer.vocoder.get("global_step", 0)
                    self.file_extension = self.sep.join(
                        (f"v_ckpt={vocoder_global_step}", self.file_extension)
                    )
                case _:
                    raise TypeError(f"We don't yet handle {type(self.synthesizer)}.")

            sampling_rate_change = (
                vocoder_config.preprocessing.audio.output_sampling_rate
                // vocoder_config.preprocessing.audio.input_sampling_rate
            )
            self.output_hop_size = (
                sampling_rate_change * vocoder_config.preprocessing.audio.fft_hop_size
            )

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
        output_value = outputs[self.output_key]
        if output_value is not None:
            wavs, sr = self.synthesizer(output_value)
        else:
            raise ValueError(
                f"{self.output_key} does not exist in the output of your model"
            )

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
