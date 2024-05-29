from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import HiFiGANConfig
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.model import HiFiGAN
from everyvoice.text.text_processor import TextProcessor
from loguru import logger
from pympi import TextGrid
from pytorch_lightning.callbacks import Callback

from .config import FastSpeech2Config
from .type_definitions import SynthesizeOutputFormats


def get_synthesis_output_callbacks(
    output_type: Sequence[SynthesizeOutputFormats],
    output_dir: Path,
    config: FastSpeech2Config,
    output_key: str,
    device: torch.device,
    global_step: int,
    vocoder_model: HiFiGAN,
    vocoder_config: HiFiGANConfig,
    vocoder_global_step: int,
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
    if SynthesizeOutputFormats.textgrid in output_type:
        callbacks.append(
            PredictionWritingTextGridCallback(
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
                vocoder_model=vocoder_model,
                vocoder_config=vocoder_config,
                vocoder_global_step=vocoder_global_step,
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
        # We don't truncate or alter the filename here because the basename is
        # already truncated/cleaned in cli/synthesize.py
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
        # We don't truncate or alter the filename here because the basename is
        # already truncated/cleaned in cli/synthesize.py
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


class PredictionWritingTextGridCallback(PredictionWritingCallbackBase):
    """
    This callback runs inference on a provided text-to-spec model and saves the resulting textgrid of the predicted durations to disk. This can be used for evaluation.
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
            file_extension=f"{config.preprocessing.audio.input_sampling_rate}-{config.preprocessing.audio.spec_type}.TextGrid",
            save_dir=output_dir / "textgrids",
        )
        self.text_processor = TextProcessor(config.text)
        self.output_key = output_key
        self.config = config
        logger.info(f"Saving pytorch output to {self.save_dir}")

    def _get_filename(self, basename: str, speaker: str, language: str) -> Path:
        # We don't truncate or alter the filename here because the basename is
        # already truncated/cleaned in cli/synthesize.py
        # the textgrid should not have the global step printed because it is used to fine-tune
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

    def frames_to_seconds(self, frames: int) -> float:
        return (
            frames * self.config.preprocessing.audio.fft_hop_size
        ) / self.config.preprocessing.audio.output_sampling_rate

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
        assert (
            "duration_prediction" in outputs
            and outputs["duration_prediction"] is not None
        )
        for basename, speaker, language, raw_text, text, duration in zip(
            batch["basename"],
            batch["speaker"],
            batch["language"],
            batch["raw_text"],
            batch["text"],  # type: ignore
            outputs["duration_prediction"],
        ):
            # Get all durations in frames
            duration_frames = (
                torch.clamp(torch.round(torch.exp(duration) - 1), min=0).int().tolist()
            )
            # Get all input labels
            text_labels = self.text_processor.decode_tokens(
                text.tolist(), join_character=None
            )
            assert len(duration_frames) == len(
                text_labels
            ), f"can't synthesize {raw_text} because the number of predicted duration steps ({len(duration_frames)}) doesn't equal the number of input text labels ({len(text_labels)})"
            # get the duration of the audio: (sum_of_frames * hop_size) / sample_rate
            xmax_seconds = self.frames_to_seconds(sum(duration_frames))
            # create new textgrid
            new_tg = TextGrid(xmax=xmax_seconds)
            # create the tiers
            words: list[tuple[float, float, str]] = []
            phones: list[tuple[float, float, str]] = []
            raw_text_words = raw_text.split()
            current_word_duration = 0.0
            last_phone_end = 0.0
            last_word_end = 0.0
            phone_tier = new_tg.add_tier("phones")
            phone_annotation_tier = new_tg.add_tier("phone annotations")
            word_tier = new_tg.add_tier("words")
            word_annotation_tier = new_tg.add_tier("word annotations")
            # skip padding
            text_labels_no_padding = [tl for tl in text_labels if tl != "\x80"]
            duration_frames_no_padding = duration_frames[: len(text_labels_no_padding)]
            for label, duration in zip(
                text_labels_no_padding, duration_frames_no_padding
            ):
                # add phone label
                phone_duration = self.frames_to_seconds(duration)
                current_phone_end = last_phone_end + phone_duration
                interval = (last_phone_end, current_phone_end, label)
                phones.append(interval)
                phone_annotation_tier.add_interval(interval[0], interval[1], "")
                phone_tier.add_interval(*interval)
                last_phone_end = current_phone_end
                # accumulate phone to word label
                current_word_duration += phone_duration
                # if label is space or the last phone, add the word and recount
                if label == " " or len(phones) == len(text_labels_no_padding):
                    current_word_end = last_word_end + current_word_duration
                    interval = (
                        last_word_end,
                        current_word_end,
                        raw_text_words[len(words)],
                    )
                    words.append(interval)
                    word_tier.add_interval(*interval)
                    word_annotation_tier.add_interval(interval[0], interval[1], "")
                    last_word_end = current_word_end
                    current_word_duration = 0
            # get the filename
            filename = self._get_filename(
                basename=basename,
                speaker=speaker,
                language=language,
            )
            # write the file
            new_tg.to_file(filename)


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
        vocoder_model: HiFiGAN,
        vocoder_config: HiFiGANConfig,
        vocoder_global_step: int,
    ):
        super().__init__(
            file_extension="pred.wav",
            global_step=global_step,
            save_dir=output_dir / "wav",
        )

        self.output_key = output_key
        self.device = device
        self.config = config
        self.vocoder_model = vocoder_model
        self.vocoder_config = vocoder_config
        sampling_rate_change = (
            vocoder_config.preprocessing.audio.output_sampling_rate
            // vocoder_config.preprocessing.audio.input_sampling_rate
        )
        self.output_hop_size = (
            sampling_rate_change * vocoder_config.preprocessing.audio.fft_hop_size
        )
        self.file_extension = self.sep.join(
            (f"v_ckpt={vocoder_global_step}", self.file_extension)
        )

        logger.info(f"Saving wav output to {self.save_dir}")

    def synthesize_audio(self, outputs):
        from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import (
            synthesize_data,
        )

        sr: int
        wavs: np.ndarray
        output_value = outputs[self.output_key]
        if output_value is not None:
            wavs, sr = synthesize_data(
                output_value, self.vocoder_model, self.vocoder_config
            )
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

        wavs, sr = self.synthesize_audio(outputs)

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
