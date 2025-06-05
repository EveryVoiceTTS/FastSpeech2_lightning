from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import numpy.typing as npt
import torch
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import HiFiGANConfig
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.model import HiFiGAN
from everyvoice.text.text_processor import TextProcessor
from loguru import logger
from pympi import TextGrid
from pytorch_lightning.callbacks import Callback
from readalongs.api import (
    Token,
    convert_prealigned_text_to_offline_html,
    convert_prealigned_text_to_readalong,
)

from .config import FastSpeech2Config
from .type_definitions import SynthesizeOutputFormats


def get_synthesis_output_callbacks(
    output_type: Sequence[SynthesizeOutputFormats],
    output_dir: Path,
    config: FastSpeech2Config,
    output_key: str,
    device: torch.device,
    global_step: int,
    vocoder_model: Optional[HiFiGAN] = None,
    vocoder_config: Optional[HiFiGANConfig] = None,
    vocoder_global_step: Optional[int] = None,
) -> dict[SynthesizeOutputFormats, Callback]:
    """
    Given a list of desired output file formats, return the proper callbacks
    that will generate those files.
    """
    callbacks: dict[SynthesizeOutputFormats, Callback] = {}
    if (
        SynthesizeOutputFormats.wav in output_type
        or SynthesizeOutputFormats.readalong_html in output_type
    ):
        if (
            vocoder_model is None
            or vocoder_config is None
            or vocoder_global_step is None
        ):
            raise ValueError(
                "We cannot synthesize waveforms without a vocoder. Please ensure that a vocoder is specified."
            )
        callbacks[SynthesizeOutputFormats.wav] = PredictionWritingWavCallback(
            config=config,
            device=device,
            global_step=global_step,
            output_dir=output_dir,
            output_key=output_key,
            vocoder_model=vocoder_model,
            vocoder_config=vocoder_config,
            vocoder_global_step=vocoder_global_step,
        )
    if SynthesizeOutputFormats.spec in output_type:
        callbacks[SynthesizeOutputFormats.spec] = PredictionWritingSpecCallback(
            config=config,
            global_step=global_step,
            output_dir=output_dir,
            output_key=output_key,
        )
    if SynthesizeOutputFormats.textgrid in output_type:
        callbacks[SynthesizeOutputFormats.textgrid] = PredictionWritingTextGridCallback(
            config=config,
            global_step=global_step,
            output_dir=output_dir,
            output_key=output_key,
        )
    if SynthesizeOutputFormats.readalong_xml in output_type:
        callbacks[SynthesizeOutputFormats.readalong_xml] = (
            PredictionWritingReadAlongCallback(
                config=config,
                global_step=global_step,
                output_dir=output_dir,
                output_key=output_key,
            )
        )
    if SynthesizeOutputFormats.readalong_html in output_type:
        wav_callback = callbacks[SynthesizeOutputFormats.wav]
        assert isinstance(wav_callback, PredictionWritingWavCallback)
        callbacks[SynthesizeOutputFormats.readalong_html] = (
            PredictionWritingOfflineRASCallback(
                config=config,
                global_step=global_step,
                output_dir=output_dir,
                output_key=output_key,
                wav_callback=wav_callback,
            )
        )

    return callbacks


class PredictionWritingCallbackBase(Callback):
    def __init__(
        self,
        config: FastSpeech2Config,
        file_extension: str,
        global_step: int,
        save_dir: Path,
        include_global_step_in_filename: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.file_extension = file_extension
        self.global_step = f"ckpt={global_step}"
        self.save_dir = save_dir
        self.sep = "--"
        self.include_global_step_in_filename = include_global_step_in_filename

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def get_filename(
        self,
        basename: str,
        speaker: str,
        language: str,
    ) -> str:
        # We don't truncate or alter the filename here because the basename is
        # already truncated/cleaned in cli/synthesize.py
        name_parts = [basename, speaker, language, self.file_extension]
        if self.include_global_step_in_filename:
            name_parts.insert(-1, self.global_step)
        path = self.save_dir / self.sep.join(name_parts)
        # synthesizing spec allows nested outputs so we may need to make subdirs
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)


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
            config=config,
            global_step=global_step,
            file_extension=f"spec-pred-{config.preprocessing.audio.input_sampling_rate}-{config.preprocessing.audio.spec_type}.pt",
            save_dir=output_dir / "synthesized_spec",
        )

        self.output_key = output_key
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
                data[:unmasked_len]
                .cpu()
                .transpose(0, 1),  # save tensors as [K (bands), T (frames)]
                self.get_filename(basename, speaker, language),
            )


class PredictionWritingAlignedTextCallback(PredictionWritingCallbackBase):
    """
    This callback runs inference on a provided text-to-spec model and saves the
    resulting time-aligned text to file. The output format depends on the subclass's
    implementation of save_aligned_text_to_file.
    """

    def __init__(
        self,
        config: FastSpeech2Config,
        global_step: int,
        output_key: str,
        file_extension: str,
        save_dir: Path,
    ):
        super().__init__(
            config=config,
            global_step=global_step,
            file_extension=file_extension,
            save_dir=save_dir,
        )
        self.text_processor = TextProcessor(config.text)
        self.output_key = output_key
        logger.info(f"Saving text output to {self.save_dir}")

    def save_aligned_text_to_file(
        self,
        max_seconds: float,
        phones: list[tuple[float, float, str]],
        words: list[tuple[float, float, str]],
        basename: str,
        speaker: str,
        language: str,
    ):  # pragma: no cover
        """Subclasses must implement this function to save the aligned text to file
        in the desired format."""
        raise NotImplementedError

    def frames_to_seconds(self, frames: int) -> float:
        return (
            frames * self.config.preprocessing.audio.fft_hop_size
        ) / self.config.preprocessing.audio.output_sampling_rate

    def get_tokens_from_duration_and_labels(
        self,
        duration_predictions: torch.Tensor,
        text: npt.NDArray[np.float32],
        raw_text: str,
    ):
        # Get all durations in frames
        duration_frames = (
            torch.clamp(torch.round(torch.exp(duration_predictions) - 1), min=0)
            .int()
            .tolist()
        )
        # Get all input labels
        tokens: list[int] = text.tolist()
        text_labels = self.text_processor.decode_tokens(tokens, join_character=None)
        assert len(duration_frames) == len(
            text_labels
        ), f"can't synthesize {raw_text} because the number of predicted duration steps ({len(duration_frames)}) doesn't equal the number of input text labels ({len(text_labels)})"
        # get the duration of the audio: (sum_of_frames * hop_size) / sample_rate
        xmax_seconds = self.frames_to_seconds(sum(duration_frames))
        # create the tiers
        words: list[tuple[float, float, str]] = []
        phones: list[tuple[float, float, str]] = []
        raw_text_words = raw_text.split()
        current_word_duration = 0.0
        last_phone_end = 0.0
        last_word_end = 0.0
        # skip padding
        text_labels_no_padding = [tl for tl in text_labels if tl != "\x80"]
        duration_frames_no_padding = duration_frames[: len(text_labels_no_padding)]
        for label, duration in zip(text_labels_no_padding, duration_frames_no_padding):
            # add phone label
            phone_duration = self.frames_to_seconds(duration)
            current_phone_end = last_phone_end + phone_duration
            interval = (last_phone_end, current_phone_end, label)
            phones.append(interval)
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
                last_word_end = current_word_end
                current_word_duration = 0
        return xmax_seconds, phones, words

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
            # Get the phone/word alignment tokens
            xmax_seconds, phones, words = self.get_tokens_from_duration_and_labels(
                duration, text, raw_text
            )

            # Save the output (the subclass has to implement this)
            self.save_aligned_text_to_file(
                xmax_seconds, phones, words, basename, speaker, language
            )


class PredictionWritingTextGridCallback(PredictionWritingAlignedTextCallback):
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
            config=config,
            global_step=global_step,
            output_key=output_key,
            file_extension=f"{config.preprocessing.audio.input_sampling_rate}-{config.preprocessing.audio.spec_type}.TextGrid",
            save_dir=output_dir / "textgrids",
        )

    def save_aligned_text_to_file(
        self,
        max_seconds: float,
        phones: list[tuple[float, float, str]],
        words: list[tuple[float, float, str]],
        basename: str,
        speaker: str,
        language: str,
    ):
        """Save the aligned text as a TextGrid with phones and words layers"""
        new_tg = TextGrid(xmax=max_seconds)
        phone_tier = new_tg.add_tier("phones")
        phone_annotation_tier = new_tg.add_tier("phone annotations")
        for interval in phones:
            phone_annotation_tier.add_interval(interval[0], interval[1], "")
            phone_tier.add_interval(*interval)

        word_tier = new_tg.add_tier("words")
        word_annotation_tier = new_tg.add_tier("word annotations")
        for interval in words:
            word_tier.add_interval(*interval)
            word_annotation_tier.add_interval(interval[0], interval[1], "")

        filename = self.get_filename(basename, speaker, language)
        new_tg.to_file(filename)


class PredictionWritingReadAlongCallback(PredictionWritingAlignedTextCallback):
    """
    This callback runs inference on a provided text-to-spec model and saves the resulting readalong of the predicted durations to disk. Combined with the .wav output, this can be loaded in the ReadAlongs Web-Component for viewing.
    """

    def __init__(
        self,
        config: FastSpeech2Config,
        global_step: int,
        output_dir: Path,
        output_key: str,
    ):
        super().__init__(
            config=config,
            global_step=global_step,
            output_key=output_key,
            file_extension=f"{config.preprocessing.audio.input_sampling_rate}-{config.preprocessing.audio.spec_type}.readalong",
            save_dir=output_dir / "readalongs",
        )
        self.text_processor = TextProcessor(config.text)
        self.output_key = output_key

    def save_aligned_text_to_file(
        self,
        max_seconds: float,
        phones: list[tuple[float, float, str]],
        words: list[tuple[float, float, str]],
        basename: str,
        speaker: str,
        language: str,
    ):
        """Save the aligned text as a .readalong file"""

        ras_tokens: list[Token] = []
        for start, end, label in words:
            if ras_tokens:
                ras_tokens.append(Token(text=" ", is_word=False))
            ras_tokens.append(Token(text=label, time=start, dur=end - start))

        readalong = convert_prealigned_text_to_readalong([ras_tokens], [language])
        filename = self.get_filename(basename, speaker, language)
        with open(filename, "w", encoding="utf8") as f:
            f.write(readalong)


class PredictionWritingOfflineRASCallback(PredictionWritingAlignedTextCallback):
    """
    This callback runs inference on a provided text-to-spec model and saves the
    resulting readalong of the predicted durations to disk as a single file
    Offline HTML. This can be loaded in the ReadAlongs Studio-Web Editor for
    further modification.
    """

    def __init__(
        self,
        config: FastSpeech2Config,
        global_step: int,
        output_dir: Path,
        output_key: str,
        wav_callback: PredictionWritingWavCallback,
    ):
        super().__init__(
            config=config,
            global_step=global_step,
            output_key=output_key,
            file_extension=f"{config.preprocessing.audio.input_sampling_rate}-{config.preprocessing.audio.spec_type}.html",
            save_dir=output_dir / "readalongs",
        )
        self.text_processor = TextProcessor(config.text)
        self.output_key = output_key
        self.wav_callback = wav_callback

    def save_aligned_text_to_file(
        self,
        max_seconds: float,
        phones: list[tuple[float, float, str]],
        words: list[tuple[float, float, str]],
        basename: str,
        speaker: str,
        language: str,
    ):
        """Save the aligned text as an Offline HTML readalong file"""

        ras_tokens: list[Token] = []
        for start, end, label in words:
            if ras_tokens:
                ras_tokens.append(Token(text=" ", is_word=False))
            ras_tokens.append(Token(text=label, time=start, dur=end - start))

        wav_file_name = self.wav_callback.get_filename(basename, speaker, language)
        readalong_html, _readalong_xml = convert_prealigned_text_to_offline_html(
            [ras_tokens],
            wav_file_name,
            [language],
            title="ReadAlong generated using EveryVoice",
        )
        filename = self.get_filename(basename, speaker, language)
        with open(filename, "w", encoding="utf8") as f:
            f.write(readalong_html)


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
            config=config,
            file_extension="pred.wav",
            global_step=global_step,
            save_dir=output_dir / "wav",
            include_global_step_in_filename=True,
        )

        self.output_key = output_key
        self.device = device
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

    def synthesize_audio(self, outputs: dict[str, torch.Tensor | None]):
        from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import (
            synthesize_data,
        )

        sr: int
        wavs: np.ndarray
        output_value = outputs[self.output_key]
        if output_value is None:
            raise ValueError(
                f"{self.output_key} does not exist in the output of your model"
            )

        wavs, sr = synthesize_data(
            output_value.transpose(1, 2), self.vocoder_model, self.vocoder_config
        )

        # wavs: [B (batch_size), C (channels), T (samples)]
        assert (
            wavs.ndim == 3
        ), f"The generated audio did not contain 3 dimensions. First dimension should be B(atch) and the second dimension should be C(hannels) and third dimension should be T(ime) in samples. Got {wavs.shape} instead."
        assert wavs.shape[0] == output_value.size(
            0
        ), f"You provided {output_value.size(0)} utterances, but {wavs.shape[0]} audio files were synthesized instead."

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
        import torchaudio

        logger.trace("Generating waveform...")

        wavs, sr = self.synthesize_audio(outputs)

        full_wav = torch.tensor(())  # Accumulates one full text input before saving
        assert "tgt_lens" in outputs and outputs["tgt_lens"] is not None
        for basename, speaker, language, flag, wav, unmasked_len in zip(
            batch["basename"],
            batch["speaker"],
            batch["language"],
            batch["end_flag"],
            wavs,
            outputs["tgt_lens"],
        ):
            # The vocoder output includes padding, so we have to remove that
            trimmed_wav = wav[:, : (unmasked_len * self.output_hop_size)]

            full_wav = torch.cat((full_wav, trimmed_wav), -1)

            # If we have reached the end of one full text input, save it as a .wav
            if flag:
                torchaudio.save(
                    self.get_filename(basename, speaker, language),
                    full_wav,
                    sr,
                    format="wav",
                    encoding="PCM_S",
                    bits_per_sample=16,
                )
                full_wav = torch.tensor(())
