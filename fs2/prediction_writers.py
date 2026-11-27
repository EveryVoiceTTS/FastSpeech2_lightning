"""Writers that save synthesis outputs to files: specs, wavs, aligned text, losses.

Get them with get_synthesis_output_writers(). Most are UtteranceWriters, which
write each synthesized text from its combined Utterance; LossWriter is a
BatchWriter, which needs whole batches. See synthesize_helper() in
core/synthesize.py, which drives them for the CLI and the demo.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from csv import DictWriter
from pathlib import Path
from typing import Any, Optional

import torch
import torchaudio
from everyvoice import logger
from everyvoice.base_cli.prediction_writer import BasePredictionWriter
from everyvoice.config.type_definitions import SynthesizeOutputFormats
from everyvoice.text.text_processor import TextProcessor
from pympi import TextGrid
from readalongs.api import (
    Token,
    convert_prealigned_text_to_offline_html,
    convert_prealigned_text_to_readalong,
)

from .config import FastSpeech2Config
from .synthesis_types import Utterance


def get_synthesis_output_writers(
    output_types: Sequence[SynthesizeOutputFormats],
    output_dir: Path,
    config: FastSpeech2Config,
    global_step: int,
    vocoder_global_step: Optional[int] = None,
    return_scores=False,
    simple_filenames: bool = False,
    loss_fn: Optional[Callable[..., dict[str, torch.Tensor]]] = None,
) -> dict[SynthesizeOutputFormats | str, "OutputWriter"]:
    """
    Given a list of desired output file formats, return the proper writers
    that will generate those files.

    loss_fn is required when return_scores is True: pass the model's loss
    module, e.g., model.loss. vocoder_global_step is required for wav and
    readalong_html output: it is part of the wav filenames.
    """
    writers: dict[SynthesizeOutputFormats | str, OutputWriter] = {}
    if return_scores:
        if loss_fn is None:
            raise ValueError("loss_fn is required to calculate scores.")
        writers["score"] = LossWriter(
            config=config,
            global_step=global_step,
            output_dir=output_dir,
            loss_fn=loss_fn,
        )
    if (
        SynthesizeOutputFormats.wav in output_types
        or SynthesizeOutputFormats.readalong_html in output_types
    ):
        if vocoder_global_step is None:
            raise ValueError("vocoder_global_step is required to name wav files.")
        writers[SynthesizeOutputFormats.wav] = WavPredictionWriter(
            config=config,
            global_step=global_step,
            output_dir=output_dir,
            vocoder_global_step=vocoder_global_step,
            simple_filenames=simple_filenames,
        )
    if SynthesizeOutputFormats.spec in output_types:
        writers[SynthesizeOutputFormats.spec] = SpecPredictionWriter(
            config=config,
            global_step=global_step,
            output_dir=output_dir,
            simple_filenames=simple_filenames,
        )
    if SynthesizeOutputFormats.textgrid in output_types:
        writers[SynthesizeOutputFormats.textgrid] = TextGridPredictionWriter(
            config=config,
            global_step=global_step,
            output_dir=output_dir,
            simple_filenames=simple_filenames,
        )
    if SynthesizeOutputFormats.readalong_xml in output_types:
        writers[SynthesizeOutputFormats.readalong_xml] = ReadAlongPredictionWriter(
            config=config,
            global_step=global_step,
            output_dir=output_dir,
            simple_filenames=simple_filenames,
        )
    if SynthesizeOutputFormats.readalong_html in output_types:
        wav_writer = writers[SynthesizeOutputFormats.wav]
        assert isinstance(wav_writer, WavPredictionWriter)
        writers[SynthesizeOutputFormats.readalong_html] = OfflineRASPredictionWriter(
            config=config,
            global_step=global_step,
            output_dir=output_dir,
            wav_writer=wav_writer,
            simple_filenames=simple_filenames,
        )

    return writers


class OutputWriter(ABC):
    """Common base of the synthesis output writers.

    A writer is either a BatchWriter, which gets each padded batch of model
    outputs, or an UtteranceWriter, which gets each synthesized text as a
    combined Utterance. Either way, the caller calls finalize() once at the end.
    """

    def finalize(self) -> None:
        """Called once, after the last batch or utterance has been processed."""


class BatchWriter(OutputWriter):
    """A writer that needs whole batches: the caller passes every batch, in
    order, to write_batch()."""

    @abstractmethod
    def write_batch(
        self,
        outputs: dict[str, torch.Tensor | None],
        batch: dict[str, Any],
    ) -> None:
        """Process one batch of model outputs, writing files as needed."""


class UtteranceWriter(OutputWriter):
    """A writer that writes each synthesized text: the caller passes every
    combined Utterance (see combine_utterance_chunks()), in order, to
    write_utterance()."""

    @abstractmethod
    def write_utterance(self, utterance: Utterance) -> None:
        """Write the files for one synthesized text."""


class LossWriter(BatchWriter):
    """
    This writer computes the losses of each batch, which needs teacher forcing, and
    saves them, sorted, to one file at the end.
    """

    def __init__(
        self,
        config: FastSpeech2Config,
        global_step: int,
        output_dir: Path,
        loss_fn: Callable[..., dict[str, torch.Tensor]],
    ):
        super().__init__()
        self.global_step = global_step
        self.save_dir = output_dir
        self.config = config
        self.loss_fn = loss_fn
        logger.info(f"Saving pytorch output to {self.save_dir}")
        self.scores: list[dict] = []

    def _get_filename(self) -> Path:
        path = self.save_dir / f"scores-{self.global_step}.psv"
        path.parent.mkdir(
            parents=True, exist_ok=True
        )  # synthesizing spec allows nested outputs
        return path

    def sort_scores(self):
        self.scores.sort(key=lambda x: (-x["total"], x["trigram_coverage_score"]))

    def finalize(self) -> None:
        self.sort_scores()
        with open(self._get_filename(), "w") as f:
            fieldnames = [
                "basename",
                "speaker",
                "language",
                "total",
                "trigram_coverage_score",
                "duration",
                "spec",
                "postnet",
                "attn_ctc",
                "attn_bin",
                "raw_text",
                "phone_coverage_score",
            ]
            writer = DictWriter(f, fieldnames=fieldnames, delimiter="|")
            writer.writeheader()
            for score in self.scores:
                writer.writerow(score)

    def write_batch(
        self,
        outputs: dict[str, torch.Tensor | None],
        batch: dict[str, Any],
    ) -> None:
        with torch.no_grad():
            # current_epoch=0: this is what model.current_epoch is during
            # inference, it only affects the attn_bin loss warmup.
            losses = self.loss_fn(outputs, batch, 0)
        score = {k: float(v) for k, v in losses.items()}
        score["basename"] = batch["basename"][0]
        score["speaker"] = batch["speaker"][0]
        score["language"] = batch["language"][0]
        score["raw_text"] = batch["raw_text"][0]
        score["phone_coverage_score"] = batch["phone_coverage_score"][0]
        score["trigram_coverage_score"] = batch["trigram_coverage_score"][0]
        self.scores.append(score)


class SpecPredictionWriter(BasePredictionWriter, UtteranceWriter):
    """
    This writer saves the Mel spectrogram of each synthesized text to disk as a
    pytorch file, [K (bands), T (frames)]. These can be used to fine-tune an
    EveryVoice spec-to-wav model.
    """

    def __init__(
        self,
        config: FastSpeech2Config,
        global_step: int,
        output_dir: Path,
        simple_filenames: bool = False,
    ):
        super().__init__(
            global_step=global_step,
            file_extension=f"spec-pred-{config.preprocessing.audio.input_sampling_rate}-{config.preprocessing.audio.spec_type}.pt",
            save_dir=output_dir / "synthesized_spec",
            simple_filenames=simple_filenames,
        )
        self.config = config

        logger.info(f"Saving pytorch output to {self.save_dir}")

    def write_utterance(self, utterance: Utterance) -> None:
        filename = self.get_filename(
            utterance.basename, utterance.speaker, utterance.language
        )
        torch.save(utterance.spec.cpu(), filename)


class AlignedTextPredictionWriter(BasePredictionWriter, UtteranceWriter):
    """
    This writer saves the time-aligned text of each synthesized text to file. The
    output format depends on the subclass's implementation of
    save_aligned_text_to_file.
    """

    def __init__(
        self,
        config: FastSpeech2Config,
        global_step: int,
        file_extension: str,
        save_dir: Path,
        simple_filenames: bool = False,
    ):
        super().__init__(
            global_step=global_step,
            file_extension=file_extension,
            save_dir=save_dir,
            simple_filenames=simple_filenames,
        )
        self.config = config
        self.text_processor = TextProcessor(
            config.text,
            target_text_representation_level=config.model.target_text_representation_level,
        )
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

    def get_phones_and_words(
        self, utterance: Utterance
    ) -> tuple[float, list[tuple[float, float, str]], list[tuple[float, float, str]]]:
        """Align utterance's tokens (phones) and words to its audio.

        Returns: (xmax_seconds, phones, words), where phones and words are lists of
            (start_seconds, end_seconds, label). Words end at each " " token, which
            includes the 0-frame space that separates the chunks of a combined
            utterance, and at the end; they are labelled from utterance.text.
        """
        # utterance.durations are the frames per token the model used to build the
        # spec, with duration_control applied
        duration_frames: list[int] = utterance.durations.tolist()
        tokens: list[int] = utterance.tokens.tolist()
        text_labels = self.text_processor.decode_tokens(tokens, join_character=None)
        assert len(duration_frames) == len(
            text_labels
        ), f"can't align {utterance.text!r} because the number of durations ({len(duration_frames)}) doesn't equal the number of input text labels ({len(text_labels)})"
        # get the duration of the audio: (sum_of_frames * hop_size) / sample_rate
        xmax_seconds = self.frames_to_seconds(sum(duration_frames))
        # create the tiers
        words: list[tuple[float, float, str]] = []
        phones: list[tuple[float, float, str]] = []
        raw_text_words = utterance.text.split()
        current_word_duration = 0.0
        last_phone_end = 0.0
        last_word_end = 0.0
        for label, duration in zip(text_labels, duration_frames):
            # add phone label
            phone_duration = self.frames_to_seconds(duration)
            current_phone_end = last_phone_end + phone_duration
            phones.append((last_phone_end, current_phone_end, label))
            last_phone_end = current_phone_end
            # accumulate phone to word label
            current_word_duration += phone_duration
            # if label is space or the last phone, add the word and recount
            if label == " " or len(phones) == len(text_labels):
                current_word_end = last_word_end + current_word_duration
                words.append(
                    (last_word_end, current_word_end, raw_text_words[len(words)])
                )
                last_word_end = current_word_end
                current_word_duration = 0
        return xmax_seconds, phones, words

    def write_utterance(self, utterance: Utterance) -> None:
        xmax_seconds, phones, words = self.get_phones_and_words(utterance)
        # Save the output to file (the subclass has to implement this)
        self.save_aligned_text_to_file(
            xmax_seconds,
            phones,
            words,
            utterance.basename,
            utterance.speaker,
            utterance.language,
        )


class TextGridPredictionWriter(AlignedTextPredictionWriter):
    """
    This writer saves the predicted durations of each synthesized text as a
    TextGrid, with phones and words tiers. This can be used for evaluation.
    """

    def __init__(
        self,
        config: FastSpeech2Config,
        global_step: int,
        output_dir: Path,
        simple_filenames: bool = False,
    ):
        super().__init__(
            config=config,
            global_step=global_step,
            file_extension=f"{config.preprocessing.audio.input_sampling_rate}-{config.preprocessing.audio.spec_type}.TextGrid",
            save_dir=output_dir / "textgrids",
            simple_filenames=simple_filenames,
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
        for i in range(len(phones)):
            interval = (
                phones[i - 1][1] if i > 0 else 0,
                phones[i][1] if i < len(phones) - 1 else max_seconds,
                phones[i][2],
            )
            phone_annotation_tier.add_interval(interval[0], interval[1], "")
            phone_tier.add_interval(*interval)

        word_tier = new_tg.add_tier("words")
        word_annotation_tier = new_tg.add_tier("word annotations")
        for i in range(len(words)):
            interval = (
                words[i - 1][1] if i > 0 else 0,
                words[i][1] if i < len(words) - 1 else max_seconds,
                words[i][2],
            )
            word_tier.add_interval(*interval)
            word_annotation_tier.add_interval(interval[0], interval[1], "")

        filename = self.get_filename(basename, speaker, language)
        new_tg.to_file(filename)


class ReadAlongPredictionWriter(AlignedTextPredictionWriter):
    """
    This writer saves the predicted durations of each synthesized text as a
    .readalong file. Combined with the .wav output, this can be loaded in the
    ReadAlongs Web-Component for viewing.
    """

    def __init__(
        self,
        config: FastSpeech2Config,
        global_step: int,
        output_dir: Path,
        simple_filenames: bool = False,
    ):
        super().__init__(
            config=config,
            global_step=global_step,
            file_extension=f"{config.preprocessing.audio.input_sampling_rate}-{config.preprocessing.audio.spec_type}.readalong",
            save_dir=output_dir / "readalongs",
            simple_filenames=simple_filenames,
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


class OfflineRASPredictionWriter(AlignedTextPredictionWriter):
    """
    This writer saves the predicted durations of each synthesized text as a
    readalong in a single Offline HTML file, which embeds the wav file written by
    the wav writer. This can be loaded in the ReadAlongs Studio-Web Editor for
    further modification.
    """

    def __init__(
        self,
        config: FastSpeech2Config,
        global_step: int,
        output_dir: Path,
        wav_writer: "WavPredictionWriter",
        simple_filenames: bool = False,
    ):
        super().__init__(
            config=config,
            global_step=global_step,
            file_extension=f"{config.preprocessing.audio.input_sampling_rate}-{config.preprocessing.audio.spec_type}.html",
            save_dir=output_dir / "readalongs",
            simple_filenames=simple_filenames,
        )
        self.wav_writer = wav_writer

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

        wav_file_name = self.wav_writer.get_filename(basename, speaker, language)
        readalong_html, _readalong_xml = convert_prealigned_text_to_offline_html(
            [ras_tokens],
            wav_file_name,
            [language],
            title="ReadAlong generated using EveryVoice",
        )
        filename = self.get_filename(basename, speaker, language)
        with open(filename, "w", encoding="utf8") as f:
            f.write(readalong_html)


class WavPredictionWriter(BasePredictionWriter, UtteranceWriter):
    """
    This writer saves the waveform of each synthesized text as a wav file: run
    spec_to_wav_synthesis() before unbatching, so the utterances have one.
    """

    def __init__(
        self,
        output_dir: Path,
        config: FastSpeech2Config,
        global_step: int,
        vocoder_global_step: int,
        simple_filenames: bool = False,
    ):
        super().__init__(
            file_extension="pred.wav",
            global_step=global_step,
            save_dir=output_dir / "wav",
            include_global_step_in_filename=True,
            simple_filenames=simple_filenames,
        )
        self.config = config

        self.last_file_written: Optional[str] = None  # Necessary for the demo
        self.file_extension: str = self.sep.join(
            (f"v_ckpt={vocoder_global_step}", self.file_extension)
        )

        logger.info(f"Saving wav output to {self.save_dir}")

    def write_utterance(self, utterance: Utterance) -> None:
        if utterance.wav is None or utterance.sample_rate is None:
            raise ValueError(
                f"Utterance {utterance.basename!r} has no waveform: "
                "run spec_to_wav_synthesis() before unbatching."
            )
        filename = self.get_filename(
            utterance.basename, utterance.speaker, utterance.language
        )
        torchaudio.save(
            filename,
            utterance.wav.cpu(),
            utterance.sample_rate,
            format="wav",
            encoding="PCM_S",
            bits_per_sample=16,
        )
        self.last_file_written = filename  # for the demo
