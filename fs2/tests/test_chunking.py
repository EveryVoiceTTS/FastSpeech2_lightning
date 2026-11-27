from functools import partial
from pathlib import Path
from string import ascii_lowercase
from typing import Callable

import pytest
import torch
from everyvoice.config.shared_types import ContactInformation
from everyvoice.config.text_config import TextConfig
from everyvoice.config.type_definitions import SynthesizeOutputFormats
from everyvoice.text.text_processor import TextProcessor
from everyvoice.utils import slugify, truncate_basename
from pydub import AudioSegment
from pympi import TextGrid

from ..config import FastSpeech2Config, FastSpeech2TrainingConfig
from ..core.synthesize import combine_utterance_chunks
from ..prediction_writers import get_synthesis_output_writers
from ..synthesis_types import Utterance

TEXT_CONFIG = TextConfig(symbols={"ascii": list(ascii_lowercase)})
TEXT_PROCESSOR = TextProcessor(TEXT_CONFIG)
HOP_SIZE = 256  # audio samples per spec frame


@pytest.fixture()
def make_writers(tmp_path, stubbed_vocoder) -> Callable:
    vocoder, vocoder_path = stubbed_vocoder
    config = FastSpeech2Config(
        contact=ContactInformation(
            contact_name="Test Runner", contact_email="info@everyvoice.ca"
        ),
        training=FastSpeech2TrainingConfig(vocoder_path=vocoder_path),
        text=TEXT_CONFIG,
    )
    return partial(
        get_synthesis_output_writers,
        config=config,
        device=torch.device("cpu"),
        global_step=77,
        output_dir=tmp_path,
        output_key="output",
        vocoder_model=vocoder,
        vocoder_config=vocoder.config,
        vocoder_global_step=10,
    )


def chunk(
    text: str,
    index: int,
    chunk_index: int,
    is_last_chunk: bool,
    speaker: str,
    language: str,
    n_frames: int = 490,
) -> Utterance:
    """One synthesized chunk of text, as unbatch() yields: its basename is its
    slugified text, as prepare_data() makes it, its spec is n_frames frames of
    ones, its tokens share the frames, and its wav is silence, HOP_SIZE samples
    per frame."""
    tokens = torch.IntTensor(TEXT_PROCESSOR.encode_text(text))
    durations = torch.full((len(tokens),), n_frames // len(tokens), dtype=torch.int)
    durations[-1] += n_frames - int(durations.sum())
    return Utterance(
        index=index,
        chunk_index=chunk_index,
        is_last_chunk=is_last_chunk,
        text=text,
        basename=truncate_basename(slugify(text)),
        speaker=speaker,
        language=language,
        spec=torch.ones(80, n_frames),
        tokens=tokens,
        durations=durations,
        space_token=TEXT_PROCESSOR.encode_string_tokens([" "])[0],
        wav=torch.zeros(1, n_frames * HOP_SIZE),
        sample_rate=22050,
    )


# Four chunks of two texts: "one", and "two three four"
CHUNKS = [
    chunk("one", 0, 0, True, "S1", "L1"),
    chunk("two", 1, 0, False, "S2", "L2"),
    chunk("three", 1, 1, False, "S2", "L2"),
    chunk("four", 1, 2, True, "S2", "L2"),
]


def write_chunks(chunks: list[Utterance], *writers) -> None:
    """Combine chunks into texts and write them with each writer, in order, as
    synthesize_helper() does: e.g. the readalong HTML writer embeds the wav file,
    so the wav writer must come first."""
    for utterance in combine_utterance_chunks(chunks):
        for writer in writers:
            writer.write_utterance(utterance)
    for writer in writers:
        writer.finalize()


class TestDuplicateFilename:
    def test_duplicate_filename(self, make_writers):
        """
        Tests that the second file is not overwritten when the first chunk is the
        same for two files.
        """
        writers = make_writers([SynthesizeOutputFormats.wav])
        writer = writers[SynthesizeOutputFormats.wav]
        chunks = [
            chunk("this is a chunk", 0, 0, False, "S1", "L1"),
            chunk("this is another chunk", 0, 1, True, "S1", "L1"),
            chunk("this is a chunk", 1, 0, True, "S1", "L1"),
        ]
        write_chunks(chunks, writer)
        output_dir = writer.save_dir
        files = sorted(path.name for path in output_dir.glob("*.wav"))
        assert len(files) == 2
        assert "this-is-a-chunk--S1--L1--ckpt=77--v_ckpt=10--pred.wav" in files


class TestWritingWav:
    """
    Testing chunking when writing wav files.
    """

    def test_wav_chunks(self, make_writers):
        """
        Tests the correctness of the output of .wavs for chunked text.
        """
        writers = make_writers([SynthesizeOutputFormats.wav])
        writer = writers[SynthesizeOutputFormats.wav]
        write_chunks(CHUNKS, writer)
        output_dir = writer.save_dir

        # Test that the correctly named files were outputted
        assert (output_dir / "one--S1--L1--ckpt=77--v_ckpt=10--pred.wav").exists()
        assert (
            output_dir / "two-three-four--S2--L2--ckpt=77--v_ckpt=10--pred.wav"
        ).exists()

        # Tests that last_file_written contains the correct most recent filename
        # written. This is important for the demo
        assert (
            output_dir / "two-three-four--S2--L2--ckpt=77--v_ckpt=10--pred.wav"
        ) == Path(writer.last_file_written)

        # Checks that the files have reasonable lengths
        output_one = AudioSegment.from_file(
            output_dir / "one--S1--L1--ckpt=77--v_ckpt=10--pred.wav"
        )
        output_two = AudioSegment.from_file(
            output_dir / "two-three-four--S2--L2--ckpt=77--v_ckpt=10--pred.wav"
        )

        # There are four chunks but two outputs.
        # Output one contains only one chunk, so output_two should be 3 times longer
        assert len(output_one) * 3 == len(output_two)


class TestWritingSpec:
    def test_spec_chunks(self, make_writers):
        """
        Tests the correctness of the output of spectrograms for chunked text.
        """
        writers = make_writers([SynthesizeOutputFormats.spec])
        writer = next(iter(writers.values()))
        write_chunks(CHUNKS, writer)
        output_dir = writer.save_dir

        # Test that the correctly named files were outputted: the chunks' texts
        # are joined with spaces, as in "two three four"
        assert (output_dir / "one--S1--L1--spec-pred-22050-mel-librosa.pt").exists()
        assert (
            output_dir / "two-three-four--S2--L2--spec-pred-22050-mel-librosa.pt"
        ).exists()

        # Checks that the files have reasonable lengths
        output_one = torch.load(
            output_dir / "one--S1--L1--spec-pred-22050-mel-librosa.pt"
        )
        output_two = torch.load(
            output_dir / "two-three-four--S2--L2--spec-pred-22050-mel-librosa.pt"
        )

        # There are four chunks but two outputs.
        # Output one contains only one chunk, so output_two should be 3 times longer
        assert output_one.size(-1) * 3 == output_two.size(-1)


class TestWritingTextGrid:
    def test_textgrid_chunks(self, make_writers):
        """
        Tests the correctness of the output of TextGrid files for chunked text.
        """
        writers = make_writers([SynthesizeOutputFormats.textgrid])
        writer = next(iter(writers.values()))
        write_chunks(CHUNKS, writer)
        output_dir = writer.save_dir

        # Test that the correctly named files were outputted
        assert (output_dir / "one--S1--L1--22050-mel-librosa.TextGrid").exists()
        assert (
            output_dir / "two-three-four--S2--L2--22050-mel-librosa.TextGrid"
        ).exists()

        # Check that the correct words were added to the first TextGrid
        tg = TextGrid(
            file_path=(output_dir / "one--S1--L1--22050-mel-librosa.TextGrid")
        )
        tiers = list(tg.get_tiers())

        phones = [interval[2] for interval in tiers[0].get_all_intervals()]
        assert phones == list("one")

        words = tiers[2].get_all_intervals()
        assert len(words) == 1
        assert words[0][2] == "one"

        # Check that the correct words were added to the second TextGrid
        tg = TextGrid(
            file_path=(
                output_dir / "two-three-four--S2--L2--22050-mel-librosa.TextGrid"
            )
        )
        tiers = list(tg.get_tiers())

        # The chunks are separated by spaces, which last 0 frames
        phone_intervals = tiers[0].get_all_intervals()
        assert [interval[2] for interval in phone_intervals] == list("two three four")
        separators = [interval for interval in phone_intervals if interval[2] == " "]
        assert len(separators) == 2
        assert all(start == end for start, end, _ in separators)

        words = tiers[2].get_all_intervals()
        assert [word[2] for word in words] == ["two", "three", "four"]
        # The three chunks have the same number of frames
        assert words[0][1] == pytest.approx(words[2][1] - words[1][1])


class TestWritingReadAlongXML:
    def test_writing_readalong(self, subtests, make_writers):
        writers = make_writers([SynthesizeOutputFormats.readalong_xml])
        writer = next(iter(writers.values()))
        write_chunks(CHUNKS, writer)
        output_dir = writer.save_dir

        # Test that the correctly named files were outputted
        output_files = (
            output_dir / "one--S1--L1--22050-mel-librosa.readalong",
            output_dir / "two-three-four--S2--L2--22050-mel-librosa.readalong",
        )
        for output_file in output_files:
            with subtests.test(output_file=output_file):
                assert output_file.exists()
                with open(output_file, "r", encoding="utf8") as f:
                    readalong = f.read()
                assert "<read-along" in readalong
                assert '<w time="0.0" dur=' in readalong


class TestWritingReadAlongHTML:
    def test_writing_readalong(self, subtests, make_writers) -> None:
        writers = make_writers([SynthesizeOutputFormats.readalong_html])
        # Also writes the wav files, which the HTML files embed
        write_chunks(CHUNKS, *writers.values())
        writer = writers[SynthesizeOutputFormats.readalong_html]
        output_dir = writer.save_dir

        # Test that the correctly named files were outputted
        output_file_basenames = (
            "one--S1--L1--22050-mel-librosa.html",
            "two-three-four--S2--L2--22050-mel-librosa.html",
        )
        for output_file_basename in output_file_basenames:
            output_file = output_dir / output_file_basename
            with subtests.test(output_file=output_file):
                assert output_file.exists()
                with open(output_file, "r", encoding="utf8") as f:
                    readalong = f.read()
                assert "<read-along" in readalong
                assert "<span slot" in readalong
