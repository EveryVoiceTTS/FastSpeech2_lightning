"""Tests for the synthesis output writers, which write whole texts (combined chunks)."""

from csv import DictReader
from functools import partial
from string import ascii_lowercase
from typing import Callable

import pytest
import torch
from everyvoice.config.shared_types import ContactInformation
from everyvoice.config.text_config import TextConfig
from everyvoice.config.type_definitions import SynthesizeOutputFormats as Formats
from everyvoice.text.text_processor import TextProcessor
from everyvoice.utils import slugify, truncate_basename
from pydub import AudioSegment
from pympi import TextGrid

from ..config import FastSpeech2Config
from ..core.synthesize import combine_utterance_chunks
from ..prediction_writers import (
    BatchWriter,
    OutputWriter,
    UtteranceWriter,
    get_synthesis_output_writers,
)
from ..synthesis_types import Utterance

try:
    # Accelerate the failing for fetching bundles online, since we don't
    # care about them here in unit testing. This only works since late
    # April 2025, though, so silently ignore if it fails.
    import readalongs.text.make_package as make_package

    make_package.FETCH_BUNDLE_TIMEOUT_SECONDS = 1
except Exception:
    pass

CONTACT = ContactInformation(
    contact_name="Test Runner", contact_email="info@everyvoice.ca"
)
TEXT_CONFIG = TextConfig(symbols={"ascii": list(ascii_lowercase)})
TEXT_PROCESSOR = TextProcessor(TEXT_CONFIG)
HOP_SIZE = 256  # audio samples per spec frame


def chunk(
    text: str,
    index: int,
    chunk_index: int,
    is_last_chunk: bool,
    speaker: str,
    language: str,
    basename: str | None = None,
    n_frames: int = 490,
) -> Utterance:
    """One synthesized chunk of text, as unbatch() yields: by default, its
    basename is its slugified text, as prepare_data() makes it; its spec is
    n_frames frames of ones, its tokens share the frames, and its wav is
    silence, HOP_SIZE samples per frame."""
    tokens = torch.IntTensor(TEXT_PROCESSOR.encode_text(text))
    durations = torch.full((len(tokens),), n_frames // len(tokens), dtype=torch.int)
    durations[-1] += n_frames - int(durations.sum())
    return Utterance(
        index=index,
        chunk_index=chunk_index,
        is_last_chunk=is_last_chunk,
        text=text,
        basename=basename or truncate_basename(slugify(text)),
        speaker=speaker,
        language=language,
        spec=torch.ones(80, n_frames),
        tokens=tokens,
        durations=durations,
        space_token=TEXT_PROCESSOR.encode_string_tokens([" "])[0],
        wav=torch.zeros(1, n_frames * HOP_SIZE),
        sample_rate=22050,
    )


# Three texts: "one"; "two three four", in three chunks; and a text whose
# basename, as given by a filelist, deliberately doesn't match its text, which
# catches writers that re-derive the basename from the text.
CHUNKS = [
    chunk("one", 0, 0, True, "S1", "L1"),
    chunk("two", 1, 0, False, "S2", "L2"),
    chunk("three", 1, 1, False, "S2", "L2"),
    chunk("four", 1, 2, True, "S2", "L2"),
    chunk("this utterance is way too long", 2, 0, True, "S3", "L3", "spk3_utt002"),
]
# The basename, speaker and language of each text, once combined
TEXTS = [
    ("one", "S1", "L1"),
    ("two-three-four", "S2", "L2"),
    ("spk3_utt002", "S3", "L3"),
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


@pytest.fixture()
def make_writers(tmp_path) -> Callable:
    """get_synthesis_output_writers(), writing to tmp_path"""
    return partial(
        get_synthesis_output_writers,
        config=FastSpeech2Config(contact=CONTACT, text=TEXT_CONFIG),
        global_step=77,
        output_dir=tmp_path,
        vocoder_global_step=10,
    )


# For each output type: its subdirectory, and its file name suffix
FILES = {
    Formats.spec: ("synthesized_spec", "spec-pred-22050-mel-librosa.pt"),
    Formats.textgrid: ("textgrids", "22050-mel-librosa.TextGrid"),
    Formats.readalong_xml: ("readalongs", "22050-mel-librosa.readalong"),
    Formats.readalong_html: ("readalongs", "22050-mel-librosa.html"),
    Formats.wav: ("wav", "ckpt=77--v_ckpt=10--pred.wav"),
}


@pytest.mark.parametrize("output_type", FILES)
def test_filenames(make_writers, tmp_path, output_type):
    """Files are named from the given basenames, not truncated nor re-derived
    from the texts, and from the speaker, language, and model details."""
    writers = make_writers([output_type])
    write_chunks(CHUNKS, *writers.values())
    subdir, suffix = FILES[output_type]
    files = {p.name for p in (tmp_path / subdir).glob(f"*{suffix}")}
    assert files == {f"{b}--{s}--{lang}--{suffix}" for b, s, lang in TEXTS}


@pytest.mark.parametrize("output_type", FILES)
def test_simple_filenames(make_writers, tmp_path, output_type):
    """With simple_filenames=True, only the basename and extension are used,
    even though e.g. the wav writer adds v_ckpt=... to its file_extension."""
    writers = make_writers([output_type], simple_filenames=True)
    write_chunks(CHUNKS, *writers.values())
    subdir, suffix = FILES[output_type]
    extension = suffix[suffix.rindex(".") :]
    files = {p.name for p in (tmp_path / subdir).glob(f"*{extension}")}
    assert files == {f"{basename}{extension}" for basename, _, _ in TEXTS}


def test_spec_lengths(make_writers, tmp_path):
    write_chunks(CHUNKS, *make_writers([Formats.spec]).values())
    spec_dir = tmp_path / "synthesized_spec"
    one = torch.load(spec_dir / "one--S1--L1--spec-pred-22050-mel-librosa.pt")
    two = torch.load(
        spec_dir / "two-three-four--S2--L2--spec-pred-22050-mel-librosa.pt"
    )
    # "two three four" is three chunks, each as long as "one"
    assert one.shape == (80, 490)
    assert two.shape == (80, 3 * 490)


def test_wav_lengths(make_writers, tmp_path):
    writer = make_writers([Formats.wav])[Formats.wav]
    write_chunks(CHUNKS, writer)
    one = tmp_path / "wav" / "one--S1--L1--ckpt=77--v_ckpt=10--pred.wav"
    two = tmp_path / "wav" / "two-three-four--S2--L2--ckpt=77--v_ckpt=10--pred.wav"
    # "two three four" is three chunks, each as long as "one"
    assert len(AudioSegment.from_file(one)) * 3 == len(AudioSegment.from_file(two))
    # The demo plays the last file written
    assert writer.last_file_written == str(
        tmp_path / "wav" / "spk3_utt002--S3--L3--ckpt=77--v_ckpt=10--pred.wav"
    )


def test_textgrid(make_writers, tmp_path):
    write_chunks(CHUNKS, *make_writers([Formats.textgrid]).values())
    textgrid_dir = tmp_path / "textgrids"

    tiers = list(
        TextGrid(
            file_path=textgrid_dir / "one--S1--L1--22050-mel-librosa.TextGrid"
        ).get_tiers()
    )
    assert [tier.name for tier in tiers] == [
        "phones",
        "phone annotations",
        "words",
        "word annotations",
    ]
    assert [label for _, _, label in tiers[0].get_all_intervals()] == list("one")
    assert [label for _, _, label in tiers[2].get_all_intervals()] == ["one"]

    tiers = list(
        TextGrid(
            file_path=textgrid_dir
            / "two-three-four--S2--L2--22050-mel-librosa.TextGrid"
        ).get_tiers()
    )
    # The chunks are separated by spaces, which last 0 frames
    phones = tiers[0].get_all_intervals()
    assert [label for _, _, label in phones] == list("two three four")
    separators = [(start, end) for start, end, label in phones if label == " "]
    assert len(separators) == 2
    assert all(start == end for start, end in separators)
    words = tiers[2].get_all_intervals()
    assert [label for _, _, label in words] == ["two", "three", "four"]
    # The three chunks have the same number of frames
    assert words[0][1] == pytest.approx(words[2][1] - words[1][1])


@pytest.mark.parametrize(
    "output_type, markup",
    [
        (Formats.readalong_xml, '<w time="0.0" dur='),
        (Formats.readalong_html, "<span slot"),
    ],
)
def test_readalongs(make_writers, tmp_path, output_type, markup):
    # For readalong_html, also writes the wav files, which the HTML files embed
    write_chunks(CHUNKS, *make_writers([output_type]).values())
    _subdir, suffix = FILES[output_type]
    for basename, speaker, language in TEXTS:
        readalong = (
            tmp_path / "readalongs" / f"{basename}--{speaker}--{language}--{suffix}"
        ).read_text(encoding="utf8")
        assert "<read-along" in readalong
        assert markup in readalong


def test_duplicate_filename(make_writers, tmp_path):
    """The second file is not overwritten when the first chunk is the same for
    two files."""
    writer = make_writers([Formats.wav])[Formats.wav]
    chunks = [
        chunk("this is a chunk", 0, 0, False, "S1", "L1"),
        chunk("this is another chunk", 0, 1, True, "S1", "L1"),
        chunk("this is a chunk", 1, 0, True, "S1", "L1"),
    ]
    write_chunks(chunks, writer)
    files = [p.name for p in (tmp_path / "wav").glob("*.wav")]
    assert len(files) == 2
    assert "this-is-a-chunk--S1--L1--ckpt=77--v_ckpt=10--pred.wav" in files


def test_writer_interfaces_are_abstract():
    with pytest.raises(TypeError):
        BatchWriter()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        UtteranceWriter()  # type: ignore[abstract]


def test_writer_kinds(make_writers):
    """The writers are plain OutputWriters, driven directly, not Lightning
    callbacks: the loss needs whole batches, the others write whole texts."""
    from pytorch_lightning.callbacks import Callback

    writers = make_writers(list(FILES), return_scores=True, loss_fn=lambda *_: {})
    assert len(writers) == len(FILES) + 1
    for name, writer in writers.items():
        assert isinstance(writer, OutputWriter)
        assert not isinstance(writer, Callback)
        expected = BatchWriter if name == "score" else UtteranceWriter
        assert isinstance(writer, expected), name


class TestScorer:
    @staticmethod
    def make_batch(basename: str) -> dict:
        return {
            "basename": [basename],
            "speaker": ["spk1"],
            "language": ["lngA"],
            "raw_text": [f"text for {basename}"],
            "phone_coverage_score": [0.5],
            "trigram_coverage_score": [0.25],
        }

    def test_loss_fn_required(self, make_writers):
        with pytest.raises(ValueError, match="loss_fn"):
            make_writers([], return_scores=True)

    def test_write_batch_and_finalize(self, make_writers, tmp_path):
        totals = {"low": 1.0, "high": 3.0, "mid": 2.0}
        calls = []

        def fake_loss_fn(outputs, batch, current_epoch):
            calls.append(current_epoch)
            loss_names = ("duration", "spec", "postnet", "attn_ctc", "attn_bin")
            losses = {name: torch.tensor(0.1) for name in loss_names}
            losses["total"] = torch.tensor(totals[batch["basename"][0]])
            return losses

        writers = make_writers([], return_scores=True, loss_fn=fake_loss_fn)
        assert list(writers.keys()) == ["score"]
        scorer = writers["score"]
        for basename in totals:
            scorer.write_batch({}, self.make_batch(basename))
        assert calls == [0, 0, 0]

        scores_file = tmp_path / "scores-77.psv"
        assert not scores_file.exists()
        scorer.finalize()
        with open(scores_file, encoding="utf8") as f:
            rows = list(DictReader(f, delimiter="|"))
        # sorted by decreasing total loss
        assert [row["basename"] for row in rows] == ["high", "mid", "low"]
        assert rows[0]["total"] == "3.0"
        assert rows[0]["raw_text"] == "text for high"
