from csv import DictReader
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from everyvoice.config.shared_types import ContactInformation
from everyvoice.config.type_definitions import SynthesizeOutputFormats
from pympi import TextGrid

from ..config import FastSpeech2Config, FastSpeech2TrainingConfig
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


class WritingTestBase:
    contact = ContactInformation(
        contact_name="Test Runner", contact_email="info@everyvoice.ca"
    )

    @staticmethod
    def utterances() -> list[Utterance]:
        # "spk2_utt002" deliberately does not match slugify(text), so these tests
        # catch writers that ignore the filelist-provided basename and re-derive
        # one from the text instead.
        def utterance(index, text, basename, speaker, language, n_frames):
            return Utterance(
                index=index,
                chunk_index=0,
                is_last_chunk=True,
                text=text,
                basename=basename,
                speaker=speaker,
                language=language,
                spec=torch.ones(80, n_frames),
                tokens=torch.IntTensor([2, 3, 4, 5, 6, 7, 8]),
                durations=torch.full((7,), n_frames // 7, dtype=torch.int),
                space_token=1,
                wav=torch.zeros(1, n_frames * 256),
                sample_rate=22050,
            )

        return [
            utterance(0, "short", "short", "spk1", "lngA", 91),
            utterance(
                1, "This utterance is way too long", "spk2_utt002", "spk2", "lngB", 490
            ),
        ]


class TestWriterClasses(WritingTestBase):
    """
    The writers are plain OutputWriters, driven directly, not Lightning callbacks.
    """

    def test_writer_interfaces_are_abstract(self):
        with pytest.raises(TypeError):
            BatchWriter()  # type: ignore[abstract]
        with pytest.raises(TypeError):
            UtteranceWriter()  # type: ignore[abstract]

    def test_writers_are_not_callbacks(self):
        from pytorch_lightning.callbacks import Callback

        with TemporaryDirectory() as tmp_dir:
            writers = get_synthesis_output_writers(
                [SynthesizeOutputFormats.spec, SynthesizeOutputFormats.textgrid],
                config=FastSpeech2Config(contact=self.contact),
                global_step=77,
                output_dir=Path(tmp_dir),
                return_scores=True,
                loss_fn=lambda *_: {},
            )
            assert len(writers) == 3
            for writer in writers.values():
                assert isinstance(writer, OutputWriter)
                assert not isinstance(writer, Callback)
            # The loss needs whole batches; the spec is written per text
            assert isinstance(writers["score"], BatchWriter)
            assert isinstance(writers[SynthesizeOutputFormats.spec], UtteranceWriter)
            assert isinstance(
                writers[SynthesizeOutputFormats.textgrid], UtteranceWriter
            )


class TestScorer:
    contact = ContactInformation(
        contact_name="Test Runner", contact_email="info@everyvoice.ca"
    )

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

    def test_loss_fn_required(self):
        with TemporaryDirectory() as tmp_dir:
            with pytest.raises(ValueError, match="loss_fn"):
                get_synthesis_output_writers(
                    [],
                    config=FastSpeech2Config(contact=self.contact),
                    global_step=77,
                    output_dir=Path(tmp_dir),
                    return_scores=True,
                )

    def test_write_batch_and_finalize(self):
        totals = {"low": 1.0, "high": 3.0, "mid": 2.0}
        calls = []

        def fake_loss_fn(outputs, batch, current_epoch):
            calls.append(current_epoch)
            loss_names = ("duration", "spec", "postnet", "attn_ctc", "attn_bin")
            losses = {name: torch.tensor(0.1) for name in loss_names}
            losses["total"] = torch.tensor(totals[batch["basename"][0]])
            return losses

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            writers = get_synthesis_output_writers(
                [],
                config=FastSpeech2Config(contact=self.contact),
                global_step=77,
                output_dir=tmp_path,
                return_scores=True,
                loss_fn=fake_loss_fn,
            )
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


class TestWritingSpec(WritingTestBase):
    """
    Testing the writer that writes pt files.
    """

    def make_writer(self, tmp_dir: Path, **kwargs):
        writers = get_synthesis_output_writers(
            [SynthesizeOutputFormats.spec],
            config=FastSpeech2Config(contact=self.contact),
            global_step=77,
            output_dir=tmp_dir,
            **kwargs,
        )
        return next(iter(writers.values()))

    def test_filenames_not_truncated(self, tmp_path):
        """
        We limit the file name's length to at most BASENAME_MAX_LENGTH in the CLI,
        but the writer does not truncate the basenames passed to it
        """
        writer = self.make_writer(tmp_path)
        for utterance in self.utterances():
            writer.write_utterance(utterance)
        output_dir = writer.save_dir
        assert (
            output_dir / "short--spk1--lngA--spec-pred-22050-mel-librosa.pt"
        ).exists()
        spec = torch.load(
            output_dir / "spk2_utt002--spk2--lngB--spec-pred-22050-mel-librosa.pt"
        )
        assert spec.shape == (80, 490)

    def test_simple_filenames(self, tmp_path):
        """
        With simple_filenames=True, only the basename and extension are used.
        """
        writer = self.make_writer(tmp_path, simple_filenames=True)
        for utterance in self.utterances():
            writer.write_utterance(utterance)
        output_dir = writer.save_dir
        assert (output_dir / "short.pt").exists()
        assert (output_dir / "spk2_utt002.pt").exists()


class TestWritingTextGrid(WritingTestBase):
    """
    Testing the writer that writes TextGrid files.
    """

    def test_filenames_not_truncated(self):
        """
        We limit the file name's length to at most BASENAME_MAX_LENGTH in the CLI,
        but the writer does not truncate the basenames passed to it
        """
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            writers = get_synthesis_output_writers(
                [SynthesizeOutputFormats.textgrid],
                config=FastSpeech2Config(contact=self.contact),
                global_step=77,
                output_dir=tmp_dir,
            )
            writer = next(iter(writers.values()))
            for utterance in self.utterances():
                writer.write_utterance(utterance)
            output_dir = writer.save_dir
            # print(output_dir, *output_dir.glob("**/*"))  # For debugging
            assert output_dir.exists()
            assert (
                output_dir / "short--spk1--lngA--22050-mel-librosa.TextGrid"
            ).exists()
            assert (
                output_dir / "spk2_utt002--spk2--lngB--22050-mel-librosa.TextGrid"
            ).exists()
            tg = TextGrid(
                file_path=(
                    output_dir / "spk2_utt002--spk2--lngB--22050-mel-librosa.TextGrid"
                )
            )
            tiers = list(tg.get_tiers())
            assert tiers[0].name == "phones"
            assert tiers[1].name == "phone annotations"
            assert tiers[2].name == "words"
            assert tiers[3].name == "word annotations"
            assert tiers[2].intervals[0][2] == "This"


class TestWritingReadAlong(WritingTestBase):
    """
    Testing the writer that writes .readalong files.
    """

    def test_writing_readalong(self, subtests):
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            writers = get_synthesis_output_writers(
                [SynthesizeOutputFormats.readalong_xml],
                config=FastSpeech2Config(contact=self.contact),
                global_step=77,
                output_dir=tmp_dir,
            )
            writer = next(iter(writers.values()))
            for utterance in self.utterances():
                writer.write_utterance(utterance)
            output_dir = writer.save_dir

            # print(output_dir, *output_dir.glob("**/*"))  # For debugging
            output_files = (
                output_dir / "short--spk1--lngA--22050-mel-librosa.readalong",
                output_dir / "spk2_utt002--spk2--lngB--22050-mel-librosa.readalong",
            )
            for output_file in output_files:
                with subtests.test(output_file=output_file):
                    assert output_file.exists()
                    with open(output_file, "r", encoding="utf8") as f:
                        readalong = f.read()
                    # print(readalong)
                    assert "<read-along" in readalong
                    assert '<w time="0.0" dur=' in readalong


class TestWritingOfflineRAS(WritingTestBase):
    """
    Testing the writer that writes Offline HTML readalong files.
    """

    def test_writing_offline_ras(self, subtests, stubbed_vocoder):
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            vocoder, vocoder_path = stubbed_vocoder
            writers = get_synthesis_output_writers(
                [SynthesizeOutputFormats.readalong_html],
                config=FastSpeech2Config(
                    contact=self.contact,
                    training=FastSpeech2TrainingConfig(vocoder_path=vocoder_path),
                ),
                global_step=77,
                output_dir=tmp_dir,
                vocoder_global_step=10,
            )
            # Also writes the wav files, which the HTML files embed
            for utterance in self.utterances():
                for writer in writers.values():
                    writer.write_utterance(utterance)
            output_dir = writers[SynthesizeOutputFormats.readalong_html].save_dir

            # print(output_dir, *output_dir.glob("**/*"))  # For debugging

            assert output_dir.exists()
            output_files = (
                output_dir / "short--spk1--lngA--22050-mel-librosa.html",
                output_dir / "spk2_utt002--spk2--lngB--22050-mel-librosa.html",
            )
            for output_file in output_files:
                with subtests.test(output_file=output_file):
                    assert output_file.exists()
                    with open(output_file, "r", encoding="utf8") as f:
                        readalong = f.read()
                    # print(readalong)
                    assert "<read-along" in readalong
                    assert "<span slot" in readalong


class TestWritingWav(WritingTestBase):
    """
    Testing the writer that writes wav files.
    Note that this test may be expensive.
    """

    def test_filenames_not_truncated(self, stubbed_vocoder):
        """
        We limit the file name's length to at most BASENAME_MAX_LENGTH in the CLI,
        but the writer does not truncate the basenames passed to it
        """
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            vocoder, vocoder_path = stubbed_vocoder

            writers = get_synthesis_output_writers(
                [SynthesizeOutputFormats.wav],
                config=FastSpeech2Config(
                    contact=self.contact,
                    training=FastSpeech2TrainingConfig(vocoder_path=vocoder_path),
                ),
                global_step=77,
                output_dir=tmp_dir,
                vocoder_global_step=10,
            )
            writer = next(iter(writers.values()))
            for utterance in self.utterances():
                writer.write_utterance(utterance)
            output_dir = writer.save_dir
            # print(output_dir, *output_dir.glob("**/*"))  # For debugging
            assert output_dir.exists()
            assert (
                output_dir / "short--spk1--lngA--ckpt=77--v_ckpt=10--pred.wav"
            ).exists()
            assert (
                output_dir / "spk2_utt002--spk2--lngB--ckpt=77--v_ckpt=10--pred.wav"
            ).exists()

    def test_simple_filenames(self, stubbed_vocoder):
        """
        With simple_filenames=True, only the basename and extension are used,
        even though the wav writer embeds extra info (v_ckpt=...) into
        file_extension after __init__.
        """
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            vocoder, vocoder_path = stubbed_vocoder

            writers = get_synthesis_output_writers(
                [SynthesizeOutputFormats.wav],
                config=FastSpeech2Config(
                    contact=self.contact,
                    training=FastSpeech2TrainingConfig(vocoder_path=vocoder_path),
                ),
                global_step=77,
                output_dir=tmp_dir,
                vocoder_global_step=10,
                simple_filenames=True,
            )
            writer = next(iter(writers.values()))
            for utterance in self.utterances():
                writer.write_utterance(utterance)
            output_dir = writer.save_dir
            assert (output_dir / "short.wav").exists()
            assert (output_dir / "spk2_utt002.wav").exists()
