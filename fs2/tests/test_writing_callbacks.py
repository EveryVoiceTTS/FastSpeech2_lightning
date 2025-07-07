from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
from everyvoice.config.shared_types import ContactInformation
from everyvoice.tests.model_stubs import get_stubbed_vocoder
from everyvoice.tests.stubs import silence_c_stderr
from pympi import TextGrid

from ..config import FastSpeech2Config, FastSpeech2TrainingConfig
from ..prediction_writing_callback import get_synthesis_output_callbacks
from ..type_definitions import SynthesizeOutputFormats
from ..utils import BASENAME_MAX_LENGTH, truncate_basename

try:
    # Accelerate the failing for fetching bundles online, since we don't
    # care about them here in unit testing. This only works since late
    # April 2025, though, so silently ignore if it fails.
    import readalongs.text.make_package as make_package

    make_package.FETCH_BUNDLE_TIMEOUT_SECONDS = 1
except Exception:
    pass


class TestTruncateBasename(TestCase):
    """
    Testing truncate_basename().
    """

    def test_short_name(self):
        """
        Short utterances should produce file names that are not truncated.
        """
        output = truncate_basename("Short utterance")
        self.assertEqual(output, "Short-utterance")

    def test_long_name(self):
        """
        Uttrerances longer than BASENAME_MAX_LENGTH should get truncated and
        should have a sha1 in case two utterances have the same prefix.
        """
        output = truncate_basename("A utterance that is too long")
        self.assertEqual(len(output), BASENAME_MAX_LENGTH + 1 + 8)
        self.assertEqual(output, "A-utterance-that-is--d607fba8")

    def test_limit(self):
        """
        Utterances exactly BASENAME_MAX_LENGTH long should not be truncated.
        """
        input = "A" * BASENAME_MAX_LENGTH
        output = truncate_basename(input)
        self.assertEqual(len(output), BASENAME_MAX_LENGTH)
        self.assertEqual(output, input)

    def test_limit_plus_one(self):
        """
        Testing the edge case where the utterance as one too many characters.
        """
        input = "A" * (BASENAME_MAX_LENGTH + 1)
        output = truncate_basename(input)
        self.assertEqual(len(output), BASENAME_MAX_LENGTH + 1 + 8)

    def test_same_prefix_different_names(self):
        """
        Two long utterances with the same prefix should yield different file names.
        """
        prefix = "A" * BASENAME_MAX_LENGTH
        input1 = prefix + "1"
        input2 = prefix + "2"
        output1 = truncate_basename(input1)
        output2 = truncate_basename(input2)
        self.assertNotEqual(output1, output2)
        self.assertRegex(output1, prefix + r"-[0-9A-Fa-f]{8}")
        self.assertRegex(output2, prefix + r"-[0-9A-Fa-f]{8}")


class WritingTestBase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contact = ContactInformation(
            contact_name="Test Runner", contact_email="info@everyvoice.ca"
        )
        cls.output_key = "output"
        cls.outputs = {
            cls.output_key: torch.ones([2, 500, 80], device="cpu"),
            "duration_prediction": torch.ones([2, 7], device="cpu"),
            "tgt_lens": [
                90,
                490,
            ],
        }
        cls.batch = {
            "basename": [
                "short",
                "This utterance is way too long",
            ],
            "duration_control": [1.0, 1.0],
            "raw_text": ["test", "W̱SÁNEĆ"],
            "text": [
                torch.IntTensor([2, 3, 4, 5, 6, 7, 8], device="cpu"),
                torch.IntTensor([2, 3, 4, 5, 6, 7, 8], device="cpu"),
            ],
            "speaker": [
                "spk1",
                "spk2",
            ],
            "language": [
                "lngA",
                "lngB",
            ],
            "is_last_input_chunk": [1, 1],
        }


class TestWritingSpec(WritingTestBase):
    """
    Testing the callback that writes pt files.
    """

    def test_filenames_not_truncated(self):
        """
        We limit the file name's length to at most BASENAME_MAX_LENGTH in the CLI,
        but the callback does not truncate the basenames passed to it
        """
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            with silence_c_stderr():
                writers = get_synthesis_output_callbacks(
                    [SynthesizeOutputFormats.spec],
                    config=FastSpeech2Config(contact=self.contact),
                    global_step=77,
                    output_dir=tmp_dir,
                    output_key=self.output_key,
                    device=torch.device("cpu"),
                )
            writer = next(iter(writers.values()))
            writer.on_predict_batch_end(
                _trainer=None,
                _pl_module=None,
                outputs=self.outputs,
                batch=self.batch,
                _batch_idx=0,
                _dataloader_idx=0,
            )
            output_dir = writer.save_dir
            # print(output_dir, *output_dir.glob("**"))  # For debugging
            self.assertTrue(output_dir.exists())
            self.assertTrue(
                (
                    output_dir / "short--spk1--lngA--spec-pred-22050-mel-librosa.pt"
                ).exists()
            )
            self.assertTrue(
                (
                    output_dir
                    / "This utterance is way too long--spk2--lngB--spec-pred-22050-mel-librosa.pt"
                ).exists()
            )


class TestWritingTextGrid(WritingTestBase):
    """
    Testing the callback that writes TextGrid files.
    """

    def test_filenames_not_truncated(self):
        """
        We limit the file name's length to at most BASENAME_MAX_LENGTH in the CLI,
        but the callback does not truncate the basenames passed to it
        """
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            with silence_c_stderr():
                writers = get_synthesis_output_callbacks(
                    [SynthesizeOutputFormats.textgrid],
                    config=FastSpeech2Config(contact=self.contact),
                    global_step=77,
                    output_dir=tmp_dir,
                    output_key=self.output_key,
                    device=torch.device("cpu"),
                )
            writer = next(iter(writers.values()))
            writer.on_predict_batch_end(
                _trainer=None,
                _pl_module=None,
                outputs=self.outputs,
                batch=self.batch,
                _batch_idx=0,
                _dataloader_idx=0,
            )
            output_dir = writer.save_dir
            # print(output_dir, *output_dir.glob("**/*"))  # For debugging
            self.assertTrue(output_dir.exists())
            self.assertTrue(
                (output_dir / "short--spk1--lngA--22050-mel-librosa.TextGrid").exists()
            )
            self.assertTrue(
                (
                    output_dir
                    / "This utterance is way too long--spk2--lngB--22050-mel-librosa.TextGrid"
                ).exists()
            )
            tg = TextGrid(
                file_path=(
                    output_dir
                    / "This utterance is way too long--spk2--lngB--22050-mel-librosa.TextGrid"
                )
            )
            tiers = list(tg.get_tiers())
            self.assertEqual(tiers[0].name, "phones")
            self.assertEqual(tiers[1].name, "phone annotations")
            self.assertEqual(tiers[2].name, "words")
            self.assertEqual(tiers[3].name, "word annotations")
            self.assertEqual(tiers[2].intervals[0][2], "W̱SÁNEĆ")


class TestWritingReadAlong(WritingTestBase):
    """
    Testing the callback that writes .readalong files.
    """

    def test_writing_readalong(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            with silence_c_stderr():
                writers = get_synthesis_output_callbacks(
                    [SynthesizeOutputFormats.readalong_xml],
                    config=FastSpeech2Config(contact=self.contact),
                    global_step=77,
                    output_dir=tmp_dir,
                    output_key=self.output_key,
                    device=torch.device("cpu"),
                )
            writer = next(iter(writers.values()))
            writer.on_predict_batch_end(
                _trainer=None,
                _pl_module=None,
                outputs=self.outputs,
                batch=self.batch,
                _batch_idx=0,
                _dataloader_idx=0,
            )
            output_dir = writer.save_dir

            # print(output_dir, *output_dir.glob("**/*"))  # For debugging
            output_files = (
                output_dir / "short--spk1--lngA--22050-mel-librosa.readalong",
                output_dir
                / "This utterance is way too long--spk2--lngB--22050-mel-librosa.readalong",
            )
            for output_file in output_files:
                with self.subTest(output_file=output_file):
                    self.assertTrue(output_file.exists())
                    with open(output_file, "r", encoding="utf8") as f:
                        readalong = f.read()
                    # print(readalong)
                    self.assertIn("<read-along", readalong)
                    self.assertIn('<w time="0.0" dur=', readalong)


class TestWritingOfflineRAS(WritingTestBase):
    """
    Testing the callback that writes Offline HTML readalong files.
    """

    def test_writing_offline_ras(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            vocoder, vocoder_path = get_stubbed_vocoder(tmp_dir)
            with silence_c_stderr():
                writers = get_synthesis_output_callbacks(
                    [SynthesizeOutputFormats.readalong_html],
                    config=FastSpeech2Config(
                        contact=self.contact,
                        training=FastSpeech2TrainingConfig(vocoder_path=vocoder_path),
                    ),
                    global_step=77,
                    output_dir=tmp_dir,
                    output_key=self.output_key,
                    device=torch.device("cpu"),
                    vocoder_model=vocoder,
                    vocoder_config=vocoder.config,
                    vocoder_global_step=10,
                )
            for writer in writers.values():
                writer.on_predict_batch_end(
                    _trainer=None,
                    _pl_module=None,
                    outputs=self.outputs,
                    batch=self.batch,
                    _batch_idx=0,
                    _dataloader_idx=0,
                )
                output_dir = writer.save_dir

            # print(output_dir, *output_dir.glob("**/*"))  # For debugging
            output_files = (
                output_dir / "short--spk1--lngA--22050-mel-librosa.html",
                output_dir
                / "This utterance is way too long--spk2--lngB--22050-mel-librosa.html",
            )
            for output_file in output_files:
                with self.subTest(output_file=output_file):
                    self.assertTrue(output_file.exists())
                    with open(output_file, "r", encoding="utf8") as f:
                        readalong = f.read()
                    # print(readalong)
                    self.assertIn("<read-along", readalong)
                    self.assertIn("<span slot", readalong)


class TestWritingWav(WritingTestBase):
    """
    Testing the callback that writes wav files.
    Note that this test may be expansive.
    """

    def test_filenames_not_truncated(self):
        """
        We limit the file name's length to at most BASENAME_MAX_LENGTH in the CLI,
        but the callback does not truncate the basenames passed to it
        """
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            vocoder, vocoder_path = get_stubbed_vocoder(tmp_dir)

            with silence_c_stderr():
                writers = get_synthesis_output_callbacks(
                    [SynthesizeOutputFormats.wav],
                    config=FastSpeech2Config(
                        contact=self.contact,
                        training=FastSpeech2TrainingConfig(vocoder_path=vocoder_path),
                    ),
                    device=torch.device("cpu"),
                    global_step=77,
                    output_dir=tmp_dir,
                    output_key=self.output_key,
                    vocoder_model=vocoder,
                    vocoder_config=vocoder.config,
                    vocoder_global_step=10,
                )
            writer = next(iter(writers.values()))
            writer.on_predict_batch_end(
                _trainer=None,
                _pl_module=None,
                outputs=self.outputs,
                batch=self.batch,
                _batch_idx=0,
                _dataloader_idx=0,
            )
            output_dir = writer.save_dir
            # print(output_dir, *output_dir.glob("**"))  # For debugging
            self.assertTrue(output_dir.exists())
            self.assertTrue(
                (
                    output_dir / "short--spk1--lngA--ckpt=77--v_ckpt=10--pred.wav"
                ).exists()
            )
            self.assertTrue(
                (
                    output_dir
                    / "This utterance is way too long--spk2--lngB--ckpt=77--v_ckpt=10--pred.wav"
                ).exists()
            )
