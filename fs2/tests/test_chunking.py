from functools import partial
from pathlib import Path
from string import ascii_lowercase
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
from everyvoice.config.shared_types import ContactInformation
from everyvoice.config.text_config import TextConfig
from everyvoice.tests.model_stubs import get_stubbed_vocoder
from everyvoice.tests.stubs import silence_c_stderr
from everyvoice.text.text_processor import TextProcessor
from pydub import AudioSegment
from pympi import TextGrid

from ..config import FastSpeech2Config, FastSpeech2TrainingConfig
from ..prediction_writing_callback import get_synthesis_output_callbacks
from ..type_definitions import SynthesizeOutputFormats


class TestDuplicateFilename(TestCase):
    def setUp(self):
        self.contact = ContactInformation(
            contact_name="Test Runner", contact_email="info@everyvoice.ca"
        )
        self.output_key = "output"
        self.outputs = {
            self.output_key: torch.ones([3, 500, 80], device="cpu"),
            "duration_prediction": torch.ones([3, 7], device="cpu"),
            "tgt_lens": [490, 490, 490],
        }
        self.batch1 = {
            "basename": ["This is a chunk", "This is another chunk", "This is a chunk"],
            "raw_text": ["This is a chunk", "This is another chunk", "This is a chunk"],
            "text": [
                torch.IntTensor([2, 3, 4, 5, 6, 7, 8], device="cpu"),
                torch.IntTensor([2, 3, 4, 5, 6, 7, 8], device="cpu"),
                torch.IntTensor([2, 3, 4, 5, 6, 7, 8], device="cpu"),
            ],
            "speaker": ["S1", "S1", "S1"],
            "language": ["L1", "L1", "L1"],
            "is_last_input_chunk": [0, 1, 1],
        }

    def test_duplicate_filename(self):
        """
        Tests that the second file is not overwritten when the first chunk is the same for two files.
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

            # Batch 1
            writer = next(iter(writers.values()))
            writer.on_predict_batch_end(
                _trainer=None,
                _pl_module=None,
                outputs=self.outputs,
                batch=self.batch1,
                _batch_idx=0,
                _dataloader_idx=0,
            )
            output_dir = writer.save_dir
            # print(output_dir, *output_dir.glob("**"))  # For debugging
            self.assertTrue(output_dir.exists())
            self.assertTrue(
                (
                    output_dir
                    / "This-is-a-chunkThis--9fc7184d--S1--L1--ckpt=77--v_ckpt=10--pred.wav"
                ).exists()
            )
            self.assertTrue(
                (
                    output_dir / "This-is-a-chunk--S1--L1--ckpt=77--v_ckpt=10--pred.wav"
                ).exists()
            )


class ChunkingTestBase(TestCase):
    @classmethod
    def setUpClass(cls):
        # Define the function that gets the callbacks, get_test_callback
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            vocoder, vocoder_path = get_stubbed_vocoder(tmp_dir)
            config = FastSpeech2Config(
                contact=ContactInformation(
                    contact_name="Test Runner", contact_email="info@everyvoice.ca"
                ),
                training=FastSpeech2TrainingConfig(vocoder_path=vocoder_path),
                text=TextConfig(symbols={"ascii": list(ascii_lowercase)}),
            )
            tp = TextProcessor(config.text)
            cls.get_test_callback = partial(
                get_synthesis_output_callbacks,
                config=config,
                device=torch.device("cpu"),
                global_step=77,
                output_dir=tmp_dir,
                output_key="output",
                vocoder_model=vocoder,
                vocoder_config=vocoder.config,
                vocoder_global_step=10,
            )

        # Define the batches
        cls.outputs = {
            "output": torch.ones([2, 500, 80], device="cpu"),
            "duration_prediction": torch.ones([2, 5], device="cpu"),
            "tgt_lens": [490, 490],
        }
        cls.batch1 = {
            "basename": ["one", "two"],
            "raw_text": ["one", "two"],
            "duration_control": [1.0, 1.0],
            "text": [
                torch.IntTensor(tp.encode_text("one\x80\x80"), device="cpu"),
                torch.IntTensor(tp.encode_text("two\x80\x80"), device="cpu"),
            ],
            "speaker": [
                "S1",
                "S2",
            ],
            "language": [
                "L1",
                "L2",
            ],
            "is_last_input_chunk": [1, 0],
        }
        cls.batch2 = {
            "basename": ["three", "four"],
            "raw_text": ["three", "four"],
            "duration_control": [1.0, 1.0],
            "text": [
                torch.IntTensor(tp.encode_text("three"), device="cpu"),
                torch.IntTensor(tp.encode_text("four\x80"), device="cpu"),
            ],
            "speaker": [
                "S1",
                "S2",
            ],
            "language": [
                "L1",
                "L2",
            ],
            "is_last_input_chunk": [0, 1],
        }


class TestWritingWav(ChunkingTestBase):
    """
    Testing chunking when writing wav files.
    """

    def test_wav_chunks(self):
        """
        Tests the correctness of the output of .wavs for chunked text over multiple batches.
        """
        writers = self.get_test_callback([SynthesizeOutputFormats.wav])

        # Batch 1
        writer = next(iter(writers.values()))
        writer.on_predict_batch_end(
            _trainer=None,
            _pl_module=None,
            outputs=self.outputs,
            batch=self.batch1,
            _batch_idx=0,
            _dataloader_idx=0,
        )
        output_dir = writer.save_dir
        # print(output_dir, *output_dir.glob("**"))  # For debugging
        self.assertTrue(output_dir.exists())

        # Batch 2
        writer = next(iter(writers.values()))
        writer.on_predict_batch_end(
            _trainer=None,
            _pl_module=None,
            outputs=self.outputs,
            batch=self.batch2,
            _batch_idx=1,
            _dataloader_idx=1,
        )

        # Test that the correctly named files were outputted
        self.assertTrue(
            (output_dir / "one--S1--L1--ckpt=77--v_ckpt=10--pred.wav").exists()
        )
        self.assertTrue(
            (output_dir / "twothreefour--S2--L2--ckpt=77--v_ckpt=10--pred.wav").exists()
        )

        # Tests that last_file_written contains the correct most recent filename written
        # This is important for the demo
        self.assertEqual(
            (output_dir / "twothreefour--S2--L2--ckpt=77--v_ckpt=10--pred.wav"),
            Path(writer.last_file_written),
        )

        # Checks that the files have reasonable lengths
        output_one = AudioSegment.from_file(
            output_dir / "one--S1--L1--ckpt=77--v_ckpt=10--pred.wav"
        )
        output_two = AudioSegment.from_file(
            output_dir / "twothreefour--S2--L2--ckpt=77--v_ckpt=10--pred.wav"
        )

        # There are four chunks but two outputs.
        # Output one contains only one chunk, so output_two should be 3 times longer
        self.assertEqual(len(output_one) * 3, len(output_two))


class TestWritingSpec(ChunkingTestBase):
    def test_spec_chunks(self):
        """
        Tests the correctness of the output of spectrograms for chunked text over multiple batches.
        """
        writers = self.get_test_callback([SynthesizeOutputFormats.spec])

        # Batch 1
        writer = next(iter(writers.values()))
        writer.on_predict_batch_end(
            _trainer=None,
            _pl_module=None,
            outputs=self.outputs,
            batch=self.batch1,
            _batch_idx=0,
            _dataloader_idx=0,
        )
        output_dir = writer.save_dir
        # print(output_dir, *output_dir.glob("**"))  # For debugging
        self.assertTrue(output_dir.exists())

        # Batch 2
        writer = next(iter(writers.values()))
        writer.on_predict_batch_end(
            _trainer=None,
            _pl_module=None,
            outputs=self.outputs,
            batch=self.batch2,
            _batch_idx=1,
            _dataloader_idx=1,
        )

        # Test that the correctly named files were outputted
        self.assertTrue(
            (output_dir / "one--S1--L1--spec-pred-22050-mel-librosa.pt").exists()
        )
        self.assertTrue(
            (
                output_dir / "twothreefour--S2--L2--spec-pred-22050-mel-librosa.pt"
            ).exists()
        )

        # Checks that the files have reasonable lengths
        output_one = torch.load(
            output_dir / "one--S1--L1--spec-pred-22050-mel-librosa.pt"
        )
        output_two = torch.load(
            output_dir / "twothreefour--S2--L2--spec-pred-22050-mel-librosa.pt"
        )

        # There are four chunks but two outputs.
        # Output one contains only one chunk, so output_two should be 3 times longer
        self.assertEqual(output_one.size(-1) * 3, output_two.size(-1))


class TestWritingTextGrid(ChunkingTestBase):
    def test_textgrid_chunks(self):
        """
        Tests the correctness of the output of TextGrid files for chunked text over multiple batches.
        """
        writers = self.get_test_callback([SynthesizeOutputFormats.textgrid])

        # Batch 1
        writer = next(iter(writers.values()))
        writer.on_predict_batch_end(
            _trainer=None,
            _pl_module=None,
            outputs=self.outputs,
            batch=self.batch1,
            _batch_idx=0,
            _dataloader_idx=0,
        )
        output_dir = writer.save_dir
        # print(output_dir, *output_dir.glob("**/*"))  # For debugging
        self.assertTrue(output_dir.exists())

        # Batch 2
        writer = next(iter(writers.values()))
        writer.on_predict_batch_end(
            _trainer=None,
            _pl_module=None,
            outputs=self.outputs,
            batch=self.batch2,
            _batch_idx=1,
            _dataloader_idx=1,
        )
        # Test that the correctly named files were outputted
        self.assertTrue(
            (output_dir / "one--S1--L1--22050-mel-librosa.TextGrid").exists()
        )
        self.assertTrue(
            (output_dir / "twothreefour--S2--L2--22050-mel-librosa.TextGrid").exists()
        )

        # Check that the correct words were added to the first TextGrid
        tg = TextGrid(
            file_path=(output_dir / "one--S1--L1--22050-mel-librosa.TextGrid")
        )
        tiers = list(tg.get_tiers())

        phones = [interval[2] for interval in tiers[0].get_all_intervals()]
        for phone, char in zip(list(phones), list("one")):
            self.assertEqual(phone, char)

        words = tiers[2].get_all_intervals()
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0][2], "one")

        # Check that the correct words were added to the second TextGrid
        tg = TextGrid(
            file_path=(output_dir / "twothreefour--S2--L2--22050-mel-librosa.TextGrid")
        )
        tiers = list(tg.get_tiers())

        phones = [interval[2] for interval in tiers[0].get_all_intervals()]
        for phone, char in zip(list(phones), list("twothreefour")):
            self.assertEqual(phone, char)

        words = tiers[2].get_all_intervals()
        self.assertEqual(len(words), 3)
        self.assertEqual(words[0][2], "two")
        self.assertEqual(words[1][2], "three")
        self.assertEqual(words[2][2], "four")
