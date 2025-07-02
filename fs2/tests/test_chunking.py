from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
from everyvoice.config.shared_types import ContactInformation
from everyvoice.tests.model_stubs import get_stubbed_vocoder
from everyvoice.tests.stubs import silence_c_stderr
from pydub import AudioSegment

from ..config import FastSpeech2Config, FastSpeech2TrainingConfig
from ..prediction_writing_callback import get_synthesis_output_callbacks
from ..type_definitions import SynthesizeOutputFormats


class ChunkingTestBase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contact = ContactInformation(
            contact_name="Test Runner", contact_email="info@everyvoice.ca"
        )
        cls.output_key = "output"
        cls.outputs = {
            cls.output_key: torch.ones([2, 500, 80], device="cpu"),
            "duration_prediction": torch.ones([2, 7], device="cpu"),
            "tgt_lens": [490, 490],
        }
        cls.batch1 = {
            "basename": ["one", "two"],
            "raw_text": ["This is chunk one.", "This is chunk two."],
            "text": [
                torch.IntTensor([2, 3, 4, 5, 6, 7, 8], device="cpu"),
                torch.IntTensor([2, 3, 4, 5, 6, 7, 8], device="cpu"),
            ],
            "speaker": [
                "S1",
                "S2",
            ],
            "language": [
                "L1",
                "L2",
            ],
            "last_input_chunk": [1, 0],
        }
        cls.batch2 = {
            "basename": ["three", "four"],
            "raw_text": ["This is chunk three.", "This is chunk four."],
            "text": [
                torch.IntTensor([2, 3, 4, 5, 6, 7, 8], device="cpu"),
                torch.IntTensor([2, 3, 4, 5, 6, 7, 8], device="cpu"),
            ],
            "speaker": [
                "S1",
                "S2",
            ],
            "language": [
                "L1",
                "L2",
            ],
            "last_input_chunk": [0, 1],
        }


class TestWritingWav(ChunkingTestBase):
    """
    Testing chunking when writing wav files.
    """

    def test_wav_chunks(self):
        """
        Tests the correctness of the output of .wavs for chunked text over multiple batches.
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

            # Test that the correctly named files were and weren't outputted
            # The outputted files should be named after the first chunk from each text input
            self.assertTrue(
                (output_dir / "one--S1--L1--ckpt=77--v_ckpt=10--pred.wav").exists()
            )
            self.assertTrue(
                (output_dir / "two--S2--L2--ckpt=77--v_ckpt=10--pred.wav").exists()
            )
            self.assertFalse(
                (output_dir / "three--S1--L1--ckpt=77--v_ckpt=10--pred.wav").exists()
            )
            self.assertFalse(
                (output_dir / "four--S2--L2--ckpt=77--v_ckpt=10--pred.wav").exists()
            )

            # Tests that last_file_written contains the correct most recent filename written
            # This is important for the demo
            self.assertEqual(
                (output_dir / "two--S2--L2--ckpt=77--v_ckpt=10--pred.wav"),
                Path(writer.last_file_written),
            )

            # Checks that the files have reasonable lengths
            output_one = AudioSegment.from_file(
                output_dir / "one--S1--L1--ckpt=77--v_ckpt=10--pred.wav"
            )
            output_two = AudioSegment.from_file(
                output_dir / "two--S2--L2--ckpt=77--v_ckpt=10--pred.wav"
            )

            # There are four chunks but two outputs.
            # Output one contains only one chunk, so output_two should be 3 times longer
            self.assertEqual(len(output_one) * 3, len(output_two))


class TestWritingSpec(ChunkingTestBase):
    def test_spec_chunks(self):
        """
        Tests the correctness of the output of spectrograms for chunked text over multiple batches.
        """
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            vocoder, vocoder_path = get_stubbed_vocoder(tmp_dir)

            with silence_c_stderr():
                writers = get_synthesis_output_callbacks(
                    [SynthesizeOutputFormats.spec],
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

            # Test that the correctly named files were and weren't outputted
            # The outputted files should be named after the first chunk from each text input
            self.assertTrue(
                (output_dir / "one--S1--L1--spec-pred-22050-mel-librosa.pt").exists()
            )
            self.assertTrue(
                (output_dir / "two--S2--L2--spec-pred-22050-mel-librosa.pt").exists()
            )
            self.assertFalse(
                (output_dir / "three--S1--L1--spec-pred-22050-mel-librosa.pt").exists()
            )
            self.assertFalse(
                (output_dir / "four--S2--L2--spec-pred-22050-mel-librosa.pt").exists()
            )

            # Tests that last_file_written contains the correct most recent filename written
            # This is important for the demo
            self.assertEqual(
                (output_dir / "two--S2--L2--spec-pred-22050-mel-librosa.pt"),
                Path(writer.last_file_written),
            )

            # Checks that the files have reasonable lengths
            output_one = torch.load(
                output_dir / "one--S1--L1--spec-pred-22050-mel-librosa.pt"
            )
            output_two = torch.load(
                output_dir / "two--S2--L2--spec-pred-22050-mel-librosa.pt"
            )

            # There are four chunks but two outputs.
            # Output one contains only one chunk, so output_two should be 3 times longer
            self.assertEqual(output_one.size(-1) * 3, output_two.size(-1))
