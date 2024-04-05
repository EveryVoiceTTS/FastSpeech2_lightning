from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
from everyvoice.config.shared_types import ContactInformation
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import HiFiGANConfig
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.model import HiFiGAN
from pytorch_lightning import Trainer

from ..config import FastSpeech2Config, FastSpeech2TrainingConfig
from ..prediction_writing_callback import (
    BASENAME_MAX_LENGTH,
    PredictionWritingNpyCallback,
    PredictionWritingPtCallback,
    PredictionWritingWavCallback,
    truncate_basename,
)


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
            "speaker": [
                "spk1",
                "spk2",
            ],
            "language": [
                "lngA",
                "lngB",
            ],
        }


class TestWritingNpy(WritingTestBase):
    """
    Testing the callback that writes npy files.
    """

    def test_truncated_filenames(self):
        """
        We limit the file name's length to at most BASENAME_MAX_LENGTH.
        """
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            writer = PredictionWritingNpyCallback(
                global_step=77,
                output_dir=tmp_dir,
                output_key=self.output_key,
            )
            writer.on_predict_batch_end(
                _trainer=None,
                _pl_module=None,
                outputs=self.outputs,
                batch=self.batch,
                _batch_idx=0,
                _dataloader_idx=0,
            )
            output_dir = writer.save_dir
            self.assertTrue(output_dir.exists())
            self.assertTrue(
                (output_dir / "short--spk1--lngA--ckpt=77--pred.npy").exists()
            )
            self.assertTrue(
                (
                    output_dir
                    / "This-utterance-is-wa-3663fb86--spk2--lngB--ckpt=77--pred.npy"
                ).exists()
            )


class TestWritingPt(WritingTestBase):
    """
    Testing the callback that writes pt files.
    """

    def test_truncated_filenames(self):
        """
        We limit the file name's length to at most BASENAME_MAX_LENGTH.
        """
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            writer = PredictionWritingPtCallback(
                config=FastSpeech2Config(contact=self.contact),
                global_step=77,
                output_dir=tmp_dir,
                output_key=self.output_key,
            )
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
                    output_dir
                    / "short--spk1--lngA--ckpt=77--spec-pred-22050-mel-librosa.pt"
                ).exists()
            )
            self.assertTrue(
                (
                    output_dir
                    / "This-utterance-is-wa-3663fb86--spk2--lngB--ckpt=77--spec-pred-22050-mel-librosa.pt"
                ).exists()
            )


class TestWritingWav(WritingTestBase):
    """
    Testing the callback that writes wav files.
    Note that this test may be expansive.
    """

    def test_truncated_filenames(self):
        """
        We limit the file name's length to at most BASENAME_MAX_LENGTH.
        """
        with TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            contact_info = ContactInformation(
                contact_name="Test Runner", contact_email="info@everyvoice.ca"
            )
            vocoder = HiFiGAN(HiFiGANConfig(contact=contact_info))
            trainer = Trainer(default_root_dir=str(tmp_dir), barebones=True)
            trainer.strategy.connect(vocoder)
            vocoder_path = Path(tmp_dir) / "vocoder"
            trainer.save_checkpoint(vocoder_path)

            writer = PredictionWritingWavCallback(
                config=FastSpeech2Config(
                    contact=self.contact,
                    training=FastSpeech2TrainingConfig(vocoder_path=vocoder_path),
                ),
                device=torch.device("cpu"),
                global_step=77,
                output_dir=tmp_dir,
                output_key=self.output_key,
            )
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
                (output_dir / "short--spk1--lngA--ckpt=77--v_ckpt=0--pred.wav").exists()
            )
            self.assertTrue(
                (
                    output_dir
                    / "This-utterance-is-wa-3663fb86--spk2--lngB--ckpt=77--v_ckpt=0--pred.wav"
                ).exists()
            )
