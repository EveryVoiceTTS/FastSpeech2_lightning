import tempfile
from pathlib import Path
from unittest import TestCase

from everyvoice.config.shared_types import ContactInformation
from everyvoice.config.type_definitions import DatasetTextRepresentation
from everyvoice.tests.stubs import silence_c_stderr
from everyvoice.text.lookups import LookupTable

from ..cli.synthesize import load_data_from_filelist
from ..config import FastSpeech2Config
from ..model import FastSpeech2
from ..type_definitions_heavy import Stats, StatsInfo

TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX = "everyvoice-text-to-spec"


class TestLoadingModel(TestCase):
    """Test loading models"""

    data_dir = Path(__file__).parent / "data"

    def setUp(self) -> None:
        super().setUp()
        self.config_dir = self.data_dir / "config"

    def test_wrong_model_type(self):
        """
        Detecting wrong model type in checkpoint.
        """
        import torch
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint

        with tempfile.TemporaryDirectory() as tmpdir_str:
            with silence_c_stderr():
                model = FastSpeech2(
                    FastSpeech2Config.load_config_from_path(
                        self.config_dir / f"{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml"
                    ),
                    stats=Stats(
                        pitch=StatsInfo(
                            min=0, max=1, std=2, mean=3, norm_min=4, norm_max=5
                        ),
                        energy=StatsInfo(
                            min=7, max=8, std=9, mean=10, norm_min=11, norm_max=12
                        ),
                    ),
                    lang2id={"foo": 0, "bar": 1},
                    speaker2id={"baz": 0, "qux": 1},
                )
            with silence_c_stderr():
                trainer = Trainer(
                    default_root_dir=tmpdir_str,
                    enable_progress_bar=False,
                    logger=False,
                    max_epochs=1,
                    limit_train_batches=1,
                    limit_val_batches=1,
                    callbacks=[
                        ModelCheckpoint(dirpath=tmpdir_str, every_n_train_steps=1)
                    ],
                )
            trainer.strategy.connect(model)
            ckpt_fn = tmpdir_str + "/checkpoint.ckpt"
            trainer.save_checkpoint(ckpt_fn)
            m = torch.load(ckpt_fn, weights_only=True)
            self.assertIn("model_info", m.keys())
            m["model_info"]["name"] = "BAD_TYPE"
            torch.save(m, ckpt_fn)
            m = torch.load(ckpt_fn, weights_only=True)
            self.assertIn("model_info", m.keys())
            self.assertEqual(m["model_info"]["name"], "BAD_TYPE")
            # self.assertEqual(m["model_info"]["version"], "1.0")
            with silence_c_stderr():
                with self.assertRaisesRegex(
                    TypeError,
                    r"Wrong model type \(BAD_TYPE\), we are expecting a 'FastSpeech2' model",
                ):
                    FastSpeech2.load_from_checkpoint(ckpt_fn)

    def test_wrong_model_version(self):
        """
        Detecting wrong model version number in checkpoint.
        """
        import torch
        from packaging.version import InvalidVersion
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint

        with tempfile.TemporaryDirectory() as tmpdir_str:
            with silence_c_stderr():
                model = FastSpeech2(
                    FastSpeech2Config.load_config_from_path(
                        self.config_dir / f"{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml"
                    ),
                    stats=Stats(
                        pitch=StatsInfo(
                            min=0, max=1, std=2, mean=3, norm_min=4, norm_max=5
                        ),
                        energy=StatsInfo(
                            min=7, max=8, std=9, mean=10, norm_min=11, norm_max=12
                        ),
                    ),
                    lang2id={"foo": 0, "bar": 1},
                    speaker2id={"baz": 0, "qux": 1},
                )
            BAD_VERSION = "BAD_VERSION"
            model._VERSION = BAD_VERSION
            with silence_c_stderr():
                trainer = Trainer(
                    default_root_dir=tmpdir_str,
                    enable_progress_bar=False,
                    logger=False,
                    max_epochs=1,
                    limit_train_batches=1,
                    limit_val_batches=1,
                    callbacks=[
                        ModelCheckpoint(dirpath=tmpdir_str, every_n_train_steps=1)
                    ],
                )
            trainer.strategy.connect(model)
            ckpt_fn = tmpdir_str + "/checkpoint.ckpt"
            trainer.save_checkpoint(ckpt_fn)
            m = torch.load(ckpt_fn, weights_only=True)
            self.assertIn("model_info", m.keys())
            self.assertEqual(m["model_info"]["name"], FastSpeech2.__name__)
            self.assertEqual(m["model_info"]["version"], BAD_VERSION)
            with silence_c_stderr():
                with self.assertRaisesRegex(
                    InvalidVersion, r"Invalid version: 'BAD_VERSION'"
                ):
                    FastSpeech2.load_from_checkpoint(ckpt_fn)

    def test_newer_model_version(self):
        """
        Detecting an incompatible version number in the checkpoint.
        """
        import torch
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint

        with tempfile.TemporaryDirectory() as tmpdir_str:
            with silence_c_stderr():
                model = FastSpeech2(
                    FastSpeech2Config.load_config_from_path(
                        self.config_dir / f"{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml"
                    ),
                    stats=Stats(
                        pitch=StatsInfo(
                            min=0, max=1, std=2, mean=3, norm_min=4, norm_max=5
                        ),
                        energy=StatsInfo(
                            min=7, max=8, std=9, mean=10, norm_min=11, norm_max=12
                        ),
                    ),
                    lang2id={"foo": 0, "bar": 1},
                    speaker2id={"baz": 0, "qux": 1},
                )
            BAD_VERSION = "100.0"
            model._VERSION = BAD_VERSION
            with silence_c_stderr():
                trainer = Trainer(
                    default_root_dir=tmpdir_str,
                    enable_progress_bar=False,
                    logger=False,
                    max_epochs=1,
                    limit_train_batches=1,
                    limit_val_batches=1,
                    callbacks=[
                        ModelCheckpoint(dirpath=tmpdir_str, every_n_train_steps=1)
                    ],
                )
            trainer.strategy.connect(model)
            ckpt_fn = tmpdir_str + "/checkpoint.ckpt"
            trainer.save_checkpoint(ckpt_fn)
            m = torch.load(ckpt_fn, weights_only=True)
            self.assertIn("model_info", m.keys())
            self.assertEqual(m["model_info"]["name"], FastSpeech2.__name__)
            self.assertEqual(m["model_info"]["version"], BAD_VERSION)
            with silence_c_stderr():
                with self.assertRaisesRegex(
                    ValueError,
                    r"Your model was created with a newer version of EveryVoice, please update your software.",
                ):
                    FastSpeech2.load_from_checkpoint(ckpt_fn)


class TestLoadingConfig(TestCase):
    """Test loading configurations"""

    data_dir = Path(__file__).parent / "data"

    def setUp(self) -> None:
        super().setUp()
        self.config_dir = self.data_dir / "config"

    def test_config_versionless(self):
        """
        Validate that we can load a config that doesn't have a `VERSION` as a version 1.0 config.
        """

        with silence_c_stderr():
            arguments = FastSpeech2Config.load_config_from_path(
                self.config_dir / f"{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml"
            ).model_dump()
        del arguments["VERSION"]

        self.assertNotIn("VERSION", arguments)
        with silence_c_stderr():
            c = FastSpeech2Config(**arguments)
        self.assertEqual(c.VERSION, "1.0")

    def test_config_newer_version(self):
        """
        Validate that we are detecting that a config is newer.
        """

        with silence_c_stderr():
            reference = FastSpeech2Config.load_config_from_path(
                self.config_dir / f"{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml"
            )
        NEWER_VERSION = "100.0"
        reference.VERSION = NEWER_VERSION

        with self.assertRaisesRegex(
            ValueError,
            r"Your config was created with a newer version of EveryVoice, please update your software.",
        ):
            FastSpeech2Config(**reference.model_dump())


class StubModelWithConfigOnly:
    def __init__(self):
        self.config = FastSpeech2Config(
            contact=ContactInformation(
                contact_name="Unit Testing Script",
                contact_email="unit_tester@mail.com",
            )
        )
        self.lang2id: LookupTable = {}
        self.speaker2id: LookupTable = {}


class TestLoadingData(TestCase):

    def write_and_load(self, file_contents: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "data_file"
            with open(data_file, "w") as f:
                f.write(file_contents)
            with silence_c_stderr():
                data = load_data_from_filelist(
                    data_file,
                    StubModelWithConfigOnly(),
                    DatasetTextRepresentation.characters,
                )
            return data

    def test_load_oneline(self):
        data = self.write_and_load("this is a test\n")
        self.assertEqual(len(data), 1)

    def test_load_twolines(self):
        data = self.write_and_load("test line 1\ntest line 2\n")
        self.assertEqual(len(data), 2)

    def test_load_psv(self):
        data = self.write_and_load("characters|language\nfoo|eng\nbar|eng\nbaz|fra\n")
        self.assertEqual(len(data), 3)
