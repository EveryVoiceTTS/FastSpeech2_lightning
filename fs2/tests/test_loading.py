import tempfile
from pathlib import Path
from unittest import TestCase

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
            trainer = Trainer(
                default_root_dir=tmpdir_str,
                enable_progress_bar=False,
                logger=False,
                max_epochs=1,
                limit_train_batches=1,
                limit_val_batches=1,
                callbacks=[ModelCheckpoint(dirpath=tmpdir_str, every_n_train_steps=1)],
            )
            trainer.strategy.connect(model)
            ckpt_fn = tmpdir_str + "/checkpoint.ckpt"
            trainer.save_checkpoint(ckpt_fn)
            m = torch.load(ckpt_fn)
            self.assertIn("model_info", m.keys())
            m["model_info"]["name"] = "BAD_TYPE"
            torch.save(m, ckpt_fn)
            m = torch.load(ckpt_fn)
            self.assertIn("model_info", m.keys())
            self.assertEqual(m["model_info"]["name"], "BAD_TYPE")
            # self.assertEqual(m["model_info"]["version"], "1.0")
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
            trainer = Trainer(
                default_root_dir=tmpdir_str,
                enable_progress_bar=False,
                logger=False,
                max_epochs=1,
                limit_train_batches=1,
                limit_val_batches=1,
                callbacks=[ModelCheckpoint(dirpath=tmpdir_str, every_n_train_steps=1)],
            )
            trainer.strategy.connect(model)
            ckpt_fn = tmpdir_str + "/checkpoint.ckpt"
            trainer.save_checkpoint(ckpt_fn)
            m = torch.load(ckpt_fn)
            self.assertIn("model_info", m.keys())
            self.assertEqual(m["model_info"]["name"], FastSpeech2.__name__)
            self.assertEqual(m["model_info"]["version"], BAD_VERSION)
            with self.assertRaisesRegex(
                InvalidVersion,
                r"Invalid version: 'BAD_VERSION'",
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
            trainer = Trainer(
                default_root_dir=tmpdir_str,
                enable_progress_bar=False,
                logger=False,
                max_epochs=1,
                limit_train_batches=1,
                limit_val_batches=1,
                callbacks=[ModelCheckpoint(dirpath=tmpdir_str, every_n_train_steps=1)],
            )
            trainer.strategy.connect(model)
            ckpt_fn = tmpdir_str + "/checkpoint.ckpt"
            trainer.save_checkpoint(ckpt_fn)
            m = torch.load(ckpt_fn)
            self.assertIn("model_info", m.keys())
            self.assertEqual(m["model_info"]["name"], FastSpeech2.__name__)
            self.assertEqual(m["model_info"]["version"], BAD_VERSION)
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

        arguments = FastSpeech2Config.load_config_from_path(
            self.config_dir / f"{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml"
        ).model_dump()
        del arguments["VERSION"]

        self.assertNotIn("VERSION", arguments)
        c = FastSpeech2Config(**arguments)
        self.assertEqual(c.VERSION, "1.0")

    def test_config_newer_version(self):
        """
        Validate that we are detecting that a config is newer.
        """

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