import tempfile
from pathlib import Path

from everyvoice.config.shared_types import ContactInformation
from everyvoice.config.text_config import Symbols, TextConfig
from everyvoice.config.type_definitions import DatasetTextRepresentation
from everyvoice.tests.stubs import TEST_CONTACT, capture_logs
from everyvoice.text.lookups import LookupTable
from everyvoice.text.utils import get_symbols_from_checkpoint_symbol_dict, symbol_sorter
from pytest import raises

from ..cli.synthesize import load_data_from_filelist
from ..config import FastSpeech2Config
from ..model import FastSpeech2
from ..type_definitions_heavy import Stats, StatsInfo

TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX = "everyvoice-text-to-spec"


class TestLoadingModel:
    """Test loading models"""

    data_dir = Path(__file__).parent / "data"
    config_dir = data_dir / "config"

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
            m = torch.load(ckpt_fn, weights_only=True)
            assert "model_info" in m.keys()
            m["model_info"]["name"] = "BAD_TYPE"
            torch.save(m, ckpt_fn)
            m = torch.load(ckpt_fn, weights_only=True)
            assert "model_info" in m.keys()
            assert m["model_info"]["name"] == "BAD_TYPE"
            # assert m["model_info"]["version"] == "1.0"
            with raises(
                TypeError,
                match=r"Wrong model type \(BAD_TYPE\), we are expecting a 'FastSpeech2' model",
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
            m = torch.load(ckpt_fn, weights_only=True)
            assert "model_info" in m.keys()
            assert m["model_info"]["name"] == FastSpeech2.__name__
            assert m["model_info"]["version"] == BAD_VERSION
            with raises(InvalidVersion, match=r"Invalid version: 'BAD_VERSION'"):
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
            m = torch.load(ckpt_fn, weights_only=True)
            assert "model_info" in m.keys()
            assert m["model_info"]["name"] == FastSpeech2.__name__
            assert m["model_info"]["version"] == BAD_VERSION
            with raises(
                ValueError,
                match=r"Your model was created with a newer version of EveryVoice, please update your software.",
            ):
                FastSpeech2.load_from_checkpoint(ckpt_fn)

    def _build_model(self, symbols_kwargs, target_text_representation_level=None):
        model_kwargs = {}
        if target_text_representation_level is not None:
            model_kwargs["target_text_representation_level"] = (
                target_text_representation_level
            )
        config = FastSpeech2Config(
            contact=TEST_CONTACT,
            text=TextConfig(symbols=Symbols(**symbols_kwargs)),
            model=model_kwargs,
        )
        return FastSpeech2(
            config,
            stats=Stats(
                pitch=StatsInfo(min=0, max=1, std=2, mean=3, norm_min=4, norm_max=5),
                energy=StatsInfo(
                    min=7, max=8, std=9, mean=10, norm_min=11, norm_max=12
                ),
            ),
            lang2id={},
            speaker2id={},
        )

    def test_check_and_upgrade_checkpoint_pre_1_2_growth(self):
        """Regression test: refactoring the pre-1.2 migration to use the
        shared _remap_embedding_weights helper must preserve its original
        growth-only behavior (checkpoint's symbol set is a subset of the
        live model's).
        """
        import torch

        model = self._build_model({"letters": ["a", "b"]})
        # the pre-1.2 branch prepends these hardcoded symbols before whatever
        # was declared, mirroring check_and_upgrade_checkpoint's own
        # old_hardcoded_symbols list for that historical format
        old_hardcoded_symbols = [
            "\x80",
            " ",
            "<EXCL>",
            "<QINT>",
            "<QUOTE>",
            "<BB>",
            "<SB>",
            "<EPS>",
        ]
        raw_symbols_dict = {"letters": ["a"]}
        old_symbols = symbol_sorter(
            get_symbols_from_checkpoint_symbol_dict(raw_symbols_dict),
            hardcoded_initial_symbols=old_hardcoded_symbols,
        )
        a_old_index = old_symbols.index("a")
        embedding_dim = model.text_input_layer.weight.size(1)
        old_weights = torch.arange(
            len(old_symbols) * embedding_dim, dtype=torch.float
        ).reshape(len(old_symbols), embedding_dim)
        checkpoint = {
            "model_info": {"name": "FastSpeech2", "version": "1.1"},
            "hyper_parameters": {
                "config": {
                    "text": {"symbols": raw_symbols_dict},
                    "model": {"target_text_representation_level": "characters"},
                },
            },
            "state_dict": {"text_input_layer.weight": old_weights},
        }
        model.check_and_upgrade_checkpoint(checkpoint)
        new_weights = checkpoint["state_dict"]["text_input_layer.weight"]
        assert new_weights.shape == model.text_input_layer.weight.shape
        a_index = model.text_processor.symbols.index("a")
        assert torch.equal(new_weights[a_index], old_weights[a_old_index])

    def test_check_and_upgrade_checkpoint_1_2_to_1_3_shrinkage(self):
        """A characters-only model whose text config also declares phone
        symbols (from a shared text config) should shrink an old checkpoint's
        embedding table, dropping the now-irrelevant phone rows, rather than
        crashing.
        """
        import torch

        model = self._build_model(
            {"ds1_characters": ["x", "y"], "ds1_phones": ["p", "q"]},
            target_text_representation_level="characters",
        )
        assert "p" not in model.text_processor.symbols
        raw_symbols_dict = {"ds1_characters": ["x", "y"], "ds1_phones": ["p", "q"]}
        # check_and_upgrade_checkpoint reconstructs the old symbol order via
        # symbol_sorter(get_symbols_from_checkpoint_symbol_dict(...)) -- compute
        # that same order here rather than assuming it, so the test doesn't
        # depend on symbol_sorter's exact sort key.
        old_symbols = symbol_sorter(
            get_symbols_from_checkpoint_symbol_dict(raw_symbols_dict)
        )
        x_old_index = old_symbols.index("x")
        embedding_dim = model.text_input_layer.weight.size(1)
        old_weights = torch.arange(
            len(old_symbols) * embedding_dim, dtype=torch.float
        ).reshape(len(old_symbols), embedding_dim)
        checkpoint = {
            "model_info": {"name": "FastSpeech2", "version": "1.2"},
            "hyper_parameters": {
                "config": {"text": {"symbols": raw_symbols_dict}},
            },
            "state_dict": {"text_input_layer.weight": old_weights},
        }
        with capture_logs() as logs:
            model.check_and_upgrade_checkpoint(checkpoint)
        new_weights = checkpoint["state_dict"]["text_input_layer.weight"]
        assert new_weights.shape == model.text_input_layer.weight.shape
        x_index = model.text_processor.symbols.index("x")
        assert torch.equal(new_weights[x_index], old_weights[x_old_index])
        assert any("p" in log and "q" in log for log in logs)

    def test_check_and_upgrade_checkpoint_uses_realized_symbols(self):
        """Once a checkpoint carries realized_symbols (persisted at save
        time), reconciliation must use that exact list rather than
        reconstructing one from the raw text config -- proven here by making
        the raw config dict deliberately wrong/unrelated.
        """
        import torch

        model = self._build_model({"letters": ["a", "b"]})
        embedding_dim = model.text_input_layer.weight.size(1)
        old_weights = torch.zeros(1, embedding_dim)
        old_weights[0] = 1.0  # recognizable value for symbol 'a'
        checkpoint = {
            "model_info": {"name": "FastSpeech2", "version": FastSpeech2._VERSION},
            "hyper_parameters": {
                # deliberately unrelated to the live config, to prove this
                # isn't what gets used for reconciliation
                "config": {"text": {"symbols": {"letters": ["z"]}}},
                "realized_symbols": ["a"],
            },
            "state_dict": {"text_input_layer.weight": old_weights},
        }
        model.check_and_upgrade_checkpoint(checkpoint)
        new_weights = checkpoint["state_dict"]["text_input_layer.weight"]
        assert new_weights.shape == model.text_input_layer.weight.shape
        a_index = model.text_processor.symbols.index("a")
        b_index = model.text_processor.symbols.index("b")
        assert torch.equal(new_weights[a_index], old_weights[0])
        assert torch.equal(new_weights[b_index], torch.zeros(embedding_dim))


class TestLoadingConfig:
    """Test loading configurations"""

    data_dir = Path(__file__).parent / "data"
    config_dir = data_dir / "config"

    def test_config_versionless(self):
        """
        Validate that we can load a config that doesn't have a `VERSION` as a version 1.0 config.
        """

        arguments = FastSpeech2Config.load_config_from_path(
            self.config_dir / f"{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml"
        ).model_dump()
        del arguments["VERSION"]

        assert "VERSION" not in arguments
        c = FastSpeech2Config(**arguments)
        assert c.VERSION == "1.0"

    def test_config_newer_version(self):
        """
        Validate that we are detecting that a config is newer.
        """

        reference = FastSpeech2Config.load_config_from_path(
            self.config_dir / f"{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml"
        )
        NEWER_VERSION = "100.0"
        reference.VERSION = NEWER_VERSION

        with raises(
            ValueError,
            match=r"Your config was created with a newer version of EveryVoice, please update your software.",
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


class TestLoadingData:

    def write_and_load(self, file_contents: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "data_file"
            with open(data_file, "w") as f:
                f.write(file_contents)
            data = load_data_from_filelist(
                data_file,
                StubModelWithConfigOnly(),
                DatasetTextRepresentation.characters,
            )
            return data

    def test_load_oneline(self):
        data = self.write_and_load("this is a test\n")
        assert len(data) == 1

    def test_load_twolines(self):
        data = self.write_and_load("test line 1\ntest line 2\n")
        assert len(data) == 2

    def test_load_psv(self):
        data = self.write_and_load("characters|language\nfoo|eng\nbar|eng\nbaz|fra\n")
        assert len(data) == 3
