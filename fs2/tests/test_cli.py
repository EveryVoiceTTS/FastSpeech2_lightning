"""
To run just this test: pytest path/to/test_cli.py
"""

import io
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

from everyvoice.config.shared_types import ContactInformation
from everyvoice.config.text_config import TextConfig
from everyvoice.config.type_definitions import (
    DatasetTextRepresentation,
    TargetTrainingTextRepresentationLevel,
)
from everyvoice.tests.model_stubs import get_stubbed_model
from everyvoice.tests.preprocessed_audio_fixture import PreprocessedAudioFixture
from everyvoice.tests.stubs import (
    TEST_DATA_DIR,
    mute_logger,
    temp_chdir,
)
from everyvoice.utils import generic_psv_filelist_reader
from pytest import approx, raises
from typer.testing import CliRunner

from ..cli.check_data_heavy import check_data_from_filelist
from ..cli.cli import app
from ..cli.synthesize import get_text_split_params
from ..cli.synthesize import prepare_data as prepare_synthesize_data
from ..cli.synthesize import validate_data_keys_with_model_keys
from ..config import FastSpeech2Config
from ..model import FastSpeech2
from ..type_definitions_heavy import Stats, StatsInfo

DEFAULT_LANG2ID: set = set()
DEFAULT_SPEAKER2ID: set = set()
CONTACT = ContactInformation(
    contact_name="Test Runner", contact_email="info@everyvoice.ca"
)


class SynthesizeTest(TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(app, ["synthesize", "--help"])
        assert "synthesize [OPTIONS] MODEL_PATH" in result.stdout

    def test_no_model(self):
        result = self.runner.invoke(app, ["synthesize"])
        assert "Missing argument 'MODEL_PATH'." in result.output

    def test_filelist_and_text(self):
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            test = tmpdir / "test.psv"
            test.touch()
            model = tmpdir / "model"
            model.touch()
            result = self.runner.invoke(
                app,
                (
                    "synthesize",
                    "--filelist",
                    str(test),
                    "--text",
                    "BAD",
                    str(model),
                ),
            )
            assert (
                "Got arguments for both text and a filelist - this will only process the text."
                " Please re-run without providing text if you want to run batch synthesis"
                in result.output
            )

    def test_no_filelist_nor_text(self):
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            model = tmpdir / "model"
            model.touch()
            result = self.runner.invoke(
                app,
                (
                    "synthesize",
                    str(model),
                ),
            )
            assert "You must define either --text or --filelist" in result.output

    def mock_synthesis(self, *_args, **_kwargs):
        print(_kwargs["model"].config)

    def test_config_args(self):
        """
        Tests the -c flag with 'everyvoice synthesize'
        """
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            _, spec_model_path = get_stubbed_model(tmpdir)

            with temp_chdir(tmpdir):
                with (
                    mock.patch(
                        self.__module__.replace(
                            "tests.test_cli", "cli.synthesize.synthesize_helper"
                        ),
                        side_effect=self.mock_synthesis,
                    ),
                    mute_logger("everyvoice.utils"),
                ):
                    result = self.runner.invoke(
                        app,
                        [
                            "synthesize",
                            str(spec_model_path),
                            "-t",
                            "hello world",
                            "-c",
                            "text.split_text=False",
                            "--output-type",
                            "spec",
                        ],
                    )
                    assert result.exit_code == 0
                    assert "split_text=False" in result.stdout


class MockModelForPrepare:
    class Dummy:
        pass

    def __init__(
        self,
        lang2id,
        speaker2id,
        filelist_loader,
        multilingual,
        multispeaker,
        target_text_representation_level,
        text_config=TextConfig(boundaries={"foo": {"strong": ".!?", "weak": ",;:"}}),
    ):
        self.lang2id = lang2id
        self.speaker2id = speaker2id
        self.config = self.Dummy()
        self.config.text = text_config
        self.config.training = self.Dummy()
        self.config.training.filelist_loader = filelist_loader
        self.config.model = self.Dummy()
        self.config.model.multilingual = multilingual
        self.config.model.multispeaker = multispeaker
        self.config.model.target_text_representation_level = (
            target_text_representation_level
        )


# TODO: Currently, an extra unwanted split occurs on the period in "Mr. Neild".
# In future versions, we would like to prevent erroneous splitting on abbreviations.
class PrepareSynthesizeDataTest(TestCase):
    """"""

    def test_filelist_language(self):
        """
        Use a different language than the one provided in the filelist.
        """
        data = prepare_synthesize_data(
            texts=[],
            language="foo",
            speaker="bar",
            duration_control=1.0,
            style_reference=None,
            filelist=Path(__file__).parent / "data/filelist.psv",
            model=MockModelForPrepare(
                lang2id={"foo": 1},
                speaker2id={"bar": 2},
                filelist_loader=generic_psv_filelist_reader,
                multilingual=True,
                multispeaker=True,
                target_text_representation_level=TargetTrainingTextRepresentationLevel.characters,
            ),
            text_representation=DatasetTextRepresentation.characters,
        )
        assert len(data) == 11
        assert all((d["language"] == "foo" for d in data))

    def test_filelist_speaker(self):
        """
        Use a different speaker than the one provided in the filelist.
        """
        data = prepare_synthesize_data(
            texts=[],
            language="foo",
            speaker="bar",
            duration_control=1.0,
            style_reference=None,
            filelist=Path(__file__).parent / "data/filelist.psv",
            model=MockModelForPrepare(
                lang2id={"foo": 1},
                speaker2id={"bar": 2},
                filelist_loader=generic_psv_filelist_reader,
                multilingual=True,
                multispeaker=True,
                target_text_representation_level=TargetTrainingTextRepresentationLevel.characters,
            ),
            text_representation=DatasetTextRepresentation.characters,
        )
        assert (
            data[-1]["basename"] == "LJ002 this is a really long basename"
        ), "Asserts that if a filelist provides a basename, it won't get slugified or truncated"
        assert len(data) == 11
        assert all((d["speaker"] == "bar" for d in data))

    def test_plain_filelist(self):
        data = prepare_synthesize_data(
            texts=[],
            language=None,
            speaker=None,
            duration_control=1.0,
            style_reference=None,
            filelist=Path(__file__).parent / "data/filelist.txt",
            model=MockModelForPrepare(
                lang2id={"foo": 1},
                speaker2id={"bar": 2},
                filelist_loader=generic_psv_filelist_reader,
                multilingual=True,
                multispeaker=True,
                target_text_representation_level=TargetTrainingTextRepresentationLevel.characters,
            ),
            text_representation=DatasetTextRepresentation.characters,
        )
        assert (
            data[-1]["basename"] == "Neild-found-a-man-na-80dfa7e5"
        ), "Asserts that basenames are truncated and slugified"
        assert len(data) == 10
        assert all((d["language"] == "foo" for d in data))
        assert all((d["speaker"] == "bar" for d in data))

    def test_chunking(self):
        """
        Provide a long text and verify that it is being chunked properly.
        """

        a = "There are approximately 70 Indigenous languages spoken in Canada from 10 distinct language families."
        b = "As a consequence of the residential school system and other policies of cultural suppression, the majority of these languages now have fewer than 500 fluent speakers remaining, most of them elderly."

        data = prepare_synthesize_data(
            texts=[a + " " + b],
            language="foo",
            speaker="bar",
            duration_control=1.0,
            style_reference=None,
            filelist=Path(),  # Does not get used in this test
            model=MockModelForPrepare(
                lang2id={"foo": 1},
                speaker2id={"bar": 2},
                filelist_loader=generic_psv_filelist_reader,
                multilingual=True,
                multispeaker=True,
                target_text_representation_level=TargetTrainingTextRepresentationLevel.characters,
            ),
            text_representation=DatasetTextRepresentation.characters,
        )
        assert len(data) == 2
        assert data[0]["characters"] == a
        assert data[1]["characters"] == b
        assert not data[0]["is_last_input_chunk"]
        assert data[1]["is_last_input_chunk"]

    def test_no_chunking(self):
        """
        Provide a long text and verify that it is not chunked when split_text = False.
        """

        a = "There are approximately 70 Indigenous languages spoken in Canada from 10 distinct language families."
        b = "As a consequence of the residential school system and other policies of cultural suppression, the majority of these languages now have fewer than 500 fluent speakers remaining, most of them elderly."

        data = prepare_synthesize_data(
            texts=[a + " " + b],
            language="foo",
            speaker="bar",
            duration_control=1.0,
            style_reference=None,
            filelist=Path(__file__).parent / "data/filelist.psv",
            model=MockModelForPrepare(
                lang2id={"foo": 1},
                speaker2id={"bar": 2},
                filelist_loader=generic_psv_filelist_reader,
                multilingual=True,
                multispeaker=True,
                target_text_representation_level=TargetTrainingTextRepresentationLevel.characters,
                text_config=TextConfig(
                    split_text=False,
                    boundaries={"foo": {"strong": ".!?", "weak": ",;:"}},
                ),
            ),
            text_representation=DatasetTextRepresentation.characters,
        )
        assert len(data) == 1
        assert data[0]["characters"] == a + " " + b
        assert data[0]["is_last_input_chunk"]

    def test_get_text_split_params(self):
        """
        Tests the helper function get_text_split_params()
        """

        contact_info = ContactInformation(
            contact_name="Test Runner", contact_email="info@everyvoice.ca"
        )

        text_config = TextConfig(
            boundaries={
                "eng": {"strong": "‽", "weak": ":?"},
                "default": {"strong": "!?", "weak": ",;"},
            }
        )

        stats = Stats(
            pitch=StatsInfo(
                min=150, max=300, std=2.0, mean=0.5, norm_max=1.0, norm_min=0.1
            ),
            energy=StatsInfo(
                min=0.1, max=10.0, std=2.0, mean=0.5, norm_max=1.0, norm_min=0.1
            ),
            character_length=StatsInfo(
                min=1, max=2, std=3, mean=4, norm_max=5, norm_min=6
            ),
            phone_length=StatsInfo(min=6, max=5, std=4, mean=3, norm_max=2, norm_min=1),
        )

        fs2_config = FastSpeech2Config(contact=contact_info, text=text_config)

        model = FastSpeech2(
            config=fs2_config,
            lang2id={"default": 0},
            speaker2id={"default": 0},
            stats=stats,
        )

        split_text, split_params = get_text_split_params(
            model, language="eng", text_representation="phones"
        )
        desired_length, max_length, strong_boundaries, weak_boundaries = split_params

        assert split_text
        assert desired_length == 3
        assert max_length == 5
        assert strong_boundaries == "‽"
        assert weak_boundaries == ":?"


class ValidateDataWithModelTest(TestCase):
    """
    Validate different combination of user provided options against model configuration.
    """

    def test_multilingual_invalid_language(self):
        """
        The model is multilingual and the user provided a language that is not supported by the model.
        """
        language = "UNSUPPORTED"
        config = FastSpeech2Config(contact=CONTACT)
        config.model.multilingual = True
        model_languages = {"L1", "L2"}
        f = io.StringIO()
        with raises(SystemExit), redirect_stderr(f):
            validate_data_keys_with_model_keys(
                data_keys={language},
                model_keys=model_languages,
                key="language",
                multi=bool(model_languages),
            )
        assert (
            f"You provided {set((language,))} which is not a language supported by the model {model_languages}."
            in f.getvalue()
        )
        language_two = "ALSO_UNSUPPORTED"
        with raises(SystemExit), redirect_stderr(f):
            validate_data_keys_with_model_keys(
                data_keys={language, language_two},
                model_keys=model_languages,
                key="language",
                multi=bool(model_languages),
            )
        assert (
            f"You provided {set((language, language_two))} which are not languages that are supported by the model {model_languages}."
            in f.getvalue()
        )

    def test_not_multilingual_with_language(self):
        """
        The model is not multilingual and the user provided a language.
        """
        language = "L3"
        config = FastSpeech2Config(contact=CONTACT)
        config.model.multilingual = False
        model_languages = DEFAULT_LANG2ID
        f = io.StringIO()
        with raises(SystemExit), redirect_stderr(f):
            validate_data_keys_with_model_keys(
                data_keys={language},
                model_keys=model_languages,
                key="language",
                multi=bool(model_languages),
            )
        assert "The current model doesn't support multiple languages" in f.getvalue()

    def test_multispeaker_invalid_speaker(self):
        """
        The model is multispeaker and the user provided a speaker that is not supported by the model.
        """
        speaker = "UNSUPPORTED"
        config = FastSpeech2Config(contact=CONTACT)
        config.model.multispeaker = True
        model_speakers = {"S1", "S2"}
        f = io.StringIO()
        with raises(SystemExit), redirect_stderr(f):
            validate_data_keys_with_model_keys(
                data_keys={speaker},
                model_keys=model_speakers,
                key="speaker",
                multi=bool(model_speakers),
            )
        assert (
            f"You provided {set((speaker,))} which is not a speaker supported by the model {model_speakers}."
            in f.getvalue()
        )

    def test_not_multispeaker_with_speaker(self):
        """
        The model is not multispeaker and the user provided a speaker.
        """
        speaker = "s3"
        config = FastSpeech2Config(contact=CONTACT)
        config.model.multispeaker = False
        model_speakers = DEFAULT_SPEAKER2ID
        f = io.StringIO()
        with raises(SystemExit), redirect_stderr(f):
            validate_data_keys_with_model_keys(
                data_keys={speaker},
                model_keys=model_speakers,
                key="speaker",
                multi=bool(model_speakers),
            )
        assert "The current model doesn't support multiple speakers" in f.getvalue()


class CLITest(PreprocessedAudioFixture, TestCase):
    """
    Validate that all subcommands are accessible.
    """

    def setUp(self) -> None:
        super().setUp()
        self.runner = CliRunner()
        self.subcommands = (
            "benchmark",
            "preprocess",
            "synthesize",
            "train",
        )

    def test_check_data(self):
        filelist = generic_psv_filelist_reader(TEST_DATA_DIR / "metadata.psv")
        checked_data = check_data_from_filelist(
            self.preprocessor, filelist, heavy_objective_evaluation=True
        )
        assert "pesq" in checked_data[0]
        assert "stoi" in checked_data[0]
        assert "si_sdr" in checked_data[0]
        assert checked_data[0]["pesq"] > 3.0
        assert checked_data[0]["pesq"] < 5.0
        assert checked_data[0]["duration"] == approx(5.17, abs=0.01)

    def test_commands_present(self):
        """
        Each subcommand is present in the the command's help message.
        """
        result = self.runner.invoke(app, ["--help"])
        for command in self.subcommands:
            with self.subTest(msg=f"Looking for {command}"):
                assert command in result.stdout

    def test_command_help_messages(self):
        """
        Each subcommand has its help message.
        """
        for subcommand in self.subcommands:
            with self.subTest(msg=f"Looking for {subcommand}'s help"):
                result = self.runner.invoke(app, [subcommand, "--help"])
                assert result.exit_code == 0
                result = self.runner.invoke(app, [subcommand, "-h"])
                assert result.exit_code == 0


class MiscTests(TestCase):
    def test_version_sync(self):
        """
        Ensure fs2 and everyvoice versions are in sync.
        """
        from everyvoice._version import VERSION as ev_version

        from .._version import VERSION as fs2_version

        assert (
            ev_version == fs2_version
        ), "Version mismatch between EveryVoice and FastSpeech2_lightning"
