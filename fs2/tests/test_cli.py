"""
If you've installed `everyvoice` and would like to run this unittest:
python -m unittest everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.tests.test_cli
"""

import io
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from everyvoice.config.shared_types import ContactInformation
from everyvoice.config.type_definitions import (
    DatasetTextRepresentation,
    TargetTrainingTextRepresentationLevel,
)
from everyvoice.tests.basic_test_case import BasicTestCase
from everyvoice.tests.preprocessed_audio_fixture import PreprocessedAudioFixture
from everyvoice.tests.stubs import capture_stderr, silence_c_stderr
from everyvoice.utils import generic_psv_filelist_reader
from typer.testing import CliRunner

from ..cli.check_data_heavy import check_data_from_filelist
from ..cli.cli import app
from ..cli.synthesize import prepare_data as prepare_synthesize_data
from ..cli.synthesize import validate_data_keys_with_model_keys
from ..config import FastSpeech2Config

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
        self.assertIn("synthesize [OPTIONS] MODEL_PATH", result.stdout)

    def test_no_model(self):
        result = self.runner.invoke(app, ["synthesize"])
        self.assertIn("Missing argument 'MODEL_PATH'.", result.stdout)

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
            self.assertIn(
                "Got arguments for both text and a filelist - this will only process the text."
                " Please re-run without providing text if you want to run batch synthesis",
                result.stdout,
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
            self.assertIn("You must define either --text or --filelist", result.stdout)


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
    ):
        self.lang2id = lang2id
        self.speaker2id = speaker2id
        self.config = self.Dummy()
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
        self.assertEqual(len(data), 11)
        self.assertTrue(all((d["language"] == "foo" for d in data)))

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
        self.assertEqual(
            data[-1]["basename"],
            "LJ002 this is a really long basename",
            "Asserts that if a filelist provides a basename, it won't get slugified or truncated",
        )
        self.assertEqual(len(data), 11)
        self.assertTrue(all((d["speaker"] == "bar" for d in data)))

    def test_plain_filelist(self):
        with silence_c_stderr():
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
        self.assertEqual(
            data[-1]["basename"],
            "Neild-found-a-man-na-80dfa7e5",
            "Asserts that basenames are truncated and slugified",
        )
        self.assertEqual(len(data), 10)
        self.assertTrue(all((d["language"] == "foo" for d in data)))
        self.assertTrue(all((d["speaker"] == "bar" for d in data)))

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
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["characters"], a)
        self.assertEqual(data[1]["characters"], b)
        self.assertFalse(data[0]["last_input_chunk"])
        self.assertTrue(data[1]["last_input_chunk"])


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
        with self.assertRaises(SystemExit), redirect_stderr(f):
            validate_data_keys_with_model_keys(
                data_keys={language},
                model_keys=model_languages,
                key="language",
                multi=bool(model_languages),
            )
        self.assertIn(
            f"You provided {set((language,))} which is not a language supported by the model {model_languages}.",
            f.getvalue(),
        )
        language_two = "ALSO_UNSUPPORTED"
        with self.assertRaises(SystemExit), redirect_stderr(f):
            validate_data_keys_with_model_keys(
                data_keys={language, language_two},
                model_keys=model_languages,
                key="language",
                multi=bool(model_languages),
            )
        self.assertIn(
            f"You provided {set((language, language_two))} which are not languages that are supported by the model {model_languages}.",
            f.getvalue(),
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
        with self.assertRaises(SystemExit), redirect_stderr(f):
            validate_data_keys_with_model_keys(
                data_keys={language},
                model_keys=model_languages,
                key="language",
                multi=bool(model_languages),
            )
        self.assertIn(
            "The current model doesn't support multiple languages",
            f.getvalue(),
        )

    def test_multispeaker_invalid_speaker(self):
        """
        The model is multispeaker and the user provided a speaker that is not supported by the model.
        """
        speaker = "UNSUPPORTED"
        config = FastSpeech2Config(contact=CONTACT)
        config.model.multispeaker = True
        model_speakers = {"S1", "S2"}
        f = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stderr(f):
            validate_data_keys_with_model_keys(
                data_keys={speaker},
                model_keys=model_speakers,
                key="speaker",
                multi=bool(model_speakers),
            )
        self.assertIn(
            f"You provided {set((speaker,))} which is not a speaker supported by the model {model_speakers}.",
            f.getvalue(),
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
        with self.assertRaises(SystemExit), redirect_stderr(f):
            validate_data_keys_with_model_keys(
                data_keys={speaker},
                model_keys=model_speakers,
                key="speaker",
                multi=bool(model_speakers),
            )
        self.assertIn(
            "The current model doesn't support multiple speakers",
            f.getvalue(),
        )


class CLITest(PreprocessedAudioFixture, BasicTestCase):
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
        filelist = generic_psv_filelist_reader(BasicTestCase.data_dir / "metadata.psv")
        with capture_stderr():
            checked_data = check_data_from_filelist(
                self.preprocessor, filelist, heavy_objective_evaluation=True
            )
        self.assertIn("pesq", checked_data[0])
        self.assertIn("stoi", checked_data[0])
        self.assertIn("si_sdr", checked_data[0])
        self.assertGreater(checked_data[0]["pesq"], 3.0)
        self.assertLess(checked_data[0]["pesq"], 5.0)
        self.assertAlmostEqual(checked_data[0]["duration"], 5.17, 2)

    # def test_compute_stats(self):
    #     feat_prediction_config = EveryVoiceConfig.load_config_from_path().feature_prediction
    #     preprocessor = Preprocessor(feat_prediction_config)
    #     preprocessor.compute_stats()
    # self.assertEqual(
    #     self.preprocessor.config["preprocessing"]["audio"]["mel_mean"],
    #     -4.018,
    #     places=3,
    # )
    # self.assertEqual(
    #     self.preprocessor.config["preprocessing"]["audio"]["mel_std"],
    #     4.017,
    #     places=3,
    # )

    def test_commands_present(self):
        """
        Each subcommand is present in the the command's help message.
        """
        result = self.runner.invoke(app, ["--help"])
        for command in self.subcommands:
            with self.subTest(msg=f"Looking for {command}"):
                self.assertIn(command, result.stdout)

    def test_command_help_messages(self):
        """
        Each subcommand has its help message.
        """
        for subcommand in self.subcommands:
            with self.subTest(msg=f"Looking for {subcommand}'s help"):
                result = self.runner.invoke(app, [subcommand, "--help"])
                self.assertEqual(result.exit_code, 0)
                result = self.runner.invoke(app, [subcommand, "-h"])
                self.assertEqual(result.exit_code, 0)


class MiscTests(TestCase):
    def test_version_sync(self):
        """
        Ensure fs2 and everyvoice versions are in sync.
        """
        from everyvoice._version import VERSION as ev_version

        from .._version import VERSION as fs2_version

        self.assertEqual(
            ev_version,
            fs2_version,
            "Version mismatch between EveryVoice and FastSpeech2_lightning",
        )
