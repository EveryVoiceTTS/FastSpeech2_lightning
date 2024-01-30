#!/usr/bin/env python

"""
If you've installed `everyvoice` and would like to run this unittest:
python -m unittest everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.tests.test_cli
"""

import io
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

from typer.testing import CliRunner

try:
    from ..cli import app, validate_data_keys_with_model_keys
    from ..config import FastSpeech2Config
except ImportError:
    from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli import (
        app,
        validate_data_keys_with_model_keys,
    )
    from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.config import (
        FastSpeech2Config,
    )

DEFAULT_LANG2ID: set = set()
DEFAULT_SPEAKER2ID: set = set()


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

    def test_filelist_language(self):
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            test = tmpdir / "test.psv"
            test.touch()
            model = tmpdir / "model"
            model.touch()
            _ = self.runner.invoke(
                app,
                (
                    "synthesize",
                    "--filelist",
                    str(test),
                    "--language",
                    "foo",
                    str(model),
                ),
            )
            # TODO: we need to actually check that the language overwrites the languages provided in the filelist
            # self.assertIn(
            #     "Loading checkpoint",  # This means the command was not invalid
            #     result.stdout,
            # )

    def test_filelist_speaker(self):
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            test = tmpdir / "test.psv"
            test.touch()
            model = tmpdir / "model"
            model.touch()
            _ = self.runner.invoke(
                app,
                (
                    "synthesize",
                    "--filelist",
                    str(test),
                    "--speaker",
                    "BAD",
                    str(model),
                ),
            )
            # TODO: we need to actually check that the speaker overwrites the speakers provided in the filelist
            # self.assertIn(
            #     "Loading checkpoint",  # This means the command was not invalid
            #     result.stdout,
            # )

    def test_plain_filelist(self):
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            test = tmpdir / "test.list"
            with open(test, "w") as f:
                f.write("\n".join(["this is a test sentence", "and another test"]))
            model = tmpdir / "model"
            model.touch()
            _ = self.runner.invoke(
                app,
                (
                    "synthesize",
                    "--filelist",
                    str(test),
                    str(model),
                ),
            )
            # TODO: we need to actually check that the default/first speaker and language was used
            #       and this should be logged to the user
            # self.assertIn(
            #     "Loading checkpoint",  # This means the command was not invalid
            #     result.stdout,
            # )


class ValidateDataWithModelTest(TestCase):
    """
    Validate different combination of user provided options against model configuration.
    """

    def test_multilingual_invalid_language(self):
        """
        The model is multilingual and the user provided a language that is not supported by the model.
        """
        language = "UNSUPPORTED"
        config = FastSpeech2Config()
        config.model.multilingual = True
        model_languages = {"L1", "L2"}
        f = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stderr(f):
            validate_data_keys_with_model_keys(
                data_keys={language}, model_keys=model_languages, key="language"
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
            )
        self.assertIn(
            f"You provided {set((language,language_two))} which are not languages that are supported by the model {model_languages}.",
            f.getvalue(),
        )

    def test_not_multilingual_with_language(self):
        """
        The model is not multilingual and the user provided a language.
        """
        language = "L3"
        config = FastSpeech2Config()
        config.model.multilingual = False
        model_languages = DEFAULT_LANG2ID
        f = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stderr(f):
            validate_data_keys_with_model_keys(
                data_keys={language}, model_keys=model_languages, key="language"
            )
        self.assertIn(
            "You provided {'L3'} which is not a language supported by the model",
            f.getvalue(),
        )

    def test_multispeaker_invalid_speaker(self):
        """
        The model is multispeaker and the user provided a speaker that is not supported by the model.
        """
        speaker = "UNSUPPORTED"
        config = FastSpeech2Config()
        config.model.multispeaker = True
        model_speakers = {"S1", "S2"}
        f = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stderr(f):
            validate_data_keys_with_model_keys(
                data_keys={speaker}, model_keys=model_speakers, key="speaker"
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
        config = FastSpeech2Config()
        config.model.multispeaker = False
        model_speakers = DEFAULT_SPEAKER2ID
        f = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stderr(f):
            validate_data_keys_with_model_keys(
                data_keys={speaker}, model_keys=model_speakers, key="speaker"
            )
        self.assertIn(
            "You provided {'s3'} which is not a speaker supported by the model",
            f.getvalue(),
        )


if __name__ == "__main__":
    main()
