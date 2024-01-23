#!/usr/bin/env python

import io
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

from typer.testing import CliRunner

from ..cli import app, validate_languages_with_model, validate_speakers_with_model
from ..config import FastSpeech2Config

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
            result = self.runner.invoke(
                app,
                (
                    "synthesize",
                    "--filelist",
                    str(test),
                    "--language",
                    "BAD",
                    str(model),
                ),
            )
            self.assertIn(
                "Specifying a language is only valid when using --text.",
                result.stdout,
            )

    def test_filelist_speaker(self):
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
                    "--speaker",
                    "BAD",
                    str(model),
                ),
            )
            self.assertIn(
                "Specifying a speaker is only valid when using --text.",
                result.stdout,
            )


class ValidateDataWithModelTest(TestCase):
    """
    Validate different combination of user provided options against model configuration.
    """

    def test_multilingual_no_language(self):
        """
        The model is multilingual but the user did not provide a language.
        """
        data = [{"language": None}]
        config = FastSpeech2Config()
        config.model.multilingual = True
        model_languages = {"L1", "L2"}
        f = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stderr(f):
            validate_languages_with_model(
                data,
                config,
                model_languages=model_languages,
            )
        self.assertIn(
            "Your model is multilingual and you've failed to provide a language for all your sentences."
            f" Available languages are {model_languages}",
            f.getvalue(),
        )

    def test_multilingual_invalid_language(self):
        """
        The model is multilingual and the user provided a language that is not supported by the model.
        """
        language = "UNSUPPORTED"
        data = [{"language": language}]
        config = FastSpeech2Config()
        config.model.multilingual = True
        model_languages = {"L1", "L2"}
        f = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stderr(f):
            validate_languages_with_model(
                data,
                config,
                model_languages=model_languages,
            )
        self.assertIn(
            f"You provided {set((language,))} which is/are not a language(s) supported by the model {model_languages}.",
            f.getvalue(),
        )

    def test_not_multilingual_with_language(self):
        """
        The model is not multilingual and the user provided a language.
        """
        language = "L3"
        data = [{"language": language}]
        config = FastSpeech2Config()
        config.model.multilingual = False
        model_languages = DEFAULT_LANG2ID
        f = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stderr(f):
            validate_languages_with_model(
                data,
                config,
                model_languages=model_languages,
            )
        self.assertIn(
            f"The current model is not multilingual but you've provide {set((language,))}.",
            f.getvalue(),
        )

    def test_multilingual_single_language_no_language(self):
        """
        The model is multilingual but only has one language and the user did not provided a language.
        We expect the data's language to be changed to the only language provided by the model.
        """
        language = None
        data = [{"language": language}]
        config = FastSpeech2Config()
        config.model.multilingual = True
        model_languages = {
            "L1",
        }
        validate_languages_with_model(
            data,
            config,
            model_languages=model_languages,
        )
        self.assertListEqual(data, [{"language": "L1"}])

    def test_multispeaker_no_speaker(self):
        """
        The model is multispeaker but the user did not provide a speaker.
        """
        data = [{"speaker": None}]
        config = FastSpeech2Config()
        config.model.multispeaker = True
        model_speakers = {"S1", "S2"}
        f = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stderr(f):
            validate_speakers_with_model(
                data,
                config,
                model_speakers=model_speakers,
            )
        self.assertIn(
            "Your model is multispeaker and you've failed to provide a speaker for all your sentences."
            f" Available speakers are {model_speakers}",
            f.getvalue(),
        )

    def test_multispeaker_invalid_speaker(self):
        """
        The model is multispeaker and the user provided a speaker that is not supported by the model.
        """
        speaker = "UNSUPPORTED"
        data = [{"speaker": speaker}]
        config = FastSpeech2Config()
        config.model.multispeaker = True
        model_speakers = {"S1", "S2"}
        f = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stderr(f):
            validate_speakers_with_model(
                data,
                config,
                model_speakers=model_speakers,
            )
        self.assertIn(
            f"You provided {set((speaker,))} which is/are not a speaker(s) supported by the model {model_speakers}.",
            f.getvalue(),
        )

    def test_not_multispeaker_with_speaker(self):
        """
        The model is not multispeaker and the user provided a speaker.
        """
        speaker = "s3"
        data = [{"speaker": speaker}]
        config = FastSpeech2Config()
        config.model.multispeaker = False
        model_speakers = DEFAULT_SPEAKER2ID
        f = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stderr(f):
            validate_speakers_with_model(
                data,
                config,
                model_speakers=model_speakers,
            )
        self.assertIn(
            f"The current model doesn't support multi speakers but you've provide {set((speaker,))}.",
            f.getvalue(),
        )

    def test_multispeaker_single_speaker_no_speaker(self):
        """
        The model is multispeaker but only has one speaker and the user did not provided a speaker.
        We expect the data's speaker to default to the only speaker provided by the model.
        """
        speaker = None
        data = [{"speaker": speaker}]
        config = FastSpeech2Config()
        config.model.multispeaker = True
        model_speakers = {
            "S1",
        }
        validate_speakers_with_model(
            data,
            config,
            model_speakers=model_speakers,
        )
        self.assertListEqual(data, [{"speaker": "S1"}])


if __name__ == "__main__":
    main()
