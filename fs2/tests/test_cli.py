#!/usr/bin/env python

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

from typer.testing import CliRunner

from ..cli import app


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


if __name__ == "__main__":
    main()
