#!/usr/bin/env python

import io
from contextlib import contextmanager, redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator
from unittest import TestCase, main

from loguru import logger
from typer.testing import CliRunner

from ..cli import app


@contextmanager
def capture_stderr() -> Generator[io.StringIO, None, None]:
    """Context manager to capture what is printed to stdout.

    Usage:
        with capture_stdout() as stdout:
            # do stuff whose stdout you want to capture
        stdout.getvalue() is what was printed to stdout during the context

        with capture_stdout():
            # do stuff with stdout suppressed

    Yields:
        stdout (io.StringIO): captured stdout
    """
    f = io.StringIO()
    with redirect_stderr(f):
        yield f


@contextmanager
def capture_logs(level="INFO", format="{level}:{name}:{message}"):
    """Capture loguru-based logs."""
    output = []
    handler_id = logger.add(output.append, level=level, format=format)
    yield output
    logger.remove(handler_id)


class SynthesizeTest(TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner(mix_stderr=True)

    def test_help(self):
        result = self.runner.invoke(app, ["synthesize", "--help"])
        self.assertIn("synthesize [OPTIONS] MODEL_PATH", result.stdout)

    def test_no_model(self):
        result = self.runner.invoke(app, ["synthesize"])
        self.assertIn("Missing argument 'MODEL_PATH'.", result.stdout)

    def test_filelist_and_text(self):
        with TemporaryDirectory() as tmpdir, capture_logs() as output:
            tmpdir = Path(tmpdir)
            test = tmpdir / "test.psv"
            test.touch()
            model = tmpdir / "model"
            model.touch()
            self.runner.invoke(
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
                output[0],
            )

    def test_no_filelist_nor_text(self):
        with TemporaryDirectory() as tmpdir, capture_logs() as output:
            tmpdir = Path(tmpdir)
            model = tmpdir / "model"
            model.touch()
            self.runner.invoke(
                app,
                (
                    "synthesize",
                    str(model),
                ),
            )
            self.assertIn("You must define either --text or --filelist", output[0])

    def test_filelist_language(self):
        with TemporaryDirectory() as tmpdir, capture_logs() as output:
            tmpdir = Path(tmpdir)
            test = tmpdir / "test.psv"
            test.touch()
            model = tmpdir / "model"
            model.touch()
            self.runner.invoke(
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
                "Specifying a language is only valid when using --text.", output[0]
            )

    def test_filelist_speaker(self):
        with TemporaryDirectory() as tmpdir, capture_logs() as output:
            tmpdir = Path(tmpdir)
            test = tmpdir / "test.psv"
            test.touch()
            model = tmpdir / "model"
            model.touch()
            self.runner.invoke(
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
                "Specifying a speaker is only valid when using --text.", output[0]
            )


if __name__ == "__main__":
    main()
