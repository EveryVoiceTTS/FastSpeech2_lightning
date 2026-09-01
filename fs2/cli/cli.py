import typer
from everyvoice.base_cli import command, default_typer_args

from .benchmark import benchmark as app_benchmark
from .preprocess import preprocess as app_preprocess
from .synthesize import synthesize as app_synthesize
from .train import train as app_train

app = typer.Typer(
    **default_typer_args,
    help="A PyTorch Lightning implementation of the FastSpeech2 Text-to-Speech Feature Prediction Model",
)

command(
    app,
    name="benchmark",
)(app_benchmark)

command(
    app,
    name="preprocess",
)(app_preprocess)

command(
    app,
    name="synthesize",
)(app_synthesize)

command(
    app,
    name="train",
)(app_train)
