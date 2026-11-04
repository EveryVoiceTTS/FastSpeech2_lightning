import typer
from everyvoice.base_cli import command, default_typer_args

from .benchmark import benchmark
from .preprocess import preprocess
from .synthesize import synthesize
from .train import train

app = typer.Typer(
    **default_typer_args,
    help="A PyTorch Lightning implementation of the FastSpeech2 Text-to-Speech Feature Prediction Model",
)

command(app)(preprocess)
command(app)(train)
command(app)(synthesize)
command(app)(benchmark)
