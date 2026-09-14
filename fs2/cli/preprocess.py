from enum import Enum

import typer
from everyvoice.base_cli.interfaces import preprocess_base_command_interface
from merge_args import merge_args

from .. import core

PreprocessCategories = Enum(  # type: ignore[misc]
    "PreprocessCategories",
    {category: category for category in core.PREPROCESS_CATEGORIES},
    type=str,
)


@merge_args(preprocess_base_command_interface)
def preprocess(
    compute_stats: bool = typer.Option(
        True, "--stats/--no-stats", "-S", help="Calculate stats for energy and pitch"
    ),
    steps: list[PreprocessCategories] = typer.Option(
        [cat.value for cat in PreprocessCategories],
        "-s",
        "--steps",
        help="Which steps of the preprocessor to use. If none are provided, all steps will be performed.",
    ),
    **kwargs,
):
    """Preprocess data for text-to-spec (FastSpeech2) training

    # Preprocess Help

    This command will preprocess all of the data you need for use with EveryVoice.

    By default every step of the preprocessor will be done by running:

    **fs2l preprocess config/everyvoice-text-to-spec.yaml**

    If you only want to process specific things, you can run specific commands by adding them as options for example:

    **fs2l preprocess config/everyvoice-text-to-spec.yaml -s energy -s pitch**
    """
    config = core.load_config(
        config_file=kwargs.pop("config_file"),
        config_args=kwargs.pop("config_args"),
    )
    core.preprocess(
        config=config,
        compute_stats=compute_stats,
        steps=[step.name for step in steps],
        **kwargs,
    )
