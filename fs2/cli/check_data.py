import json
import sys
from pathlib import Path
from typing import Optional

import typer
from everyvoice.base_cli.interfaces import complete_path
from everyvoice.config.type_definitions import DatasetTextRepresentation
from everyvoice.utils import generic_psv_filelist_reader, spinner
from loguru import logger

from .synthesize import get_global_step, synthesize_helper


def check_data_command(  # noqa: C901
    config_file: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="The path to your text-to-spec model configuration file.",
        shell_complete=complete_path,
    ),
    model_path: Optional[Path] = typer.Argument(
        ...,
        file_okay=True,
        exists=True,
        dir_okay=False,
        help="The path to a trained text-to-spec (i.e., feature prediction) or e2e EveryVoice model.",
        shell_complete=complete_path,
    ),
    output_dir: Path = typer.Option(
        "checked_data",
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="The directory where your synthesized audio should be written",
        shell_complete=complete_path,
    ),
    style_reference: Optional[Path] = typer.Option(
        None,
        "--style-reference",
        "-S",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="The path to an audio file containing a style reference. Your text-to-spec must have been trained with the global style token module to use this feature.",
        shell_complete=complete_path,
    ),
    accelerator: str = typer.Option("auto", "--accelerator", "-a"),
    devices: str = typer.Option(
        "auto", "--devices", "-d", help="The number of GPUs to use"
    ),
    filelist: Path = typer.Option(
        None,
        "--filelist",
        "-f",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="The path to a file containing a list of utterances (a.k.a filelist). Use --text if you want to just synthesize one sample.",
        shell_complete=complete_path,
    ),
    text_representation: DatasetTextRepresentation = typer.Option(
        DatasetTextRepresentation.characters,
        help="The representation of the text you are synthesizing. Can be either 'characters', 'phones', or 'arpabet'. The input type must be compatible with your model.",
    ),
    teacher_forcing_directory: Path = typer.Option(
        "preprocessed",
        "--preprocessed-directory",
        "-p",
        help="The path to the folder containing all of your preprocessed data.",
        dir_okay=True,
        file_okay=False,
        shell_complete=complete_path,
    ),
    num_workers: int = typer.Option(
        4,
        "--num-workers",
        "-n",
        help="Number of workers to process the data.",
    ),
    calculate_stats: bool = typer.Option(
        True,
        "--calculate-stats/--no-calculate-stats",
        help="Whether to calculate basic statistics on your dataset.",
    ),
    objective_evaluation: bool = typer.Option(
        True,
        "--objective-evaluation/--no-objective-evaluation",
        help="Whether to perform objective evaluation on your dataset using TorchSquim. This is time-consuming.",
    ),
    clip_detection: bool = typer.Option(
        False,
        "--clip-detection/--no-clip-detection",
        help="Whether to detect clipping in your audio. This is expensive so we do not do this by default.",
    ),
):
    """
    Given a filelist and some preprocessed data, check some basic statistics on the data.
    If a checkpoint is provided, also calculate the loss for each datapoint with respect to the model.

    Note: this function was written by restricting the synthesize command.
    """

    with spinner():
        from everyvoice.base_cli.helpers import MODEL_CONFIGS, load_unknown_config
        from everyvoice.preprocessor import Preprocessor
        from everyvoice.utils.heavy import get_device_from_accelerator

        from ..model import FastSpeech2
        from .check_data_heavy import check_data_from_filelist
        from .synthesize import load_data_from_filelist

    config = load_unknown_config(config_file)
    preprocessor = Preprocessor(config)
    if not any((isinstance(config, x) for x in MODEL_CONFIGS)):
        print(
            "Sorry, your file does not appear to be a valid model configuration. Please choose another model config file."
        )
        sys.exit(1)

    output_dir.mkdir(exist_ok=True, parents=True)

    if filelist is None:
        training_filelist = generic_psv_filelist_reader(
            config.training.training_filelist
        )
        val_filelist = generic_psv_filelist_reader(config.training.validation_filelist)
        combined_filelist_data = training_filelist + val_filelist
    else:
        combined_filelist_data = generic_psv_filelist_reader(filelist)

    # process stats
    if calculate_stats:
        stats = check_data_from_filelist(
            preprocessor,
            combined_filelist_data,
            heavy_clip_detection=clip_detection,
            heavy_objective_evaluation=objective_evaluation,
        )
        if not stats:
            print(
                f"Sorry, the data at {config.training.training_filelist} and {config.training.validation_filelist} is empty so there is nothing to check."
            )
            sys.exit(1)
        else:
            with open(output_dir / "checked-data.json", "w", encoding="utf8") as f:
                json.dump(stats, f)

    if model_path:
        # NOTE: We want to be able to put the vocoder on the proper accelerator for
        # it to be compatible with the vocoder's input device.
        # We could misuse the trainer's API and use the private variable
        # trainer._accelerator_connector._accelerator_flag but that value is
        # computed when instantiating a trainer and that is exactly when we need
        # the information to create the callbacks.
        device = get_device_from_accelerator(accelerator)

        # Load checkpoints
        print(f"Loading checkpoint from {model_path}", file=sys.stderr)

        from pydantic import ValidationError

        try:
            model: FastSpeech2 = FastSpeech2.load_from_checkpoint(model_path).to(device)  # type: ignore
        except (TypeError, ValidationError) as e:
            logger.error(f"Unable to load {model_path}: {e}")
            sys.exit(1)
        model.eval()

        if filelist is None:
            training_filelist = load_data_from_filelist(
                config.training.training_filelist, model, text_representation
            )
            val_filelist = load_data_from_filelist(
                config.training.training_filelist, model, text_representation
            )
            combined_filelist_data = training_filelist + val_filelist
        else:
            combined_filelist_data = None

        # get global step
        # We can't just use model.global_step because it gets reset by lightning
        global_step = get_global_step(model_path)

        synthesize_helper(
            model=model,
            texts=None,
            style_reference=style_reference,
            language=None,
            speaker=None,
            duration_control=1.0,
            global_step=global_step,
            output_type=[],
            text_representation=text_representation,
            accelerator=accelerator,
            devices=devices,
            device=device,
            batch_size=1,
            num_workers=num_workers,
            filelist=filelist,
            filelist_data=combined_filelist_data,
            teacher_forcing_directory=teacher_forcing_directory,
            output_dir=output_dir,
            vocoder_model=None,
            vocoder_config=None,
            vocoder_global_step=None,
            return_scores=True,
        )
