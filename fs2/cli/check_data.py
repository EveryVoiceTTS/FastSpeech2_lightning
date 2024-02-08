import json
from pathlib import Path

import typer
from everyvoice.base_cli.interfaces import load_config_base_command_interface
from merge_args import merge_args


@merge_args(load_config_base_command_interface)
def check_data(
    filelist: Path = typer.Option(
        None, "--filelist", "-f", exists=True, dir_okay=False, file_okay=True
    ),
    **kwargs,
):
    from everyvoice.base_cli.helpers import load_config_base_command
    from everyvoice.preprocessor import Preprocessor
    from everyvoice.utils import generic_dict_loader

    from ..config import FastSpeech2Config

    config = load_config_base_command(
        model_config=FastSpeech2Config,
        **kwargs,
    )
    filelist = generic_dict_loader(filelist)
    preprocessor = Preprocessor(config)
    checked_data = preprocessor.check_data(filelist=filelist)
    with open("datapoints_sil_removed.json", "w", encoding="utf8") as f:
        json.dump(checked_data, f)
