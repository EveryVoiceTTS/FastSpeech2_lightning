import sys

import typer
from everyvoice.base_cli.interfaces import train_base_command_interface
from everyvoice.exceptions import EveryVoiceError
from loguru import logger
from merge_args import merge_args

from .. import core


@merge_args(train_base_command_interface)
def train(**kwargs):
    """Train your Text-to-Spec (FastSpeech2) model

    For example:

    **fs2l train config/everyvoice-text-to-spec.yaml**
    """

    debug = kwargs.pop("debug")
    try:
        config = core.load_config(
            config_args=kwargs.pop("config_args"),
            config_file=kwargs.pop("config_file"),
        )
        core.train(config=config, **kwargs)
    except typer.BadParameter as e:
        raise e
    except (EveryVoiceError, Exception) as e:
        logger.error(f"{e.__class__.__name__}: {e}")
        if debug:
            raise e
        else:
            logger.info("Rerun with --debug to see the full stack trace.")
            sys.exit(1)
