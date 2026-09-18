from everyvoice.base_cli.interfaces import train_base_command_interface
from merge_args import merge_args

from .. import core


@merge_args(train_base_command_interface)
def train(**kwargs):
    """Train your Text-to-Spec (FastSpeech2) model

    For example:

    **fs2l train config/everyvoice-text-to-spec.yaml**
    """

    config = core.load_config(
        config_args=kwargs.pop("config_args"),
        config_file=kwargs.pop("config_file"),
    )
    core.train(config=config, **kwargs)
