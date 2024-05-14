import json

from everyvoice.base_cli.interfaces import train_base_command_interface
from merge_args import merge_args


@merge_args(train_base_command_interface)
def train(**kwargs):
    from everyvoice.base_cli.helpers import load_config_base_command, train_base_command
    from everyvoice.text.lookups import lookuptables_from_config

    from ..config import FastSpeech2Config
    from ..dataset import FastSpeech2DataModule
    from ..model import FastSpeech2
    from ..type_definitions_heavy import Stats

    config_args = kwargs["config_args"]
    config_file = kwargs["config_file"]
    config = load_config_base_command(FastSpeech2Config, config_args, config_file)
    lang2id, speaker2id = lookuptables_from_config(config)

    # TODO: What about when we are fine-tuning? Do the bins in the Variance Adaptor not change? https://github.com/roedoejet/FastSpeech2_lightning/issues/28
    with open(config.preprocessing.save_dir / "stats.json") as f:
        stats: Stats = Stats(**json.load(f))

    model_kwargs = {"lang2id": lang2id, "speaker2id": speaker2id, "stats": stats}

    train_base_command(
        model_config=FastSpeech2Config,
        model=FastSpeech2,
        data_module=FastSpeech2DataModule,
        monitor="validation/total_loss",
        gradient_clip_val=1.0,
        model_kwargs=model_kwargs,
        **kwargs,
    )
