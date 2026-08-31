from pathlib import Path
from typing import Annotated

from everyvoice.base_cli.interfaces import (
    AcceleratorOption,
    ConfigArgsOption,
    ConfigFileArgument,
    DevicesOption,
    NodesOption,
    StrategyOption,
)
from everyvoice.utils import spinner
from everyvoice.wizard import TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX


def train(
    config_file: Annotated[Path, ConfigFileArgument],
    config_args: Annotated[list[str], ConfigArgsOption] = [],
    accelerator: Annotated[str, AcceleratorOption] = "auto",
    devices: Annotated[str, DevicesOption] = "auto",
    nodes: Annotated[int, NodesOption] = 1,
    strategy: Annotated[str, StrategyOption] = "ddp",
) -> None:
    """Train your FastSpeech2 text-to-spec model.

    For example:

    **everyvoice train text-to-spec config/everyvoice-text-to-spec.yaml**
    """
    with spinner():
        import json

        from everyvoice.base_cli.helpers import (
            load_config_base_command,
            train_base_command,
        )
        from everyvoice.text.lookups import lookuptables_from_config

        from ..config import FastSpeech2Config
        from ..dataset import FastSpeech2DataModule
        from ..model import FastSpeech2
        from ..type_definitions_heavy import Stats

    config = load_config_base_command(FastSpeech2Config, config_args, config_file)
    lang2id, speaker2id = lookuptables_from_config(config)

    # TODO: What about when we are fine-tuning? Do the bins in the Variance Adaptor not change? https://github.com/EveryVoiceTTS/FastSpeech2_lightning/issues/28
    with open(config.preprocessing.save_dir / "stats.json", encoding="utf8") as f:
        stats: Stats = Stats(**json.load(f))

    model_kwargs = {"lang2id": lang2id, "speaker2id": speaker2id, "stats": stats}

    train_base_command(
        model_config=FastSpeech2Config,
        model=FastSpeech2,
        data_module=FastSpeech2DataModule,
        monitor="validation/total_loss",
        gradient_clip_val=1.0,
        model_kwargs=model_kwargs,
        config_file=config_file,
        config_args=config_args,
        accelerator=accelerator,
        devices=devices,
        nodes=nodes,
        strategy=strategy,
    )


# docstrings cannot be f-strings, so assert they're still in sync
assert (
    f"config/{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml" in train.__doc__  # type: ignore
), "docstring out of sync with everyvoice.wizard config file names"
