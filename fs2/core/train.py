from typing import TYPE_CHECKING

from everyvoice.utils import spinner

if TYPE_CHECKING:
    from ..config import FastSpeech2Config


def train(
    config: "FastSpeech2Config",
    accelerator: str,
    devices: str,
    nodes: int,
    strategy: str,
):
    with spinner():
        import json

        from everyvoice.base_cli.helpers import train_base_command
        from everyvoice.text.lookups import lookuptables_from_config

        from ..dataset import FastSpeech2DataModule
        from ..model import FastSpeech2
        from ..type_definitions_heavy import Stats

    lang2id, speaker2id = lookuptables_from_config(config)

    # TODO: What about when we are fine-tuning? Do the bins in the Variance Adaptor not change? https://github.com/EveryVoiceTTS/FastSpeech2_lightning/issues/28
    with open(config.preprocessing.save_dir / "stats.json", encoding="utf8") as f:
        stats: Stats = Stats(**json.load(f))

    model_kwargs = {"lang2id": lang2id, "speaker2id": speaker2id, "stats": stats}

    train_base_command(
        config=config,
        model=FastSpeech2,
        data_module=FastSpeech2DataModule,
        monitor="validation/total_loss",
        gradient_clip_val=1.0,
        model_kwargs=model_kwargs,
        accelerator=accelerator,
        devices=devices,
        nodes=nodes,
        strategy=strategy,
    )
