import os
from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from smts.preprocessor import Preprocessor
from smts.utils import update_config_from_cli_args, update_config_from_path

from .config import CONFIGS, FastSpeech2Config
from .dataset import FastSpeech2DataModule
from .model import FastSpeech2

app = typer.Typer(pretty_exceptions_show_locals=False)

_config_keys = {k: k for k in CONFIGS.keys()}

CONFIGS_ENUM = Enum("CONFIGS", _config_keys)  # type: ignore


class PreprocessCategories(str, Enum):
    audio = "audio"
    mel = "mel"
    text = "text"


@app.command()
def preprocess(
    name: CONFIGS_ENUM,
    data: Optional[List[PreprocessCategories]] = typer.Option(None, "-d", "--data"),
    output_path: Optional[Path] = typer.Option(
        "processed_filelist.psv", "-o", "--output"
    ),
    overwrite: bool = typer.Option(False, "-O", "--overwrite"),
):
    config = FastSpeech2Config.load_config_from_path(CONFIGS[name.value])
    preprocessor = Preprocessor(config)
    to_preprocess = {k: k in data for k in PreprocessCategories.__members__.keys()}  # type: ignore
    if not data:
        logger.info(
            f"No specific preprocessing data requested, processing everything (pitch, mel, energy, durations, inputs) from dataset '{name}'"
        )
    else:
        preprocessor.preprocess(
            output_path=output_path,
            process_audio=to_preprocess["audio"],
            process_spec=to_preprocess["mel"],
            process_text=to_preprocess["text"],
            overwrite=overwrite,
        )


@app.command()
def train(
    name: CONFIGS_ENUM,
    accelerator: str = typer.Option("auto"),
    devices: str = typer.Option("auto"),
    strategy: str = typer.Option(None),
    config_args: List[str] = typer.Option(None, "--config", "-c"),
    config_path: Path = typer.Option(None, exists=True, dir_okay=False, file_okay=True),
):
    original_config = FastSpeech2Config.load_config_from_path(CONFIGS[name.value])
    config = update_config_from_cli_args(config_args, original_config)
    config: FastSpeech2Config = update_config_from_path(config_path, config)
    tensorboard_logger = TensorBoardLogger(**(config.training.logger.dict()))
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger.info("Starting training for alignment model.")
    ckpt_callback = ModelCheckpoint(
        monitor="validation/loss",
        mode="min",
        save_last=True,
        save_top_k=config.training.save_top_k_ckpts,
        every_n_train_steps=config.training.ckpt_steps,
        every_n_epochs=config.training.ckpt_epochs,
    )
    trainer = Trainer(
        gradient_clip_val=1.0,
        logger=tensorboard_logger,
        accelerator=accelerator,
        devices=devices,
        max_epochs=config.training.max_epochs,
        callbacks=[ckpt_callback, lr_monitor],
        strategy=strategy,
        detect_anomaly=False,  # used for debugging, but triples training time
    )
    aligner = FastSpeech2(config)
    data = FastSpeech2DataModule(config)
    last_ckpt = (
        config.training.finetune_checkpoint
        if config.training.finetune_checkpoint is not None
        and os.path.exists(config.training.finetune_checkpoint)
        else None
    )
    tensorboard_logger.log_hyperparams(config.dict())
    trainer.fit(aligner, data, ckpt_path=last_ckpt)


if __name__ == "__main__":
    app()
