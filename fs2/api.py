import multiprocessing as mp
from pathlib import Path

import torch
from everyvoice.config.type_definitions import DatasetTextRepresentation

from . import core
from .config import FastSpeech2Config
from .model import FastSpeech2


def load_config(
    config_file: Path | str,
) -> FastSpeech2Config:
    """Load a FastSpeech2 configuration from config_file.

    Your fs2 config file is called "config/everyvoice-text-to-spec.yaml" if it
    was created using the "everyvoice new-project" wizard.

    If you need to override any values, change them in the returned config object.

    Args:
        config_file (Path|str): FastSpeech2 configuration filename
    """
    return core.load_config(config_file=Path(config_file))


def preprocess(
    config: FastSpeech2Config,
    compute_stats: bool = True,
    steps: list[str] = core.PREPROCESS_CATEGORIES,
    cpus: int = min(4, mp.cpu_count()),
    overwrite: bool = False,
    debug: bool = False,
) -> None:
    """Preprocess data for text-to-spec (FastSpeech2) training.

    The datasets to process are described in config.

    Args:
        config (FastSpeech2Config): your FastSpeech2 configuration
        compute_stats (bool): if set, calculate energe and pitch statistics
        steps (list[str]): steps to process, one or more of "audio", "spec", "attn", "text", "pitch", "energy"
        cpus (int): how many CPUs to use for preprocessing
        overwrite (bool): if false, existing files will be kept and only new files will be generated;
                   if true, redo all preprocessing, even if files already exist
        debug (bool): enable debugging
    """
    core.preprocess(
        config=config,
        compute_stats=compute_stats,
        steps=steps,
        cpus=cpus,
        overwrite=overwrite,
        debug=debug,
    )


def train(
    config: FastSpeech2Config,
    accelerator: str = "auto",
    devices: str | int = "auto",
    nodes: int = 1,
    strategy: str = "ddp",
):
    """Train your Text-to-Spec (FastSpeech2) model

    Args:
        config (FastSpeech2Config): your FastSpeech2 configuration
        accelerator (str): PyTorch Lightning Accelerator to use: https://pytorch-lightning.readthedocs.io/en/stable/extensions/accelerator.html
        devices ("auto" | str | int): the number of GPUs to use on each node as a str or int; use "auto" to let pytoch-lightning decide
        nodes (int): the number of nodes to use
        strategy (str): the strategy for data parallelization: https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_intermediate.html"
    """
    core.train(
        config=config,
        accelerator=accelerator,
        devices=str(devices),
        nodes=nodes,
        strategy=strategy,
    )


def synthesize(
    model: FastSpeech2,
    texts: list[str],
    *,
    language: str | None = None,
    duration_control: float = 1.0,
    style_reference: Path | None = None,
    speaker: str | None = None,
    accelerator: str = "auto",
    devices: str | int = "auto",
    text_representation: str | DatasetTextRepresentation = "characters",
    teacher_forcing_directory: Path | None = None,
    batch_size: int = 4,
    num_workers: int = 4,
) -> list[torch.Tensor]:
    """Synthesize one or more utterences in texts using model.

    Returns MEL spectrograms. Pass them through a Vocoder to generate wavs.

    Args:
        model (FastSpeech2): the model/checkpoint to use for synthesis
        texts (list[str]]): a list of sentences/utterences to synthesize
        language (str|None): language to use in a multi-lingual model
        duration_control (float): systhesis speaking rate, multiplies durations so
            lower numbers produce quicker speaking rates
        style_reference (Path|None): audio file containing a style reference; model
            must have been trainined with the global style token module to use this
        speaker (str|None): speaker to use in a multi-speaker model

    Returns: torch.Tensor: synthesized mel spectrogram"""
    return []


synthesize()
