import multiprocessing as mp
from pathlib import Path

from . import core
from .config import FastSpeech2Config


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
