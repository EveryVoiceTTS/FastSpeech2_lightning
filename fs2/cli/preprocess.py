import multiprocessing as mp
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from everyvoice.base_cli.interfaces import (
    ConfigArgsOption,
    ConfigFileArgument,
    CPUsOption,
    DebugFlag,
    OverwriteFlag,
)
from everyvoice.utils import spinner
from everyvoice.wizard import TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX


class PreprocessCategories(str, Enum):
    audio = "audio"
    spec = "spec"
    attn = "attn"
    text = "text"
    pitch = "pitch"
    energy = "energy"


ComputeStatsToggle = typer.Option(
    "-S", "--stats/--no-stats", help="Calculate stats for energy and pitch"
)
PreprocessStepsOption = typer.Option(
    "-s",
    "--steps",
    help="Which steps of the preprocessor to use. If none are provided, all steps will be performed.",
)


def preprocess(
    *,
    compute_stats: Annotated[bool, ComputeStatsToggle] = True,
    steps: Annotated[list[PreprocessCategories], PreprocessStepsOption] = list(
        PreprocessCategories
    ),
    config_file: Annotated[Path, ConfigFileArgument],
    config_args: Annotated[list[str], ConfigArgsOption] = [],
    cpus: Annotated[int, CPUsOption] = min(4, mp.cpu_count()),
    overwrite: Annotated[bool, OverwriteFlag] = False,
    debug: Annotated[bool, DebugFlag] = False,
) -> None:
    """
    Preprocess data for a FastSpeech2 text-to-spec model.

    By default every step of the preprocessor will be done by running:

    **everyvoice preprocess text-to-spec config/everyvoice-text-to-spec.yaml**

    To run only specific steps:

    **everyvoice preprocess text-to-spec config/everyvoice-text-to-spec.yaml -s energy -s pitch**
    """
    print("STATS", compute_stats)
    with spinner():
        import json

        from everyvoice.base_cli.helpers import preprocess_base_command

        from ..config import FastSpeech2Config

    try:
        my_steps = [PreprocessCategories(step).name for step in steps]
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    preprocessor, config, processed = preprocess_base_command(
        model_config=FastSpeech2Config,
        steps=my_steps,
        config_file=config_file,
        config_args=config_args,
        cpus=cpus,
        overwrite=overwrite,
        debug=debug,
    )

    if compute_stats:
        # NOTE that these stats are computed over all datasets in a project, regardless of whether they are all the same language
        stats_path = config.preprocessing.save_dir / "stats.json"
        e_scaler, p_scaler, cl_scaler, pl_scaler = preprocessor.compute_stats(
            energy="energy" in processed,
            pitch="pitch" in processed,
            char_length="text" in processed,
            phone_length="text" in processed,
        )
        stats = {}
        if e_scaler:
            e_stats = e_scaler.calculate_stats()
            stats["energy"] = e_stats
        if p_scaler:
            p_stats = p_scaler.calculate_stats()
            stats["pitch"] = p_stats
        if cl_scaler:
            cl_stats = cl_scaler.calculate_stats()
            stats["character_length"] = cl_stats
        if pl_scaler:
            pl_stats = pl_scaler.calculate_stats()
            stats["phone_length"] = pl_stats

        preprocessor.normalize_stats(e_scaler, p_scaler)

        # Merge with existing stats
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf8") as f:
                previous_stats = json.load(f)
        else:
            previous_stats = {}
        stats = {**previous_stats, **stats}
        with open(stats_path, "w", encoding="utf8") as f:
            json.dump(stats, f)


# docstrings cannot be f-strings, so assert they're still in sync
assert f"config/{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml" in (
    preprocess.__doc__ or ""
), "docstring out of sync with everyvoice.wizard config file names"
