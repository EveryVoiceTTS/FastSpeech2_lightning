import json
from enum import Enum

import typer
from everyvoice.base_cli.interfaces import preprocess_base_command_interface
from everyvoice.utils import spinner
from merge_args import merge_args


class PreprocessCategories(str, Enum):
    audio = "audio"
    spec = "spec"
    attn = "attn"
    text = "text"
    pitch = "pitch"
    energy = "energy"


@merge_args(preprocess_base_command_interface)
def preprocess(
    compute_stats: bool = typer.Option(
        True, "-S", "--stats", help="Calculate stats for energy and pitch"
    ),
    steps: list[PreprocessCategories] = typer.Option(
        [cat.value for cat in PreprocessCategories],
        "-s",
        "--steps",
        help="Which steps of the preprocessor to use. If none are provided, all steps will be performed.",
    ),
    **kwargs,
):
    with spinner():
        from everyvoice.base_cli.helpers import preprocess_base_command

        from ..config import FastSpeech2Config

    preprocessor, config, processed = preprocess_base_command(
        model_config=FastSpeech2Config,
        steps=[step.name for step in steps],
        **kwargs,
    )

    if compute_stats:
        stats_path = config.preprocessing.save_dir / "stats.json"
        e_scaler, p_scaler = preprocessor.compute_stats(
            energy="energy" in processed, pitch="pitch" in processed
        )
        stats = {}
        if e_scaler:
            e_stats = e_scaler.calculate_stats()
            stats["energy"] = e_stats
        if p_scaler:
            p_stats = p_scaler.calculate_stats()
            stats["pitch"] = p_stats
        preprocessor.normalize_stats(e_scaler, p_scaler)
        # Merge with existing stats
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf8") as f:
                previous_stats = json.load(f)
        else:
            previous_stats = {}
        stats = {**previous_stats, **stats}
        with open(
            config.preprocessing.save_dir / "stats.json", "w", encoding="utf8"
        ) as f:
            json.dump(stats, f)
