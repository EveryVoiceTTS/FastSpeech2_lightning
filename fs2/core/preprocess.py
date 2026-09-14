from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import FastSpeech2Config


def load_config(
    config_file: Path,
    config_args: list[str] = [],
) -> "FastSpeech2Config":
    """Load FastSpeech2 configuration from config_file, possibly overriding some parameters"""
    from everyvoice.utils import spinner

    with spinner():
        from everyvoice.base_cli.helpers import load_config_base_command

        from ..config import FastSpeech2Config

    return load_config_base_command(
        model_config=FastSpeech2Config,
        config_file=config_file,
        config_args=config_args,
    )


PREPROCESS_CATEGORIES = ["audio", "spec", "attn", "text", "pitch", "energy"]


def preprocess(
    config: "FastSpeech2Config",
    compute_stats: bool,
    steps: list[str],
    cpus: int,
    overwrite: bool,
    debug: bool,
) -> None:
    import json

    from everyvoice.base_cli.helpers import preprocess_base_command

    for step in steps:
        if step not in PREPROCESS_CATEGORIES:
            raise ValueError(f"Unknown step '{step}'")

    preprocessor, processed = preprocess_base_command(
        config=config,
        steps=steps,
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
