import json
import os
import sys
from glob import glob
from pathlib import Path

import typer
from everyvoice.base_cli.interfaces import complete_path
from tqdm import tqdm


def audit(
    config_file: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="The path to your model configuration file.",
        autocompletion=complete_path,
    ),
    should_check_stats: bool = True,
    dimensions: bool = True,
):
    import torch

    from ..config import FastSpeech2Config
    from ..type_definitions_heavy import Stats, StatsInfo

    def check_stats(data, path, stats: StatsInfo):
        import torch

        data_min = torch.min(data)
        data_max = torch.max(data)
        assert (
            data_min >= stats.norm_min
        ), f"Data in {path} had min of {data_min} despite stats min being {stats.norm_min}"
        assert (
            data_max <= stats.norm_max
        ), f"Data in {path} had max of {data_min} despite stats max being {stats.norm_max}"

    original_config: FastSpeech2Config = FastSpeech2Config.load_config_from_path(
        config_file
    )
    if should_check_stats:
        with open(original_config.preprocessing.save_dir / "stats.json") as f:
            stats: Stats = Stats(**json.load(f))
    files = []
    for dataset in original_config.preprocessing.source_data:
        files += dataset.filelist_loader(dataset.filelist)
    for x in tqdm(files):
        duration_files = (
            glob(
                os.path.join(
                    (original_config.preprocessing.save_dir / "duration"),
                    f"**/*{x['basename']}*.pt",
                ),
                recursive=True,
            )
            if dimensions
            else []
        )
        energy_files = (
            glob(
                os.path.join(
                    (original_config.preprocessing.save_dir / "energy"),
                    f"**/*{x['basename']}*.pt",
                ),
                recursive=True,
            )
            if dimensions or should_check_stats
            else []
        )
        pitch_files = (
            glob(
                os.path.join(
                    (original_config.preprocessing.save_dir / "pitch"),
                    f"**/*{x['basename']}*.pt",
                ),
                recursive=True,
            )
            if dimensions or should_check_stats
            else []
        )
        text_files = (
            glob(
                os.path.join(
                    (original_config.preprocessing.save_dir / "text"),
                    f"**/*{x['basename']}*.pt",
                ),
                recursive=True,
            )
            if dimensions or should_check_stats
            else []
        )
        if dimensions:
            for dur_path, e_path, p_path, t_path in zip(
                duration_files, energy_files, pitch_files, text_files
            ):
                duration = torch.load(dur_path)
                text = torch.load(t_path)
                assert duration.size(0) == text.size(0)
                e_asserted_duration = (
                    duration.size(0)
                    if original_config.model.variance_predictors.energy.level == "phone"
                    else torch.sum(duration)
                )
                p_asserted_duration = (
                    duration.size(0)
                    if original_config.model.variance_predictors.pitch.level == "phone"
                    else torch.sum(duration)
                )
                e_data = torch.load(e_path)
                assert e_asserted_duration == e_data.size(
                    0
                ), f"Data in {e_path} had duration of {e_data.size(0)} but should have been {e_asserted_duration}"
                if should_check_stats is not None:
                    check_stats(e_data, e_path, stats.energy)
                p_data = torch.load(p_path)
                assert p_asserted_duration == p_data.size(
                    0
                ), f"Data in {p_path} had duration of {p_data.size(0)} but should have been {p_asserted_duration}"
                if stats is not None:
                    check_stats(p_data, p_path, stats.pitch)
        elif should_check_stats:
            for path in energy_files:
                data = torch.load(path)
                check_stats(data, path, stats.energy)
            for path in pitch_files:
                data = torch.load(path)
                check_stats(data, path, stats.pitch)
        else:
            print(
                "Nothing to check. Please re-run with --should_check_stats or --dimensions",
                file=sys.stderr,
            )
