import json
import os
import sys
from enum import Enum
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from everyvoice.base_cli.interfaces import (
    load_config_base_command_interface,
    preprocess_base_command_interface,
    train_base_command_interface,
)
from loguru import logger
from merge_args import merge_args
from tqdm import tqdm

from .synthesis_outputs import SynthesisOutputs

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    help="A PyTorch Lightning implementation of the FastSpeech2 Text-to-Speech Feature Prediction Model",
)


class PreprocessCategories(str, Enum):
    audio = "audio"
    spec = "spec"
    attn = "attn"
    text = "text"
    pitch = "pitch"
    energy = "energy"


class BenchmarkType(str, Enum):
    training = "training"
    inference = "inference"


@app.command()
def benchmark(
    config_file: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="The path to your model configuration file.",
    ),
    benchmark_type: BenchmarkType = BenchmarkType.training,
    gpu: bool = True,
    warmup_reps: int = 10,
    repetitions: int = 300,
):
    import time
    from functools import partial

    import numpy as np
    import torch

    from .config import FastSpeech2Config
    from .dataset import FastSpeech2DataModule
    from .model import FastSpeech2

    config = FastSpeech2Config.load_config_from_path(config_file)
    loader = FastSpeech2DataModule(config)
    loader.prepare_data()
    batch = loader.collate_method(
        [loader.train_dataset[i] for i in range(config.training.batch_size)]
    )
    model = FastSpeech2(config=config, lang2id={}, speaker2id={})
    device = "cpu"
    if gpu:
        device = "cuda"
    model.to(device)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    if benchmark_type == BenchmarkType.training:
        benchmark_fn = model.forward
    else:
        benchmark_fn = partial(model.forward, inference=True)

    # INIT LOGGERS
    starter, ender = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(warmup_reps):
        _ = benchmark_fn(batch)
    # Forward
    for rep in range(repetitions):
        if gpu:
            starter.record()
            _ = benchmark_fn(batch)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
        else:
            t0 = time.time()
            _ = benchmark_fn(batch)
            t1 = time.time()
            curr_time = (t1 - t0) * 1000
            timings[rep] = curr_time
    print(
        f"Average forward pass for {benchmark_type.value} duration after {repetitions} repetitions: {np.sum(timings) / repetitions} ms Standard Deviation: {np.std(timings)}"
    )


@app.command()
@merge_args(load_config_base_command_interface)
def check_data(
    filelist: Path = typer.Option(
        None, "--filelist", "-f", exists=True, dir_okay=False, file_okay=True
    ),
    **kwargs,
):
    from everyvoice.base_cli.helpers import load_config_base_command
    from everyvoice.preprocessor import Preprocessor
    from everyvoice.utils import generic_dict_loader

    from .config import FastSpeech2Config

    config = load_config_base_command(
        model_config=FastSpeech2Config,
        **kwargs,
    )
    filelist = generic_dict_loader(filelist)
    preprocessor = Preprocessor(config)
    checked_data = preprocessor.check_data(filelist=filelist)
    with open("datapoints_sil_removed.json", "w", encoding="utf8") as f:
        json.dump(checked_data, f)


@app.command()
@merge_args(preprocess_base_command_interface)
def preprocess(
    compute_stats: bool = typer.Option(
        True, "-S", "--stats", help="Calculate stats for energy and pitch"
    ),
    steps: List[PreprocessCategories] = typer.Option(
        [cat.value for cat in PreprocessCategories],
        "-s",
        "--steps",
        help="Which steps of the preprocessor to use. If none are provided, all steps will be performed.",
    ),
    **kwargs,
):
    from everyvoice.base_cli.helpers import preprocess_base_command

    from .config import FastSpeech2Config

    preprocessor, config, processed = preprocess_base_command(
        model_config=FastSpeech2Config,
        steps=[step.name for step in steps],
        **kwargs,
    )

    if compute_stats:
        stats_path = config.preprocessing.save_dir / "stats.json"
        if stats_path.exists() and not kwargs["overwrite"]:
            logger.info(f"{stats_path} exists, please re-run with --overwrite flag")
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


@app.command()
@merge_args(train_base_command_interface)
def train(**kwargs):
    from everyvoice.base_cli.helpers import load_config_base_command, train_base_command
    from everyvoice.text.lookups import lookuptables_from_config

    from .config import FastSpeech2Config
    from .dataset import FastSpeech2DataModule
    from .model import FastSpeech2

    config_args = kwargs["config_args"]
    config_file = kwargs["config_file"]
    config = load_config_base_command(FastSpeech2Config, config_args, config_file)
    lang2id, speaker2id = lookuptables_from_config(config)
    model_kwargs = {
        "lang2id": lang2id,
        "speaker2id": speaker2id,
    }

    train_base_command(
        model_config=FastSpeech2Config,
        model=FastSpeech2,
        data_module=FastSpeech2DataModule,
        monitor="training/total_loss",
        gradient_clip_val=1.0,
        model_kwargs=model_kwargs,
        **kwargs,
    )


@app.command()
def audit(
    config_file: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="The path to your model configuration file.",
    ),
    should_check_stats: bool = True,
    dimensions: bool = True,
):
    import torch

    from .config import FastSpeech2Config
    from .type_definitions import Stats, StatsInfo

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
                    "**/*{x['basename']}*.pt",
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
                    "**/*{x['basename']}*.pt",
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
                    "**/*{x['basename']}*.pt",
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
                    "**/*{x['basename']}*.pt",
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
            logger.info(
                "Nothing to check. Please re-run with --should_check_stats or --dimensions"
            )


@app.command()
def synthesize(  # noqa: C901
    model_path: Path = typer.Argument(
        ...,
        file_okay=True,
        exists=True,
        dir_okay=False,
        help="The path to a trained text-to-spec or e2e EveryVoice model.",
    ),
    output_dir: Path = typer.Option(
        "synthesis_output",
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="The directory where your synthesized audio should be written",
    ),
    texts: List[str] = typer.Option(
        [],
        "--text",
        "-t",
        help="Some text to synthesize. Choose --filelist if you want to synthesize more than one sample at a time.",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help="Specify which language to use in a multilingual system. [requires --text]",
    ),
    speaker: Optional[str] = typer.Option(
        None,
        "--speaker",
        "-s",
        help="Specify which speaker to use in a multispeaker system. [requires --text]",
    ),
    accelerator: str = typer.Option("auto", "--accelerator", "-a"),
    devices: str = typer.Option(
        "auto", "--devices", "-d", help="The number of GPUs to use"
    ),
    filelist: Path = typer.Option(
        None,
        "--filelist",
        "-f",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Synthesize all audio in a given filelist. Use --text if you want to just synthesize one sample.",
    ),
    output_type: List[SynthesisOutputs] = typer.Option(
        [SynthesisOutputs.wav.value],
        "-O",
        "--output-type",
        help="Which format to synthesize to. **wav** is the default and will synthesize to a playable audio file. **npy** will generate spectrograms required to fine-tune [HiFiGAN](https://github.com/jik876/hifi-gan) (Mel-band oriented tensors, K, T). **pt** will generate predicted Mel spectrograms in the EveryVoice format (time-oriented Tensors, T, K)",
    ),
    vocoder_path: Path = typer.Option(
        None,
        "--vocoder-path",
        "-v",
        help="The path to a trained vocoder in case one was not specified in your model configuration.",
    ),
):
    """Given some text and a trained model, generate some audio. i.e. perform typical speech synthesis"""
    # TODO: allow for changing of language/speaker and variance control
    import torch
    from everyvoice.preprocessor import Preprocessor
    from everyvoice.wizard.utils import sanitize_path

    from .model import FastSpeech2
    from .synthesize_text_dataset import SynthesizeTextDataSet

    if model_path is None:
        logger.error("Model path is required.")
        sys.exit(1)

    if texts and filelist:
        logger.warning(
            "Got arguments for both text and a filelist - this will only process the text. Please re-run without out providing text if you want to run batch synthesis"
        )
    if not texts and not filelist:
        logger.error("You must define either --text or --filelist")
        sys.exit(1)

    if not texts:
        if language is not None:
            logger.error("Specifying a language is only valid when using --text.")
            sys.exit(1)
        if speaker is not None:
            logger.error("Specifying a speaker is only valid when using --text.")
            sys.exit(1)

    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load checkpoints
    logger.info(f"Loading checkpoint from {model_path}")
    model: FastSpeech2 = FastSpeech2.load_from_checkpoint(model_path).to(device)
    model.eval()
    if SynthesisOutputs.wav in output_type:
        if vocoder_path:
            model.config.training.vocoder_path = vocoder_path
        if not model.config.training.vocoder_path:
            logger.error(
                "Sorry, no vocoder was provided, please add it to model.config.training.vocoder_path or as --vocoder-path /path/to/vocoder in the command line"
            )
            sys.exit(1)

    data: List[Dict[str, Any]]
    if texts:
        logger.info(f"Processing text '{texts}'")
        data = [
            {
                "basename": sanitize_path(text),
                "text": text,
                "language": language,
                "speaker": speaker,
            }
            for text in texts
        ]
    elif filelist:
        data = model.config.training.filelist_loader(filelist)
        data = [
            {
                "basename": sanitize_path(d["basename"]),
                "text": d["text"],
                "language": d.get("language", None),
                "speaker": d.get("speaker", None),
            }
            for d in data
        ]

    languages = set(d["language"] for d in data if d["language"] is not None)
    extra_languages = languages.difference(model.lang2id.keys())
    if len(extra_languages) > 0:
        logger.error(
            f"You provide '{languages}' which is/are not a language(s) supported by the model {set(model.lang2id.keys())}"
        )
        sys.exit(1)

    speakers = set(d["speaker"] for d in data if d["speaker"] is not None)
    extra_speakers = speakers.difference(model.speaker2id.keys())
    if len(extra_speakers) > 0:
        logger.error(
            f"You provide '{speakers}' which is/are not a speaker(s) supported by the model {set(model.speaker2id.keys())}"
        )
        sys.exit(1)

    dataset = SynthesizeTextDataSet(
        data,
        preprocessor=Preprocessor(model.config),
        lang2id=model.lang2id,
        speaker2id=model.speaker2id,
        device=device,
    )

    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger

    from .prediction_writing_callback import get_synthesis_output_callbacks

    tensorboard_logger = TensorBoardLogger(
        **(model.config.training.logger.model_dump(exclude={"sub_dir_callable"}))
    )
    trainer = Trainer(
        logger=tensorboard_logger,
        accelerator=accelerator,
        devices=devices,
        max_epochs=model.config.training.max_epochs,
        callbacks=get_synthesis_output_callbacks(
            output_type=output_type,
            output_dir=output_dir,
            config=model.config,
            output_key=model.output_key,
            device=device,
        ),
    )
    trainer.predict(model, dataset)


if __name__ == "__main__":
    app()
