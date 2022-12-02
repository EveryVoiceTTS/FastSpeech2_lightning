import json
import multiprocessing as mp
import os
from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger
from tqdm import tqdm

from .config import CONFIGS, FastSpeech2Config
from .variance_adaptor import Stats, StatsInfo

app = typer.Typer(pretty_exceptions_show_locals=False)

_config_keys = {k: k for k in CONFIGS.keys()}

CONFIGS_ENUM = Enum("CONFIGS", _config_keys)  # type: ignore


class PreprocessCategories(str, Enum):
    audio = "audio"
    mel = "mel"
    text = "text"
    pitch = "pitch"
    energy = "energy"


class SynthesisOutputs(str, Enum):
    wav = "wav"
    npy = "npy"
    pt = "pt"


@app.command()
def preprocess(
    name: CONFIGS_ENUM,
    data: Optional[List[PreprocessCategories]] = typer.Option(None, "-d", "--data"),
    output_path: Optional[Path] = typer.Option(
        "processed_filelist.psv", "-o", "--output"
    ),
    cpus: Optional[int] = typer.Option(mp.cpu_count(), "-c", "--cpus"),
    overwrite: bool = typer.Option(False, "-O", "--overwrite"),
):
    from smts.preprocessor import Preprocessor

    config = FastSpeech2Config.load_config_from_path(CONFIGS[name.value])
    preprocessor = Preprocessor(config)
    to_preprocess = {k: k in data for k in PreprocessCategories.__members__.keys()}  # type: ignore
    if not data:
        logger.info(
            f"No specific preprocessing data requested, processing everything (pitch, mel, energy, durations, inputs) from dataset '{name}'"
        )
        to_preprocess = {k: True for k in to_preprocess}
    preprocessor.preprocess(
        output_path=output_path,
        process_audio=to_preprocess["audio"],
        process_spec=to_preprocess["mel"],
        process_text=to_preprocess["text"],
        process_pitch=to_preprocess["pitch"],
        process_energy=to_preprocess["energy"],
        compute_stats=True,
        cpus=cpus,
        overwrite=overwrite,
    )


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


@app.command()
def audit(name: CONFIGS_ENUM, should_check_stats: bool = True, dimensions: bool = True):
    import torch

    original_config: FastSpeech2Config = FastSpeech2Config.load_config_from_path(
        CONFIGS[name.value]
    )
    if should_check_stats:
        with open(original_config.preprocessing.save_dir / "stats.json") as f:
            stats: Stats = Stats(**json.load(f))
    files = []
    for dataset in original_config.preprocessing.source_data:
        files += dataset.filelist_loader(dataset.filelist)
    for x in tqdm(files):
        duration_files = (
            (original_config.preprocessing.save_dir / "duration").glob(
                f"*{x['basename']}*.pt"
            )
            if dimensions
            else []
        )
        energy_files = (
            (original_config.preprocessing.save_dir / "energy").glob(
                f"*{x['basename']}*.pt"
            )
            if dimensions or should_check_stats
            else []
        )
        pitch_files = (
            (original_config.preprocessing.save_dir / "pitch").glob(
                f"*{x['basename']}*.pt"
            )
            if dimensions or should_check_stats
            else []
        )
        text_files = (
            (original_config.preprocessing.save_dir / "text").glob(
                f"*{x['basename']}*.pt"
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
                    if original_config.model.variance_adaptor.variance_predictors.energy.level
                    == "phone"
                    else torch.sum(duration)
                )
                p_asserted_duration = (
                    duration.size(0)
                    if original_config.model.variance_adaptor.variance_predictors.pitch.level
                    == "phone"
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
def synthesize(
    model_path: Path = typer.Argument(
        ...,
        file_okay=True,
        exists=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(
        "synthesis_output",
        "--output_dir",
        "-o",
        file_okay=False,
        dir_okay=True,
    ),
    text: str = typer.Option("", "--text", "-t"),
    accelerator: str = typer.Option("auto", "--accelerator", "-a"),
    devices: str = typer.Option("auto", "--devices", "-d"),
    strategy: str = typer.Option(None),
    filelist: Path = typer.Option(
        None, "--filelist", "-f", exists=True, file_okay=True, dir_okay=False
    ),
    name: CONFIGS_ENUM = typer.Option(None, "--name", "-n"),
    output_type: List[SynthesisOutputs] = typer.Option(..., "-O", "--output_type"),
    vocoder_path: Path = typer.Option(None, "--vocoder_path", "-v"),
):
    # TODO: allow for changing of language/speaker and variance control
    import torch
    from slugify import slugify
    from smts.preprocessor import Preprocessor

    from .model import FastSpeech2

    if model_path is None:
        logger.error
        exit()
    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load checkpoints
    logger.info(f"Loading checkpoint from {model_path}")
    model: FastSpeech2 = FastSpeech2.load_from_checkpoint(model_path).to(device)
    model.eval()
    preprocessor = Preprocessor(model.config)
    if "wav" in output_type:
        if vocoder_path:
            model.config.training.vocoder_path = vocoder_path
        assert (
            model.config.training.vocoder_path
        ), "Sorry, no vocoder was provided, please add it to model.config.training.vocoder_path or as --vocoder-path /path/to/vocoder in the command line"

    if text and (name or filelist):
        logger.warning(
            "Got arguments for both text and a config name or filelist - this will only process the text. Please re-run without out providing text if you want to run batch synthesis"
        )

    # Single Inference
    if text:
        logger.info(f"Processing text '{text}'")
        text_len = min(10, len(text))
        data_path = output_dir / slugify(text[:text_len])
        text_tensor = preprocessor.extract_text_inputs(text)
        # Create Batch
        logger.info("Creating batch")
        src_lens = torch.LongTensor([text_tensor.size(0)])
        max_src_len = max(src_lens)
        batch = {
            "text": text_tensor,
            "src_lens": src_lens,
            "max_src_len": max_src_len,
            "speaker_id": torch.LongTensor([0]),
            "language_id": torch.LongTensor([0]),
        }
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["max_mel_len"] = 1_000_000
        batch["mel_lens"] = None
        # Run model
        with torch.no_grad():
            logger.info("Predicting spectral features")
            spec = model.forward(batch, inference=True)["postnet_output"]
        if "wav" in output_type:
            from hfgl.utils import synthesize_data
            from scipy.io.wavfile import write

            logger.info(f"Loading Vocoder from {model.config.training.vocoder_path}")
            ckpt = torch.load(model.config.training.vocoder_path)
            logger.info("Generating waveform...")
            wav, sr = synthesize_data(spec, ckpt)
            logger.info(f"Writing file {data_path}")
            write(f"{data_path}.wav", sr, wav)
        if "npy" in output_type:
            import numpy as np

            logger.info("Formatting for use with original HiFiGAN")
            spec = spec.squeeze().transpose(0, 1).cpu().numpy()
            np.save(f"{data_path}.npy", spec)
        if "pt" in output_type:
            torch.save(spec, f"{data_path}.pt")

    elif filelist or name:
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import Callback
        from pytorch_lightning.loggers import TensorBoardLogger

        from .dataset import FastSpeech2DataModule

        class PredictionWritingCallback(Callback):
            def __init__(self, output_types, output_dir, config: FastSpeech2Config):
                self.save_dir = output_dir
                self.config = config
                self.sep = config.preprocessing.value_separator
                self.output_types: List[SynthesisOutputs] = output_types
                logger.info(f"Saving output to {self.save_dir / 'synthesized_spec'}")
                if "pt" in self.output_types:
                    (self.save_dir / "synthesized_spec").mkdir(
                        parents=True, exist_ok=True
                    )
                if "npy" in self.output_types:
                    (self.save_dir / "original_hifigan_spec").mkdir(
                        parents=True, exist_ok=True
                    )
                if "wav" in self.output_types:
                    (self.save_dir / "wav").mkdir(parents=True, exist_ok=True)

            def on_predict_batch_end(
                self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            ):
                if "wav" in self.output_types:
                    from hfgl.utils import synthesize_data
                    from scipy.io.wavfile import write
                    from smts.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import (
                        HiFiGANConfig,
                    )

                    ckpt = torch.load(self.config.training.vocoder_path)
                    vocoder_config: HiFiGANConfig = ckpt["config"]
                    sampling_rate_change = (
                        vocoder_config.preprocessing.audio.output_sampling_rate
                        // vocoder_config.preprocessing.audio.input_sampling_rate
                    )
                    output_hop_size = (
                        sampling_rate_change
                        * vocoder_config.preprocessing.audio.fft_hop_frames
                    )
                    wavs, sr = synthesize_data(outputs["postnet_output"], ckpt)
                if "npy" in self.output_types:
                    import numpy as np

                    specs = outputs["postnet_output"].transpose(0, 1).cpu().numpy()
                for b in range(batch["text"].size(0)):
                    basename = batch["basename"][b]
                    speaker = batch["speaker"][b]
                    language = batch["language"][b]
                    unmasked_len = outputs["tgt_lens"][
                        b
                    ]  # the vocoder output includes padding so we have to remove that
                    if "pt" in self.output_types:
                        torch.save(
                            outputs["postnet_output"][b][:unmasked_len]
                            .transpose(0, 1)
                            .cpu(),
                            self.save_dir
                            / "synthesized_spec"
                            / self.sep.join(
                                [
                                    basename,
                                    speaker,
                                    language,
                                    f"spec-pred-{self.config.preprocessing.audio.input_sampling_rate}-{self.config.preprocessing.audio.spec_type}.pt",
                                ]
                            ),
                        )
                    if "wav" in self.output_types:
                        write(
                            self.save_dir
                            / "wav"
                            / self.sep.join([basename, speaker, language, "pred.wav"]),
                            sr,
                            wavs[b][: (unmasked_len * output_hop_size)],
                        )
                    if "npy" in self.output_types:
                        np.save(
                            self.save_dir
                            / "original_hifigan_spec"
                            / self.sep.join([basename, speaker, language, "pred.npy"]),
                            specs[b][:, :unmasked_len].squeeze(),
                        )

        if filelist:
            model.config.training.filelist = filelist
        else:
            original_config: FastSpeech2Config = (
                FastSpeech2Config.load_config_from_path(CONFIGS[name.value])
            )
            model.config.training.filelist = original_config.training.filelist
        data = FastSpeech2DataModule(model.config)
        tensorboard_logger = TensorBoardLogger(**(model.config.training.logger.dict()))
        trainer = Trainer(
            logger=tensorboard_logger,
            accelerator=accelerator,
            devices=devices,
            max_epochs=model.config.training.max_epochs,
            strategy=strategy,
            callbacks=[
                PredictionWritingCallback(
                    output_types=output_type, output_dir=output_dir, config=model.config
                )
            ],
        )
        trainer.predict(model, data)


@app.command()
def train(
    name: CONFIGS_ENUM,
    accelerator: str = typer.Option("auto", "--accelerator", "-a"),
    devices: str = typer.Option("auto", "--devices", "-d"),
    strategy: str = typer.Option(None),
    config_args: List[str] = typer.Option(None, "--config", "-c"),
    config_path: Path = typer.Option(None, exists=True, dir_okay=False, file_okay=True),
):
    from smts.utils import update_config_from_cli_args, update_config_from_path

    original_config = FastSpeech2Config.load_config_from_path(CONFIGS[name.value])
    config: FastSpeech2Config = update_config_from_cli_args(
        config_args, original_config
    )
    config = update_config_from_path(config_path, config)

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    from .dataset import FastSpeech2DataModule
    from .model import FastSpeech2

    tensorboard_logger = TensorBoardLogger(**(config.training.logger.dict()))
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger.info("Starting training for feature prediction model.")
    ckpt_callback = ModelCheckpoint(
        monitor="train/total_loss",
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
    model = FastSpeech2(config)
    data = FastSpeech2DataModule(config)
    last_ckpt = (
        config.training.finetune_checkpoint
        if config.training.finetune_checkpoint is not None
        and os.path.exists(config.training.finetune_checkpoint)
        else None
    )
    tensorboard_logger.log_hyperparams(config.dict())
    trainer.fit(model, data, ckpt_path=last_ckpt)


if __name__ == "__main__":
    app()
