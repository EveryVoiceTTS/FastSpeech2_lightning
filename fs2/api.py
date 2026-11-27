"""
Programmatic API to the EveryVoice FastSpeech2_lightning module

This API is intended for use as a stand-alone script or a Jupyter notebook or a
service like HuggingFace.

Before using this API, you must have prepared your data and configuration files using
the EveryVoice new-project wizard:

    everyvoice new-project

From there, you can start using the API, which largely parallels the CLI

from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2 import api
# preprocess and train
config = api.load_config("config/everyvoice-text-to-spec.yaml")
api.preprocess(config)
api.train(config)

# load models
model_path = "logs_and_checkpoints/FeaturePredictionExperiment/base/checkpoints/last.ckpt"
model = api.load_model(model_path)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl import api as hfgl_api
vocoder_path = "logs_and_checkpoints/VocoderExperiment/base/checkpoints/last.ckpt"
vocoder = hfgl_api.load_model(vocoder_path)

# synthesize via the single-function wrapper
texts = ["utterance 1", "utturance 2"]
api.synthesize_to_files(model, texts, "output_dir", ["wav"], vocoder=vocoder, )

# synthesize step by step (lets you control what you want)
# copy the body of synthesize_to_files and adjust it to your needs
"""

import multiprocessing as mp
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from everyvoice.config.type_definitions import (
    DatasetTextRepresentation,
    SynthesizeOutputFormats,
)

from . import core
from .config import FastSpeech2Config
from .model import FastSpeech2
from .synthesis_types import SynthesisBatch, Utterance

if TYPE_CHECKING:
    import torch
    from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.model import (
        HiFiGAN,
        HiFiGANGenerator,
    )


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
        devices ("auto" | str | int): the number of GPUs to use on each node as a str or int; use "auto" to let PyTorch Lightning decide
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


def load_model(
    model_path: Path | str,
    accelerator: str = "auto",
) -> FastSpeech2:
    """Load a FastSpeech2 model from a checkpoint, ready for generating mel specs.

    Its configuration is available as model.config, and the training step the
    checkpoint was saved at as model.global_step_at_load.

    Args:
        model_path (Path|str): the FastSpeech2 checkpoint, e.g. the "last.ckpt" saved by train()
        accelerator (str): determines the device the model is loaded on, one of
            "auto", "cpu", "gpu" or "mps"; see train()
    """
    from everyvoice.utils.heavy import get_device_from_accelerator

    return core.load_model(
        Path(model_path), device=get_device_from_accelerator(accelerator)
    )


def text_to_spec(
    model: FastSpeech2,
    texts: Sequence[str] | str,
    *,
    language: str | None = None,
    speaker: str | None = None,
    duration_control: float = 1.0,
    style_reference: Path | None = None,
    text_representation: str | DatasetTextRepresentation = "characters",
    split_text: bool | None = None,
    desired_length: int | None = None,
    max_length: int | None = None,
    batch_size: int = 4,
    num_workers: int = 0,
) -> Iterator[SynthesisBatch]:
    """The text-to-spec stage: synthesize the spectrograms of texts with model, one
    batch at a time.

    Long texts are split into chunks, which are synthesized separately, in order.
    Each batch holds chunks in the order of texts, and can mix chunks from
    different texts.

    Filter the result through unbatch() to get one Utterance per chunk, without
    padding. The tensors of each batch stay on the model's device.

    Args:
        model (FastSpeech2): the model to use, see load_model(); synthesis runs on
            the device the model is on, and puts the model in eval mode
        texts (Sequence[str] | str): the utterances to synthesize, or a single one
        language (str|None): language to use in a multilingual model
        speaker (str|None): speaker to use in a multispeaker model
        duration_control (float): multiplies durations, so lower numbers produce
            quicker speaking rates
        style_reference (Path|None): audio file containing a style reference; the
            model must have been trained with the global style token module
        text_representation: the representation of the utterances in texts
        split_text (bool|None): whether to split long texts into chunks; None means
            use the model's configuration (text.split_text)
        desired_length, max_length (int|None): the desired and maximum chunk
            lengths; None means use the lengths seen in the training data
        batch_size (int): how many chunks to process at once; there must be enough
            accelerator RAM to process a batch in parallel
        num_workers (int): number of worker processes to prepare the data; 0 means
            prepare it in the main process

    Yields: a SynthesisBatch for each batch of up to batch_size chunks.

    Raises:
        ValueError: if a text is empty or only whitespace, since there is nothing
            to synthesize; checked before any synthesis starts.
    """
    from .core.synthesize import (
        check_text_representation,
        prepare_data,
        text_to_spec_synthesis,
    )

    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return
    blank = [index for index, text in enumerate(texts) if not text.strip()]
    if blank:
        raise ValueError(
            f"The texts at indices {blank} are empty or blank: "
            "there is nothing to synthesize."
        )

    text_representation = DatasetTextRepresentation(text_representation)
    check_text_representation(model, text_representation)
    data = prepare_data(
        texts=list(texts),
        language=language,
        speaker=speaker,
        filelist=None,
        model=model,
        text_representation=text_representation,
        duration_control=duration_control,
        style_reference=style_reference,
        split_text=split_text,
        desired_length=desired_length,
        max_length=max_length,
    )
    yield from text_to_spec_synthesis(
        model,
        data,
        model.device,
        style_reference=style_reference is not None,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def spec_to_wav(
    batches: Iterable[SynthesisBatch],
    vocoder: "HiFiGAN | HiFiGANGenerator",
) -> Iterator[SynthesisBatch]:
    """Generate the waveforms of each batch from text_to_spec() with vocoder.

    Each batch's chunks are vocoded together: long texts are split into chunks
    so that each fits in memory. Filter the result through unbatch() to get each
    chunk's waveform as Utterance.wav, and through combine_chunks() to get each
    text's, its chunks' waveforms joined end to end.

    Args:
        batches: as yielded by text_to_spec()
        vocoder: the spec-to-wav model, see everyvoice.api.hfgl.load_model()

    Yields: each batch, with its wavs, wav_lens and sample_rate set; the wavs are
        on the CPU.
    """
    from .core.synthesize import spec_to_wav_synthesis

    yield from spec_to_wav_synthesis(batches, vocoder)


def unbatch(
    batches: Iterable[SynthesisBatch],
    device: "torch.device | str | None" = "cpu",
) -> Iterator[Utterance]:
    """Split each batch from text_to_spec() into one Utterance per chunk, without padding.

    Args:
        batches: as yielded by text_to_spec()
        device: where to put the tensors of each Utterance; None leaves them on
            the model's device

    Yields: one Utterance per chunk of input text, in the order of the texts,
        with its wav if spec_to_wav() was applied to batches.
    """
    from .core.synthesize import unbatch_synthesis

    yield from unbatch_synthesis(batches, device)


def combine_chunks(utterances: Iterable[Utterance]) -> Iterator[Utterance]:
    """Combine the chunks of each input text from unbatch() into one Utterance.

    Each combined Utterance is yielded as soon as the last chunk of its text
    comes in. Its spec and wav (if the chunks have one) are its chunks'
    concatenated; its text is its chunks' texts joined with single spaces, and
    its tokens and durations are joined to match, with a space token lasting 0
    frames between chunks. Texts that were not split pass through unchanged.

    Args:
        utterances: as yielded by unbatch()

    Yields: one Utterance per input text, in the order of the texts.

    Raises:
        ValueError: if a text's chunks are missing, out of order, or interleaved
            with another text's.
    """
    from .core.synthesize import combine_utterance_chunks

    yield from combine_utterance_chunks(utterances)


def write_to_files(
    utterances: Iterable[Utterance],
    model: FastSpeech2,
    output_dir: Path | str,
    output_types: Sequence[SynthesizeOutputFormats | str],
    *,
    vocoder: "HiFiGAN | HiFiGANGenerator | None" = None,
    simple_filenames: bool = False,
) -> Iterator[Utterance]:
    """Write the files of each synthesized text from combine_chunks(), as the
    `everyvoice synthesize` command does, and yield it unchanged.

    The files are written as the utterances go through: iterate over the result,
    to actually write the files.

    Args:
        utterances: as yielded by combine_chunks()
        model: the model that synthesized utterances, for its configuration and
            training step, which are part of the file names
        output_dir: the directory to write to, in subdirectories by output type
        output_types: one or more of "spec", "wav", "textgrid", "readalong-xml"
            and "readalong-html"; "wav" and "readalong-html" require the
            utterances to have waveforms, see spec_to_wav()
        vocoder: the vocoder that generated the waveforms, for its training
            step, which is part of the wav file names
        simple_filenames: if set, name files only by their basenames

    Yields: each utterance, after writing its files.

    Raises:
        ValueError: if an utterance is only a chunk of a text: apply
            combine_chunks() first.
    """
    from .core.synthesize import write_utterances
    from .prediction_writers import UtteranceWriter, get_synthesis_output_writers

    formats = [SynthesizeOutputFormats(t) for t in output_types]
    if vocoder is None and (
        SynthesizeOutputFormats.wav in formats
        or SynthesizeOutputFormats.readalong_html in formats
    ):
        raise ValueError("Writing wav or readalong-html files requires the vocoder.")
    writers = get_synthesis_output_writers(
        output_types=formats,
        output_dir=Path(output_dir),
        config=model.config,
        global_step=model.global_step_at_load or 0,
        vocoder_global_step=(
            None if vocoder is None else vocoder.global_step_at_load or 0
        ),
        simple_filenames=simple_filenames,
    )
    utterance_writers = [w for w in writers.values() if isinstance(w, UtteranceWriter)]
    yield from write_utterances(utterances, utterance_writers)
    # Only reached once utterances is exhausted, not if the iteration stops early
    for writer in utterance_writers:
        writer.finalize()


def synthesize_to_files(
    model: FastSpeech2,
    texts: Sequence[str] | str,
    output_dir: Path | str,
    output_types: Sequence[SynthesizeOutputFormats | str],
    *,
    vocoder: "HiFiGAN | HiFiGANGenerator | None" = None,
    simple_filenames: bool = False,
    **text_to_spec_args,
) -> None:
    """Synthesize texts and write their files: the whole pipeline, from
    text_to_spec() to write_to_files(), in one call.

    If the need arises, this function could return the names of the files written.

    This function is both a convenience wrapper for the whole synthesis pipeline
    and a sample you can copy and modify for your needs, since it uses only
    functions from this API.

    Args:
        model, texts, and **text_to_spec_args: see text_to_spec()
        output_dir, output_types, simple_filenames: see write_to_files()
        vocoder: the spec-to-wav model, required for "wav" and "readalong-html"
            output
    """
    # Prepare the synthesis generators -- nothing actually happens yet here
    prediction_batches = text_to_spec(model, texts, **text_to_spec_args)
    if vocoder is not None:
        prediction_batches = spec_to_wav(prediction_batches, vocoder)
    utterances = unbatch(prediction_batches)
    utterances = combine_chunks(utterances)
    written_utterances = write_to_files(
        utterances,
        model,
        output_dir,
        output_types,
        vocoder=vocoder,
        simple_filenames=simple_filenames,
    )
    # The following loop actually causes the stack of generators above to do
    # their work:
    for _ in written_utterances:
        pass
