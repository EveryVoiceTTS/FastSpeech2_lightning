import multiprocessing as mp
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from everyvoice.config.type_definitions import DatasetTextRepresentation

from . import core
from .config import FastSpeech2Config
from .model import FastSpeech2
from .synthesis_types import SynthesisBatch, Utterance

if TYPE_CHECKING:
    import torch


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
    model_path: Path,
    accelerator: str = "auto",
) -> FastSpeech2:
    """Returns the model"""
    from everyvoice.utils.heavy import get_device_from_accelerator

    return core.load_model(model_path, device=get_device_from_accelerator(accelerator))


def synthesize(
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
    """Synthesize the utterances in texts with model, one batch at a time.

    Long texts are split into chunks, which are synthesized separately, in order.
    Each batch holds chunks in the order of texts, and can mix chunks from
    different texts.

    Filter the result through unbatch() to get one Utterance per chunk, without
    padding. The tensors of each batch stay on the model's device.

    Args:
        model (FastSpeech2): the model to use, see load_model(); synthesis runs on
            the device the model is on
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
    """
    from .core.synthesize import (
        check_text_representation,
        iter_synthesis,
        prepare_data,
    )

    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return

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
    for outputs, batch in iter_synthesis(
        model,
        data,
        model.device,
        style_reference=style_reference is not None,
        batch_size=batch_size,
        num_workers=num_workers,
    ):
        yield SynthesisBatch(outputs=outputs, batch=batch, output_key=model.output_key)


def unbatch(
    batches: Iterable[SynthesisBatch],
    device: "torch.device | str | None" = "cpu",
) -> Iterator[Utterance]:
    """Split each batch from synthesize() into one Utterance per chunk, without padding.

    Args:
        batches: as yielded by synthesize()
        device: where to put the tensors of each Utterance; None leaves them on
            the model's device

    Yields: one Utterance per chunk of input text, in the order of the texts.
    """
    from .core.synthesize import unbatch_synthesis

    yield from unbatch_synthesis(batches, device)
