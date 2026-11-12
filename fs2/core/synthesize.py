import math
import sys
import textwrap
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import typer
from everyvoice import logger
from everyvoice.base_cli.checkpoint import check_file_not_empty
from everyvoice.config.type_definitions import (
    DatasetTextRepresentation,
    SynthesizeOutputFormats,
    TargetTrainingTextRepresentationLevel,
)
from everyvoice.text.textsplit import chunk_text, resolve_split_params
from everyvoice.utils import slugify, truncate_basename

if TYPE_CHECKING:
    import torch.device
    from torch.utils.data import DataLoader

    from ..model import FastSpeech2, FastSpeech2Config
    from ..prediction_writing_callback import OutputWriter


def validate_data_keys_with_model_keys(
    data_keys: set[str], model_keys: set[str], key: str, multi: bool
) -> None:
    """
    Make sure the user-supplied language or speaker specification is compatible with the model.
    In current version of model, we can validate either key="language" or key="speaker"
    Args:
        data_keys: values for key found in the data
        model_keys: values for key found in the model
        key: "language" or "speaker"
        multi: whether the model was trained in multilingual/speaker mode
    """
    if multi:
        if None in data_keys:
            raise typer.BadParameter(
                f"You have not specified a {key} for all your sentences."
                f" Available values are {model_keys}"
            )

        extras = data_keys.difference(model_keys)
        if extras:
            is_or_are_not = (
                f"are not {key}s that are" if len(data_keys) > 1 else f"is not a {key}"
            )
            raise typer.BadParameter(
                f"You provided {data_keys} which {is_or_are_not} supported by the model {model_keys or {}}."
            )
    else:
        # NOTE: Even in non multiX, the model has a default value.
        # Looking at a filelist.psv
        # basename|characters|language|speaker
        # LJ002-0234|In the yard behind the prison|und|default
        # TODO: Instead, should we check that `data_keys == model_keys`?
        extras = data_keys.difference(model_keys | {None})
        if extras:
            raise typer.BadParameter(
                f"The current model doesn't support multiple {key}s but your data has {key}s {extras}.\n"
                f"Please retrain your model with multi{'lingual' if key == 'language' else key} set to True."
            )


def get_text_split_params(
    model: "FastSpeech2",
    language: str | None,
    text_representation: DatasetTextRepresentation,
) -> tuple[bool, tuple[int, int, str, str]]:
    """
    Calculates the parameters for text splitting (desired_length, max_length, strong_boundaries, weak_boundaries)
    """
    from everyvoice.config.text_config import TextConfig

    from ..type_definitions_heavy import Stats, StatsInfo

    text_config: TextConfig = model.config.text

    desired_length: float = 100
    max_length: float = 200

    if text_config.split_text:
        # Get desired and max length from Stats
        stats: Stats | None = getattr(model, "stats", None)
        length_stats: StatsInfo | None = None
        if stats is not None:
            match text_representation:
                case DatasetTextRepresentation.characters:
                    length_stats = stats.character_length
                case DatasetTextRepresentation.ipa_phones:
                    length_stats = stats.phone_length
                case DatasetTextRepresentation.arpabet:
                    logger.warning(
                        "Arpabet stats are not yet supported. Chunking with default lengths."
                    )
        if length_stats is not None:
            desired_length = length_stats.mean
            max_length = length_stats.max
        elif stats is None or text_representation != DatasetTextRepresentation.arpabet:
            # This is for unit testing or older models
            logger.warning(
                f"Length stats for {text_representation} could not be found. Chunking with default lengths."
            )

    return resolve_split_params(
        text_config,
        language,
        desired_length=int(desired_length),
        max_length=int(max_length),
    )


def load_data_from_filelist(
    filelist: Path,
    model: "FastSpeech2",
    text_representation: DatasetTextRepresentation,
    language: str | None = None,
    speaker: str | None = None,
    default_language: str | None = None,
    default_speaker: str | None = None,
):
    if default_language is None:
        default_language = next(iter(model.lang2id.keys()), None)
    if default_speaker is None:
        default_speaker = next(iter(model.speaker2id.keys()), None)

    split_text, split_params = get_text_split_params(
        model, language or default_language, text_representation
    )

    try:
        data = []
        for d in model.config.training.filelist_loader(filelist):
            # Chunk longer texts, for better longform audio synthesis
            line = d[text_representation.value]
            chunks = chunk_text(line, *split_params) if split_text else [line]
            for i, chunk in enumerate(chunks):
                data.append(
                    {
                        "basename": d.get(
                            "basename",
                            truncate_basename(slugify(chunk)),
                        ),  # Only truncate the basename if the basename doesn't already exist in the filelist.
                        text_representation.value: chunk,
                        "language": language or d.get("language", default_language),
                        "speaker": speaker or d.get("speaker", default_speaker),
                        "is_last_input_chunk": (i == len(chunks) - 1),
                    }
                )

            print(f"Processing text: {chunks}", file=sys.stderr)

        if not data:
            # If there is no data, it means we had a one-line input file. Raise KeyError
            # so we enter the except block below and read it as a plain text file.
            raise KeyError
    except KeyError:
        # TODO: Errors should have better formatting:
        #       https://github.com/EveryVoiceTTS/FastSpeech2_lightning/issues/26
        logger.info(textwrap.dedent("""
            EveryVoice only accepts filelists in PSV format as in:

                basename|characters|language|speaker
                LJ0001|Hello|eng|LJ

            Or in a format where each new line is an utterance:

                This is a sentence.
                Here is another sentence.

            Your filelist did not contain the correct keys so we will assume it is in the plain text format.
            Text can either be defined as 'characters' or 'phones'.
            """))
        data = []
        with open(filelist, encoding="utf8") as file:
            for line in file:
                # Chunk longer texts, for better longform audio synthesis
                chunks = chunk_text(line, *split_params) if split_text else [line]
                for i, chunk in enumerate(chunks):
                    data.append(
                        {
                            "basename": truncate_basename(slugify(chunk.strip())),
                            text_representation.value: chunk.strip(),
                            "language": language or default_language,
                            "speaker": speaker or default_speaker,
                            "is_last_input_chunk": (i == len(chunks) - 1),
                        }
                    )
                print(f"Processing text: {chunks}", file=sys.stderr)
    return data


def prepare_data(
    texts: Optional[list[str]],
    language: str | None,
    speaker: str | None,
    filelist: Path | None,
    model: "FastSpeech2",
    text_representation: DatasetTextRepresentation,
    duration_control: float,
    style_reference: Path | None,
) -> list[dict[str, Any]]:
    """"""
    data: list[dict[str, Any]]
    # NOTE: The wizard adds a default speaker=`default` to the data.
    # It also asks for a language that the user choses from a list which then becomes the default lanaguage, like `und`.
    # Knowing this, model.*2id should always have a default value thus DEFAULT_* should never be `None`.
    DEFAULT_LANGUAGE = next(iter(model.lang2id.keys()), None)
    DEFAULT_SPEAKER = next(iter(model.speaker2id.keys()), None)

    split_text, split_params = get_text_split_params(
        model, language or DEFAULT_LANGUAGE, text_representation
    )

    if (texts and filelist) or (not texts and not filelist):
        raise ValueError("Provide exactly one of filelist or texts")
    if texts:
        data = []
        for text in texts:
            # Chunk longer texts, for better longform audio synthesis
            chunks = chunk_text(text, *split_params) if split_text else [text]
            for i, chunk in enumerate(chunks):
                data.append(
                    {
                        "basename": truncate_basename(slugify(chunk)),
                        text_representation.value: chunk,
                        "language": language or DEFAULT_LANGUAGE,
                        "speaker": speaker or DEFAULT_SPEAKER,
                        "is_last_input_chunk": (
                            i == len(chunks) - 1
                        ),  # True if end of a text_input, False otherwise
                    }
                )
            print(f"Processing text: {chunks}", file=sys.stderr)
    else:
        assert isinstance(filelist, Path)
        data = load_data_from_filelist(
            filelist,
            model,
            text_representation,
            language,
            speaker,
            DEFAULT_LANGUAGE,
            DEFAULT_SPEAKER,
        )

    validate_data_keys_with_model_keys(
        data_keys=set(d["language"] for d in data),
        model_keys=set(model.lang2id.keys()),
        key="language",
        multi=model.config.model.multilingual,
    )
    validate_data_keys_with_model_keys(
        data_keys=set(d["speaker"] for d in data),
        model_keys=set(model.speaker2id.keys()),
        key="speaker",
        multi=model.config.model.multispeaker,
    )

    # We only allow a single style reference right now, so it's fine to load it once here.
    if style_reference:
        from everyvoice.utils.heavy import get_spectral_transform

        spectral_transform = get_spectral_transform(
            model.config.preprocessing.audio.spec_type,
            model.config.preprocessing.audio.n_fft,
            model.config.preprocessing.audio.fft_window_size,
            model.config.preprocessing.audio.fft_hop_size,
            f_min=model.config.preprocessing.audio.f_min,
            f_max=model.config.preprocessing.audio.f_max,
            sample_rate=model.config.preprocessing.audio.output_sampling_rate,
            n_mels=model.config.preprocessing.audio.n_mels,
        )
        import torchaudio

        style_reference_audio, style_reference_sr = torchaudio.load(
            str(style_reference)
        )
        if style_reference_sr != model.config.preprocessing.audio.input_sampling_rate:
            style_reference_audio = torchaudio.functional.resample(
                style_reference_audio,
                style_reference_sr,
                model.config.preprocessing.audio.input_sampling_rate,
            )
        style_reference_spec = spectral_transform(style_reference_audio)
    # Add duration_control
    for item in data:
        item["duration_control"] = duration_control
        # Add style reference
        if style_reference:
            item["mel_style_reference"] = style_reference_spec

    return data


def get_global_step(model_path: Path) -> int:
    """
    Extract the `global_step` from a model.
    Note: we have to do this because the `global_step` gets reset to 0 when we call load_from_checkpoint().
    """
    import torch

    m = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    return m["global_step"]


def load_model(model_path: Path, device: "torch.device") -> "FastSpeech2":
    check_file_not_empty(model_path)
    from ..model import FastSpeech2

    model = FastSpeech2.load_from_checkpoint(model_path).to(device)  # type: ignore
    return model


def check_text_representation(
    model: "FastSpeech2", text_representation: DatasetTextRepresentation
) -> None:
    """Raise ValueError if text_representation is incompatible with the model."""
    if (
        model.config.model.target_text_representation_level
        == TargetTrainingTextRepresentationLevel.characters
        and text_representation != DatasetTextRepresentation.characters
    ):
        raise ValueError(
            f"Your model was trained on {model.config.model.target_text_representation_level} but you provided {text_representation.value} which is incompatible."
        )


def warn_if_multiple_devices(devices: str | int) -> None:
    """Synthesis runs on a single device: warn if more than one was requested.

    devices follows the PyTorch Lightning conventions: "auto", a number of
    devices ("-1" meaning all of them), or a comma-separated list of device ids.
    """
    devices_str = str(devices).strip()
    if "," in devices_str:
        requested = len([d for d in devices_str.split(",") if d.strip()])
    else:
        try:
            requested = int(devices_str)
        except ValueError:  # "auto" or anything else Lightning might accept
            return
    if requested == -1 or requested > 1:
        logger.warning(
            f"Synthesis only uses a single device, ignoring devices={devices!r}."
        )


def make_synthesis_dataloader(
    model: "FastSpeech2",
    data: list[dict[str, Any]],
    *,
    teacher_forcing: bool = False,
    style_reference: bool = False,
    batch_size: int = 4,
    num_workers: int = 0,
) -> "DataLoader":
    """Create the DataLoader that tokenizes and collates data for synthesis.

    Batches are produced in the order of data, without shuffling.
    With teacher_forcing, model.config.preprocessing.save_dir must point to
    the preprocessed data to force the output to match.
    """
    from functools import partial

    from torch.utils.data import DataLoader

    from ..dataset import FastSpeech2DataModule, FastSpeechDataset

    dataset = FastSpeechDataset(
        data,
        model.config,
        model.lang2id,
        model.speaker2id,
        teacher_forcing=teacher_forcing,
        inference=True,
        style_reference=style_reference,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        collate_fn=partial(
            FastSpeech2DataModule.collate_method,
            learn_alignment=model.config.model.learn_alignment,
        ),
    )


def iter_synthesis(
    model: "FastSpeech2",
    data: list[dict[str, Any]],
    device: "torch.device",
    *,
    teacher_forcing: bool = False,
    style_reference: bool = False,
    batch_size: int = 4,
    num_workers: int = 0,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Run the model's forward pass in inference mode over data, one batch at a time.

    Args:
        model: the text-to-spec model; it gets moved to device and set to eval mode
        data: the utterances to synthesize, as produced by prepare_data()
        device: the device to run on
        teacher_forcing, style_reference, batch_size, num_workers:
            see make_synthesis_dataloader()

    Yields: (outputs, batch) for each batch, with the tensors left on device.
        Batches come in the order of data, which the output writers rely on
        to reassemble chunked texts.
    """
    import torch
    from lightning_fabric.utilities.apply_func import move_data_to_device

    model.to(device)
    model.eval()
    dataloader = make_synthesis_dataloader(
        model,
        data,
        teacher_forcing=teacher_forcing,
        style_reference=style_reference,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    for batch in dataloader:
        batch = move_data_to_device(batch, device)
        with torch.no_grad():
            outputs = model(batch, inference=True)
        yield outputs, batch


def synthesize_helper(
    model: "FastSpeech2",
    texts: Optional[list[str]],
    style_reference: Path | None,
    language: Optional[str],
    speaker: Optional[str],
    duration_control: Optional[float],
    global_step: int,
    output_type: Sequence[SynthesizeOutputFormats],
    text_representation: DatasetTextRepresentation,
    accelerator: str,  # unused: device is what matters, kept for compatibility
    devices: str,  # unused: synthesis runs on a single device
    device: "torch.device",  # set to: get_device_from_accelerator(accelerator)
    batch_size: int,
    num_workers: int,
    filelist: Path | None,
    filelist_data: Optional[list[dict]],
    output_dir: Path,
    teacher_forcing_directory: Path | None,
    vocoder_global_step: Optional[int] = None,
    vocoder_model=None,
    vocoder_config=None,
    return_scores=False,
    simple_filenames: bool = False,
) -> tuple[
    "FastSpeech2Config",
    dict[SynthesizeOutputFormats | str, "OutputWriter"],
]:
    """This is a helper to perform synthesis once the model has been loaded.
    It allows us to use the same command for synthesis via the CLI and
    via the gradio demo.

    Synthesis runs on the single device `device`. TODO: remove accelerator
    and devices, or use them to support multiple devices.

    Returns: a tuple of:
        config: the config that was used -- model.config, modified in place
        writers: the output writers created to handle the output_types specified
    """
    import torch
    from tqdm import tqdm

    warn_if_multiple_devices(devices)
    check_text_representation(model, text_representation)

    sources = sum(1 if source else 0 for source in (filelist_data, filelist, texts))
    if sources != 1:
        raise ValueError("Provide exactly one of texts, filelist or filelist_data")
    if filelist_data is None:
        data: list[dict[Any, Any]] = prepare_data(
            texts=texts,
            language=language,
            speaker=speaker,
            duration_control=duration_control if duration_control else 1.0,
            filelist=filelist,
            model=model,
            text_representation=text_representation,
            style_reference=style_reference,
        )
    else:
        data = filelist_data
    if return_scores:
        from nltk.util import ngrams

        token_counter: Counter = Counter()
        trigram_counter: Counter = Counter()
        for line in tqdm(
            data, desc="calculating filelist statistics for score calculation"
        ):
            tokens = line[f"{text_representation.value[:-1]}_tokens"].split("/")
            for t in tokens:
                token_counter[t] += 1
            tokens.insert(0, "<BOS>")
            tokens.append("<EOS>")
            for trigram in ngrams(tokens, 3):
                trigram_counter[trigram] += 1
        for line in tqdm(data, desc="scoring utterances"):
            tokens = line[f"{text_representation.value[:-1]}_tokens"].split("/")
            line["phone_coverage_score"] = sum((1 / token_counter[t]) for t in tokens)
            line["trigram_coverage_score"] = sum(
                (1 / trigram_counter[n]) for n in ngrams(tokens, 3)
            )

    if teacher_forcing_directory is not None:
        teacher_forcing = True
        model.config.preprocessing.save_dir = teacher_forcing_directory
    else:
        if return_scores:
            raise ValueError(
                "In order to return the scores, we also need access to the directory containing your ground truth audio and preprocessed data. Please pass this in using the --teacher-forcing-directory option. e.g. --teacher-forcing-directory ./preprocessed"
            )
        teacher_forcing = False
    # overwrite batch_size and num_workers
    model.config.training.batch_size = batch_size
    model.config.training.train_data_workers = num_workers

    from ..prediction_writing_callback import get_synthesis_output_callbacks

    writers = get_synthesis_output_callbacks(
        output_type=output_type,
        output_dir=output_dir,
        config=model.config,
        output_key=model.output_key,
        device=device,
        global_step=global_step,
        vocoder_model=vocoder_model,
        vocoder_config=vocoder_config,
        vocoder_global_step=vocoder_global_step,
        return_scores=return_scores,
        simple_filenames=simple_filenames,
        loss_fn=model.loss,
    )

    batches = iter_synthesis(
        model,
        data,
        device,
        teacher_forcing=teacher_forcing,
        style_reference=style_reference is not None,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    # Like trainer.predict() did, run the writers (including the vocoder) in
    # inference mode too.
    with torch.inference_mode():
        for outputs, batch in tqdm(
            batches, desc="Synthesizing", total=math.ceil(len(data) / batch_size)
        ):
            for writer in writers.values():
                writer.write_batch(outputs, batch)
        for writer in writers.values():
            writer.finalize()

    return model.config, writers
