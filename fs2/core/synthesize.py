import math
import sys
import textwrap
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from everyvoice import logger
from everyvoice.config.type_definitions import (
    DatasetTextRepresentation,
    SynthesizeOutputFormats,
    TargetTrainingTextRepresentationLevel,
)
from everyvoice.text.textsplit import chunk_text, resolve_split_params
from everyvoice.utils import check_file_not_empty, slugify, truncate_basename

if TYPE_CHECKING:
    import torch.device
    from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import (
        HiFiGANConfig,
    )
    from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.model import (
        HiFiGAN,
        HiFiGANGenerator,
    )
    from torch.utils.data import DataLoader

    from ..model import FastSpeech2, FastSpeech2Config
    from ..prediction_writers import OutputWriter, UtteranceWriter
    from ..synthesis_types import SynthesisBatch, Utterance


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
            raise ValueError(
                f"You have not specified a {key} for all your sentences."
                f" Available values are {model_keys}"
            )

        extras = data_keys.difference(model_keys)
        if extras:
            is_or_are_not = (
                f"are not {key}s that are" if len(data_keys) > 1 else f"is not a {key}"
            )
            raise ValueError(
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
            raise ValueError(
                f"The current model doesn't support multiple {key}s but your data has {key}s {extras}.\n"
                f"Please retrain your model with multi{'lingual' if key == 'language' else key} set to True."
            )


def get_text_split_params(
    model: "FastSpeech2",
    language: str | None,
    text_representation: DatasetTextRepresentation,
    split_text: bool | None = None,
    desired_length: int | None = None,
    max_length: int | None = None,
) -> tuple[bool, tuple[int, int, str, str]]:
    """
    Calculates the parameters for text splitting (desired_length, max_length, strong_boundaries, weak_boundaries)

    Args:
        split_text: whether to split long texts into chunks; None means use the
            model's text config (text.split_text)
        desired_length, max_length: override the chunk lengths, which otherwise
            come from the model's length statistics, or default to 100 and 200
    """
    from everyvoice.config.text_config import TextConfig

    from ..type_definitions_heavy import Stats, StatsInfo

    text_config: TextConfig = model.config.text
    if split_text is not None and split_text != text_config.split_text:
        # resolve_split_params() only looks up boundaries when text.split_text is set
        text_config = text_config.model_copy(update={"split_text": split_text})

    stats_desired_length: float = 100
    stats_max_length: float = 200

    if text_config.split_text and (desired_length is None or max_length is None):
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
            stats_desired_length = length_stats.mean
            stats_max_length = length_stats.max
        elif stats is None or text_representation != DatasetTextRepresentation.arpabet:
            # This is for unit testing or older models
            logger.warning(
                f"Length stats for {text_representation} could not be found. Chunking with default lengths."
            )

    return resolve_split_params(
        text_config,
        language,
        desired_length=int(
            stats_desired_length if desired_length is None else desired_length
        ),
        max_length=int(stats_max_length if max_length is None else max_length),
    )


def load_data_from_filelist(
    filelist: Path,
    model: "FastSpeech2",
    text_representation: DatasetTextRepresentation,
    language: str | None = None,
    speaker: str | None = None,
    default_language: str | None = None,
    default_speaker: str | None = None,
    split_text: bool | None = None,
    desired_length: int | None = None,
    max_length: int | None = None,
):
    """Load and chunk the utterances in filelist.

    Each chunk gets "index" (the row or line number in filelist, counting from 0)
    and "chunk_index" (its position within that text).
    See get_text_split_params() for split_text, desired_length and max_length.
    """
    if default_language is None:
        default_language = next(iter(model.lang2id.keys()), None)
    if default_speaker is None:
        default_speaker = next(iter(model.speaker2id.keys()), None)

    split_text, split_params = get_text_split_params(
        model,
        language or default_language,
        text_representation,
        split_text=split_text,
        desired_length=desired_length,
        max_length=max_length,
    )

    try:
        data = []
        for index, d in enumerate(model.config.training.filelist_loader(filelist)):
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
                        "index": index,
                        "chunk_index": i,
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
            for index, line in enumerate(file):
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
                            "index": index,
                            "chunk_index": i,
                        }
                    )
                print(f"Processing text: {chunks}", file=sys.stderr)
    return data


def combine_filelist_data(*filelists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Concatenate the data of several filelists, keeping each text's "index" unique.

    Each filelist from load_data_from_filelist() numbers its texts from 0: here,
    each one's indices are offset by the number of texts in the ones before it.
    The items are copied, not modified.
    """
    combined: list[dict[str, Any]] = []
    offset = 0
    for data in filelists:
        n_texts = 0
        for item in data:
            combined.append({**item, "index": item["index"] + offset})
            n_texts = max(n_texts, item["index"] + 1)
        offset += n_texts
    return combined


def prepare_data(
    texts: Optional[list[str]],
    language: str | None,
    speaker: str | None,
    filelist: Path | None,
    model: "FastSpeech2",
    text_representation: DatasetTextRepresentation,
    duration_control: float,
    style_reference: Path | None,
    split_text: bool | None = None,
    desired_length: int | None = None,
    max_length: int | None = None,
) -> list[dict[str, Any]]:
    """Build the list of items to synthesize, one per chunk of input text.

    Each item has "index", the position of its text in texts (or its row or line
    in filelist), and "chunk_index", its position among that text's chunks.
    See get_text_split_params() for split_text, desired_length and max_length.
    """
    data: list[dict[str, Any]]
    # NOTE: The wizard adds a default speaker=`default` to the data.
    # It also asks for a language that the user choses from a list which then becomes the default lanaguage, like `und`.
    # Knowing this, model.*2id should always have a default value thus DEFAULT_* should never be `None`.
    DEFAULT_LANGUAGE = next(iter(model.lang2id.keys()), None)
    DEFAULT_SPEAKER = next(iter(model.speaker2id.keys()), None)

    split_kwargs: dict[str, Any] = dict(
        split_text=split_text, desired_length=desired_length, max_length=max_length
    )
    split_text, split_params = get_text_split_params(
        model, language or DEFAULT_LANGUAGE, text_representation, **split_kwargs
    )

    if (texts and filelist) or (not texts and not filelist):
        raise ValueError("Provide exactly one of filelist or texts")
    if texts:
        data = []
        for index, text in enumerate(texts):
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
                        "index": index,
                        "chunk_index": i,
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
            **split_kwargs,
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


def text_to_spec_synthesis(
    model: "FastSpeech2",
    data: list[dict[str, Any]],
    device: "torch.device",
    *,
    teacher_forcing: bool = False,
    style_reference: bool = False,
    batch_size: int = 4,
    num_workers: int = 0,
) -> Iterator["SynthesisBatch"]:
    """The text-to-spec stage of the synthesis pipeline: run the model's forward
    pass in inference mode over data, one batch at a time.

    The next stages are spec_to_wav_synthesis() (optional), unbatch_synthesis()
    and combine_utterance_chunks().

    Args:
        model: the text-to-spec model; it gets moved to device and set to eval mode
        data: the utterances to synthesize, as produced by prepare_data()
        device: the device to run on
        teacher_forcing, style_reference, batch_size, num_workers:
            see make_synthesis_dataloader()

    Yields: a SynthesisBatch of spectrograms for each batch, with the tensors left
        on device. Batches come in the order of data, which
        combine_utterance_chunks() relies on to reassemble chunked texts.
    """
    import torch
    from lightning_fabric.utilities.apply_func import move_data_to_device

    from ..synthesis_types import SynthesisBatch

    model.to(device)
    model.eval()
    (space_token,) = model.text_processor.encode_string_tokens([" "])
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
        yield SynthesisBatch(
            outputs=outputs,
            batch=batch,
            output_key=model.output_key,
            space_token=space_token,
        )


def get_output_hop_size(vocoder_config: "HiFiGANConfig") -> int:
    """The number of output audio samples the vocoder generates per spec frame."""
    audio_config = vocoder_config.preprocessing.audio
    sampling_rate_change = (
        audio_config.output_sampling_rate // audio_config.input_sampling_rate
    )
    return sampling_rate_change * audio_config.fft_hop_size


def spec_to_wav_batch(
    specs: "torch.Tensor",
    frame_lens: "torch.Tensor | Sequence[int]",
    vocoder: "HiFiGAN | HiFiGANGenerator",
    vocoder_config: "HiFiGANConfig",
) -> tuple["torch.Tensor", "torch.Tensor", int]:
    """Generate the waveforms of a padded batch of spectrograms.

    Args:
        specs: the spectrograms, [B (batch size), T (frames), K (mel bands)], as
            output by the text-to-spec model
        frame_lens: the number of frames of each spectrogram, [B], as a tensor or
            a sequence of ints
        vocoder: the spec-to-wav model
        vocoder_config: its configuration, normally vocoder.config

    Returns: (wavs, wav_lens, sample_rate), where wavs is [B, C (channels),
        T (samples)], still padded, on the CPU, and wav_lens is the number of
        samples of each item, [B].
    """
    import torch
    from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import (
        synthesize_data,
    )

    wavs, sample_rate = synthesize_data(specs.transpose(1, 2), vocoder, vocoder_config)
    assert (
        wavs.ndim == 3
    ), f"The generated audio did not contain 3 dimensions. First dimension should be B(atch) and the second dimension should be C(hannels) and third dimension should be T(ime) in samples. Got {wavs.shape} instead."
    assert wavs.shape[0] == specs.size(
        0
    ), f"You provided {specs.size(0)} utterances, but {wavs.shape[0]} audio files were synthesized instead."

    wav_lens = torch.as_tensor(frame_lens).cpu() * get_output_hop_size(vocoder_config)
    return wavs, wav_lens, sample_rate


def spec_to_wav_synthesis(
    batches: Iterable["SynthesisBatch"],
    vocoder: "HiFiGAN | HiFiGANGenerator",
) -> Iterator["SynthesisBatch"]:
    """Add the waveforms generated by vocoder to each SynthesisBatch.

    Each batch's padded spectrograms are vocoded together, chunk by chunk: long
    texts are split into chunks precisely so that each fits in memory.

    Args:
        batches: as yielded by api.text_to_spec()
        vocoder: the spec-to-wav model, e.g., from hfgl.api.load_model()

    Yields: a copy of each batch with wavs, wav_lens and sample_rate set; the
        wavs are on the CPU.
    """
    import dataclasses

    for synthesis_batch in batches:
        wavs, wav_lens, sample_rate = spec_to_wav_batch(
            synthesis_batch.outputs[synthesis_batch.output_key],
            synthesis_batch.outputs["tgt_lens"],
            vocoder,
            vocoder.config,
        )
        yield dataclasses.replace(
            synthesis_batch, wavs=wavs, wav_lens=wav_lens, sample_rate=sample_rate
        )


def unbatch_synthesis(
    batches: Iterable["SynthesisBatch"],
    device: "torch.device | str | None" = "cpu",
) -> Iterator["Utterance"]:
    """Split each SynthesisBatch into one Utterance per item, removing the padding.

    Args:
        batches: as yielded by api.text_to_spec()
        device: where to put the tensors of each Utterance; None leaves them where
            they are. They are copied either way, so an Utterance doesn't keep its
            whole padded batch in memory.
    """
    import torch

    from ..synthesis_types import Utterance

    def copy_to_device(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(
            device=device, copy=True, memory_format=torch.contiguous_format
        )

    for synthesis_batch in batches:
        outputs, batch = synthesis_batch.outputs, synthesis_batch.batch
        specs = outputs[synthesis_batch.output_key]
        wavs, wav_lens = synthesis_batch.wavs, synthesis_batch.wav_lens
        for i in range(len(synthesis_batch)):
            n_frames = int(outputs["tgt_lens"][i])  # for unpadding specs
            n_tokens = int(batch["src_lens"][i])  # for unpadding text and durations
            # for unpadding wavs, if spec_to_wav_synthesis() was applied
            n_samples = None if wav_lens is None else int(wav_lens[i])
            yield Utterance(
                index=int(batch["index"][i]),
                chunk_index=int(batch["chunk_index"][i]),
                is_last_chunk=bool(batch["is_last_input_chunk"][i]),
                text=batch["raw_text"][i],
                basename=batch["basename"][i],
                speaker=batch["speaker"][i],
                language=batch["language"][i],
                # [T, K] -> [K, T]
                # unpad by subscripting here with :n_frames / :n_tokens / :n_samples
                spec=copy_to_device(specs[i, :n_frames].transpose(0, 1)),
                tokens=copy_to_device(batch["text"][i, :n_tokens]),
                durations=copy_to_device(outputs["duration_rounded"][i, :n_tokens]),
                space_token=synthesis_batch.space_token,
                wav=(
                    None
                    if wavs is None or n_samples is None
                    else copy_to_device(wavs[i, :, :n_samples])
                ),
                sample_rate=synthesis_batch.sample_rate,
            )


def combine_utterance_chunks(
    utterances: Iterable["Utterance"],
) -> Iterator["Utterance"]:
    """Combine the chunks of each input text into a single Utterance.

    Each combined Utterance is yielded as soon as its last chunk comes in:
    - its spec and wav (if the chunks have one) are its chunks' concatenated;
    - its text is their texts joined with single spaces (chunking strips the
      whitespace between chunks);
    - likewise, its tokens are their tokens joined with space_token, whose
      duration is 0 frames;
    - its basename is the one shared by all chunks, or else a slug of its text,
      as the output writers do;
    - it has chunk_index 0 and is_last_chunk True, like a text that was not split.
    A text with a single chunk is yielded as is.

    Args:
        utterances: as yielded by unbatch_synthesis(): the chunks of each text must
            be consecutive and in order, and the last one must come in.

    Raises:
        ValueError: if utterances doesn't respect the conditions above.
    """
    import dataclasses

    import torch
    from everyvoice.utils import resolve_chunked_basename

    def combine(chunks: list["Utterance"]) -> "Utterance":
        if len(chunks) == 1:
            return chunks[0]
        first = chunks[0]
        wavs = [chunk.wav for chunk in chunks if chunk.wav is not None]

        def join_tensors(tensors: list[torch.Tensor], separator: int) -> torch.Tensor:
            """Concatenate tensors, with separator between consecutive ones"""
            sep = torch.tensor(
                [separator], dtype=tensors[0].dtype, device=tensors[0].device
            )
            pieces = [tensors[0]]
            for tensor in tensors[1:]:
                pieces += [sep, tensor]
            return torch.cat(pieces)

        text = " ".join(chunk.text for chunk in chunks)
        return dataclasses.replace(
            first,
            is_last_chunk=True,
            text=text,
            basename=resolve_chunked_basename(
                [chunk.basename for chunk in chunks], text
            ),
            spec=torch.cat([chunk.spec for chunk in chunks], dim=-1),
            # The space between chunks, as in text, takes no frames
            tokens=join_tensors([chunk.tokens for chunk in chunks], first.space_token),
            durations=join_tensors([chunk.durations for chunk in chunks], 0),
            wav=torch.cat(wavs, dim=-1) if len(wavs) == len(chunks) else None,
        )

    pending: list["Utterance"] = []
    for utterance in utterances:
        expected_chunk_index = len(pending)
        if pending and utterance.index != pending[0].index:
            raise ValueError(
                f"Text {pending[0].index} is missing its last chunk: got a chunk of "
                f"text {utterance.index} after its chunk {pending[-1].chunk_index}."
            )
        if utterance.chunk_index != expected_chunk_index:
            raise ValueError(
                f"Chunks of text {utterance.index} are out of order: expected chunk "
                f"{expected_chunk_index}, got chunk {utterance.chunk_index}."
            )
        pending.append(utterance)
        if utterance.is_last_chunk:
            yield combine(pending)
            pending = []
    if pending:
        raise ValueError(
            f"Text {pending[0].index} is missing its last chunk: the input ended "
            f"after its chunk {pending[-1].chunk_index}."
        )


def write_utterances(
    utterances: Iterable["Utterance"],
    writers: Sequence["UtteranceWriter"],
) -> Iterator["Utterance"]:
    """The last stage of the synthesis pipeline: write each synthesized text with
    every writer, in order, then yield it unchanged.

    The caller finalizes the writers, once the iteration is done.

    Args:
        utterances: whole texts, as yielded by combine_utterance_chunks()
        writers: as from get_synthesis_output_writers()

    Raises:
        ValueError: if an utterance is only a chunk of a text.
    """
    for utterance in utterances:
        if utterance.chunk_index > 0 or not utterance.is_last_chunk:
            raise ValueError(
                f"Utterance {utterance.basename!r} is chunk {utterance.chunk_index} "
                f"of text {utterance.index}: apply combine_utterance_chunks() "
                "before writing."
            )
        for writer in writers:
            writer.write_utterance(utterance)
        yield utterance


def synthesize_helper(
    model: "FastSpeech2",
    texts: Optional[list[str]],
    style_reference: Path | None,
    language: Optional[str],
    speaker: Optional[str],
    duration_control: Optional[float],
    global_step: int,
    output_types: Sequence[SynthesizeOutputFormats],
    text_representation: DatasetTextRepresentation,
    devices: str,  # unused: synthesis runs on a single device
    device: "torch.device",  # set to: get_device_from_accelerator(accelerator)
    batch_size: int,
    num_workers: int,
    filelist: Path | None,
    filelist_data: Optional[list[dict]],
    output_dir: Path,
    teacher_forcing_directory: Path | None,
    vocoder_global_step: Optional[int] = None,
    vocoder_model: "HiFiGAN | HiFiGANGenerator | None" = None,
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

    needs_wav = (
        SynthesizeOutputFormats.wav in output_types
        or SynthesizeOutputFormats.readalong_html in output_types
    )
    if needs_wav and vocoder_model is None:
        raise ValueError(
            "We cannot synthesize waveforms without a vocoder. Please ensure that a vocoder is specified."
        )

    from ..prediction_writers import (
        BatchWriter,
        UtteranceWriter,
        get_synthesis_output_writers,
    )

    writers = get_synthesis_output_writers(
        output_types=output_types,
        output_dir=output_dir,
        config=model.config,
        global_step=global_step,
        vocoder_global_step=vocoder_global_step,
        return_scores=return_scores,
        simple_filenames=simple_filenames,
        loss_fn=model.loss,
    )

    batch_writers = [w for w in writers.values() if isinstance(w, BatchWriter)]
    utterance_writers = [w for w in writers.values() if isinstance(w, UtteranceWriter)]

    def text_to_spec_with_batch_writers() -> Iterator["SynthesisBatch"]:
        """The text-to-spec stage, also passing each batch to the batch writers."""
        for synthesis_batch in tqdm(
            text_to_spec_synthesis(
                model,
                data,
                device,
                teacher_forcing=teacher_forcing,
                style_reference=style_reference is not None,
                batch_size=batch_size,
                num_workers=num_workers,
            ),
            desc="Synthesizing",
            total=math.ceil(len(data) / batch_size),
        ):
            for batch_writer in batch_writers:
                batch_writer.write_batch(synthesis_batch.outputs, synthesis_batch.batch)
            yield synthesis_batch

    # Like trainer.predict() did, run the writers (including the vocoder) in
    # inference mode too.
    with torch.inference_mode():
        # The pipeline stages are lazy iterators: everything runs in the loop below
        spec_batches = text_to_spec_with_batch_writers()
        if SynthesizeOutputFormats.wav in writers:  # also there for readalong_html
            assert vocoder_model is not None  # checked above
            spec_batches = spec_to_wav_synthesis(spec_batches, vocoder_model)
        chunk_utterances = unbatch_synthesis(spec_batches)
        text_utterances = combine_utterance_chunks(chunk_utterances)
        written_utterances = write_utterances(text_utterances, utterance_writers)
        for _ in written_utterances:
            pass
        for writer in writers.values():
            writer.finalize()

    return model.config, writers
