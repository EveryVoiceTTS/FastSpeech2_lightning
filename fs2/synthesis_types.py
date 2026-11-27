"""Types yielded by the fs2 synthesis API (see api.text_to_spec and api.unbatch).

Deliberately light: torch is only imported for type checking.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass
class SynthesisBatch:
    """The model's outputs for one batch of input chunks, still padded.

    Attributes:
        outputs: the model's forward() outputs, e.g. outputs[output_key] is the
            [B, T (frames), K (mel bands)] spectrogram batch, "tgt_lens" the
            number of frames of each item, "duration_rounded" the number of frames
            of each input token
        batch: the model's input batch, with the metadata needed downstream:
            "raw_text", "basename", "speaker", "language", "index",
            "chunk_index", "is_last_input_chunk", and "src_lens", the number of
            input tokens of each item
        output_key: the key of the spectrogram in outputs ("postnet_output", or
            "output" for models without a postnet)
        space_token: the model's token id for " ", which separates the chunks of
            a text when they are combined
        wavs: after api.spec_to_wav(), the waveforms, [B, C (channels),
            T (samples)], still padded, on the CPU; None before
        wav_lens: after api.spec_to_wav(), the number of samples of each item
        sample_rate: after api.spec_to_wav(), the sample rate of wavs
    """

    outputs: dict[str, Any]
    batch: dict[str, Any]
    output_key: str
    space_token: int
    wavs: "torch.Tensor | None" = None
    wav_lens: "torch.Tensor | None" = None
    sample_rate: int | None = None

    def __len__(self) -> int:
        return len(self.batch["basename"])


@dataclass
class Utterance:
    """One synthesized chunk of input text, without padding.

    Attributes:
        index: which input text this is a chunk of (its position in the texts
            given to api.text_to_spec)
        chunk_index: the position of this chunk within its input text
        is_last_chunk: whether this is the last chunk of its input text
        text: the text of this chunk
        basename, speaker, language: as used to synthesize this chunk
        spec: the spectrogram, [K (mel bands), T (frames)], the same layout as the
            spec files written by EveryVoice
        tokens: the input token ids, [N (tokens)]
        durations: the number of frames of each input token, [N (tokens)];
            durations.sum() == spec.size(-1)
        space_token: the model's token id for " ": api.combine_chunks() puts it,
            with a duration of 0, between the tokens of consecutive chunks, so
            tokens match text
        wav: the waveform, [C (channels), T (samples)], if api.spec_to_wav() was
            applied before api.unbatch(); None otherwise
        sample_rate: the sample rate of wav, or None
    """

    index: int
    chunk_index: int
    is_last_chunk: bool
    text: str
    basename: str
    speaker: str
    language: str
    spec: "torch.Tensor"
    tokens: "torch.Tensor"
    durations: "torch.Tensor"
    space_token: int
    wav: "torch.Tensor | None" = None
    sample_rate: int | None = None
