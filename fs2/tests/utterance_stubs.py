"""Hand-made synthesized Utterances, for tests that don't need a model."""

from string import ascii_lowercase

import torch
from everyvoice.config.text_config import TextConfig
from everyvoice.text.text_processor import TextProcessor
from everyvoice.utils import slugify, truncate_basename

from ..synthesis_types import Utterance

# Texts must use these symbols: lowercase ASCII letters and spaces
TEXT_CONFIG = TextConfig(symbols={"ascii": list(ascii_lowercase)})
TEXT_PROCESSOR = TextProcessor(TEXT_CONFIG)
SPACE_TOKEN = TEXT_PROCESSOR.encode_string_tokens([" "])[0]
HOP_SIZE = 256  # audio samples per spec frame


def make_utterance(
    text: str,
    index: int = 0,
    chunk_index: int = 0,
    is_last_chunk: bool = True,
    *,
    basename: str | None = None,
    speaker: str = "default",
    language: str = "default",
    n_frames: int = 490,
) -> Utterance:
    """A synthesized text, or chunk of text, as unbatch() yields.

    By default, its basename is its slugified text, as prepare_data() makes it.
    Its spec is n_frames frames of 80 mel bands of ones; its tokens, from
    TEXT_PROCESSOR, share the frames; and its wav is silence, HOP_SIZE samples
    per frame.
    """
    tokens = torch.IntTensor(TEXT_PROCESSOR.encode_text(text))
    durations = torch.full((len(tokens),), n_frames // len(tokens), dtype=torch.int)
    durations[-1] += n_frames - int(durations.sum())
    return Utterance(
        index=index,
        chunk_index=chunk_index,
        is_last_chunk=is_last_chunk,
        text=text,
        basename=basename or truncate_basename(slugify(text)),
        speaker=speaker,
        language=language,
        spec=torch.ones(80, n_frames),
        tokens=tokens,
        durations=durations,
        space_token=SPACE_TOKEN,
        wav=torch.zeros(1, n_frames * HOP_SIZE),
        sample_rate=22050,
    )
