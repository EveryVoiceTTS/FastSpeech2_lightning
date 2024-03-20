from typing import Sequence

import torch
from everyvoice.preprocessor import Preprocessor
from everyvoice.text.lookups import LookupTable
from everyvoice.text.text_processor import TextProcessor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .config import FastSpeech2Config


def collator(data: list):
    batch = {
        "text": pad_sequence(
            [d["text"] for d in data], batch_first=True, padding_value=0.0
        ),
        "src_lens": torch.cat([d["src_lens"] for d in data]),
        "max_src_len": max([d["max_src_len"] for d in data]),  # Must be a single value
        "language_id": torch.cat([d["language_id"] for d in data]),
        "speaker_id": torch.cat([d["speaker_id"] for d in data]),
        "basename": [d["basename"] for d in data],
        "language": [d["language"] for d in data],
        "speaker": [d["speaker"] for d in data],
        "max_mel_len": max([d["max_mel_len"] for d in data]),
        "mel_lens": [d["mel_lens"] for d in data],
    }

    return batch


class SynthesizeTextDataSet(Dataset):
    def __init__(
        self,
        items: Sequence[dict[str, str]],
        config: FastSpeech2Config,
        lang2id: LookupTable,
        speaker2id: LookupTable,
    ):
        self.items = items
        self.config = config
        self.text_processor = TextProcessor(config.text)
        self.lang2id = lang2id
        self.speaker2id = speaker2id

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        text_tensor = Preprocessor.extract_text_inputs(
            item["text"],
            self.text_processor,
            use_pfs=self.config.model.use_phonological_feats,
        )
        # Create Batch
        src_lens = text_tensor.size(0)
        max_src_len = src_lens
        if (language_id := self.lang2id.get(item["language"], None)) is None:
            language_id = 0
        if (speaker_id := self.speaker2id.get(item["speaker"], None)) is None:
            speaker_id = 0
        batch = {
            "text": text_tensor,
            "src_lens": torch.LongTensor([src_lens]),
            "language_id": torch.LongTensor([language_id]),
            "speaker_id": torch.LongTensor([speaker_id]),
            "basename": item["basename"],
            "language": item["language"],
            "speaker": item["speaker"],
            "max_mel_len": 1_000_000,
            "mel_lens": None,
            "max_src_len": max_src_len,  # Must be a int
            # "duration": duration,   # loaded, skipped in inference mode
            # "energy": energy,   # loaded, skipped in inference mode
            # "label": item.get("label", "default"), # TODO: determine proper label
            # "mel": mel,  # loaded
            # "pfs": pfs,   # loaded
            # "pitch": pitch,   # loaded, skipped in inference mode
            # "raw_text": raw_text,
        }

        return batch
