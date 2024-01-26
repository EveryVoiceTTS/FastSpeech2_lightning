from typing import Sequence

import torch
from everyvoice.preprocessor import Preprocessor
from everyvoice.text.lookups import LookupTable
from loguru import logger
from torch.utils.data import Dataset


class SynthesizeTextDataSet(Dataset):
    def __init__(
        self,
        items: Sequence[dict[str, str]],
        preprocessor: Preprocessor,
        lang2id: LookupTable,
        speaker2id: LookupTable,
        device: torch.device,
    ):
        self.items = items
        self.preprocessor = preprocessor
        self.lang2id = lang2id
        self.speaker2id = speaker2id
        self.device = device

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        logger.info("Creating batch")
        item = self.items[idx]
        text_tensor = self.preprocessor.extract_text_inputs(item["text"])
        # Create Batch
        src_lens = torch.LongTensor([text_tensor.size(0)])
        max_src_len = max(src_lens)
        if (language_id := self.lang2id.get(item["language"], None)) is None:
            language_id = 0
        if (speaker_id := self.speaker2id.get(item["speaker"], None)) is None:
            speaker_id = 0
        batch = {
            "text": text_tensor,
            "src_lens": src_lens,
            "max_src_len": max_src_len,
            "language_id": torch.LongTensor([language_id]),
            "speaker_id": torch.LongTensor([speaker_id]),
            # "basename": basename,   # from item["basename"]
            # "duration": duration,   # loaded, skipped in inference mode
            # "energy": energy,   # loaded, skipped in inference mode
            # "label": item.get("label", "default"), # TODO: determine proper label
            # "language": language,   # from item["language"]
            # "mel": mel,  # loaded
            # "pfs": pfs,   # loaded
            # "pitch": pitch,   # loaded, skipped in inference mode
            # "raw_text": raw_text,
            # "speaker": speaker,   # from item["speaker"]
        }
        batch = {k: v.to(self.device) for k, v in batch.items()}
        batch["basename"] = item["basename"]
        batch["language"] = item["language"]
        batch["speaker"] = item["speaker"]
        batch["max_mel_len"] = 1_000_000
        batch["mel_lens"] = None

        return batch
