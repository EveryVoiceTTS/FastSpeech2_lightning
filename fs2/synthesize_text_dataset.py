from typing import Dict, Sequence

import torch
from everyvoice.preprocessor import Preprocessor
from everyvoice.text.lookups import LookupTable
from loguru import logger
from torch.utils.data import Dataset


class SynthesizeTextDataSet(Dataset):
    def __init__(
        self,
        data: Sequence[Dict[str, str]],
        preprocessor: Preprocessor,
        lang2id: LookupTable,
        speaker2id: LookupTable,
        device: torch.device,
    ):
        self.data = data
        self.preprocessor = preprocessor
        self.lang2id = lang2id
        self.speaker2id = speaker2id
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        logger.info("Creating batch")
        data = self.data[idx]
        text_tensor = self.preprocessor.extract_text_inputs(data["text"])
        # Create Batch
        src_lens = torch.LongTensor([text_tensor.size(0)])
        max_src_len = max(src_lens)
        if (language_id := self.lang2id.get(data["language"], None)) is None:
            language_id = 0
            logger.error(f"{data['language']} not in {set(self.lang2id.keys())}")
        if (speaker_id := self.speaker2id.get(data["speaker"], None)) is None:
            speaker_id = 0
            logger.error(f"{data['speaker']} not in {set(self.speaker2id.keys())}")
        batch = {
            "text": text_tensor,
            "src_lens": src_lens,
            "max_src_len": max_src_len,
            "language_id": torch.LongTensor([language_id]),
            "speaker_id": torch.LongTensor([speaker_id]),
            # "energy": None,
            # "pitch": None,
        }
        batch = {k: v.to(self.device) for k, v in batch.items()}
        batch["max_mel_len"] = 1_000_000
        batch["mel_lens"] = None

        return batch
