import random
from pathlib import Path

import numpy as np
import torch
from smts.dataloader import BaseDataModule
from smts.text import TextProcessor
from smts.text.lookups import LookupTables
from smts.utils import collate_fn, expand
from torch.utils.data import Dataset, random_split

from .config import FastSpeech2Config


class FastSpeechDataset(Dataset):
    """
    To debug, set num_workers=0 and batch_size=1
    """

    def __init__(self, dataset, config: FastSpeech2Config):
        self.dataset = dataset
        self.config = config
        self.lookup = LookupTables(config)
        self.sep = config.preprocessing.value_separator
        self.text_processor = TextProcessor(config)
        self.preprocessed_dir = Path(self.config.preprocessing.save_dir)
        random.seed(self.config.training.seed)
        self.sampling_rate = self.config.preprocessing.audio.input_sampling_rate
        self.speaker2id = self.lookup.speaker2id
        self.lang2id = self.lookup.lang2id

    def _load_file(self, bn, spk, lang, fn):
        return torch.load(self.preprocessed_dir / self.sep.join([bn, spk, lang, fn]))

    def __getitem__(self, index):
        """
        Returns dict with keys: {
            "phones",
            "duration",
            "silence_mask",
            "unexpanded_silence_mask",
            "priors",
            "audio",
            "speaker",
            "text",
            "basename",
            "variances",
            "labels",
        }
        """
        item = self.dataset[index]
        speaker = "default" if "speaker" not in item else item["speaker"]
        language = "default" if "language" not in item else item["language"]
        speaker_id = self.speaker2id[speaker]
        language_id = self.lang2id[language]
        basename = item["basename"]
        mel = self._load_file(
            basename,
            speaker,
            language,
            f"spec-{self.sampling_rate}-{self.config.preprocessing.audio.spec_type}.npy",
        ).transpose(
            0, 1
        )  # [mel_bins, frames] -> [frames, mel_bins]
        duration = self._load_file(basename, speaker, language, "duration.npy")
        text = self._load_file(basename, speaker, language, "text.npy")
        raw_text = item["raw_text"]
        pfs = None
        if self.config.model.use_phonological_feats:
            pfs = self._load_file(basename, speaker, language, "pfs.npy")
        variances = {
            vp.variance_type: self._load_file(
                basename, speaker, language, f"{vp.variance_type}.npy"
            )
            for vp in self.config.model.variance_adaptor.variance_predictors
        }

        # TODO: Fix text processor and resolve how to deal with punctuation and
        # potential mismatch between duration/textgrid and text
        # DONE: This was resolved by the use of an aligner that uses the same text processor
        # TODO: silence masks
        # DONE: There won't really be any silences if we aren't using MFA, this might be a problem
        # Durations & Silence Mask
        silence_ids = [
            self.text_processor.cleaned_text_to_sequence(x)[0]
            for x in self.config.text.symbols.silence
        ]
        silence_masks = [np.array(text) == s for s in silence_ids]
        unexpanded_silence_mask = np.logical_or.reduce(silence_masks)
        silence_mask = expand(unexpanded_silence_mask, duration)

        # Priors

        return {
            "mel": mel,
            "duration": duration,
            "silence_mask": silence_mask,
            "unexpanded_silence_mask": unexpanded_silence_mask,
            "pfs": pfs,
            "text": text,
            "raw_text": raw_text,
            "speaker": speaker,
            "speaker_id": speaker_id,
            "language": language,
            "language_id": language_id,
            "label": item["label"],
            "variances": variances,
        }

    def __len__(self):
        return len(self.dataset)

    def get_labels(self):
        return [x["label"] for x in self.dataset]


class FastSpeech2DataModule(BaseDataModule):
    # TODO: look into compatibility with base data module; ie pin_memory, drop_last, etc
    def __init__(self, config: FastSpeech2Config):
        super().__init__(config=config)
        self.collate_fn = collate_fn
        self.use_weighted_sampler = config.training.use_weighted_sampler
        self.batch_size = config.training.batch_size
        self.train_split = self.config.training.train_split
        self.load_dataset()

    def load_dataset(self):
        self.dataset = self.config.training.filelist_loader(
            self.config.training.filelist
        )

    def prepare_data(self):
        train_split = int(len(self.dataset) * self.train_split)

        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_split, len(self.dataset) - train_split]
        )
        self.train_dataset = FastSpeechDataset(self.train_dataset, self.config)
        self.val_dataset = FastSpeechDataset(self.val_dataset, self.config)
        # save it to disk
        torch.save(self.train_dataset, self.train_path)
        torch.save(self.val_dataset, self.val_path)
