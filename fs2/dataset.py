from pathlib import Path

import numpy as np
import torch
from everyvoice.config.preprocessing_config import DatasetTextRepresentation
from everyvoice.config.shared_types import TargetTrainingTextRepresentationLevel
from everyvoice.dataloader import BaseDataModule
from everyvoice.text.lookups import lookuptables_from_config
from everyvoice.text.text_processor import TextProcessor
from everyvoice.utils import _flatten, check_dataset_size
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .config import FastSpeech2Config


class FastSpeechDataset(Dataset):
    """
    To debug, set num_workers=0 and batch_size=1
    """

    def __init__(self, dataset, config: FastSpeech2Config):
        self.dataset = dataset
        self.config = config
        self.sep = "--"
        self.text_processor = TextProcessor(config.text)
        self.preprocessed_dir = Path(self.config.preprocessing.save_dir)
        self.sampling_rate = self.config.preprocessing.audio.input_sampling_rate
        self.lang2id, self.speaker2id = lookuptables_from_config(self.config)

    def _load_file(self, bn, spk, lang, dir, fn):
        return torch.load(
            self.preprocessed_dir / dir / self.sep.join([bn, spk, lang, fn])
        )

    def __getitem__(self, index):
        """
        Returns dict with keys: {
            "mel"
            "duration"
            "pfs"
            "text"
            "raw_text"
            "basename"
            "speaker"
            "speaker_id"
            "language"
            "language_id"
            "label"
            "energy"
            "pitch"
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
            "spec",
            f"spec-{self.sampling_rate}-{self.config.preprocessing.audio.spec_type}.pt",
        ).transpose(
            0, 1
        )  # [mel_bins, frames] -> [frames, mel_bins]
        if self.config.model.learn_alignment:
            if (
                self.config.model.target_text_representation_level
                == TargetTrainingTextRepresentationLevel.characters
            ):
                duration = self._load_file(
                    basename,
                    speaker,
                    language,
                    "attn",
                    f"{DatasetTextRepresentation.characters.value}-attn-prior.pt",
                )
            elif self.config.model.target_text_representation_level in [
                TargetTrainingTextRepresentationLevel.ipa_phones,
                TargetTrainingTextRepresentationLevel.phonological_features,
            ]:
                duration = self._load_file(
                    basename,
                    speaker,
                    language,
                    "attn",
                    f"{DatasetTextRepresentation.ipa_phones.value}-attn-prior.pt",
                )
        else:
            duration = self._load_file(
                basename, speaker, language, "duration", "duration.pt"
            )
        if (
            self.config.model.target_text_representation_level
            == TargetTrainingTextRepresentationLevel.characters
        ):
            text = torch.Tensor(
                self.text_processor.encode_escaped_string_sequence(
                    item["character_tokens"]
                )
            ).long()
        elif self.config.model.target_text_representation_level in [
            TargetTrainingTextRepresentationLevel.ipa_phones,
            TargetTrainingTextRepresentationLevel.phonological_features,
        ]:
            text = torch.Tensor(
                self.text_processor.encode_escaped_string_sequence(item["phone_tokens"])
            ).long()

        raw_text = item.get("raw_text", "clean_text", "text")
        pfs = None
        if self.config.model.use_phonological_feats:
            pfs = self._load_file(basename, speaker, language, "text", "pfs.pt")

        energy = self._load_file(basename, speaker, language, "energy", "energy.pt")
        pitch = self._load_file(basename, speaker, language, "pitch", "pitch.pt")

        return {
            "mel": mel,
            "duration": duration,
            "pfs": pfs,
            "text": text,
            "raw_text": raw_text,
            "basename": basename,
            "speaker": speaker,
            "speaker_id": speaker_id,
            "language": language,
            "language_id": language_id,
            # "label": item.get("label", "default"), # TODO: determine proper label
            "energy": energy,
            "pitch": pitch,
        }

    def __len__(self):
        return len(self.dataset)

    def get_labels(self):
        return [x["label"] for x in self.dataset]


class FastSpeech2DataModule(BaseDataModule):
    def __init__(self, config: FastSpeech2Config):
        super().__init__(config=config)
        self.collate_fn = self.collate_method
        self.use_weighted_sampler = config.training.use_weighted_sampler
        self.batch_size = config.training.batch_size
        self.load_dataset()
        self.dataset_length = len(self.train_dataset) + len(self.val_dataset)

    def collate_method(self, data):
        data = [_flatten(x) for x in data]
        data = {k: [dic[k] for dic in data] for k in data[0]}
        text_lens = torch.LongTensor([text.size(0) for text in data["text"]])
        mel_lens = torch.LongTensor([mel.size(0) for mel in data["mel"]])
        max_mel = max(mel_lens)
        max_text = max(text_lens)
        for key in data:
            if isinstance(data[key][0], np.ndarray):
                data[key] = [torch.tensor(x) for x in data[key]]
            if torch.is_tensor(data[key][0]):
                if key == "duration" and self.config.model.learn_alignment:
                    # in this case we need to pad both the src and target dimensions
                    dur_padded = torch.zeros(len(text_lens), max_mel, max_text)
                    dur_padded.zero_()
                    for i in range(len(data[key])):
                        dur = data[key][i]
                        dur_padded[i, : dur.size(0), : dur.size(1)] = dur
                    data[key] = dur_padded
                else:
                    data[key] = pad_sequence(
                        data[key], batch_first=True, padding_value=0
                    )
            if isinstance(data[key][0], int):
                data[key] = torch.tensor(data[key]).long()
        data["src_lens"] = text_lens
        data["mel_lens"] = mel_lens
        data["max_src_len"] = max_text
        data["max_mel_len"] = max_mel
        return data

    def load_dataset(self):
        self.train_dataset = self.config.training.filelist_loader(
            self.config.training.training_filelist
        )
        self.val_dataset = self.config.training.filelist_loader(
            self.config.training.validation_filelist
        )

    def prepare_data(self):
        train_samples = len(self.train_dataset)
        val_samples = len(self.val_dataset)
        check_dataset_size(self.batch_size, train_samples, "training")
        check_dataset_size(self.batch_size, val_samples, "validation")
        self.train_dataset = FastSpeechDataset(self.train_dataset, self.config)
        self.val_dataset = FastSpeechDataset(self.val_dataset, self.config)
        # save it to disk
        torch.save(self.train_dataset, self.train_path)
        torch.save(self.val_dataset, self.val_path)
