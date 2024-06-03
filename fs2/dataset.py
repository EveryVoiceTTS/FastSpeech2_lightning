from functools import partial
from pathlib import Path

import numpy as np
import torch
from everyvoice.config.type_definitions import (
    DatasetTextRepresentation,
    TargetTrainingTextRepresentationLevel,
)
from everyvoice.dataloader import BaseDataModule
from everyvoice.preprocessor import Preprocessor
from everyvoice.text.lookups import LookupTable, lookuptables_from_config
from everyvoice.text.text_processor import TextProcessor
from everyvoice.utils import (
    _flatten,
    check_dataset_size,
    filter_dataset_based_on_target_text_representation_level,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .config import FastSpeech2Config


class FastSpeechDataset(Dataset):
    """
    To debug, set num_workers=0 and batch_size=1
    """

    def __init__(
        self,
        dataset,
        config: FastSpeech2Config,
        lang2id: LookupTable,
        speaker2id: LookupTable,
        teacher_forcing=False,
        inference=False,
    ):
        self.dataset = dataset
        self.config = config
        self.sep = "--"
        self.text_processor = TextProcessor(config.text)
        self.preprocessed_dir = Path(self.config.preprocessing.save_dir)
        self.sampling_rate = self.config.preprocessing.audio.input_sampling_rate
        self.teacher_forcing = teacher_forcing
        self.inference = inference
        self.lang2id = lang2id
        self.speaker2id = speaker2id

    def _load_file(self, bn, spk, lang, dir, fn):
        return torch.load(
            self.preprocessed_dir / dir / self.sep.join([bn, spk, lang, fn])
        )

    def __getitem__(self, index):
        """
        Returns dict with keys: {
            "mel"
            "duration"
            "duration_control"
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
        duration_control = (
            1.0 if "duration_control" not in item else item["duration_control"]
        )
        speaker_id = self.speaker2id[speaker]
        language_id = self.lang2id[language]
        basename = item["basename"]
        if self.inference:
            # TODO: we shouldn't calculate all possible representations at synthesis time,
            #       despite that's what we do at preprocessing time.
            character_tokens, phone_tokens, _ = Preprocessor.process_text(
                item,
                text_processor=self.text_processor,
                use_pfs=False,
                encode_as_string=True,
            )
            item["character_tokens"] = character_tokens
            item["phone_tokens"] = phone_tokens
        if self.teacher_forcing or not self.inference:
            mel = self._load_file(
                basename,
                speaker,
                language,
                "spec",
                f"spec-{self.sampling_rate}-{self.config.preprocessing.audio.spec_type}.pt",
            ).transpose(
                0, 1
            )  # [mel_bins, frames] -> [frames, mel_bins]
        else:
            mel = None
        if (
            self.teacher_forcing or not self.inference
        ) and self.config.model.learn_alignment:
            match self.config.model.target_text_representation_level:
                case TargetTrainingTextRepresentationLevel.characters:
                    duration = self._load_file(
                        basename,
                        speaker,
                        language,
                        "attn",
                        f"{DatasetTextRepresentation.characters.value}-attn-prior.pt",
                    )
                case TargetTrainingTextRepresentationLevel.ipa_phones | TargetTrainingTextRepresentationLevel.phonological_features:
                    duration = self._load_file(
                        basename,
                        speaker,
                        language,
                        "attn",
                        f"{DatasetTextRepresentation.ipa_phones.value}-attn-prior.pt",
                    )
                case _:
                    raise NotImplementedError(
                        f"{self.config.model.target_text_representation_level} have not yet been implemented."
                    )
        elif self.teacher_forcing or not self.inference:
            duration = self._load_file(
                basename, speaker, language, "duration", "duration.pt"
            )
        else:
            duration = None
        match self.config.model.target_text_representation_level:
            case TargetTrainingTextRepresentationLevel.characters:
                text = torch.IntTensor(
                    self.text_processor.encode_escaped_string_sequence(
                        item["character_tokens"]
                    )
                )
            case TargetTrainingTextRepresentationLevel.ipa_phones | TargetTrainingTextRepresentationLevel.phonological_features:
                text = torch.IntTensor(
                    self.text_processor.encode_escaped_string_sequence(
                        item["phone_tokens"]
                    )
                )
            case _:
                raise NotImplementedError(
                    f"{self.config.model.target_text_representation_level} have not yet been implemented."
                )

        if TargetTrainingTextRepresentationLevel.characters.value in item:
            raw_text = item[TargetTrainingTextRepresentationLevel.characters.value]
        else:
            raw_text = item.get(
                TargetTrainingTextRepresentationLevel.ipa_phones.value, "text"
            )
        pfs = None
        if (
            self.config.model.target_text_representation_level
            == TargetTrainingTextRepresentationLevel.phonological_features
        ):
            pfs = self._load_file(basename, speaker, language, "pfs", "pfs.pt")
        if not self.inference:
            energy = self._load_file(basename, speaker, language, "energy", "energy.pt")
            pitch = self._load_file(basename, speaker, language, "pitch", "pitch.pt")
        else:
            energy = None
            pitch = None

        return {
            "mel": mel,
            "duration": duration,
            "duration_control": duration_control,
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
    def __init__(
        self,
        config: FastSpeech2Config,
        inference=False,
        teacher_forcing=False,
        inference_output_dir=Path("synthesis_output"),
    ):
        super().__init__(config=config, inference_output_dir=inference_output_dir)
        self.inference = inference
        self.prepared = False
        self.teacher_forcing = teacher_forcing
        self.collate_fn = partial(
            self.collate_method, learn_alignment=config.model.learn_alignment
        )
        self.use_weighted_sampler = config.training.use_weighted_sampler
        self.batch_size = config.training.batch_size
        if not inference:
            self.load_dataset()
            self.dataset_length = len(self.train_dataset) + len(self.val_dataset)
            self.lang2id, self.speaker2id = lookuptables_from_config(config)

    @staticmethod
    def collate_method(data, learn_alignment=True):
        data = [_flatten(x) for x in data]
        data = {k: [dic[k] for dic in data] for k in data[0]}
        text_lens = torch.IntTensor([text.size(0) for text in data["text"]])
        max_text = max(text_lens)
        if data["mel"][0] is not None:
            mel_lens = torch.IntTensor([mel.size(0) for mel in data["mel"]])
            max_mel = max(mel_lens)
        else:
            mel_lens = None
            max_mel = 1_000_000

        for key in data:
            if isinstance(data[key][0], np.ndarray):
                data[key] = [torch.tensor(x) for x in data[key]]
            if torch.is_tensor(data[key][0]):
                if key == "duration" and learn_alignment:
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
                data[key] = torch.IntTensor(data[key])

        data["src_lens"] = text_lens
        data["max_src_len"] = max_text
        data["mel_lens"] = mel_lens
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
        if self.inference:
            self.predict_dataset = FastSpeechDataset(
                self.predict_dataset,
                self.config,
                self.lang2id,
                self.speaker2id,
                inference=self.inference,
                teacher_forcing=self.teacher_forcing,
            )
            torch.save(self.predict_dataset, self.predict_path)
        elif not self.prepared:
            self.train_dataset = (
                filter_dataset_based_on_target_text_representation_level(
                    self.config.model.target_text_representation_level,
                    self.train_dataset,
                    "training",
                    self.batch_size,
                )
            )
            self.val_dataset = filter_dataset_based_on_target_text_representation_level(
                self.config.model.target_text_representation_level,
                self.val_dataset,
                "validation",
                self.batch_size,
            )
            train_samples = len(self.train_dataset)
            val_samples = len(self.val_dataset)
            check_dataset_size(self.batch_size, train_samples, "training")
            check_dataset_size(self.batch_size, val_samples, "validation")
            self.train_dataset = FastSpeechDataset(
                self.train_dataset,
                self.config,
                self.lang2id,
                self.speaker2id,
                inference=self.inference,
            )
            self.val_dataset = FastSpeechDataset(
                self.val_dataset,
                self.config,
                self.lang2id,
                self.speaker2id,
                inference=self.inference,
            )
            # save it to disk
            torch.save(self.train_dataset, self.train_path)
            torch.save(self.val_dataset, self.val_path)
            self.prepared = True


class FastSpeech2SynthesisDataModule(FastSpeech2DataModule):
    def __init__(
        self,
        config: FastSpeech2Config,
        data: list[dict],
        lang2id: LookupTable,
        speaker2id: LookupTable,
        teacher_forcing: bool = False,
    ):
        super().__init__(config=config, inference=True, teacher_forcing=teacher_forcing)
        self.inference = True
        self.data = data
        self.collate_fn = partial(
            self.collate_method, learn_alignment=config.model.learn_alignment
        )
        self.use_weighted_sampler = config.training.use_weighted_sampler
        self.batch_size = config.training.batch_size
        self.load_dataset(data)
        self.predict_dataset_length = len(self.predict_dataset)
        self.lang2id = lang2id
        self.speaker2id = speaker2id

    def load_dataset(self, data):
        self.predict_dataset = data
