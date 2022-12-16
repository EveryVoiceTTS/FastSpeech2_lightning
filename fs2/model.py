import json
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
from smts.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import synthesize_data
from smts.text import TextProcessor
from smts.text.lookups import LookupTables
from smts.utils.heavy import expand
from torch import nn
from torchaudio.models import Conformer

from .config import FastSpeech2Config
from .layers import PositionalEmbedding, PostNet
from .loss import FastSpeech2Loss
from .noam import NoamLR
from .type_definitions import InferenceControl, Stats
from .utils import mask_from_lens, plot_mel
from .variance_adaptor import VarianceAdaptor


class FastSpeech2(pl.LightningModule):
    def __init__(self, config: FastSpeech2Config):
        super().__init__()
        self.config = config
        self.batch_size = config.training.batch_size
        self.text_processor = TextProcessor(config)
        self.embedding_lookup = LookupTables(config)
        self.save_hyperparameters(ignore=[])
        self.loss = FastSpeech2Loss(config=config)
        self.text_input_layer: Union[nn.Linear, nn.Embedding]
        with open(self.config.preprocessing.save_dir / "stats.json") as f:
            self.stats: Stats = Stats(**json.load(f))
        if self.config.model.use_phonological_feats:
            self.text_input_layer = nn.Linear(
                self.config.model.phonological_feats_size,
                self.config.model.encoder.hidden_dim,
                bias=False,
            )
        else:
            self.text_input_layer = nn.Embedding(
                len(self.text_processor.symbols),
                self.config.model.encoder.hidden_dim,
                padding_idx=self.text_processor.text_to_sequence(
                    self.text_processor._pad_symbol
                )[0],
            )
        self.position_embedding = PositionalEmbedding(
            self.config.model.encoder.hidden_dim
        )

        if self.config.model.encoder.conformer:
            self.encoder = Conformer(
                input_dim=self.config.model.encoder.hidden_dim,
                num_heads=self.config.model.encoder.heads,
                ffn_dim=self.config.model.encoder.feedforward_dim,
                num_layers=self.config.model.encoder.layers,
                depthwise_conv_kernel_size=self.config.model.encoder.conv_kernel_size,
                dropout=self.config.model.encoder.dropout,
            )
        else:
            raise NotImplementedError(
                "Only a standard TorchAudio Conformer is currently supported."
            )
        self.variance_adaptor = VarianceAdaptor(config)

        if self.config.model.decoder.conformer:
            self.decoder = Conformer(
                input_dim=self.config.model.decoder.hidden_dim,
                num_heads=self.config.model.decoder.heads,
                ffn_dim=self.config.model.decoder.feedforward_dim,
                num_layers=self.config.model.decoder.layers,
                depthwise_conv_kernel_size=self.config.model.decoder.conv_kernel_size,
                dropout=self.config.model.decoder.dropout,
            )
        else:
            raise NotImplementedError(
                "Only a standard TorchAudio Conformer is currently supported."
            )

        self.mel_linear = nn.Linear(
            self.config.model.decoder.hidden_dim, self.config.preprocessing.audio.n_mels
        )  # TODO: replace with option for linear spec or complex
        self.postnet = PostNet(
            n_mel_channels=self.config.preprocessing.audio.n_mels
        )  # TODO: allow for postnet parameterization in config
        self.speaker_embedding = nn.Embedding(
            len(self.embedding_lookup.speaker2id), self.config.model.encoder.hidden_dim
        )  # TODO: replace with d_vector multispeaker embedding
        self.language_embedding = nn.Embedding(
            len(self.embedding_lookup.lang2id), self.config.model.encoder.hidden_dim
        )
        # Freeze Layers
        if self.config.training.freeze_layers.encoder:
            self.encoder.freeze()
        if self.config.training.freeze_layers.decoder:
            self.decoder.freeze()
        if self.config.training.freeze_layers.postnet:
            self.encoder.freeze()

    def forward(self, batch, control=InferenceControl(), inference=False):
        # For model diagram see https://github.com/ming024/FastSpeech2/blob/master/img/model.png
        src_lens = batch["src_lens"]
        max_src_len = batch["max_src_len"]
        mel_lens = batch["mel_lens"]
        max_mel_len = batch["max_mel_len"]
        text_inputs = batch["text"]
        src_mask = mask_from_lens(src_lens, max_src_len)
        speaker_ids = batch["speaker_id"]
        language_ids = batch["language_id"]
        # Text Embedding
        inputs = self.text_input_layer(text_inputs)

        # Positional Embedding
        enc_pos_seq = torch.arange(
            max_src_len, device=inputs.device, requires_grad=False
        ).to(inputs.dtype)

        enc_pos_emb = self.position_embedding(enc_pos_seq) * src_mask.unsqueeze(2)

        # Encoder
        x, _ = self.encoder(inputs + enc_pos_emb, src_lens)  # expects B, T, K
        # Speaker Embedding
        if self.speaker_embedding:
            speaker_emb = self.speaker_embedding(speaker_ids)
            x = x + speaker_emb.unsqueeze(1)

        # Language Embedding
        if self.language_embedding:
            lang_emb = self.language_embedding(language_ids)
            x = x + lang_emb.unsqueeze(1)

        # VarianceAdaptor out
        variance_adaptor_out = self.variance_adaptor(
            x, batch, src_mask, control, inference=inference
        )

        # Create inference Mel lens
        if inference:
            mel_lens = torch.LongTensor(
                [x.nonzero().size(0) for x in variance_adaptor_out["target_mask"]]
            ).to(text_inputs.device)
            max_mel_len = max(mel_lens)

        # Positional Embedding
        dec_pos_seq = torch.arange(max_mel_len, device=text_inputs.device).to(
            torch.float32 if inference else batch["mel"].dtype
        )
        dec_pos_emb = self.position_embedding(dec_pos_seq) * variance_adaptor_out[
            "target_mask"
        ].unsqueeze(2)

        # Decoder
        x, _ = self.decoder(variance_adaptor_out["output"] + dec_pos_emb, mel_lens)

        # Mel Linear
        output = self.mel_linear(x)

        # Postnet
        postnet_output = output + self.postnet(output)

        return {
            "output": output,
            "postnet_output": postnet_output,
            "src_mask": src_mask,
            "tgt_mask": variance_adaptor_out["target_mask"],
            "tgt_lens": mel_lens,
            "duration_prediction": variance_adaptor_out["duration_prediction"],
            "energy_prediction": variance_adaptor_out["energy_prediction"],
            "pitch_prediction": variance_adaptor_out["pitch_prediction"],
        }

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            return self(batch, inference=False)

    def training_step(self, batch, batch_idx):
        output = self(batch)
        losses = self.loss(output, batch)
        self.log_dict({f"training/{k}_loss": v.item() for k, v in losses.items()})
        return losses["total"]

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            audio = torch.load(
                self.config.preprocessing.save_dir
                / "audio"
                / self.config.preprocessing.value_separator.join(
                    [
                        batch["basename"][0],
                        batch["speaker"][0],
                        batch["language"][0],
                        f"audio-{self.config.preprocessing.audio.input_sampling_rate}.pt",
                    ]
                )
            )
            # Log ground truth audio
            self.logger.experiment.add_audio(
                f"gt/wav_{batch['basename'][0]}",
                audio,
                self.global_step,
                self.config.preprocessing.audio.output_sampling_rate,
            )
            if self.config.training.vocoder_path:
                checkpoint = torch.load(
                    self.config.training.vocoder_path, map_location=batch["mel"].device
                )
                gt_wav, gt_sr = synthesize_data(batch["mel"], checkpoint)
                self.logger.experiment.add_audio(
                    f"copy-synthesis/wav_{batch['basename'][0]}",
                    gt_wav,
                    self.global_step,
                    gt_sr,
                )
        output = self(batch)
        if batch_idx == 0:
            duration_np = batch["duration"][0].cpu().numpy()
            self.logger.experiment.add_figure(
                f"pred/spec_{batch['basename'][0]}",
                plot_mel(
                    [
                        {
                            "mel": np.swapaxes(batch["mel"][0].cpu().numpy(), 0, 1),
                            "pitch": expand(
                                batch["pitch"][0].cpu().numpy(), duration_np
                            ),
                            "energy": expand(
                                batch["energy"][0].cpu().numpy(), duration_np
                            ),
                        },
                        {
                            "mel": np.swapaxes(
                                output["postnet_output"][0].cpu().numpy(), 0, 1
                            ),
                            "pitch": expand(
                                output["pitch_prediction"][0].cpu().numpy(), duration_np
                            ),
                            "energy": expand(
                                output["energy_prediction"][0].cpu().numpy(),
                                duration_np,
                            ),
                        },
                    ],
                    self.stats,
                    ["Ground-Truth Spectrogram", "Synthesized Spectrogram"],
                ),
                self.global_step,
            )
            if self.config.training.vocoder_path:
                checkpoint = torch.load(
                    self.config.training.vocoder_path, map_location=batch["mel"].device
                )
                wav, sr = synthesize_data(output["postnet_output"], checkpoint)
                self.logger.experiment.add_audio(
                    f"pred/wav_{batch['basename'][0]}", wav, self.global_step, sr
                )
        losses = self.loss(output, batch)
        self.log_dict(
            {f"validation/{k}_loss": v.item() for k, v in losses.items()},
            batch_size=self.batch_size,
            sync_dist=True,
        )

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            self.config.training.optimizer.learning_rate,
            betas=self.config.training.optimizer.betas,
            eps=self.config.training.optimizer.eps,
            weight_decay=self.config.training.optimizer.weight_decay,
        )

        self.scheduler = NoamLR(
            self.optimizer,
            self.config.training.optimizer.warmup_steps,
        )

        sched = {
            "scheduler": self.scheduler,
            "interval": "step",
        }

        return [self.optimizer], [sched]
