import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
from everyvoice.config.type_definitions import TargetTrainingTextRepresentationLevel
from everyvoice.model.feature_prediction.config import FeaturePredictionConfig
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import (
    load_hifigan_from_checkpoint,
    synthesize_data,
)
from everyvoice.text.features import N_PHONOLOGICAL_FEATURES
from everyvoice.text.lookups import LookupTable
from everyvoice.text.text_processor import TextProcessor
from everyvoice.utils.heavy import expand
from loguru import logger
from torch import nn
from torchaudio.models import Conformer

from .config import FastSpeech2Config
from .layers import PositionalEmbedding, PostNet
from .loss import FastSpeech2Loss
from .noam import NoamLR
from .type_definitions_heavy import InferenceControl, Stats
from .utils.heavy import mask_from_lens, plot_attn_maps, plot_mel
from .variance_adaptor import VarianceAdaptor

DEFAULT_LANG2ID: LookupTable = {}
DEFAULT_SPEAKER2ID: LookupTable = {}


class FastSpeech2(pl.LightningModule):
    def __init__(
        self,
        config: dict | FastSpeech2Config,
        stats: Optional[dict | Stats] = None,
        lang2id: LookupTable = DEFAULT_LANG2ID,
        speaker2id: LookupTable = DEFAULT_SPEAKER2ID,
    ):
        """ """
        super().__init__()
        if not isinstance(config, FastSpeech2Config):
            config = FastSpeech2Config(**config)
        if stats is not None and not isinstance(stats, Stats):
            stats = Stats(**stats)
        self.config = config
        self.batch_size = config.training.batch_size
        self.text_processor = TextProcessor(config.text)
        # TODO Should we fallback to a default lang2id/speaker2id if we are loading an old model?
        self.lang2id = lang2id
        self.speaker2id = speaker2id
        self.stats = stats
        self.save_hyperparameters(ignore=[])
        self.loss = FastSpeech2Loss(config=config)
        self.text_input_layer: nn.Linear | nn.Embedding
        if (
            self.config.model.target_text_representation_level
            == TargetTrainingTextRepresentationLevel.phonological_features
        ):
            self.text_input_layer = nn.Linear(
                N_PHONOLOGICAL_FEATURES,
                self.config.model.encoder.input_dim,
                bias=False,
            )
        else:
            self.text_input_layer = nn.Embedding(
                len(self.text_processor.symbols),
                self.config.model.encoder.input_dim,
                padding_idx=self.text_processor.encode_text(
                    self.text_processor._pad_symbol
                )[0],
            )
        self.position_embedding = PositionalEmbedding(
            self.config.model.encoder.input_dim
        )

        self.encoder = Conformer(
            input_dim=self.config.model.encoder.input_dim,
            num_heads=self.config.model.encoder.heads,
            ffn_dim=self.config.model.encoder.feedforward_dim,
            num_layers=self.config.model.encoder.layers,
            depthwise_conv_kernel_size=self.config.model.encoder.conv_kernel_size,
            dropout=self.config.model.encoder.dropout,
        )
        if self.stats is None:
            logger.error(
                """Your model doesn't have a value for self.stats either because the file is missing or the checkpoint didn't save them.
                              We cannot initialize the variance adaptors without variance predictor statistics."""
            )
            self.variance_adaptor = None
        else:
            self.variance_adaptor = VarianceAdaptor(self.config, self.stats)

        self.decoder = Conformer(
            input_dim=self.config.model.decoder.input_dim,
            num_heads=self.config.model.decoder.heads,
            ffn_dim=self.config.model.decoder.feedforward_dim,
            num_layers=self.config.model.decoder.layers,
            depthwise_conv_kernel_size=self.config.model.decoder.conv_kernel_size,
            dropout=self.config.model.decoder.dropout,
        )

        self.mel_linear = nn.Linear(
            self.config.model.decoder.input_dim, self.config.preprocessing.audio.n_mels
        )  # TODO: replace with option for linear spec or complex
        if self.config.model.use_postnet:
            self.postnet = PostNet(
                n_mel_channels=self.config.preprocessing.audio.n_mels
            )  # TODO: allow for postnet parameterization in config
            self.output_key = "postnet_output"
        else:
            self.output_key = "output"
        self.speaker_embedding = None
        if self.config.model.multispeaker:
            if len(self.speaker2id) == 0:
                logger.error(
                    "Your model is multispeaker but speaker2id LookupTable is empty"
                )
                sys.exit(1)
            self.speaker_embedding = nn.Embedding(
                len(self.speaker2id),
                self.config.model.encoder.input_dim,
            )  # TODO: replace with d_vector multispeaker embedding
        self.language_embedding = None
        if self.config.model.multilingual:
            if len(self.lang2id) == 0:
                logger.error(
                    "Your model is multilingual but language2id LookupTable is empty"
                )
                sys.exit(1)
            self.language_embedding = nn.Embedding(
                len(self.lang2id), self.config.model.encoder.input_dim
            )

    def forward(self, batch, control=InferenceControl(), inference=False):
        # Add duration control. TODO: currently we just take the first item of the batch
        # and update the control accordingly, but this is hacky and should be fixed
        if "duration_control" in batch and batch["duration_control"][0]:
            control.duration = batch["duration_control"][0]

        # Determine whether we're teacher forcing or not
        # To do so, we need to be in inference mode and
        # the data loader should have loaded some Mel lengths
        # to force the output to match.
        if inference and batch["mel_lens"] is not None:
            teacher_forcing = True
        else:
            teacher_forcing = False

        # For model diagram see https://github.com/ming024/FastSpeech2/blob/master/img/model.png
        src_lens = batch["src_lens"]
        max_src_len = batch["max_src_len"]
        mel_lens = batch["mel_lens"]
        max_mel_len = batch["max_mel_len"]
        if (
            self.config.model.target_text_representation_level
            == TargetTrainingTextRepresentationLevel.phonological_features
        ):
            text_inputs = batch["pfs"]
        else:
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
        if self.config.model.multispeaker and self.speaker_embedding:
            speaker_emb = self.speaker_embedding(speaker_ids)
            x = x + speaker_emb.unsqueeze(1)

        # Language Embedding
        if self.config.model.multilingual and self.language_embedding:
            lang_emb = self.language_embedding(language_ids)
            x = x + lang_emb.unsqueeze(1)

        # VarianceAdaptor out
        variance_adaptor_out = self.variance_adaptor(
            inputs,
            x,
            batch,
            src_mask,
            control,
            inference=inference,
            teacher_forcing=teacher_forcing,
        )
        # Create inference Mel lens
        if inference and not teacher_forcing:
            mel_lens = torch.IntTensor(
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
        postnet_output = None
        if self.config.model.use_postnet:
            postnet_output = output + self.postnet(output)

        return {
            "output": output,
            "postnet_output": postnet_output,
            "src_mask": src_mask,
            "src_lens": src_lens,
            "tgt_mask": variance_adaptor_out["target_mask"],
            "tgt_lens": mel_lens,
            "attn_logprob": variance_adaptor_out["attn_logprob"],
            "attn_soft": variance_adaptor_out["attn_soft"],
            "attn_hard": variance_adaptor_out["attn_hard"],
            "duration_prediction": variance_adaptor_out["duration_prediction"],
            "duration_target": variance_adaptor_out["duration_target"],
            "energy_prediction": variance_adaptor_out["energy_prediction"],
            "energy_target": variance_adaptor_out["energy_target"],
            "pitch_prediction": variance_adaptor_out["pitch_prediction"],
            "pitch_target": variance_adaptor_out["pitch_target"],
        }

    def on_load_checkpoint(self, checkpoint):
        """Deserialize the checkpoint hyperparameters.
        Note, this shouldn't fail on different versions of pydantic anymore,
        but it will fail on breaking changes to the config. We should catch those exceptions
        and handle them appropriately."""
        self.config = FeaturePredictionConfig(
            **checkpoint["hyper_parameters"]["config"]
        )
        if (
            "stats" in checkpoint["hyper_parameters"]
            and checkpoint["hyper_parameters"]["stats"] is not None
        ):
            self.stats = Stats(**checkpoint["hyper_parameters"]["stats"])

    def on_save_checkpoint(self, checkpoint):
        """Serialize the checkpoint hyperparameters"""
        # Convert the config to a checkpoint-safe config
        checkpoint["hyper_parameters"]["config"] = self.config.model_checkpoint_dump()
        if self.stats is not None:
            checkpoint["hyper_parameters"]["stats"] = self.stats.model_dump(mode="json")

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            return self(batch, inference=True)

    def training_step(self, batch, batch_idx):
        output = self(batch)
        losses = self.loss(output, batch, self.current_epoch)
        self.log_dict(
            {f"training/{k}_loss": v.item() for k, v in losses.items()}, prog_bar=True
        )
        return losses["total"]

    def _validation_global_step_0(self, batch, batch_idx) -> None:
        if self.global_step != 0:
            return

        assert self.logger is not None
        audio, _ = torchaudio.load(
            (
                Path(self.config.preprocessing.save_dir)
                / "audio"
                / "--".join(
                    [
                        batch["basename"][0],
                        batch["speaker"][0],
                        batch["language"][0],
                        f"audio-{self.config.preprocessing.audio.input_sampling_rate}.wav",
                    ]
                )
            ).__str__()
        )

        audio = audio.squeeze()
        # Log ground truth audio
        self.logger.experiment.add_audio(
            f"gt/wav_{batch['basename'][0]}",
            audio,
            self.global_step,
            self.config.preprocessing.audio.output_sampling_rate,
        )
        if self.config.training.vocoder_path:
            input_ = batch["mel"]
            vocoder_ckpt = torch.load(
                self.config.training.vocoder_path, map_location=input_.device
            )
            vocoder_model, vocoder_config = load_hifigan_from_checkpoint(
                vocoder_ckpt, input_.device
            )
            wav, sr = synthesize_data(input_, vocoder_model, vocoder_config)

            self.logger.experiment.add_audio(
                f"copy-synthesis/wav_{batch['basename'][0]}",
                wav,
                self.global_step,
                sr,
            )

    def _validation_batch_idx_0(self, batch, batch_idx, output) -> None:
        if batch_idx != 0:
            return

        assert self.logger is not None

        # Currently only plots the first one, but the function is writte to support plotting multiple
        if self.config.model.learn_alignment:
            figs = plot_attn_maps(
                output["attn_soft"],
                output["attn_hard"],
                output["tgt_lens"],
                output["src_lens"],
                n=1,
            )
            for i, fig in enumerate(figs):
                self.logger.experiment.add_figure(
                    f"attention/{batch['basename'][i]}", fig, self.global_step
                )

        duration_np = output["duration_target"][0].cpu().numpy()
        gt_pitch_for_plotting = batch["pitch"][0].cpu().numpy()
        gt_energy_for_plotting = batch["energy"][0].cpu().numpy()
        pred_pitch_for_plotting = output["pitch_prediction"][0].cpu().numpy()
        pred_energy_for_plotting = output["energy_prediction"][0].cpu().numpy()

        if self.config.model.variance_predictors.pitch.level == "phone":
            pred_pitch_for_plotting = expand(pred_pitch_for_plotting, duration_np)
            if not self.config.model.learn_alignment:
                # pitch targets are frame-wise if alignment is learned
                gt_pitch_for_plotting = expand(gt_pitch_for_plotting, duration_np)

        if self.config.model.variance_predictors.energy.level == "phone":
            pred_energy_for_plotting = expand(pred_energy_for_plotting, duration_np)
            if not self.config.model.learn_alignment:
                # energy targets are frame-wise if alignment is learned
                gt_energy_for_plotting = expand(gt_energy_for_plotting, duration_np)
        if self.stats is not None:
            self.logger.experiment.add_figure(
                f"pred/spec_{batch['basename'][0]}",
                plot_mel(
                    [
                        {
                            "mel": np.swapaxes(batch["mel"][0].cpu().numpy(), 0, 1),
                            "pitch": gt_pitch_for_plotting,
                            "energy": gt_energy_for_plotting,
                        },
                        {
                            "mel": np.swapaxes(
                                output[self.output_key][0].cpu().numpy(), 0, 1
                            ),
                            "pitch": pred_pitch_for_plotting,
                            "energy": pred_energy_for_plotting,
                        },
                    ],
                    self.stats,
                    ["Ground-Truth Spectrogram", "Synthesized Spectrogram"],
                ),
                self.global_step,
            )

        if self.config.training.vocoder_path:
            input_ = output[self.output_key]
            vocoder_ckpt = torch.load(
                self.config.training.vocoder_path, map_location=input_.device
            )
            vocoder_model, vocoder_config = load_hifigan_from_checkpoint(
                vocoder_ckpt, input_.device
            )
            wav, sr = synthesize_data(input_, vocoder_model, vocoder_config)
            self.logger.experiment.add_audio(
                f"pred/wav_{batch['basename'][0]}", wav, self.global_step, sr
            )

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            self._validation_global_step_0(batch, batch_idx)

        output = self(batch)
        if batch_idx == 0:
            self._validation_batch_idx_0(batch, batch_idx, output)

        losses = self.loss(output, batch, self.current_epoch)
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
