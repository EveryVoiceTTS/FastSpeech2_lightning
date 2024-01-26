import torch
from torch import nn

from .attn.attention_loss import AttentionBinarizationLoss, AttentionCTCLoss
from .config import FastSpeech2Config


class FastSpeech2Loss(nn.Module):
    def __init__(self, config: FastSpeech2Config):
        super().__init__()
        self.config = config
        self.loss_fns = {
            "mse": nn.MSELoss(),
            "mae": nn.L1Loss(),
        }
        self.attn_ctc_loss = AttentionCTCLoss()
        self.attn_bin_loss = AttentionBinarizationLoss()

    def forward(self, output, batch, current_epoch, frozen_components=None):
        # sourcery skip: merge-dict-assign, move-assign-in-block
        log_duration_prediction = output["duration_prediction"]
        duration_target = output["duration_target"]
        energy_target = output["energy_target"]
        pitch_target = output["pitch_target"]
        energy_prediction = output["energy_prediction"]
        pitch_prediction = output["pitch_prediction"]
        spec_target = batch["mel"]
        spec_prediction = output["output"]
        spec_postnet_prediction = output["postnet_output"]

        if frozen_components is None:
            frozen_components = []

        # Get masks
        src_mask = output["src_mask"]
        tgt_mask = output["tgt_mask"]

        # Don't calculate grad on target
        duration_target.requires_grad = False
        energy_target.requires_grad = False
        spec_target.requires_grad = False
        pitch_target.requires_grad = False

        losses = {}

        # Calculate pitch loss
        if self.config.model.variance_predictors.pitch.level == "phone":
            pitch_mask = src_mask
        else:
            pitch_mask = tgt_mask

        pitch_prediction = pitch_prediction * pitch_mask
        pitch_target = pitch_target * pitch_mask
        pitch_loss_fn = self.config.model.variance_predictors.pitch.loss
        losses["pitch"] = (
            self.loss_fns[pitch_loss_fn](pitch_prediction, pitch_target)
            * self.config.training.pitch_loss_weight
        )

        # Calculate energy loss
        if self.config.model.variance_predictors.energy.level == "phone":
            energy_mask = src_mask
        else:
            energy_mask = tgt_mask

        energy_prediction = energy_prediction * energy_mask
        energy_target = energy_target * energy_mask
        energy_loss_fn = self.config.model.variance_predictors.energy.loss
        losses["energy"] = (
            self.loss_fns[energy_loss_fn](energy_prediction, energy_target)
            * self.config.training.energy_loss_weight
        )

        # Calculate duration loss
        log_duration_target = torch.log(duration_target.float() + 1) * src_mask
        log_duration_prediction = log_duration_prediction * src_mask
        duration_loss_fn = self.config.model.variance_predictors.duration.loss
        losses["duration"] = (
            self.loss_fns[duration_loss_fn](
                log_duration_prediction, log_duration_target
            )
            * self.config.training.duration_loss_weight
        )

        # Calculate Mel-spectrogram loss
        tgt_mask = tgt_mask.unsqueeze(2)
        spec_prediction = spec_prediction * tgt_mask
        spec_target = spec_target * tgt_mask
        losses["spec"] = (
            self.loss_fns[self.config.model.mel_loss](spec_prediction, spec_target)
            * self.config.training.mel_loss_weight
        )
        if self.config.model.use_postnet:
            spec_postnet_prediction = spec_postnet_prediction * tgt_mask
            losses["postnet"] = (
                self.loss_fns[self.config.model.mel_loss](
                    spec_postnet_prediction, spec_target
                )
                * self.config.training.postnet_loss_weight
            )

        # Calculate attention loss if using
        if self.config.model.learn_alignment:
            ctc_loss = self.attn_ctc_loss(
                output["attn_logprob"], batch["src_lens"], batch["mel_lens"]
            )
            losses["attn_ctc"] = ctc_loss * self.config.training.attn_ctc_loss_weight
            bin_loss_weight = (
                min(
                    current_epoch / self.config.training.attn_bin_loss_warmup_epochs,
                    1.0,
                )
                * self.config.training.attn_bin_loss_weight
            )
            bin_loss = self.attn_bin_loss(output["attn_hard"], output["attn_soft"])
            losses["attn_bin"] = bin_loss * bin_loss_weight

        # Calculate total loss
        losses["total"] = sum(losses.values())
        return losses
