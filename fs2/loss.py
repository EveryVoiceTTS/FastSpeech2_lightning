import torch
from torch import nn

from .attention_loss import AttentionCTCLoss
from .config import FastSpeech2Config


class FastSpeech2Loss(nn.Module):
    def __init__(self, config: FastSpeech2Config):
        super().__init__()
        self.config = config
        self.loss_fns = {
            "mse": nn.MSELoss(),
            "l1": nn.L1Loss(),
        }
        self.attn_ctc_loss = AttentionCTCLoss()

    def forward(self, output, batch, frozen_components=None):
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
        if (
            self.config.model.variance_adaptor.variance_predictors.pitch.level
            == "phone"
        ):
            pitch_mask = src_mask
        else:
            pitch_mask = tgt_mask

        pitch_prediction = pitch_prediction * pitch_mask
        pitch_target = pitch_target * pitch_mask
        pitch_loss_fn = (
            self.config.model.variance_adaptor.variance_predictors.pitch.loss
        )
        losses["pitch"] = self.loss_fns[pitch_loss_fn](pitch_prediction, pitch_target)

        # Calculate energy loss
        if (
            self.config.model.variance_adaptor.variance_predictors.energy.level
            == "phone"
        ):
            energy_mask = src_mask
        else:
            energy_mask = tgt_mask

        energy_prediction = energy_prediction * energy_mask
        energy_target = energy_target * energy_mask
        energy_loss_fn = (
            self.config.model.variance_adaptor.variance_predictors.energy.loss
        )
        losses["energy"] = self.loss_fns[energy_loss_fn](
            energy_prediction, energy_target
        )

        # Calculate duration loss
        log_duration_target = torch.log(duration_target.float() + 1) * src_mask
        log_duration_prediction = log_duration_prediction * src_mask
        duration_loss_fn = (
            self.config.model.variance_adaptor.variance_predictors.duration.loss
        )
        losses["duration"] = self.loss_fns[duration_loss_fn](
            log_duration_prediction, log_duration_target
        )

        # Calculate Mel-spectrogram loss
        tgt_mask = tgt_mask.unsqueeze(2)
        spec_prediction = spec_prediction * tgt_mask
        spec_postnet_prediction = spec_postnet_prediction * tgt_mask
        spec_target = spec_target * tgt_mask
        losses["spec"] = self.loss_fns[self.config.model.mel_loss](
            spec_prediction, spec_target
        )
        losses["postnet"] = self.loss_fns[self.config.model.mel_loss](
            spec_postnet_prediction, spec_target
        )

        # Calculate attention loss if using
        if self.config.model.learn_alignment:
            attn_loss = self.attn_ctc_loss(
                output["attn_logprob"], batch["src_lens"], batch["mel_lens"]
            )
            losses["attn"] = attn_loss

        # Calculate total loss
        losses["total"] = sum(losses.values())
        return losses
