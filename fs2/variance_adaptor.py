import sys

import numpy as np
import torch
import torch.nn.functional as F
from everyvoice.exceptions import BadDataError
from loguru import logger
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .attn.alignment import mas_width1
from .attn.attention import ConvAttention
from .config import FastSpeech2Config
from .layers import VarianceConvolutionLayer
from .type_definitions_heavy import InferenceControl, Stats, StatsInfo


class VariancePredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_layers=5,
        n_channels=384,
        output_dim=1,
        kernel_size=5,
        dropout_rate=0.1,
        depthwise: bool = False,
    ):
        """Initialize VariancePredictor

        Args:
            input_dim (int): dimension of input
            n_layers (int, optional): number of layers. Defaults to 5.
            n_channels (int, optional): number of channels in convolutional layers. Defaults to 384.
            output_dim (int, optional): output dimension. Defaults to 2.
            kernel_size (int, optional): kernel size of convolution layers. Defaults to 5.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.conv = nn.ModuleList()
        self.kernel_size = kernel_size
        for idx in range(n_layers):
            in_chans = input_dim if idx == 0 else n_channels
            self.conv.append(
                VarianceConvolutionLayer(
                    in_channels=in_chans,
                    out_channels=n_channels,
                    kernel_size=kernel_size,
                    dropout=dropout_rate,
                    depthwise=depthwise,
                )
            )
        self.linear = nn.Linear(n_channels, output_dim)

    def forward(self, x, mask=None):
        for m in self.conv:
            x = m(x)
        out = self.linear(x)
        out = out.squeeze(-1)
        if mask is not None:
            out = out * mask
        return out


class LengthRegulator(nn.Module):
    def forward(self, x, durations, max_length=None):
        repeated_list = [
            torch.repeat_interleave(x[i], durations[i], dim=0)
            for i in range(x.shape[0])
        ]
        lengths = torch.IntTensor([t.shape[0] for t in repeated_list])
        # FIXME: int(max_length) when max_length is None is invalid
        max_length = min(lengths.max(), int(max_length))
        mask = (
            torch.arange(max_length).expand(len(lengths), max_length)
            < lengths.unsqueeze(1)
        ).to(x.device)
        out = pad_sequence(repeated_list, batch_first=True, padding_value=0)
        if max_length is not None:
            out = out[:, :max_length]
        return out, mask


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, config: FastSpeech2Config, stats: Stats):
        super().__init__()
        self.config = config
        self.stats = stats
        # Duration Predictor
        self.duration_predictor = VariancePredictor(
            input_dim=self.config.model.encoder.input_dim,
            n_layers=self.config.model.variance_predictors.duration.n_layers,
            n_channels=self.config.model.variance_predictors.duration.input_dim,
            output_dim=1,
            kernel_size=self.config.model.variance_predictors.duration.kernel_size,
            dropout_rate=self.config.model.variance_predictors.duration.dropout,
            depthwise=self.config.model.variance_predictors.duration.depthwise,
        )
        self.length_regulator = LengthRegulator()
        # Pitch Predictor
        self.pitch_predictor = VariancePredictor(
            input_dim=self.config.model.encoder.input_dim,
            n_layers=self.config.model.variance_predictors.pitch.n_layers,
            n_channels=self.config.model.variance_predictors.pitch.input_dim,
            output_dim=1,
            kernel_size=self.config.model.variance_predictors.pitch.kernel_size,
            dropout_rate=self.config.model.variance_predictors.pitch.dropout,
            depthwise=self.config.model.variance_predictors.pitch.depthwise,
        )
        self.pitch_embedding = nn.Embedding(
            self.config.model.variance_predictors.pitch.n_bins,
            self.config.model.variance_predictors.pitch.input_dim,
            # padding_idx=0,
        )
        self.pitch_bins = nn.Parameter(
            torch.linspace(
                self.stats.pitch.norm_min,
                self.stats.pitch.norm_max,
                self.config.model.variance_predictors.pitch.n_bins - 1,
            ),
            requires_grad=False,
        )
        # Energy Predictor
        self.energy_predictor = VariancePredictor(
            input_dim=self.config.model.encoder.input_dim,
            n_layers=self.config.model.variance_predictors.energy.n_layers,
            n_channels=self.config.model.variance_predictors.energy.input_dim,
            output_dim=1,
            kernel_size=self.config.model.variance_predictors.energy.kernel_size,
            dropout_rate=self.config.model.variance_predictors.energy.dropout,
            depthwise=self.config.model.variance_predictors.energy.depthwise,
        )
        self.energy_embedding = nn.Embedding(
            self.config.model.variance_predictors.energy.n_bins,
            self.config.model.variance_predictors.energy.input_dim,
            # padding_idx=0,
        )

        self.energy_bins = nn.Parameter(
            torch.linspace(
                self.stats.energy.norm_min,
                self.stats.energy.norm_max,
                self.config.model.variance_predictors.energy.n_bins - 1,
            ),
            requires_grad=False,
        )

        # Attention
        if self.config.model.learn_alignment:
            self.attention = ConvAttention(
                self.config.preprocessing.audio.n_mels,
                0,
                self.config.model.encoder.input_dim,
                use_query_proj=True,
                align_query_enc_type="3xconv",
            )

    def binarize_attention(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
           These will no longer recieve a gradient.
        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        b_size = attn.shape[0]
        with torch.no_grad():
            attn_out_cpu = np.zeros(attn.data.shape, dtype=np.float32)
            log_attn_cpu = torch.log(attn.data).to(device="cpu", dtype=torch.float32)
            log_attn_cpu = log_attn_cpu.numpy()
            out_lens_cpu = out_lens.cpu()
            in_lens_cpu = in_lens.cpu()
            for ind in range(b_size):
                hard_attn = mas_width1(
                    log_attn_cpu[ind, 0, : out_lens_cpu[ind], : in_lens_cpu[ind]]
                )
                attn_out_cpu[
                    ind, 0, : out_lens_cpu[ind], : in_lens_cpu[ind]
                ] = hard_attn
            attn_out = torch.tensor(attn_out_cpu, device=attn.device, dtype=attn.dtype)
        return attn_out

    def get_variance_embedding(
        self,
        x,
        target,
        mask,
        predictor,
        embedding,
        stats: StatsInfo,
        bins,
        control,
        inference,
    ):
        prediction = predictor(x, mask)
        if not inference:
            buckets = torch.bucketize(target, bins)
            # max_b = max([max(b) for b in buckets])
            # min_b = min([min(b) for b in buckets])
            # breakpoint()
            embed = embedding(buckets.to(x.device))
        else:
            prediction = prediction * control
            embed = embedding(torch.bucketize(prediction, bins).to(x.device))
        return prediction, embed

    def average_variance(self, var, durs):
        durs_cums_ends = torch.cumsum(durs, dim=1).long()
        durs_cums_starts = F.pad(durs_cums_ends[:, :-1], (1, 0))
        var_nonzero_cums = F.pad(torch.cumsum(var != 0.0, dim=1), (1, 0))
        var_cums = F.pad(torch.cumsum(var, dim=1), (1, 0))

        var_sums = (
            torch.gather(var_cums, 1, durs_cums_ends)
            - torch.gather(var_cums, 1, durs_cums_starts)
        ).float()
        var_nelems = (
            torch.gather(var_nonzero_cums, 1, durs_cums_ends)
            - torch.gather(var_nonzero_cums, 1, durs_cums_starts)
        ).float()
        var_avg = torch.where(var_nelems == 0.0, var_nelems, var_sums / var_nelems)
        return var_avg

    def forward(
        self,
        text_emb,
        encoder_output,
        batch,
        src_mask,
        control=InferenceControl(),
        inference=False,
        teacher_forcing=False,
    ):  # sourcery skip: swap-if-expression
        # Get information from batch
        x = encoder_output.clone()
        energy_target = batch["energy"] if not inference else None
        pitch_target = batch["pitch"] if not inference else None
        duration_target = (
            batch["duration"] if batch["duration"][0] is not None else None
        )
        max_target_len = batch["max_mel_len"]
        src_mask = src_mask
        attn_logprob = None  # Overwritten if alignment is learned
        attn_soft = None
        attn_hard = None
        # speaker embedding is handled in main model
        # Alignment
        if (teacher_forcing or not inference) and self.config.model.learn_alignment:
            # make sure to do the alignments before folding
            attn_mask = src_mask[..., None] == 0
            # attn_mask should be 1 for unused timesteps in the text_enc_w_spkvec tensor
            attn_soft, attn_logprob = self.attention(
                batch["mel"].transpose(1, 2),
                text_emb.transpose(1, 2),
                batch["mel_lens"],
                attn_mask,
                key_lens=batch["src_lens"],
                keys_encoded=x,
                attn_prior=batch["duration"],
            )

            attn_hard = self.binarize_attention(
                attn_soft, batch["src_lens"], batch["mel_lens"]
            )

            # Viterbi --> durations
            attn_hard_dur = attn_hard.sum(2)[:, 0, :]
            duration_target = attn_hard_dur.int()
            if (
                pitch_target is not None and (pitch_target.size(1) == text_emb.size(1))
            ) or (
                energy_target is not None
                and (energy_target.size(1) == text_emb.size(1))
            ):
                logger.error(
                    "Your pitch and/or energy targets are already averaged across phones, but when you are learning alignment with phone-level energy or pitch modelling, you must have an un-averaged target for these as the duration of phones changes during training. This should happen automatically if you re-run the preprocessing step for energy and pitch."
                )
                sys.exit(1)
            if (
                energy_target is not None
                and self.config.model.variance_predictors.energy.level == "phone"
            ):
                energy_target = self.average_variance(energy_target, duration_target)
            if (
                pitch_target is not None
                and self.config.model.variance_predictors.pitch.level == "phone"
            ):
                pitch_target = self.average_variance(pitch_target, duration_target)
            try:
                equal_dur_targets = torch.eq(
                    duration_target.sum(dim=1), batch["mel_lens"]
                )
                assert torch.all(equal_dur_targets)
            except AssertionError as e:
                from itertools import compress

                mismatches = list(
                    compress(
                        batch["basename"], [not x for x in equal_dur_targets.tolist()]
                    )
                )
                raise BadDataError(
                    f"Something failed with the following items, please check them for errors: {mismatches}"
                ) from e
                sys.exit(1)
        # If phone-level variance, use src_mask before duration predictor and upsampling of x
        # using length regulator
        # otherwise use tgt_mask after upsampling
        if self.config.model.variance_predictors.energy.level == "phone":
            energy_prediction, energy_embedding = self.get_variance_embedding(
                x,
                energy_target,
                src_mask,
                self.energy_predictor,
                self.energy_embedding,
                self.stats.energy,
                self.energy_bins,
                control.energy,
                inference,
            )
            try:
                x = x + energy_embedding
            except RuntimeError as e:
                print(batch["basename"])
                print(f"energy target is size {energy_target.size()}")
                print(x.size())
                print(energy_embedding.size())
                breakpoint()
                raise e
        if self.config.model.variance_predictors.pitch.level == "phone":
            pitch_prediction, pitch_embedding = self.get_variance_embedding(
                x,
                pitch_target,
                src_mask,
                self.pitch_predictor,
                self.pitch_embedding,
                self.stats.pitch,
                self.pitch_bins,
                control.pitch,
                inference,
            )
            try:
                x = x + pitch_embedding
            except RuntimeError as e:
                print(x.size())
                print(batch["basename"])
                print(f"pitch target is size {pitch_target.size()}")
                print(pitch_embedding.size())
                breakpoint()
                raise e

        log_duration_prediction = self.duration_predictor(x, mask=src_mask)
        # upsampling from text time steps to mel time steps
        if teacher_forcing or not inference:
            x, tgt_mask = self.length_regulator(
                x, duration_target, max_length=max_target_len
            )
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (
                    torch.round(torch.exp(log_duration_prediction) - 1)
                    * control.duration
                ),
                min=0,
            ).int()
            x, tgt_mask = self.length_regulator(
                x, duration_rounded, max_length=max_target_len
            )

        if self.config.model.variance_predictors.energy.level == "frame":
            energy_prediction, energy_embedding = self.get_variance_embedding(
                x,
                energy_target,
                tgt_mask,
                self.energy_predictor,
                self.energy_embedding,
                self.stats.energy,
                self.energy_bins,
                control.energy,
                inference,
            )
            x = x + energy_embedding

        if self.config.model.variance_predictors.pitch.level == "frame":
            pitch_prediction, pitch_embedding = self.get_variance_embedding(
                x,
                pitch_target,
                tgt_mask,
                self.pitch_predictor,
                self.pitch_embedding,
                self.stats.pitch,
                self.pitch_bins,
                control.pitch,
                inference,
            )
            x = x + pitch_embedding

        return {
            "output": x,
            "attn_logprob": attn_logprob,
            "attn_soft": attn_soft,
            "attn_hard": attn_hard,
            "duration_prediction": log_duration_prediction,
            "duration_target": duration_target,
            "pitch_prediction": pitch_prediction,
            "pitch_target": pitch_target,
            "energy_prediction": energy_prediction,
            "energy_target": energy_target,
            "duration_rounded": duration_rounded,
            "target_mask": tgt_mask,
        }
