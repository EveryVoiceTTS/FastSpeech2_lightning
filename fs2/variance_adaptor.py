import json

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .config import FastSpeech2Config
from .layers import VarianceConvolutionLayer
from .type_definitions import InferenceControl, Stats, StatsInfo


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
        lengths = torch.tensor([t.shape[0] for t in repeated_list]).long()
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

    def __init__(self, config: FastSpeech2Config):
        super().__init__()
        self.config = config
        with open(self.config.preprocessing.save_dir / "stats.json") as f:
            self.stats = Stats(**json.load(f))
        # Duration Predictor
        self.duration_predictor = VariancePredictor(
            input_dim=self.config.model.encoder.hidden_dim,
            n_layers=self.config.model.variance_adaptor.variance_predictors.duration.n_layers,
            n_channels=self.config.model.variance_adaptor.variance_predictors.duration.hidden_dim,
            output_dim=1,
            kernel_size=self.config.model.variance_adaptor.variance_predictors.duration.kernel_size,
            dropout_rate=self.config.model.variance_adaptor.variance_predictors.duration.dropout,
            depthwise=self.config.model.variance_adaptor.variance_predictors.duration.depthwise,
        )
        self.length_regulator = LengthRegulator()
        # Pitch Predictor
        self.pitch_predictor = VariancePredictor(
            input_dim=self.config.model.encoder.hidden_dim,
            n_layers=self.config.model.variance_adaptor.variance_predictors.pitch.n_layers,
            n_channels=self.config.model.variance_adaptor.variance_predictors.pitch.hidden_dim,
            output_dim=1,
            kernel_size=self.config.model.variance_adaptor.variance_predictors.pitch.kernel_size,
            dropout_rate=self.config.model.variance_adaptor.variance_predictors.pitch.dropout,
            depthwise=self.config.model.variance_adaptor.variance_predictors.pitch.depthwise,
        )
        self.pitch_embedding = nn.Embedding(
            self.config.model.variance_adaptor.variance_predictors.pitch.n_bins,
            self.config.model.variance_adaptor.variance_predictors.pitch.hidden_dim,
            padding_idx=0,
        )
        self.pitch_bins = nn.Parameter(
            torch.linspace(
                self.stats.pitch.norm_min,
                self.stats.pitch.norm_max,
                self.config.model.variance_adaptor.variance_predictors.pitch.n_bins,
                requires_grad=False,
            )
        )
        # Energy Predictor
        self.energy_predictor = VariancePredictor(
            input_dim=self.config.model.encoder.hidden_dim,
            n_layers=self.config.model.variance_adaptor.variance_predictors.energy.n_layers,
            n_channels=self.config.model.variance_adaptor.variance_predictors.energy.hidden_dim,
            output_dim=1,
            kernel_size=self.config.model.variance_adaptor.variance_predictors.energy.kernel_size,
            dropout_rate=self.config.model.variance_adaptor.variance_predictors.energy.dropout,
            depthwise=self.config.model.variance_adaptor.variance_predictors.energy.depthwise,
        )
        self.energy_embedding = nn.Embedding(
            self.config.model.variance_adaptor.variance_predictors.energy.n_bins,
            self.config.model.variance_adaptor.variance_predictors.energy.hidden_dim,
            padding_idx=0,
        )
        self.energy_bins = nn.Parameter(
            torch.linspace(
                self.stats.energy.norm_min,
                self.stats.energy.norm_max,
                self.config.model.variance_adaptor.variance_predictors.energy.n_bins,
                requires_grad=False,
            )
        )

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
            embed = embedding(torch.bucketize(target, bins).to(x.device))
        else:
            prediction = prediction * control
            embed = embedding(torch.bucketize(prediction, bins).to(x.device))
        return prediction, embed

    def forward(
        self,
        encoder_output,
        batch,
        src_mask,
        control=InferenceControl(),
        inference=False,
    ):  # sourcery skip: swap-if-expression
        # Get information from batch
        x = encoder_output.clone()
        energy_target = batch["energy"] if not inference else None
        pitch_target = batch["pitch"] if not inference else None
        duration_target = batch["duration"] if not inference else None
        max_target_len = batch["max_mel_len"]
        src_mask = src_mask
        # If phone-level variance, use src_mask before duration predictor and upsampling
        # otherwise use tgt_mask after upsampling
        if (
            self.config.model.variance_adaptor.variance_predictors.energy.level
            == "phone"
        ):
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
        if (
            self.config.model.variance_adaptor.variance_predictors.pitch.level
            == "phone"
        ):
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
        # speaker embedding is handled in main model
        log_duration_prediction = self.duration_predictor(x, mask=src_mask)
        # upsampling from text time steps to mel time steps
        if not inference:
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
            ).long()
            x, tgt_mask = self.length_regulator(
                x, duration_rounded, max_length=max_target_len
            )

        if (
            self.config.model.variance_adaptor.variance_predictors.energy.level
            == "frame"
        ):
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

        if (
            self.config.model.variance_adaptor.variance_predictors.pitch.level
            == "frame"
        ):
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
            "duration_prediction": log_duration_prediction,
            "pitch_prediction": pitch_prediction,
            "energy_prediction": energy_prediction,
            "duration_rounded": duration_rounded,
            "target_mask": tgt_mask,
        }
