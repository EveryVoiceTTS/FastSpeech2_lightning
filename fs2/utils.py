from typing import Optional

import torch
import torch.distributions as D
import torch.nn.functional as F
from matplotlib import pyplot as plt

from .type_definitions import Stats

def get_mels_from_tvcgmm_prediction(spec_prediction, spec_target, tgt_mask, n_mels):
    # Same as the implementation here: https://github.com/sony/ai-research-code/blob/master/tvc-gmm/model/utils/tools.py
    # except does batch-wise transformations, is generalized to varying numbers of Mel bands and assumes a B T K
    # oriented spectral prediction tensor.
    mixture = return_mixture_model(spec_prediction, spec_target, tgt_mask)
    mel_prediction = mixture.sample().reshape(spec_prediction.shape[0], -1, (n_mels * 3)) # B, T, K * 3
    mel_prediction[:, 1:, :n_mels] += mel_prediction[:, :-1, n_mels:(n_mels * 2)]
    mel_prediction[:, :, 1:n_mels] += mel_prediction[:, :, (n_mels * 2):-1]
    mel_prediction[:, 1:, 1:n_mels] /= 3
    mel_prediction[:, 1:, 0] /= 2
    mel_prediction[:, 0, 1:] /= 2
    return mel_prediction[:, :, :n_mels]

def return_mixture_model(spec_prediction, spec_target, tgt_mask, k=5, min_var=1.0e-3):
    # spec_prediction: B, T, K * 50
    # spec_target: B, T, K
    # tgt_mask: B, T
    param_predictions = spec_prediction.reshape(*spec_target.shape, k, 10)[
        :, : tgt_mask.shape[1]
    ]
    # in practice we predict the scale_tril (lower triangular factor of the covariance matrix)
    # we predict the parameters for every t,f bin
    # at every bin we predict the joint distribution of t,f t+1,f and t,f+1
    # --> later in sampling we have overlap of one bin with the next time and the next freq bin
    scale_tril = torch.diag_embed(
        torch.nn.functional.softplus(param_predictions[..., 4:7]) + min_var, offset=0
    )
    scale_tril += torch.diag_embed(param_predictions[..., 7:9], offset=-1)
    scale_tril += torch.diag_embed(param_predictions[..., 9:10], offset=-2)

    mix = D.Categorical(torch.nn.functional.softmax(param_predictions[..., 0], dim=-1))
    comp = D.MultivariateNormal(param_predictions[..., 1:4], scale_tril=scale_tril)
    return D.MixtureSameFamily(mix, comp)


def plot_attn_maps(attn_softs, attn_hards, mel_lens, text_lens, n=4):
    bs = len(attn_softs)
    n = min(n, bs)
    s = bs // n
    attn_softs = attn_softs[::s].cpu().numpy()
    attn_hards = attn_hards[::s].cpu().numpy()
    figs = []
    for attn_soft, attn_hard, mel_len, text_len in zip(
        attn_softs, attn_hards, mel_lens, text_lens
    ):
        attn_soft = attn_soft[:, :mel_len, :text_len].squeeze(0).transpose()
        attn_hard = attn_hard[:, :mel_len, :text_len].squeeze(0).transpose()
        fig, axs = plt.subplots(2, 1)
        axs[0].imshow(attn_soft, aspect="auto", origin="lower")
        axs[1].imshow(attn_hard, aspect="auto", origin="lower")
        fig.canvas.draw()
        figs.append(fig)
    return figs


def plot_mel(data, stats: Stats, titles):
    data_len = len(data)
    fig, axes = plt.subplots(data_len, 1, squeeze=False)
    fig.tight_layout(pad=2.0)
    if titles is None:
        titles = [None for i in range(data_len)]

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(data_len):
        mel = data[i]["mel"]
        energy = data[i]["energy"] * stats.energy.std + stats.energy.mean
        pitch = data[i]["pitch"] * stats.pitch.std + stats.pitch.mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, stats.pitch.max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(stats.energy.min, stats.energy.max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


def mask_from_lens(lens, max_len: Optional[int] = None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    return torch.lt(ids, lens.unsqueeze(1))


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = max(lengths).item()

    ids = (
        torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    )
    return ids >= lengths.unsqueeze(1).expand(-1, max_len)


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn"t know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


def dur_to_mel2ph(dur, dur_padding=None, alpha=1.0):
    """
    Example (no batch dim version):
        1. dur = [2,2,3]
        2. token_idx = [[1],[2],[3]], dur_cumsum = [2,4,7], dur_cumsum_prev = [0,2,4]
        3. token_mask = [[1,1,0,0,0,0,0],
                            [0,0,1,1,0,0,0],
                            [0,0,0,0,1,1,1]]
        4. token_idx * token_mask = [[1,1,0,0,0,0,0],
                                        [0,0,2,2,0,0,0],
                                        [0,0,0,0,3,3,3]]
        5. (token_idx * token_mask).sum(0) = [1,1,2,2,3,3,3]
    :param dur: Batch of durations of each frame (B, T_txt)
    :param dur_padding: Batch of padding of each frame (B, T_txt)
    :param alpha: duration rescale coefficient
    :return:
        mel2ph (B, T_speech)
    """
    assert alpha > 0
    dur = torch.round(dur.float() * alpha).long()
    if dur_padding is not None:
        dur = dur * (1 - dur_padding.long())
    token_idx = torch.arange(1, dur.shape[1] + 1)[None, :, None].to(dur.device)
    dur_cumsum = torch.cumsum(dur, 1)
    dur_cumsum_prev = F.pad(dur_cumsum, [1, -1], mode="constant", value=0)

    pos_idx = torch.arange(dur.sum(-1).max())[None, None].to(dur.device)
    token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (
        pos_idx < dur_cumsum[:, :, None]
    )
    return (token_idx * token_mask.long()).sum(1)


def mel2ph_to_dur(mel2ph, T_txt, max_dur=None):
    B, _ = mel2ph.shape
    dur = mel2ph.new_zeros(B, T_txt + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    return dur
