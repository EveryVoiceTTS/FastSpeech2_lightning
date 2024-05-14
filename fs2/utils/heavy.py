from typing import Optional

import torch
from matplotlib import pyplot as plt

from ..type_definitions_heavy import Stats

BASENAME_MAX_LENGTH = 20


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
