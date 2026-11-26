"""Spec-to-wav helpers, shared by the synthesis API and the wav writer.

Deliberately light at import time: torch and the vocoder are imported lazily.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.config import (
        HiFiGANConfig,
    )
    from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.model import (
        HiFiGAN,
        HiFiGANGenerator,
    )


def get_output_hop_size(vocoder_config: "HiFiGANConfig") -> int:
    """The number of output audio samples the vocoder generates per spec frame."""
    audio_config = vocoder_config.preprocessing.audio
    sampling_rate_change = (
        audio_config.output_sampling_rate // audio_config.input_sampling_rate
    )
    return sampling_rate_change * audio_config.fft_hop_size


def spec_to_wav_batch(
    specs: "torch.Tensor",
    frame_lens: "torch.Tensor | Sequence[int]",
    vocoder: "HiFiGAN | HiFiGANGenerator",
    vocoder_config: "HiFiGANConfig",
) -> tuple["torch.Tensor", "torch.Tensor", int]:
    """Generate the waveforms of a padded batch of spectrograms.

    Args:
        specs: the spectrograms, [B (batch size), T (frames), K (mel bands)], as
            output by the text-to-spec model
        frame_lens: the number of frames of each spectrogram, [B], as a tensor or
            a sequence of ints
        vocoder: the spec-to-wav model
        vocoder_config: its configuration, normally vocoder.config

    Returns: (wavs, wav_lens, sample_rate), where wavs is [B, C (channels),
        T (samples)], still padded, on the CPU, and wav_lens is the number of
        samples of each item, [B].
    """
    import torch
    from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import (
        synthesize_data,
    )

    wavs, sample_rate = synthesize_data(specs.transpose(1, 2), vocoder, vocoder_config)
    assert (
        wavs.ndim == 3
    ), f"The generated audio did not contain 3 dimensions. First dimension should be B(atch) and the second dimension should be C(hannels) and third dimension should be T(ime) in samples. Got {wavs.shape} instead."
    assert wavs.shape[0] == specs.size(
        0
    ), f"You provided {specs.size(0)} utterances, but {wavs.shape[0]} audio files were synthesized instead."

    wav_lens = torch.as_tensor(frame_lens).cpu() * get_output_hop_size(vocoder_config)
    return wavs, wav_lens, sample_rate
