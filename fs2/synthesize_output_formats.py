from enum import Enum


class SynthesizeOutputFormats(str, Enum):
    """Valid output formats for synthesize"""

    wav = "wav"
    npy = "npy"
    pt = "pt"
