from enum import Enum


class SynthesisOutputs(str, Enum):
    wav = "wav"
    npy = "npy"
    pt = "pt"
