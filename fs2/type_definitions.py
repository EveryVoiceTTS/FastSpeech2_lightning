"""
This file is for light-weight type definitions with no dependencies.
This should load in milliseconds.
Expensive type definitions, e.g., requiring pydantic, belong in type_definitions_heavy.py.
"""

from enum import Enum


class SynthesizeOutputFormats(str, Enum):
    """Valid output formats for synthesize"""

    wav = "wav"
    spec = "spec"
    textgrid = "textgrid"
