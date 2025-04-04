"""
This file turns the package into a module that can be run as a script with
    python -m fs2 ...
or, for coverage analysis, with
    coverage run -m fs2 ...
"""

from .cli import app

app()
