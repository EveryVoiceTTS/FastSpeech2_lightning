""" Setup for FastSpeech2
"""

import datetime as dt
from os import path

from setuptools import find_packages, setup

build_no = dt.datetime.now().strftime("%Y%m%d")

# Ugly hack to read the current version number without importing g2p:
# (works by )
VERSION = "0.0" + "." + build_no

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, "readme.md"), encoding="utf8") as f:
    long_description = f.read()

with open(path.join(this_directory, "requirements.txt"), encoding="utf8") as f:
    REQS = f.read().splitlines()

setup(
    name="fs2l",
    python_requires=">=3.10",
    version=VERSION,
    author="Aidan Pine",
    author_email="hello@aidanpine.ca",
    license="MIT",
    url="https://github.com/EveryVoiceTTS/FastSpeech2_lightning",
    description="Text-to-Speech Synthesis for the Speech Generation for Indigenous Language Education Small Teams Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    platform=["any"],
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQS,
    entry_points={"console_scripts": ["fs2l = fs2.cli:app"]},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
