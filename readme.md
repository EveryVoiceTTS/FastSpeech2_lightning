# FastSpeech2 written in PyTorch Lightning

<!-- [![codecov](https://codecov.io/gh/roedoejet/g2p/branch/master/graph/badge.svg)](https://codecov.io/gh/roedoejet/g2p) -->
<!-- [![Build Status](https://github.com/roedoejet/g2p/actions/workflows/tests.yml/badge.svg)](https://github.com/roedoejet/g2p/actions) -->
<!-- [![PyPI package](https://img.shields.io/pypi/v/hfgl.svg)](https://pypi.org/project/g2p/) -->
[![license](https://img.shields.io/badge/Licence-MIT-green)](LICENSE)
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/EveryVoiceTTS/FastSpeech2_lightning)

🚧 Under Construction! This repo is not expected to work fully. Please check back later for a stable release. 🚧

> A PyTorch Lightning implementation of the FastSpeech2 Text-to-Speech Feature Prediction Model

This library is for training [FastSpeech2](https://arxiv.org/abs/2006.04558) / [FastPitch](https://arxiv.org/pdf/2006.06873.pdf) style feature prediction networks for speech synthesis. This implementation is one part of the [Speech Generation for Indigenous Language Education (SGILE)](#background) Project.

This repo has been separated in case you would like to use it separately from the broader SGILE system, but if you are looking to build speech synthesis systems from scratch, please visit [the main repository](https://github.com/EveryVoiceTTS/EveryVoice)

## Table of Contents
- FastSpeech2
  - [Table of Contents](#table-of-contents)
  - [Background](#background)
  - [Install](#install)
  - [Usage](#usage)
  <!-- - [How to Cite](#citation)
  - [License](#license) -->

See also:
  - [SGILE HiFiGAN](https://github.com/EveryVoiceTTS/HiFiGAN_iSTFT_lightning)
  - [Requirements and Motivations of Low-Resource Speech Synthesis for Language Revitalization](https://aclanthology.org/2022.acl-long.507/)

## Background

There are approximately 70 Indigenous languages spoken in Canada from 10 distinct language families.  As a consequence of the residential school system and other policies of cultural suppression, the majority of these languages now have fewer than 500 fluent speakers remaining, most of them elderly.

Despite this, Indigenous people have resisted colonial policies and continued speaking their languages, with interest by students and parents in Indigenous language education continuing to grow. Teachers are often overwhelmed by the number of students, and the trend towards online education means many students who have not previously had access to language classes now do. Supporting these growing cohorts of students comes with unique challenges in languages with few fluent first-language speakers. Teachers are particularly concerned with providing their students with opportunities to hear the language outside of class.

While there is no replacement for a speaker of an Indigenous language, there are possible applications for speech synthesis (text-to-speech) to supplement existing text-based tools like verb conjugators, dictionaries and phrasebooks.

The National Research Council has partnered with the Onkwawenna Kentyohkwa Kanyen’kéha immersion school, W̱SÁNEĆ School Board, University nuhelot’įne thaiyots’į nistameyimâkanak Blue Quills, and the University of Edinburgh to research and develop state-of-the-art speech synthesis (text-to-speech) systems and techniques for Indigenous languages in Canada, with a focus on how to integrate text-to-speech technology into the classroom.

This

## Installation

Clone clone the repo and pip install it locally:

```sh
$ git clone https://github.com/EveryVoiceTTS/FastSpeech2_lightning.git
$ cd FastSpeech2_lightning
$ pip install -e .
```

## Usage

### Configuration

You can change the base configuration in `fs2/config/base.yaml`.

You can also create a new config.yaml file and add it to the `CONFIGS` object in `fs2/config/__init__.py` and then use that key.

For example if you created a new config file at `myconfig.yaml` then you would update the `CONFIGS` object like so:

```python
CONFIGS: Dict[str, Path] = {
    "base": Path(__file__).parent / "base.yaml",
    "myconfig": Path(__file__).parent / "myconfig.yaml",
}
```

You can then use the `myconfig` config with any of the following commands like `fs2l train myconfig` or `fs2l preprocess myconfig -d mel -d audio -d text -d pitch -d energy` etc...

### Preprocessing

Preprocess by running: `fs2l preprocess myconfig -d mel -d audio -d text -d pitch -d energy -d duration` to generate the preprocessed data required for the model using the base configuration.

### Training

Train by running `fs2l train base` to use the base configuration.

You can pass updates to the configuration through the command line like so:

`fs2l train base --config preprocessing.save_dir=/my/new/path --config training.batch_size=16`

### Synthesis

Coming...


## Contributing

Feel free to dive in!
 - [Open an issue](https://github.com/EveryVoiceTTS/EveryVoice/issues/new) in the main EveryVoice repo with the tag `[FastSpeech2]`,
 - submit PRs to this repo with a corresponding submodule update PR to [EveryVoice](https://github.com/EveryVoiceTTS/EveryVoice).

This repo follows the [Contributor Covenant](http://contributor-covenant.org/version/1/3/0/) Code of Conduct.

You can install our standard Git hooks by running these commands in your sandbox:

```sh
pip install -r requirements.dev.txt
pre-commit install
gitlint install-hook
```

Have a look at [Contributing.md](https://github.com/EveryVoiceTTS/EveryVoice/blob/main/Contributing.md)
for the full details on the Conventional Commit messages we prefer, our code
formatting conventions, and our Git hooks.

You can then interactively install the package by running the following command from the project root:

```sh
pip install -e .
```


## Acknowledgements

This project is only possible because of the many open source implementations available. This repo draws heavily from, and is grateful to the work of [Chung-Ming Chien](https://github.com/ming024/FastSpeech2), [Christoph Minixhofer](https://github.com/MiniXC/LightningFastSpeech2), [Keon Lee](https://github.com/keonlee9420/Comprehensive-Transformer-TTS), and the [Nvidia FastPitch Implementation](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch). Please cite their work and star their repositories.
