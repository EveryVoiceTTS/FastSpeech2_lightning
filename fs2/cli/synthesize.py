import sys
import textwrap
from pathlib import Path
from typing import Any, Optional

import typer
from everyvoice.base_cli.interfaces import complete_path
from everyvoice.config.preprocessing_config import DatasetTextRepresentation
from everyvoice.config.shared_types import TargetTrainingTextRepresentationLevel
from loguru import logger

from ..type_definitions import SynthesizeOutputFormats


def validate_data_keys_with_model_keys(
    data_keys: set[str], model_keys: set[str], key: str, multi: bool
) -> None:
    """
    Make sure the user-supplied language or speaker specification is compatible with the model.
    In current version of model, we can validate either key="language" or key="speaker"
    Args:
        data_keys: values for key found in the data
        model_keys: values for key found in the model
        key: "language" or "speaker"
        multi: whether the model was trained in multilingual/speaker mode
    """
    if multi:
        if None in data_keys:
            print(
                f"You have not specified a {key} for all your sentences."
                f" Available values are {model_keys}",
                file=sys.stderr,
            )
            sys.exit(1)

        extras = data_keys.difference(model_keys)
        if extras:
            is_or_are_not = (
                f"are not {key}s that are" if len(data_keys) > 1 else f"is not a {key}"
            )
            print(
                f"You provided {data_keys} which {is_or_are_not} supported by the model {model_keys or {}}.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        # NOTE: Even in non multiX, the model has a default value.
        # Looking at a filelist.psv
        # basename|characters|language|speaker
        # LJ002-0234|In the yard behind the prison|und|default
        # TODO: Instead, should we check that `data_keys == model_keys`?
        extras = data_keys.difference(model_keys | {None})
        if extras:
            print(
                f"The current model doesn't support multiple {key}s but your data has {key}s {extras}.\n"
                f"Please retrain your model with multi{'lingual' if key == 'language' else key} set to True.",
                file=sys.stderr,
            )
            sys.exit(1)


def prepare_data(
    texts: list[str],
    language: str | None,
    speaker: str | None,
    filelist: Path,
    # model is of type ..model.FastSpeech2, but we make it Any to keep the CLI
    # fast and enable mocking in unit testing.
    model: Any,
    text_type: DatasetTextRepresentation,
) -> list[dict[str, Any]]:
    """"""
    from everyvoice.utils import slugify

    data: list[dict[str, Any]]
    # NOTE: The wizard adds a default speaker=`default` to the data.
    # It also asks for a language that the user choses from a list which then becomes the default lanaguage, like `und`.
    # Knowing this, model.*2id should always have a default value thus DEFAULT_* should never be `None`.
    DEFAULT_LANGUAGE = next(iter(model.lang2id.keys()), None)
    DEFAULT_SPEAKER = next(iter(model.speaker2id.keys()), None)
    if texts:
        print(f"Processing text {texts}", file=sys.stderr)
        data = [
            {
                "basename": slugify(text),
                text_type.value: text,
                "language": language or DEFAULT_LANGUAGE,
                "speaker": speaker or DEFAULT_SPEAKER,
            }
            for text in texts
        ]
    else:
        data = model.config.training.filelist_loader(filelist)
        try:
            data = [
                dict(
                    d,
                    **{
                        "basename": slugify(
                            d.get("basename", d[text_type.value]),
                            limit_to_n_characters=30,
                        ),  # if 'basename' doesn't exist, create a basename from the first 30 chars of the text slug
                        "language": language or d.get("language", DEFAULT_LANGUAGE),
                        "speaker": speaker or d.get("speaker", DEFAULT_SPEAKER),
                    },
                )
                for d in data
            ]
        except KeyError:
            # TODO: Errors should have better formatting:
            #       https://github.com/roedoejet/FastSpeech2_lightning/issues/26
            logger.info(
                textwrap.dedent(
                    """
                EveryVoice only accepts filelists in PSV format as in:

                    basename|characters|language|speaker
                    LJ0001|Hello|eng|LJ

                Or in a format where each new line is an utterance:

                    This is a sentence.
                    Here is another sentence.

                Your filelist did not contain the correct keys so we will assume it is in the plain text format.
                Text can either be defined as 'characters' or 'phones'.
                        """
                )
            )
            with open(filelist, encoding="utf8") as f:
                data = [
                    {
                        "basename": slugify(line.strip(), limit_to_n_characters=30),
                        text_type.value: line.strip(),
                        "language": language or DEFAULT_LANGUAGE,
                        "speaker": speaker or DEFAULT_SPEAKER,
                    }
                    for line in f
                ]

    validate_data_keys_with_model_keys(
        data_keys=set(d["language"] for d in data),
        model_keys=set(model.lang2id.keys()),
        key="language",
        multi=model.config.model.multilingual,
    )
    validate_data_keys_with_model_keys(
        data_keys=set(d["speaker"] for d in data),
        model_keys=set(model.speaker2id.keys()),
        key="speaker",
        multi=model.config.model.multispeaker,
    )

    return data


def get_global_step(model_path: Path) -> int:
    """
    Extract the `global_step` from a model.
    Note: we have to do this because the `global_step` gets reset to 0 when we call load_from_checkpoint().
    """
    import torch

    m = torch.load(model_path, map_location=torch.device("cpu"))
    return m["global_step"]


def synthesize(  # noqa: C901
    model_path: Path = typer.Argument(
        ...,
        file_okay=True,
        exists=True,
        dir_okay=False,
        help="The path to a trained text-to-spec or e2e EveryVoice model.",
        autocompletion=complete_path,
    ),
    output_dir: Path = typer.Option(
        "synthesis_output",
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="The directory where your synthesized audio should be written",
        autocompletion=complete_path,
    ),
    texts: list[str] = typer.Option(
        [],
        "--text",
        "-t",
        help="Some text to synthesize.  This option can be repeated to synthesize multiple sentences."
        " It is recommended to use --filelist if you want to synthesize a lot of sentences or have different speaker/language per sentence.",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help="Specify which language to use in a multilingual system. [requires --text]",
    ),
    speaker: Optional[str] = typer.Option(
        None,
        "--speaker",
        "-s",
        help="Specify which speaker to use in a multispeaker system. [requires --text]",
    ),
    accelerator: str = typer.Option("auto", "--accelerator", "-a"),
    devices: str = typer.Option(
        "auto", "--devices", "-d", help="The number of GPUs to use"
    ),
    filelist: Path = typer.Option(
        None,
        "--filelist",
        "-f",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Synthesize all audio in a given filelist. Use --text if you want to just synthesize one sample.",
        autocompletion=complete_path,
    ),
    text_representation: DatasetTextRepresentation = typer.Option(
        DatasetTextRepresentation.characters,
        help="The representation of the text you are synthesizing. Can be either 'characters', 'phones', or 'arpabet'. The input type must be compatible with your model.",
    ),
    output_type: list[SynthesizeOutputFormats] = typer.Option(
        [SynthesizeOutputFormats.wav.value],
        "-O",
        "--output-type",
        help="Which format to synthesize to. **wav** is the default and will synthesize to a playable audio file. **npy** will generate spectrograms required to fine-tune [HiFiGAN](https://github.com/jik876/hifi-gan) (Mel-band oriented tensors, K, T). **pt** will generate predicted Mel spectrograms in the EveryVoice format (time-oriented Tensors, T, K)",
    ),
    vocoder_path: Path = typer.Option(
        None,
        "--vocoder-path",
        "-v",
        help="The path to a trained vocoder in case one was not specified in your model configuration.",
        dir_okay=False,
        file_okay=True,
        autocompletion=complete_path,
    ),
    batch_size: int = typer.Option(
        4,
        "--batch-size",
        "-b",
        help="Batch size.",
    ),
    num_workers: int = typer.Option(
        4,
        "--num-workers",
        "-n",
        help="Number of workers to process the data.",
    ),
):
    """Given some text and a trained model, generate some audio. i.e. perform typical speech synthesis"""
    # TODO: allow for changing of language/speaker and variance control
    import torch
    from everyvoice.text.phonemizer import AVAILABLE_G2P_ENGINES

    from ..model import FastSpeech2
    from ..synthesize_text_dataset import SynthesizeTextDataSet

    if texts and filelist:
        print(
            "Got arguments for both text and a filelist - this will only process the text."
            " Please re-run without providing text if you want to run batch synthesis on the provided file.",
            file=sys.stderr,
        )
    if not texts and not filelist:
        print("You must define either --text or --filelist", file=sys.stderr)
        sys.exit(1)

    # output to .wav will require a valid spec-to-wav model
    if SynthesizeOutputFormats.wav in output_type and not vocoder_path:
        print(
            "Missing --vocoder-path option, which is required when the output type is 'wav'.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load checkpoints
    print(f"Loading checkpoint from {model_path}", file=sys.stderr)
    model: FastSpeech2 = FastSpeech2.load_from_checkpoint(model_path).to(device)
    model.eval()

    if SynthesizeOutputFormats.wav in output_type:
        model.config.training.vocoder_path = vocoder_path

    if (
        model.config.model.target_text_representation_level
        == TargetTrainingTextRepresentationLevel.characters
        and text_representation != DatasetTextRepresentation.characters
    ):
        raise ValueError(
            f"Your model was trained on {model.config.model.target_text_representation_level} but you provided {text_representation.value} which is incompatible."
        )
    if (
        model.config.model.target_text_representation_level
        != TargetTrainingTextRepresentationLevel.characters
        and text_representation == DatasetTextRepresentation.characters
        and language not in AVAILABLE_G2P_ENGINES
    ):
        raise ValueError(
            f"Your model was trained on {model.config.model.target_text_representation_level} but you provided {text_representation.value} and there is no available grapheme-to-phoneme engine available for {language}. Please see <TODO: Docs!> for more information on how to add one."
        )

    data = prepare_data(
        texts=texts,
        language=language,
        speaker=speaker,
        filelist=filelist,
        model=model,
        text_type=text_representation,
    )

    dataset = SynthesizeTextDataSet(
        data,
        config=model.config,
        lang2id=model.lang2id,
        speaker2id=model.speaker2id,
        target_text_representation_level=model.config.model.target_text_representation_level,
    )

    from pytorch_lightning import Trainer

    from ..prediction_writing_callback import get_synthesis_output_callbacks

    trainer = Trainer(
        logger=False,  # We don't need to log things to tensorboard during inference
        accelerator=accelerator,
        devices=devices,
        max_epochs=model.config.training.max_epochs,
        callbacks=get_synthesis_output_callbacks(
            output_type=output_type,
            output_dir=output_dir,
            config=model.config,
            output_key=model.output_key,
            device=device,
            global_step=get_global_step(model_path),
        ),
    )

    from torch.utils.data import DataLoader

    from ..synthesize_text_dataset import collator

    trainer.predict(
        model,
        DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collator,
        ),
    )
