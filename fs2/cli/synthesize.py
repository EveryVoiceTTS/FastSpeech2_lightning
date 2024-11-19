import sys
import textwrap
from pathlib import Path
from typing import Any, Optional

import typer
from everyvoice.base_cli.interfaces import complete_path
from everyvoice.config.type_definitions import (
    DatasetTextRepresentation,
    TargetTrainingTextRepresentationLevel,
)
from everyvoice.utils import spinner
from loguru import logger

from ..type_definitions import SynthesizeOutputFormats
from ..utils import truncate_basename


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
    text_representation: DatasetTextRepresentation,
    duration_control: float,
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
                "basename": truncate_basename(slugify(text)),
                text_representation.value: text,
                "language": language or DEFAULT_LANGUAGE,
                "speaker": speaker or DEFAULT_SPEAKER,
            }
            for text in texts
        ]
    else:
        data = model.config.training.filelist_loader(filelist)
        try:
            data = [
                d
                | {
                    "basename": d.get(
                        "basename",
                        truncate_basename(slugify(d[text_representation.value])),
                    ),  # Only truncate the basename if the basename doesn't already exist in the filelist.
                    "language": language or d.get("language", DEFAULT_LANGUAGE),
                    "speaker": speaker or d.get("speaker", DEFAULT_SPEAKER),
                }
                for d in data
            ]
        except KeyError:
            # TODO: Errors should have better formatting:
            #       https://github.com/EveryVoiceTTS/FastSpeech2_lightning/issues/26
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
                        "basename": truncate_basename(slugify(line.strip())),
                        text_representation.value: line.strip(),
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

    # Add duration_control
    for item in data:
        item["duration_control"] = duration_control

    return data


def get_global_step(model_path: Path) -> int:
    """
    Extract the `global_step` from a model.
    Note: we have to do this because the `global_step` gets reset to 0 when we call load_from_checkpoint().
    """
    import torch

    m = torch.load(model_path, map_location=torch.device("cpu"))
    return m["global_step"]


def synthesize_helper(
    model,
    texts: list[str],
    language: Optional[str],
    speaker: Optional[str],
    duration_control: Optional[float],
    global_step: int,
    output_type: list[SynthesizeOutputFormats],
    text_representation: DatasetTextRepresentation,
    accelerator: str,
    devices: str,
    device,
    batch_size: int,
    num_workers: int,
    filelist: Path,
    output_dir: Path,
    teacher_forcing_directory: Path,
    vocoder_global_step: Optional[int] = None,
    vocoder_model=None,
    vocoder_config=None,
):
    """This is a helper to perform synthesis once the model has been loaded.
    It allows us to use the same command for synthesis via the CLI and
    via the gradio demo.
    """
    from everyvoice.text.phonemizer import AVAILABLE_G2P_ENGINES

    from ..dataset import FastSpeech2SynthesisDataModule

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
        duration_control=duration_control if duration_control else 1.0,
        filelist=filelist,
        model=model,
        text_representation=text_representation,
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
            global_step=global_step,
            vocoder_model=vocoder_model,
            vocoder_config=vocoder_config,
            vocoder_global_step=vocoder_global_step,
        ),
    )
    if teacher_forcing_directory is not None:
        teacher_forcing = True
        model.config.preprocessing.save_dir = teacher_forcing_directory
    else:
        teacher_forcing = False
    # overwrite batch_size and num_workers
    model.config.training.batch_size = batch_size
    model.config.training.train_data_workers = num_workers
    return (
        model.config,
        device,
        trainer.predict(
            model,
            FastSpeech2SynthesisDataModule(
                model.config,
                data,
                model.lang2id,
                model.speaker2id,
                teacher_forcing=teacher_forcing,
            ),
            return_predictions=True,
        ),
    )


def synthesize(  # noqa: C901
    model_path: Path = typer.Argument(
        ...,
        file_okay=True,
        exists=True,
        dir_okay=False,
        help="The path to a trained text-to-spec (i.e., feature prediction) or e2e EveryVoice model.",
        shell_complete=complete_path,
    ),
    output_dir: Path = typer.Option(
        "synthesis_output",
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="The directory where your synthesized audio should be written",
        shell_complete=complete_path,
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
    duration_control: Optional[float] = typer.Option(
        1.0,
        "--duration-control",
        "-D",
        help="Control the speaking rate of the synthesis. Set a value to multily the durations by, lower numbers produce quicker speaking rates, larger numbers produce slower speaking rates. Default is 1.0",
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
        help="The path to a file containing a list of utterances (a.k.a filelist). Use --text if you want to just synthesize one sample.",
        shell_complete=complete_path,
    ),
    text_representation: DatasetTextRepresentation = typer.Option(
        DatasetTextRepresentation.characters,
        help="The representation of the text you are synthesizing. Can be either 'characters', 'phones', or 'arpabet'. The input type must be compatible with your model.",
    ),
    output_type: list[SynthesizeOutputFormats] = typer.Option(
        [SynthesizeOutputFormats.wav.value],
        "-O",
        "--output-type",
        help="""Which format(s) to synthesize to.
        Multiple formats can be provided by repeating `--output-type`.
        **wav** is the default and will synthesize to a playable audio file;
        **spec** will generate predicted Mel spectrograms. Tensors are time-oriented (T, K) where T is equal to the number of frames and K is equal to the number of Mel bands.
        **textgrid** will generate a Praat TextGrid with alignment labels. This can be helpful for evaluation.
        """,
    ),
    teacher_forcing_directory: Path = typer.Option(
        None,
        "--teacher-forcing-directory",
        "-T",
        help="ADVANCED. The path to preprocessed folder containing spec and duration folders to use for teacher-forcing the synthesized outputs.",
        dir_okay=True,
        file_okay=False,
        shell_complete=complete_path,
    ),
    vocoder_path: Path = typer.Option(
        None,
        "--vocoder-path",
        "-v",
        help="The path to a trained vocoder (aka spec-to-wav model).",
        dir_okay=False,
        file_okay=True,
        shell_complete=complete_path,
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

    # Do argument error checking before doing expensive imports
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

    with spinner():
        import torch
        from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import (
            load_hifigan_from_checkpoint,
        )
        from everyvoice.utils.heavy import get_device_from_accelerator

        from ..model import FastSpeech2

    output_dir.mkdir(exist_ok=True, parents=True)
    # NOTE: We want to be able to put the vocoder on the proper accelerator for
    # it to be compatible with the vocoder's input device.
    # We could misuse the trainer's API and use the private variable
    # trainer._accelerator_connector._accelerator_flag but that value is
    # computed when instantiating a trainer and that is exactly when we need
    # the information to create the callbacks.
    device = get_device_from_accelerator(accelerator)

    # Load checkpoints
    print(f"Loading checkpoint from {model_path}", file=sys.stderr)

    from pydantic import ValidationError

    try:
        model: FastSpeech2 = FastSpeech2.load_from_checkpoint(model_path).to(device)  # type: ignore
    except (TypeError, ValidationError) as e:
        logger.error(f"Unable to load {model_path}: {e}")
        sys.exit(1)
    model.eval()

    # get global step
    # We can't just use model.global_step because it gets reset by lightning
    global_step = get_global_step(model_path)

    # load vocoder
    logger.info(f"Loading Vocoder from {vocoder_path}")
    if vocoder_path is not None:
        vocoder_ckpt = torch.load(vocoder_path, map_location=device)
        try:
            vocoder_model, vocoder_config = load_hifigan_from_checkpoint(
                vocoder_ckpt, device
            )
        except (TypeError, ValidationError) as e:
            logger.error(f"Unable to load {vocoder_path}: {e}")
            sys.exit(1)
        # We can't just use model.global_step because it gets reset by lightning
        vocoder_global_step = get_global_step(vocoder_path)
    else:
        vocoder_model = None
        vocoder_config = None
        vocoder_global_step = None
    return synthesize_helper(
        model=model,
        texts=texts,
        language=language,
        speaker=speaker,
        duration_control=duration_control,
        global_step=global_step,
        output_type=output_type,
        text_representation=text_representation,
        accelerator=accelerator,
        devices=devices,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        filelist=filelist,
        teacher_forcing_directory=teacher_forcing_directory,
        output_dir=output_dir,
        vocoder_model=vocoder_model,
        vocoder_config=vocoder_config,
        vocoder_global_step=vocoder_global_step,
    )
