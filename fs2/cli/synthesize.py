import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import typer
from everyvoice.base_cli.interfaces import inference_base_command_interface
from everyvoice.config.type_definitions import (
    DatasetTextRepresentation,
    TargetTrainingTextRepresentationLevel,
)
from everyvoice.text.textsplit import chunk_text
from everyvoice.utils import spinner
from loguru import logger
from merge_args import merge_args
from tqdm import tqdm

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


def load_data_from_filelist(
    filelist: Path,
    # model is of type ..model.FastSpeech2, but we make it Any to keep the CLI
    # fast and enable mocking in unit testing.
    model: Any,
    text_representation: DatasetTextRepresentation,
    language: str | None = None,
    speaker: str | None = None,
    default_language: str | None = None,
    default_speaker: str | None = None,
    output_type: list[SynthesizeOutputFormats] = [],
):

    if default_language is None:
        default_language = next(iter(model.lang2id.keys()), None)
    if default_speaker is None:
        default_speaker = next(iter(model.speaker2id.keys()), None)

    from everyvoice.config.text_config import TextConfig
    from everyvoice.utils import slugify

    # TODO: Implement Text Splitting for TextGrid and Readalong files
    split_text: bool
    if (
        SynthesizeOutputFormats.textgrid in output_type
        or SynthesizeOutputFormats.readalong_html in output_type
        or SynthesizeOutputFormats.readalong_xml in output_type
    ):
        split_text = False
        logger.warning(
            "EveryVoice does not currently support text splitting for TextGrid or Readalong files. Config variable split_text has been set to False."
        )
    else:
        text_config: TextConfig = model.config.text
        split_text = text_config.split_text

    try:
        data = []
        for d in model.config.training.filelist_loader(filelist):
            # Chunk longer texts, for better longform audio synthesis
            text_line = d[text_representation.value]
            chunks = chunk_text(text_line) if split_text else [text_line]
            for i, chunk in enumerate(chunks):
                data.append(
                    {
                        "basename": d.get(
                            "basename",
                            truncate_basename(slugify(chunk)),
                        ),  # Only truncate the basename if the basename doesn't already exist in the filelist.
                        text_representation.value: chunk,
                        "language": language or d.get("language", default_language),
                        "speaker": speaker or d.get("speaker", default_speaker),
                        "is_last_input_chunk": (i == len(chunks) - 1),
                    }
                )

            print(f"Processing text: {chunks}", file=sys.stderr)

        if not data:
            # If there is no data, it means we had a one-line input file. Raise KeyError
            # so we enter the except block below and read it as a plain text file.
            raise KeyError
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
        data = []
        with open(filelist, encoding="utf8") as file:
            for line in file:
                # Chunk longer texts, for better longform audio synthesis
                chunks = chunk_text(line) if split_text else [line]
                for i, chunk in enumerate(chunks):
                    data.append(
                        {
                            "basename": truncate_basename(slugify(chunk.strip())),
                            text_representation.value: chunk.strip(),
                            "language": language or default_language,
                            "speaker": speaker or default_speaker,
                            "is_last_input_chunk": (i == len(chunks) - 1),
                        }
                    )
                print(f"Processing text: {chunks}", file=sys.stderr)
    return data


def prepare_data(
    texts: Optional[list[str]],
    language: str | None,
    speaker: str | None,
    filelist: Path,
    # model is of type ..model.FastSpeech2, but we make it Any to keep the CLI
    # fast and enable mocking in unit testing.
    model: Any,
    text_representation: DatasetTextRepresentation,
    duration_control: float,
    style_reference: Path | None,
    output_type: list[SynthesizeOutputFormats] = [],
) -> list[dict[str, Any]]:
    """"""
    from everyvoice.config.text_config import TextConfig
    from everyvoice.utils import slugify

    # TODO: Implement Text Splitting for TextGrid and Readalong files
    split_text: bool
    if (
        SynthesizeOutputFormats.textgrid in output_type
        or SynthesizeOutputFormats.readalong_html in output_type
        or SynthesizeOutputFormats.readalong_xml in output_type
    ):
        split_text = False
        logger.warning(
            "EveryVoice does not currently support text splitting for TextGrid or Readalong files. Config variable split_text has been set to False."
        )
    else:
        text_config: TextConfig = model.config.text
        split_text = text_config.split_text

    data: list[dict[str, Any]]
    # NOTE: The wizard adds a default speaker=`default` to the data.
    # It also asks for a language that the user choses from a list which then becomes the default lanaguage, like `und`.
    # Knowing this, model.*2id should always have a default value thus DEFAULT_* should never be `None`.
    DEFAULT_LANGUAGE = next(iter(model.lang2id.keys()), None)
    DEFAULT_SPEAKER = next(iter(model.speaker2id.keys()), None)
    if texts:
        data = []
        for text_input in texts:
            # Chunk longer texts, for better longform audio synthesis
            chunks = chunk_text(text_input) if split_text else [text_input]
            for i, chunk in enumerate(chunks):
                data.append(
                    {
                        "basename": truncate_basename(slugify(chunk)),
                        text_representation.value: chunk,
                        "language": language or DEFAULT_LANGUAGE,
                        "speaker": speaker or DEFAULT_SPEAKER,
                        "is_last_input_chunk": (
                            i == len(chunks) - 1
                        ),  # True if end of a text_input, False otherwise
                    }
                )
            print(f"Processing text: {chunks}", file=sys.stderr)
    else:
        data = load_data_from_filelist(
            filelist,
            model,
            text_representation,
            language,
            speaker,
            DEFAULT_LANGUAGE,
            DEFAULT_SPEAKER,
            output_type,
        )

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

    # We only allow a single style reference right now, so it's fine to load it once here.
    if style_reference:
        from everyvoice.utils.heavy import get_spectral_transform

        spectral_transform = get_spectral_transform(
            model.config.preprocessing.audio.spec_type,
            model.config.preprocessing.audio.n_fft,
            model.config.preprocessing.audio.fft_window_size,
            model.config.preprocessing.audio.fft_hop_size,
            f_min=model.config.preprocessing.audio.f_min,
            f_max=model.config.preprocessing.audio.f_max,
            sample_rate=model.config.preprocessing.audio.output_sampling_rate,
            n_mels=model.config.preprocessing.audio.n_mels,
        )
        import torchaudio

        style_reference_audio, style_reference_sr = torchaudio.load(style_reference)
        if style_reference_sr != model.config.preprocessing.audio.input_sampling_rate:
            style_reference_audio = torchaudio.functional.resample(
                style_reference_audio,
                style_reference_sr,
                model.config.preprocessing.audio.input_sampling_rate,
            )
        style_reference_spec = spectral_transform(style_reference_audio)
    # Add duration_control
    for item in data:
        item["duration_control"] = duration_control
        # Add style reference
        if style_reference:
            item["mel_style_reference"] = style_reference_spec

    return data


def get_global_step(model_path: Path) -> int:
    """
    Extract the `global_step` from a model.
    Note: we have to do this because the `global_step` gets reset to 0 when we call load_from_checkpoint().
    """
    import torch

    m = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    return m["global_step"]


def synthesize_helper(
    model,
    texts: Optional[list[str]],
    style_reference: Optional[Path],
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
    filelist_data: Optional[list[dict]],
    output_dir: Path,
    teacher_forcing_directory: Path,
    vocoder_global_step: Optional[int] = None,
    vocoder_model=None,
    vocoder_config=None,
    return_scores=False,
):
    """This is a helper to perform synthesis once the model has been loaded.
    It allows us to use the same command for synthesis via the CLI and
    via the gradio demo.
    """

    from ..dataset import FastSpeech2SynthesisDataModule

    if (
        model.config.model.target_text_representation_level
        == TargetTrainingTextRepresentationLevel.characters
        and text_representation != DatasetTextRepresentation.characters
    ):
        raise ValueError(
            f"Your model was trained on {model.config.model.target_text_representation_level} but you provided {text_representation.value} which is incompatible."
        )

    if filelist_data is None:
        data: list[dict[Any, Any]] = prepare_data(
            texts=texts,
            language=language,
            speaker=speaker,
            duration_control=duration_control if duration_control else 1.0,
            filelist=filelist,
            model=model,
            text_representation=text_representation,
            style_reference=style_reference,
            output_type=output_type,
        )
    else:
        data = filelist_data
    if return_scores:
        from nltk.util import ngrams

        token_counter: Counter = Counter()
        trigram_counter: Counter = Counter()
        for line in tqdm(
            data, desc="calculating filelist statistics for score calculation"
        ):
            tokens = line[f"{text_representation.value[:-1]}_tokens"].split("/")
            for t in tokens:
                token_counter[t] += 1
            tokens.insert(0, "<BOS>")
            tokens.append("<EOS>")
            for trigram in ngrams(tokens, 3):
                trigram_counter[trigram] += 1
        for line in tqdm(data, desc="scoring utterances"):
            tokens = line[f"{text_representation.value[:-1]}_tokens"].split("/")
            line["phone_coverage_score"] = sum((1 / token_counter[t]) for t in tokens)
            line["trigram_coverage_score"] = sum(
                (1 / trigram_counter[n]) for n in ngrams(tokens, 3)
            )

    from pytorch_lightning import Trainer

    from ..prediction_writing_callback import get_synthesis_output_callbacks

    callbacks = get_synthesis_output_callbacks(
        output_type=output_type,
        output_dir=output_dir,
        config=model.config,
        output_key=model.output_key,
        device=device,
        global_step=global_step,
        vocoder_model=vocoder_model,
        vocoder_config=vocoder_config,
        vocoder_global_step=vocoder_global_step,
        return_scores=return_scores,
    )
    trainer = Trainer(
        logger=False,  # We don't need to log things to tensorboard during inference
        accelerator=accelerator,
        devices=devices,
        max_epochs=model.config.training.max_epochs,
        callbacks=list(callbacks.values()),
    )
    if teacher_forcing_directory is not None:
        teacher_forcing = True
        model.config.preprocessing.save_dir = teacher_forcing_directory
    else:
        if return_scores:
            raise ValueError(
                "In order to return the scores, we also need access to the directory containing your ground truth audio and preprocessed data. Please pass this in using the --teacher-forcing-directory option. e.g. --teacher-forcing-directory ./preprocessed"
            )
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
                style_reference=style_reference is not None,
            ),
            return_predictions=True,
        ),
        callbacks,
    )


@merge_args(inference_base_command_interface)
def synthesize(  # noqa: C901
    model_path: Path = typer.Argument(
        ...,
        file_okay=True,
        exists=True,
        dir_okay=False,
        help="The path to a trained text-to-spec (i.e., feature prediction) or e2e EveryVoice model.",
    ),
    output_dir: Path = typer.Option(
        "synthesis_output",
        "--output-dir",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="The directory where your synthesized audio should be written",
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
    style_reference: Optional[Path] = typer.Option(
        None,
        "--style-reference",
        "-S",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="The path to an audio file containing a style reference. Your text-to-spec must have been trained with the global style token module to use this feature.",
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


        '**wav**' is the default and will synthesize to a playable audio file. Requires --vocoder-path.


        '**spec**' will generate predicted Mel spectrograms. Tensors are Mel band-oriented (K, T) where K is equal to the number of Mel bands and T is equal to the number of frames.


        '**textgrid**' will generate a Praat TextGrid with alignment labels. This can be helpful for evaluation.


        '**readalong-xml**' will generate a ReadAlong from the given text and synthesized audio in XML .readalong format (see https://github.com/ReadAlongs).


        '**readalong-html**' will generate a single file Offline HTML ReadAlong that can be further edited in the ReadAlong Studio Editor, and opened by itself. Also implies '--output-type wav'. Requires --vocoder-path.
        """,
    ),
    teacher_forcing_directory: Path = typer.Option(
        None,
        "--teacher-forcing-directory",
        "-T",
        help="ADVANCED. The path to preprocessed folder containing spec and duration folders to use for teacher-forcing the synthesized outputs.",
        dir_okay=True,
        file_okay=False,
    ),
    vocoder_path: Path = typer.Option(
        None,
        "--vocoder-path",
        "-v",
        help="The path to a trained vocoder (aka spec-to-wav model).",
        dir_okay=False,
        file_okay=True,
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
    **kwargs,
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
    if (
        SynthesizeOutputFormats.wav in output_type
        or SynthesizeOutputFormats.readalong_html in output_type
    ) and not vocoder_path:
        print(
            "Missing --vocoder-path option, which is required when the output type includes 'wav' or 'offline-ras'.",
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

    from everyvoice.base_cli.helpers import inference_base_command
    from pydantic import ValidationError

    try:
        model: FastSpeech2 = FastSpeech2.load_from_checkpoint(model_path).to(device)  # type: ignore
    except (TypeError, ValidationError) as e:
        logger.error(f"Unable to load {model_path}: {e}")
        sys.exit(1)
    model.eval()

    inference_base_command(model, **kwargs)

    # get global step
    # We can't just use model.global_step because it gets reset by lightning
    global_step = get_global_step(model_path)

    # load vocoder
    if vocoder_path is not None:
        logger.info(f"Loading Vocoder from {vocoder_path}")
        vocoder_ckpt = torch.load(vocoder_path, map_location=device, weights_only=True)
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

    synthesize_helper(
        model=model,
        texts=texts,
        style_reference=style_reference,
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
        filelist_data=None,
        teacher_forcing_directory=teacher_forcing_directory,
        output_dir=output_dir,
        vocoder_model=vocoder_model,
        vocoder_config=vocoder_config,
        vocoder_global_step=vocoder_global_step,
    )
