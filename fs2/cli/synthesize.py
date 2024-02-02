import sys
import textwrap
from pathlib import Path
from typing import Any, Optional

import typer
from loguru import logger

from ..type_definitions import LookupTable, SynthesizeOutputFormats

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="""Given some text and a trained model, generate some audio. i.e. perform typical speech synthesis""",
)


def validate_data_keys_with_model_keys(
    data_keys: set[str], model_keys: set[str], key: str
) -> None:
    """
    Make sure the user-supplied language or speaker specification is compatible with the model.
    In current version of model, we can validate either key="language" or key="speaker"
    """
    if key not in ["language", "speaker"]:
        raise ValueError("We can currently only validate language and speaker.")

    if None in data_keys:
        print(
            f"You have not specified a {key} for all your sentences."
            f" Available values are {model_keys}",
            file=sys.stderr,
        )
        sys.exit(1)

    extra = data_keys.difference(model_keys)
    if len(extra) > 0:
        print(
            f"You provided {data_keys} which are not {key}s that are supported by the model {model_keys or {}}."
            if len(data_keys) > 1
            else f"You provided {data_keys} which is not a {key} supported by the model {model_keys or {}}.",
            file=sys.stderr,
        )
        sys.exit(1)


def prepare_data(
    texts: list[str],
    language: str | None,
    speaker: str | None,
    model_lang2id: LookupTable,
    model_speaker2id: LookupTable,
    filelist: Path,
    filelist_loader,
) -> list[dict[str, Any]]:
    """"""
    from everyvoice.utils import slugify

    data: list[dict[str, Any]]
    DEFAULT_LANGUAGE = next(iter(model_lang2id.keys()), None)
    DEFAULT_SPEAKER = next(iter(model_speaker2id.keys()), None)
    if texts:
        print(f"Processing text {texts}", file=sys.stderr)
        data = [
            {
                "basename": slugify(text),
                "text": text,
                "language": language or DEFAULT_LANGUAGE,
                "speaker": speaker or DEFAULT_SPEAKER,
            }
            for text in texts
        ]
    else:
        data = filelist_loader(filelist)
        try:
            data = [
                {
                    "basename": slugify(
                        d.get("basename", d["text"]), limit_to_n_characters=30
                    ),  # if 'basename' doesn't exist, create a basename from the first 30 chars of the text slug
                    "text": d["text"],
                    "language": language or d.get("language", DEFAULT_LANGUAGE),
                    "speaker": speaker or d.get("speaker", DEFAULT_SPEAKER),
                }
                for d in data
            ]
        except KeyError:
            # TODO: Errors should have better formatting:
            #       https://github.com/roedoejet/FastSpeech2_lightning/issues/26
            logger.info(
                textwrap.dedent(
                    """
                EveryVoice only accepts filelists in PSV format as in:

                    basename|text|language|speaker
                    LJ0001|Hello|eng|LJ

                Or in a format where each new line is an utterance:

                    This is a sentence.
                    Here is another sentence.

                Your filelist did not contain the correct keys so we will assume it is in the plain text format.
                        """
                )
            )
            with open(filelist, encoding="utf8") as f:
                data = [
                    {
                        "basename": slugify(line.strip(), limit_to_n_characters=30),
                        "text": line.strip(),
                        "language": language or DEFAULT_LANGUAGE,
                        "speaker": speaker or DEFAULT_SPEAKER,
                    }
                    for line in f
                ]

    validate_data_keys_with_model_keys(
        data_keys=set(d["language"] for d in data),
        model_keys=set(model_lang2id.keys()),
        key="language",
    )
    validate_data_keys_with_model_keys(
        data_keys=set(d["speaker"] for d in data),
        model_keys=set(model_speaker2id.keys()),
        key="speaker",
    )

    return data


@app.command()
def synthesize(  # noqa: C901
    model_path: Path = typer.Argument(
        ...,
        file_okay=True,
        exists=True,
        dir_okay=False,
        help="The path to a trained text-to-spec or e2e EveryVoice model.",
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
    from everyvoice.preprocessor import Preprocessor

    from ..model import FastSpeech2
    from ..synthesize_text_dataset import SynthesizeTextDataSet

    if model_path is None:
        # TODO A CLI Shouldn't not using logging to communicate with the user.
        print("Model path is required.", file=sys.stderr)
        sys.exit(1)

    if texts and filelist:
        print(
            "Got arguments for both text and a filelist - this will only process the text."
            " Please re-run without providing text if you want to run batch synthesis on the provided file.",
            file=sys.stderr,
        )
    if not texts and not filelist:
        print("You must define either --text or --filelist", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load checkpoints
    print(f"Loading checkpoint from {model_path}", file=sys.stderr)
    model: FastSpeech2 = FastSpeech2.load_from_checkpoint(model_path).to(device)
    model.eval()
    # output to .wav will require a valid spec-to-wav model
    if SynthesizeOutputFormats.wav in output_type:
        if vocoder_path:
            model.config.training.vocoder_path = vocoder_path
        if not model.config.training.vocoder_path:
            print(
                "Sorry, no vocoder was provided, please add it to model.config.training.vocoder_path or as --vocoder-path /path/to/vocoder in the command line",
                file=sys.stderr,
            )
            sys.exit(1)

    data = prepare_data(
        texts=texts,
        language=language,
        speaker=speaker,
        model_lang2id=model.lang2id,
        model_speaker2id=model.speaker2id,
        filelist=filelist,
        filelist_loader=model.config.training.filelist_loader,
    )

    dataset = SynthesizeTextDataSet(
        data,
        preprocessor=Preprocessor(model.config),
        lang2id=model.lang2id,
        speaker2id=model.speaker2id,
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
