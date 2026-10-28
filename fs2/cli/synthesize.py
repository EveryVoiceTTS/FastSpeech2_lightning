from pathlib import Path
from typing import Annotated, Optional

import typer
from everyvoice.base_cli.interfaces import (
    inference_base_command_interface,
    typer_directory_option,
    typer_file_argument,
    typer_file_option,
)
from everyvoice.config.type_definitions import (
    DatasetTextRepresentation,
    SynthesizeOutputFormats,
)
from merge_args import merge_args


@merge_args(inference_base_command_interface)
def synthesize(  # noqa: C901
    model_path: Annotated[
        Path,
        typer_file_argument(
            help="The path to a trained text-to-spec (i.e., feature prediction) or e2e EveryVoice model.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer_directory_option(
            "--output-dir",
            "-o",
            exists=False,
            help="The directory where your synthesized audio should be written."
            " By default, filenames include the basename, speaker, language, and"
            " other metadata, e.g."
            " 'LJ050-0269--LJ--eng--ckpt=100000--v_ckpt=100000--pred.wav'."
            " Use --simple-filenames to write just 'LJ050-0269.wav' instead.",
        ),
    ] = Path("synthesis_output"),
    texts: Annotated[
        list[str],
        typer.Option(
            "--text",
            "-t",
            help="Some text to synthesize.  This option can be repeated to synthesize multiple sentences."
            " It is recommended to use --filelist if you want to synthesize a lot of sentences or have different speaker/language per sentence.",
        ),
    ] = [],
    language: Annotated[
        Optional[str],
        typer.Option(
            "--language",
            "-l",
            help="Specify which language to use in a multilingual system. [requires --text]",
        ),
    ] = None,
    duration_control: Annotated[
        Optional[float],
        typer.Option(
            "--duration-control",
            "-D",
            help="Control the speaking rate of the synthesis. Set a value to multily the durations by, lower numbers produce quicker speaking rates, larger numbers produce slower speaking rates. Default is 1.0",
        ),
    ] = 1.0,
    style_reference: Annotated[
        Path | None,
        typer_file_option(
            "--style-reference",
            "-S",
            help="The path to an audio file containing a style reference. Your text-to-spec must have been trained with the global style token module to use this feature.",
        ),
    ] = None,
    speaker: Annotated[
        Optional[str],
        typer.Option(
            "--speaker",
            "-s",
            help="Specify which speaker to use in a multispeaker system. [requires --text]",
        ),
    ] = None,
    accelerator: Annotated[str, typer.Option("--accelerator", "-a")] = "auto",
    devices: Annotated[
        str, typer.Option("--devices", "-d", help="The number of GPUs to use")
    ] = "auto",
    filelist: Annotated[
        Path | None,
        typer_file_option(
            "--filelist",
            "-f",
            help="The path to a file containing a list of utterances (a.k.a filelist). Use --text if you want to just synthesize one sample.",
        ),
    ] = None,
    text_representation: Annotated[
        DatasetTextRepresentation,
        typer.Option(
            help="The representation of the text you are synthesizing. Can be either 'characters', 'phones', or 'arpabet'. The input type must be compatible with your model.",
        ),
    ] = DatasetTextRepresentation.characters,
    output_type: Annotated[
        list[SynthesizeOutputFormats],
        typer.Option(
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
    ] = [SynthesizeOutputFormats.wav],
    teacher_forcing_directory: Annotated[
        Path | None,
        typer_directory_option(
            "--teacher-forcing-directory",
            "-T",
            help="ADVANCED. The path to preprocessed folder containing spec and duration folders to use for teacher-forcing the synthesized outputs.",
        ),
    ] = None,
    vocoder_path: Annotated[
        Path | None,
        typer_file_option(
            "--vocoder-path",
            "-v",
            help="The path to a trained vocoder (aka spec-to-wav model).",
        ),
    ] = None,
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="Batch size.")
    ] = 4,
    num_workers: Annotated[
        int,
        typer.Option(
            "--num-workers", "-n", help="Number of workers to process the data."
        ),
    ] = 4,
    simple_filenames: Annotated[
        bool,
        typer.Option(
            "--simple-filenames",
            help="Write output filenames as just the basename and extension"
            " (e.g. 'LJ050-0269.wav') instead of the default."
            " Only use this if your basenames are unique across speakers and"
            " languages in your filelist, otherwise outputs can overwrite each other.",
        ),
    ] = False,
    **kwargs,
):
    """Given some text and a trained model, generate some audio. i.e. perform typical speech synthesis"""
    # TODO: allow for changing of language/speaker and variance control

    # Do argument error checking before doing expensive imports
    if texts and filelist:
        raise typer.BadParameter(
            "Got arguments for both --text and --filelist, which are mutually exclusive."
        )
    if not texts and not filelist:
        raise typer.BadParameter("You must define either --text or --filelist")

    # output to .wav will require a valid spec-to-wav model
    if (
        SynthesizeOutputFormats.wav in output_type
        or SynthesizeOutputFormats.readalong_html in output_type
    ) and not vocoder_path:
        raise typer.BadParameter(
            "Missing --vocoder-path option, which is required when the output type includes 'wav' or 'offline-ras'."
        )

    from everyvoice.utils import spinner

    with spinner():
        import torch
        from everyvoice import logger
        from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import (
            load_hifigan_from_checkpoint,
        )
        from everyvoice.utils.heavy import get_device_from_accelerator

        from ..core.synthesize import get_global_step, synthesize_helper
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
    logger.info(f"Loading checkpoint from {model_path}")

    from everyvoice.base_cli.helpers import inference_base_command
    from pydantic import ValidationError

    try:
        model: FastSpeech2 = FastSpeech2.load_from_checkpoint(model_path).to(device)  # type: ignore
    except (TypeError, ValidationError) as e:
        raise typer.BadParameter(f"Unable to load {model_path}: {e}")
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
            raise typer.BadParameter(f"Unable to load {vocoder_path}: {e}")
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
        simple_filenames=simple_filenames,
    )
