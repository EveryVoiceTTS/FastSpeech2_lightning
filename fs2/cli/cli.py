import typer
from everyvoice.wizard import TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX

try:
    from fs2.cli.audit import audit as app_audit
    from fs2.cli.benchmark import benchmark as app_benchmark
    from fs2.cli.check_data import check_data as app_check_data
    from fs2.cli.preprocess import preprocess as app_preprocess
    from fs2.cli.synthesize import synthesize as app_synthesize
    from fs2.cli.train import train as app_train
except ImportError:
    from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.audit import (
        audit as app_audit,
    )
    from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.benchmark import (
        benchmark as app_benchmark,
    )
    from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.check_data import (
        check_data as app_check_data,
    )
    from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.preprocess import (
        preprocess as app_preprocess,
    )
    from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.synthesize import (
        synthesize as app_synthesize,
    )
    from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.train import (
        train as app_train,
    )

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="A PyTorch Lightning implementation of the FastSpeech2 Text-to-Speech Feature Prediction Model",
)

app.command(
    name="audit",
    short_help="",
)(app_audit)

app.command(
    name="benchmark",
    short_help="",
)(app_benchmark)

app.command(
    name="check_data",
    short_help="",
)(app_check_data)

app.command(
    name="preprocess",
    short_help="Preprocess your data",
    help=f"""
    # Preprocess Help

    This command will preprocess all of the data you need for use with EveryVoice.

    By default every step of the preprocessor will be done by running:
    \n\n
    **everyvoice preprocess config/{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml**
    \n\n
    If you only want to process specific things, you can run specific commands by adding them as options for example:
    \n\n
    **everyvoice preprocess config/{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml -s energy -s pitch**
    """,
)(app_preprocess)

app.command(
    name="synthesize",
    short_help="""Given some text and a trained model, generate some audio. i.e. perform typical speech synthesis""",
)(app_synthesize)

app.command(
    name="train",
    short_help="Train your Text-to-Spec model",
    help=f"""Train your text-to-spec model.  For example:

    **everyvoice train text-to-spec config/{TEXT_TO_SPEC_CONFIG_FILENAME_PREFIX}.yaml**
    """,
)(app_train)


if __name__ == "__main__":
    app()
