from enum import Enum
from pathlib import Path

import typer
from everyvoice.base_cli.interfaces import complete_path
from everyvoice.utils import spinner


class BenchmarkType(str, Enum):
    training = "training"
    inference = "inference"


def benchmark(
    config_file: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="The path to your model configuration file.",
        shell_complete=complete_path,
    ),
    benchmark_type: BenchmarkType = BenchmarkType.training,
    gpu: bool = True,
    warmup_reps: int = 10,
    repetitions: int = 300,
):
    import time
    from functools import partial

    with spinner():
        import numpy as np
        import torch

        from ..config import FastSpeech2Config
        from ..dataset import FastSpeech2DataModule
        from ..model import FastSpeech2

    config = FastSpeech2Config.load_config_from_path(config_file)
    loader = FastSpeech2DataModule(config)
    loader.prepare_data()
    batch = loader.collate_method(
        [loader.train_dataset[i] for i in range(config.training.batch_size)]
    )
    model = FastSpeech2(config=config, lang2id={}, speaker2id={})
    device = "cpu"
    if gpu:
        device = "cuda"
    model.to(device)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    if benchmark_type == BenchmarkType.training:
        benchmark_fn = model.forward
    else:
        benchmark_fn = partial(model.forward, inference=True)

    # INIT LOGGERS
    starter, ender = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(warmup_reps):
        _ = benchmark_fn(batch)
    # Forward
    for rep in range(repetitions):
        if gpu:
            starter.record()
            _ = benchmark_fn(batch)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
        else:
            t0 = time.time()
            _ = benchmark_fn(batch)
            t1 = time.time()
            curr_time = (t1 - t0) * 1000
            timings[rep] = curr_time
    print(
        f"Average forward pass for {benchmark_type.value} duration after {repetitions} repetitions: {np.sum(timings) / repetitions} ms Standard Deviation: {np.std(timings)}"
    )
