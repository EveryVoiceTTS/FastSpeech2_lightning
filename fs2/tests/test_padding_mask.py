"""Padding added to batch an item with longer ones must not change its outputs."""

import string
from functools import partial
from typing import Any

import pytest
import torch
from everyvoice.config.text_config import Symbols, TextConfig
from everyvoice.tests.stubs import TEST_CONTACT
from torch.utils.data import DataLoader
from torchaudio.models import Conformer as TorchaudioConformer

from ..blocks import MaskedBatchNorm1d
from ..config import FastSpeech2Config
from ..conformer import Conformer
from ..dataset import FastSpeech2DataModule, FastSpeechDataset
from ..layers import PostNet
from ..model import FastSpeech2
from ..type_definitions_heavy import Stats, StatsInfo
from ..variance_adaptor import VariancePredictor

DIM = 8


def lengths_to_mask(lengths: list[int]) -> torch.Tensor:
    """(B, T) mask, True at valid positions"""
    return torch.arange(max(lengths)).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)


def pad_with_garbage(short: torch.Tensor, long: torch.Tensor) -> torch.Tensor:
    """Batch `short` (T1, D) with `long` (T2, D), T1 < T2, padding `short` with
    random values: padded positions of real batches are not zero either."""
    padding = torch.randn(long.shape[0] - short.shape[0], short.shape[1])
    return torch.stack([torch.cat([short, padding]), long])


def make_conformers(seed=0) -> tuple[TorchaudioConformer, Conformer]:
    kwargs: dict[str, Any] = dict(
        input_dim=DIM,
        num_heads=2,
        ffn_dim=16,
        num_layers=2,
        depthwise_conv_kernel_size=3,
    )
    torch.manual_seed(seed)
    conformer = TorchaudioConformer(**kwargs).eval()
    torch.manual_seed(seed)
    masked = Conformer(**kwargs).eval()
    return conformer, masked


def test_masked_conformer_is_a_drop_in_replacement():
    conformer, masked = make_conformers()
    # Same state_dict, so checkpoints load as before; and same initial weights
    # for a given seed, so seeded models are unchanged.
    ref_state = conformer.state_dict()
    state = masked.state_dict()
    assert state.keys() == ref_state.keys()
    for key, value in ref_state.items():
        assert torch.equal(state[key], value), key
    # Without padding, the outputs are the same.
    x = torch.randn(3, 7, DIM)
    lengths = torch.tensor([7, 7, 7])
    with torch.no_grad():
        torch.testing.assert_close(masked(x, lengths)[0], conformer(x, lengths)[0])


def test_masked_conformer_ignores_padding():
    conformer, masked = make_conformers()
    short, long = torch.randn(5, DIM), torch.randn(12, DIM)
    batch = pad_with_garbage(short, long)
    lengths = torch.tensor([5, 12])
    with torch.no_grad():
        alone = masked(short.unsqueeze(0), torch.tensor([5]))[0][0]
        batched = masked(batch, lengths)[0][0, :5]
        torch.testing.assert_close(batched, alone)
        # torchaudio's Conformer leaks padding, which is why we need ours
        ref_alone = conformer(short.unsqueeze(0), torch.tensor([5]))[0][0]
        ref_batched = conformer(batch, lengths)[0][0, :5]
        assert not torch.allclose(ref_batched, ref_alone)


def test_masked_conformer_training_ignores_padding():
    # In training mode, batch norm uses batch statistics: how much padding the
    # batch has must change neither the outputs nor the running statistics.
    short, long = torch.randn(5, DIM), torch.randn(12, DIM)
    lengths = torch.tensor([5, 12])
    batch = pad_with_garbage(short, long)
    extra_padding = torch.randn(2, 8, DIM)
    batches = [batch, torch.cat([batch, extra_padding], dim=1)]
    outputs, running_stats = [], []
    for batch in batches:
        _, masked = make_conformers()
        masked.train()
        with torch.no_grad():
            output = masked(batch, lengths)[0]
        outputs.append(torch.cat([output[0, :5], output[1, :12]]))
        norms = [layer.conv_module.sequential[3] for layer in masked.conformer_layers]
        running_stats.append(
            [norm.running_mean for norm in norms] + [norm.running_var for norm in norms]
        )
    torch.testing.assert_close(outputs[0], outputs[1])
    for first, second in zip(*running_stats):
        torch.testing.assert_close(first, second)


def test_variance_predictor_ignores_padding():
    torch.manual_seed(0)
    predictor = VariancePredictor(
        input_dim=DIM, n_layers=2, n_channels=DIM, kernel_size=3
    ).eval()
    short, long = torch.randn(5, DIM), torch.randn(12, DIM)
    with torch.no_grad():
        alone = predictor(short.unsqueeze(0), mask=lengths_to_mask([5]))[0]
        batched = predictor(
            pad_with_garbage(short, long), mask=lengths_to_mask([5, 12])
        )
    torch.testing.assert_close(batched[0, :5], alone)
    assert torch.all(batched[0, 5:] == 0)


def test_postnet_ignores_padding():
    torch.manual_seed(0)
    postnet = PostNet(n_mel_channels=4, postnet_embedding_dim=DIM).eval()
    short, long = torch.randn(5, 4), torch.randn(12, 4)
    with torch.no_grad():
        alone = postnet(short.unsqueeze(0), mask=lengths_to_mask([5]))[0]
        batched = postnet(pad_with_garbage(short, long), mask=lengths_to_mask([5, 12]))
        torch.testing.assert_close(batched[0, :5], alone)
        # without a mask, the padding leaks into the valid frames
        unmasked = postnet(pad_with_garbage(short, long))
        assert not torch.allclose(unmasked[0, :5], alone)


def test_masked_batch_norm_ignores_padding():
    short, long = torch.randn(5, DIM) * 3 + 1, torch.randn(12, DIM)
    batch = pad_with_garbage(short, long).transpose(1, 2)  # (B, C, T)
    mask = lengths_to_mask([5, 12])
    valid = torch.cat([short, long]).T.unsqueeze(0)  # (1, C, 5 + 12)

    reference = torch.nn.BatchNorm1d(DIM).train()
    masked = MaskedBatchNorm1d(DIM).train()
    expected = reference(valid)
    output = masked(batch, mask)
    # Same normalization and running stats as BatchNorm1d on the valid
    # positions alone, whatever the padding holds.
    torch.testing.assert_close(output[0, :, :5], expected[0, :, :5])
    torch.testing.assert_close(output[1], expected[0, :, 5:])
    torch.testing.assert_close(masked.running_mean, reference.running_mean)
    torch.testing.assert_close(masked.running_var, reference.running_var)
    # Without a mask, it's BatchNorm1d.
    torch.testing.assert_close(masked(batch), reference(batch))


@pytest.fixture
def model() -> FastSpeech2:
    torch.manual_seed(0)
    config = FastSpeech2Config(
        contact=TEST_CONTACT,
        text=TextConfig(symbols=Symbols(letters=list(string.ascii_lowercase))),
    )
    stats = StatsInfo(min=0.1, max=10.0, std=2.0, mean=0.5, norm_max=1.0, norm_min=0.1)
    model = FastSpeech2(
        config=config,
        lang2id={"default": 0},
        speaker2id={"default": 0},
        stats=Stats(pitch=stats, energy=stats),
    ).eval()
    # The untrained model predicts ~0 frames per token; shift its log durations
    # so it predicts a few frames per token and the decoder has some input.
    with torch.no_grad():
        model.variance_adaptor.duration_predictor.linear.bias.fill_(1.5)
    return model


def synthesize(model: FastSpeech2, texts: list[str]) -> tuple[dict, dict]:
    data = [
        {
            "basename": f"utt{i}",
            "characters": text,
            "language": "default",
            "speaker": "default",
            "duration_control": 1.0,
            "is_last_input_chunk": True,
        }
        for i, text in enumerate(texts)
    ]
    dataset = FastSpeechDataset(
        data, model.config, model.lang2id, model.speaker2id, inference=True
    )
    collate = partial(
        FastSpeech2DataModule.collate_method,
        learn_alignment=model.config.model.learn_alignment,
    )
    batch = next(iter(DataLoader(dataset, batch_size=len(texts), collate_fn=collate)))
    with torch.no_grad():
        return model(batch, inference=True), batch


def test_model_output_does_not_depend_on_batch_padding(model):
    short = "hello there"
    long = "this is a much longer sentence so the short one gets lots of padding"
    alone, batch = synthesize(model, [short])
    batched, _ = synthesize(model, [short, long])
    n_tokens = int(batch["src_lens"][0])
    assert n_tokens == len(short)
    torch.testing.assert_close(
        batched["duration_prediction"][0, :n_tokens],
        alone["duration_prediction"][0],
    )
    n_frames = alone["postnet_output"].shape[1]
    assert int(batched["tgt_mask"][0].sum()) == n_frames
    torch.testing.assert_close(
        batched["postnet_output"][0, :n_frames],
        alone["postnet_output"][0],
        atol=1e-4,
        rtol=1e-4,
    )
