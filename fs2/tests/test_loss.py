import pytest
import torch
from everyvoice.tests.stubs import TEST_CONTACT

from ..config import FastSpeech2Config
from ..loss import FastSpeech2Loss


@pytest.fixture
def loss_fn():
    """Test that padded frames don't leak into the FastSpeech2 loss."""
    config = FastSpeech2Config(contact=TEST_CONTACT)
    config.model.learn_alignment = False
    return FastSpeech2Loss(config)


def _make_batch(src_lens, tgt_lens, n_mels=4):
    max_src_len = max(src_lens)
    max_tgt_len = max(tgt_lens)
    src_mask = torch.arange(max_src_len).unsqueeze(0) < torch.tensor(
        src_lens
    ).unsqueeze(1)
    tgt_mask = torch.arange(max_tgt_len).unsqueeze(0) < torch.tensor(
        tgt_lens
    ).unsqueeze(1)
    batch_size = len(src_lens)
    output = {
        "duration_prediction": torch.randn(batch_size, max_src_len),
        "duration_target": torch.randint(1, 5, (batch_size, max_src_len)),
        "energy_target": torch.randn(batch_size, max_src_len) * src_mask,
        "pitch_target": torch.randn(batch_size, max_src_len) * src_mask,
        "energy_prediction": torch.randn(batch_size, max_src_len),
        "pitch_prediction": torch.randn(batch_size, max_src_len),
        "output": torch.randn(batch_size, max_tgt_len, n_mels),
        "postnet_output": torch.randn(batch_size, max_tgt_len, n_mels),
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
    }
    batch = {"mel": torch.randn(batch_size, max_tgt_len, n_mels)}
    return output, batch


def test_padded_frames_do_not_affect_loss(loss_fn):
    """Changing values in padded (masked-out) positions should not change the loss."""
    output, batch = _make_batch(src_lens=[5, 2], tgt_lens=[8, 3])

    losses_before = loss_fn(output, batch, current_epoch=1)

    # Corrupt every padded position in the predictions and targets
    src_pad = ~output["src_mask"]
    tgt_pad = ~output["tgt_mask"]
    output["pitch_prediction"][src_pad] = 1e6
    output["pitch_target"][src_pad] = -1e6
    output["energy_prediction"][src_pad] = 1e6
    output["energy_target"][src_pad] = -1e6
    output["duration_prediction"][src_pad] = 1e6
    output["duration_target"][src_pad] = 1
    output["output"][tgt_pad] = 1e6
    output["postnet_output"][tgt_pad] = 1e6
    batch["mel"][tgt_pad] = -1e6

    losses_after = loss_fn(output, batch, current_epoch=1)

    for key in losses_before:
        assert losses_before[key].item() == pytest.approx(
            losses_after[key].item(), abs=1e-4
        )


def test_loss_matches_manual_mask_select(loss_fn):
    """The mel loss should equal MSE computed only over the valid (unpadded) frames."""
    output, batch = _make_batch(src_lens=[5, 2], tgt_lens=[8, 3])

    losses = loss_fn(output, batch, current_epoch=1)

    tgt_mask = output["tgt_mask"].unsqueeze(2)
    expected_spec_loss = torch.nn.functional.mse_loss(
        output["output"].masked_select(tgt_mask),
        batch["mel"].masked_select(tgt_mask),
    )
    assert losses["spec"].item() == pytest.approx(
        (expected_spec_loss * loss_fn.config.training.mel_loss_weight).item(),
        abs=1e-4,
    )
