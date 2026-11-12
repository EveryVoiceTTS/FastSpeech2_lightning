import numpy as np
import pytest

from ..attn.alignment import b_mas, mas_width1


def original_mas_width1(log_attn_map):
    """Pure Python copy of mas_width1 before the single-token fix, as a
    reference: it is only valid for inputs with at least two text tokens."""
    neg_inf = log_attn_map.dtype.type(-np.inf)
    log_p = log_attn_map.copy()
    log_p[0, 1:] = neg_inf
    for i in range(1, log_p.shape[0]):
        prev_log1 = neg_inf
        for j in range(log_p.shape[1]):
            prev_log2 = log_p[i - 1, j]
            log_p[i, j] += max(prev_log1, prev_log2)
            prev_log1 = prev_log2
    opt = np.zeros_like(log_p)
    j = log_p.shape[1] - 1
    for i in range(log_p.shape[0] - 1, 0, -1):
        opt[i, j] = 1
        if log_p[i - 1, j - 1] >= log_p[i - 1, j]:
            j -= 1
            if j == 0:
                opt[1:i, j] = 1
                break
    opt[0, j] = 1
    return opt


def random_log_attn_map(rng, n_frames, n_tokens):
    return np.log(rng.random((n_frames, n_tokens), dtype=np.float32))


def assert_monotonic_alignment(opt):
    """Each frame aligns to exactly one token, starting at the first token,
    ending at the last one, and moving forward by at most one token per frame,
    so every token gets at least one frame."""
    n_frames, n_tokens = opt.shape
    assert np.all(opt.sum(axis=1) == 1)
    tokens = opt.argmax(axis=1)
    assert tokens[0] == 0
    assert tokens[-1] == n_tokens - 1
    assert set(np.diff(tokens)) <= {0, 1}


@pytest.mark.parametrize("n_frames", [1, 2, 7])
def test_single_token(n_frames):
    log_attn_map = random_log_attn_map(np.random.default_rng(0), n_frames, 1)
    opt = mas_width1(log_attn_map)
    assert opt.shape == (n_frames, 1)
    assert np.all(opt == 1)


@pytest.mark.parametrize("n_frames, n_tokens", [(2, 2), (5, 2), (12, 5), (40, 11)])
def test_matches_original_on_multiple_tokens(n_frames, n_tokens):
    rng = np.random.default_rng(n_frames * 100 + n_tokens)
    for _ in range(20):
        log_attn_map = random_log_attn_map(rng, n_frames, n_tokens)
        opt = mas_width1(log_attn_map)
        np.testing.assert_array_equal(opt, original_mas_width1(log_attn_map))
        assert_monotonic_alignment(opt)


def test_b_mas_with_single_token_item():
    rng = np.random.default_rng(1)
    in_lens = np.array([1, 4])
    out_lens = np.array([6, 9])
    b_log_attn_map = np.log(rng.random((2, 1, 9, 4), dtype=np.float32))
    attn_out = b_mas(b_log_attn_map, in_lens, out_lens)
    for b in range(2):
        opt = attn_out[b, 0, : out_lens[b], : in_lens[b]]
        assert_monotonic_alignment(opt)
        # nothing written outside the unpadded region
        assert attn_out[b, 0].sum() == out_lens[b]
