from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from everyvoice.config.text_config import LanguageBoundaries
from everyvoice.config.type_definitions import (
    DatasetTextRepresentation,
    SynthesizeOutputFormats,
)
from everyvoice.tests.grad_checks import assert_no_grad_tracking, assert_no_param_grads
from everyvoice.utils import slugify, truncate_basename

from .. import api, prediction_writers
from ..core import synthesize as core_synthesize
from ..core.synthesize import iter_synthesis, synthesize_helper
from ..synthesis_types import SynthesisBatch, Utterance


@pytest.fixture()
def model(stubbed_model):
    """A private copy of the session-wide stubbed model: synthesis modifies the
    model's config in place."""
    model, _ = stubbed_model
    return deepcopy(model)


def make_item(basename: str, text: str, is_last_input_chunk: bool) -> dict:
    return {
        "basename": basename,
        "characters": text,
        "language": "default",
        "speaker": "default",
        "is_last_input_chunk": is_last_input_chunk,
        "duration_control": 1.0,
    }


# Utterance uttA is split in three chunks, uttB is a single chunk.
# With batch_size=2, uttA spans the first two batches.
CHUNKED_DATA = [
    make_item("uttA", "hello there ", False),
    make_item("uttA", "big wide ", False),
    make_item("uttA", "world out there", True),
    make_item("uttB", "a second text", True),
]


class TestIterSynthesis:
    def test_batches(self, model):
        model.train()
        batches = list(
            iter_synthesis(
                model, deepcopy(CHUNKED_DATA), torch.device("cpu"), batch_size=3
            )
        )
        assert not model.training
        assert len(batches) == 2
        (outputs1, batch1), (outputs2, batch2) = batches
        assert batch1["basename"] == ["uttA", "uttA", "uttA"]
        assert batch2["basename"] == ["uttB"]
        assert outputs1[model.output_key].size(0) == 3
        assert outputs2[model.output_key].size(0) == 1
        assert outputs1[model.output_key].device == torch.device("cpu")
        assert batch1["text"].device == torch.device("cpu")
        assert not outputs1[model.output_key].requires_grad

    def test_no_grad_tracking(self, model):
        assert any(param.requires_grad for param in model.parameters())
        for i, (outputs, batch) in enumerate(
            iter_synthesis(
                model, deepcopy(CHUNKED_DATA), torch.device("cpu"), batch_size=2
            )
        ):
            # no_grad must not leak out to the code consuming the iterator
            assert torch.is_grad_enabled()
            assert_no_grad_tracking(outputs, f"batch {i} outputs")
            assert_no_grad_tracking(batch, f"batch {i} batch")
        assert_no_param_grads(model, "fs2 model")


class TestSynthesizeHelper:
    def run_helper(self, model, output_dir: Path, **kwargs):
        defaults = dict(
            model=model,
            texts=None,
            style_reference=None,
            language=None,
            speaker=None,
            duration_control=1.0,
            global_step=5,
            output_types=[SynthesizeOutputFormats.spec],
            text_representation=DatasetTextRepresentation.characters,
            devices="1",
            device=torch.device("cpu"),
            batch_size=2,
            num_workers=0,
            filelist=None,
            filelist_data=None,
            output_dir=output_dir,
            teacher_forcing_directory=None,
        )
        defaults.update(kwargs)
        return synthesize_helper(**defaults)

    def test_chunks_reassembled_across_batches(self, model, tmp_path):
        config, writers = self.run_helper(
            model, tmp_path, filelist_data=deepcopy(CHUNKED_DATA)
        )
        assert config is model.config
        assert list(writers.keys()) == [SynthesizeOutputFormats.spec]

        spec_dir = tmp_path / "synthesized_spec"
        files = sorted(p.name for p in spec_dir.glob("*.pt"))
        assert files == [
            "uttA--default--default--spec-pred-22050-mel-librosa.pt",
            "uttB--default--default--spec-pred-22050-mel-librosa.pt",
        ]

        # The full spec for uttA must be the concatenation of its three chunks.
        # Use the same batch_size: with the untrained stub model, predicted
        # durations can depend on the padding within a batch.
        chunk_lens = [
            int(length)
            for outputs, _ in iter_synthesis(
                model, deepcopy(CHUNKED_DATA), torch.device("cpu"), batch_size=2
            )
            for length in outputs["tgt_lens"]
        ]
        spec_a = torch.load(spec_dir / files[0])
        spec_b = torch.load(spec_dir / files[1])
        assert spec_a.size(-1) == sum(chunk_lens[:3])
        assert spec_b.size(-1) == chunk_lens[3]

    def test_writers_driven_directly(self, model, tmp_path):
        """write_batch is called once per batch, finalize once at the end"""
        spec_writer_class = prediction_writers.SpecPredictionWriter
        with (
            patch.object(spec_writer_class, "write_batch") as mock_write_batch,
            patch.object(spec_writer_class, "finalize") as mock_finalize,
        ):
            self.run_helper(model, tmp_path, filelist_data=deepcopy(CHUNKED_DATA))
        assert mock_write_batch.call_count == 2
        mock_finalize.assert_called_once_with()

    def test_incompatible_text_representation(self, model, tmp_path):
        with pytest.raises(ValueError, match="incompatible"):
            self.run_helper(
                model,
                tmp_path,
                texts=["hello there"],
                text_representation=DatasetTextRepresentation.ipa_phones,
            )


@pytest.mark.parametrize(
    "devices, warns",
    [
        ("auto", False),
        ("1", False),
        (1, False),
        ("0,", False),
        ("2", True),
        (2, True),
        ("-1", True),
        ("0,1", True),
    ],
)
def test_warn_if_multiple_devices(devices, warns):
    with patch.object(core_synthesize, "logger") as mock_logger:
        core_synthesize.warn_if_multiple_devices(devices)
    assert mock_logger.warning.called == warns


@pytest.fixture()
def api_model(model):
    """The stubbed model, made usable for whole-text synthesis: the untrained
    model predicts 0 frames for every token, and has no chunking boundaries
    for its "default" language."""
    with torch.no_grad():
        model.variance_adaptor.duration_predictor.linear.bias.fill_(1.5)
    model.config.text.boundaries = {"default": LanguageBoundaries()}
    return model


SHORT_TEXT = "hello there"
# Chunked in three with desired_length=10, max_length=25
LONG_TEXT = "This is chunk one. This is chunk two. This is chunk three."
CHUNKING = dict(split_text=True, desired_length=10, max_length=25)


class TestApiSynthesize:
    def test_yields_synthesis_batches(self, api_model):
        texts = [SHORT_TEXT, "a second text", "and a third one"]
        results = list(api.synthesize(api_model, texts, batch_size=2))
        assert all(isinstance(r, SynthesisBatch) for r in results)
        assert [len(r) for r in results] == [2, 1]
        for i, r in enumerate(results):
            assert r.output_key == api_model.output_key
            assert r.outputs[r.output_key].size(0) == len(r)
            assert r.outputs[r.output_key].device == api_model.device
            assert_no_grad_tracking(r.outputs, f"batch {i} outputs")
        assert_no_param_grads(api_model, "fs2 model")

    def test_single_string(self, api_model):
        results = list(api.synthesize(api_model, SHORT_TEXT))
        assert len(results) == 1
        assert results[0].batch["raw_text"] == [SHORT_TEXT]

    def test_no_texts(self, api_model):
        assert list(api.synthesize(api_model, [])) == []

    @pytest.mark.parametrize("split_text, n_chunks", [(True, 3), (False, 1), (None, 1)])
    def test_split_text(self, api_model, split_text, n_chunks):
        api_model.config.text.split_text = False
        chunking = CHUNKING | dict(split_text=split_text)
        results = list(api.synthesize(api_model, LONG_TEXT, **chunking))
        assert sum(len(r) for r in results) == n_chunks


class TestApiUnbatch:
    def test_utterances(self, api_model):
        texts = [SHORT_TEXT, LONG_TEXT, "another one"]
        batches = list(api.synthesize(api_model, texts, batch_size=2, **CHUNKING))
        utterances = list(api.unbatch(batches))

        assert [(u.index, u.chunk_index, u.is_last_chunk) for u in utterances] == [
            (0, 0, True),
            (1, 0, False),
            (1, 1, False),
            (1, 2, True),
            (2, 0, True),
        ]
        assert [u.text for u in utterances] == [
            SHORT_TEXT,
            "This is chunk one.",
            "This is chunk two.",
            "This is chunk three.",
            "another one",
        ]
        n_mels = api_model.config.preprocessing.audio.n_mels
        for u in utterances:
            assert u.speaker == "default" and u.language == "default"
            assert u.spec.size(0) == n_mels
            assert u.spec.size(1) > 0
            assert int(u.durations.sum()) == u.spec.size(1)
            assert u.tokens.shape == u.durations.shape
            for tensor in (u.spec, u.tokens, u.durations):
                assert tensor.device == torch.device("cpu")
                assert tensor.is_contiguous()
            assert_no_grad_tracking([u.spec, u.tokens, u.durations])

    def test_matches_batch_without_padding(self, api_model):
        texts = [SHORT_TEXT, "a much longer second text, to get some padding"]
        (batch,) = api.synthesize(api_model, texts, batch_size=2)
        utterances = list(api.unbatch([batch], device=None))
        specs = batch.outputs[batch.output_key]
        for i, u in enumerate(utterances):
            n_frames = int(batch.outputs["tgt_lens"][i])
            n_tokens = int(batch.batch["src_lens"][i])
            assert torch.equal(u.spec, specs[i, :n_frames].transpose(0, 1))
            assert torch.equal(u.tokens, batch.batch["text"][i, :n_tokens])
            # A copy, so an utterance doesn't keep its padded batch in memory
            assert (
                u.spec.untyped_storage().data_ptr()
                != specs.untyped_storage().data_ptr()
            )
        assert utterances[0].spec.size(1) < specs.size(1)

    def test_empty(self):
        assert list(api.unbatch([])) == []


def make_utterance(
    index: int,
    chunk_index: int,
    is_last_chunk: bool,
    basename: str = "base",
    n_frames: int = 3,
) -> Utterance:
    """A small hand-made Utterance: 2 tokens, n_frames frames of 4 mel bands."""
    return Utterance(
        index=index,
        chunk_index=chunk_index,
        is_last_chunk=is_last_chunk,
        text=f"text {index} chunk {chunk_index}",
        basename=basename,
        speaker="default",
        language="default",
        spec=torch.full((4, n_frames), float(chunk_index)),
        tokens=torch.tensor([index, chunk_index]),
        durations=torch.tensor([n_frames - 1, 1]),
    )


class TestApiCombineChunks:
    def test_combines_chunks(self, api_model):
        texts = [SHORT_TEXT, LONG_TEXT, "another one"]
        batches = api.synthesize(api_model, texts, batch_size=2, **CHUNKING)
        chunks = list(api.unbatch(batches))
        assert len(chunks) == 5
        combined = list(api.combine_chunks(chunks))

        assert [u.index for u in combined] == [0, 1, 2]
        assert all(u.chunk_index == 0 and u.is_last_chunk for u in combined)
        # Texts that weren't split pass through unchanged
        assert combined[0] is chunks[0]
        assert combined[2] is chunks[4]

        long = combined[1]
        long_chunks = chunks[1:4]
        assert long.text == LONG_TEXT
        assert long.basename == truncate_basename(slugify(LONG_TEXT))
        # Same as the writers, which join chunk texts without spaces
        assert long.basename == truncate_basename(
            slugify("".join(c.text for c in long_chunks))
        )
        assert torch.equal(long.spec, torch.cat([c.spec for c in long_chunks], -1))
        assert torch.equal(long.tokens, torch.cat([c.tokens for c in long_chunks]))
        assert torch.equal(
            long.durations, torch.cat([c.durations for c in long_chunks])
        )
        assert int(long.durations.sum()) == long.spec.size(-1)

    def test_shared_basename_is_kept(self):
        chunks = [
            make_utterance(0, 0, False, "utt1"),
            make_utterance(0, 1, True, "utt1"),
        ]
        (combined,) = api.combine_chunks(chunks)
        assert combined.basename == "utt1"
        assert combined.text == "text 0 chunk 0 text 0 chunk 1"
        assert combined.spec.shape == (4, 6)

    def test_yields_as_soon_as_a_text_is_complete(self):
        consumed = []

        def source():
            for u in [
                make_utterance(0, 0, False),
                make_utterance(0, 1, True),
                make_utterance(1, 0, True),
            ]:
                consumed.append(u)
                yield u

        combined = api.combine_chunks(source())
        assert next(combined).index == 0
        assert len(consumed) == 2
        assert next(combined).index == 1
        assert len(consumed) == 3

    @pytest.mark.parametrize(
        "chunks, message",
        [
            pytest.param(
                [make_utterance(0, 0, False)],
                "Text 0 is missing its last chunk: the input ended",
                id="ends-early",
            ),
            pytest.param(
                [make_utterance(0, 0, False), make_utterance(1, 0, True)],
                "Text 0 is missing its last chunk: got a chunk of text 1",
                id="interleaved",
            ),
            pytest.param(
                [make_utterance(0, 1, True)],
                "expected chunk 0, got chunk 1",
                id="first-chunk-missing",
            ),
            pytest.param(
                [make_utterance(0, 0, False), make_utterance(0, 2, True)],
                "expected chunk 1, got chunk 2",
                id="chunk-skipped",
            ),
        ],
    )
    def test_errors(self, chunks, message):
        with pytest.raises(ValueError, match=message):
            list(api.combine_chunks(chunks))

    def test_empty(self):
        assert list(api.combine_chunks([])) == []
