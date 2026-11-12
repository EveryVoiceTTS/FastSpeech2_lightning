from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from everyvoice.config.type_definitions import (
    DatasetTextRepresentation,
    SynthesizeOutputFormats,
)

from .. import api, prediction_writing_callback
from ..core import synthesize as core_synthesize
from ..core.synthesize import iter_synthesis, synthesize_helper


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
            output_type=[SynthesizeOutputFormats.spec],
            text_representation=DatasetTextRepresentation.characters,
            accelerator="cpu",
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
        spec_writer_class = prediction_writing_callback.PredictionWritingSpecCallback
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


class TestApiSynthesize:
    def test_yields_cpu_outputs_per_batch(self, model):
        texts = ["hello there", "a second text", "and a third one"]
        results = list(api.synthesize(model, texts, batch_size=2, num_workers=0))
        assert len(results) == 2
        assert [r[model.output_key].size(0) for r in results] == [2, 1]
        for r in results:
            for value in r.values():
                if torch.is_tensor(value):
                    assert value.device == torch.device("cpu")

    def test_no_texts(self, model):
        assert list(api.synthesize(model, [])) == []
