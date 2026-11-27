from copy import deepcopy
from pathlib import Path
from typing import Any
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
from ..core.synthesize import (
    get_output_hop_size,
    synthesize_helper,
    text_to_spec_synthesis,
)
from ..synthesis_types import SynthesisBatch, Utterance


@pytest.fixture()
def model(stubbed_model):
    """A private copy of the session-wide stubbed model: synthesis modifies the
    model's config in place."""
    model, _ = stubbed_model
    return deepcopy(model)


def make_item(
    basename: str, text: str, index: int, chunk_index: int, is_last_input_chunk: bool
) -> dict:
    """One chunk of input text, as prepare_data() produces"""
    return {
        "basename": basename,
        "characters": text,
        "language": "default",
        "speaker": "default",
        "index": index,
        "chunk_index": chunk_index,
        "is_last_input_chunk": is_last_input_chunk,
        "duration_control": 1.0,
    }


# Utterance uttA is split in three chunks, uttB is a single chunk.
# With batch_size=2, uttA spans the first two batches.
CHUNKED_DATA = [
    make_item("uttA", "hello there", 0, 0, False),
    make_item("uttA", "big wide", 0, 1, False),
    make_item("uttA", "world out there", 0, 2, True),
    make_item("uttB", "a second text", 1, 0, True),
]


class TestTextToSpecSynthesis:
    def test_batches(self, model):
        model.train()
        batches = list(
            text_to_spec_synthesis(
                model, deepcopy(CHUNKED_DATA), torch.device("cpu"), batch_size=3
            )
        )
        assert not model.training
        assert len(batches) == 2
        outputs1, batch1 = batches[0].outputs, batches[0].batch
        outputs2, batch2 = batches[1].outputs, batches[1].batch
        assert batch1["basename"] == ["uttA", "uttA", "uttA"]
        assert batch2["basename"] == ["uttB"]
        assert outputs1[model.output_key].size(0) == 3
        assert outputs2[model.output_key].size(0) == 1
        assert outputs1[model.output_key].device == torch.device("cpu")
        assert batch1["text"].device == torch.device("cpu")
        assert not outputs1[model.output_key].requires_grad

    def test_no_grad_tracking(self, model):
        assert any(param.requires_grad for param in model.parameters())
        for i, synthesis_batch in enumerate(
            text_to_spec_synthesis(
                model, deepcopy(CHUNKED_DATA), torch.device("cpu"), batch_size=2
            )
        ):
            # no_grad must not leak out to the code consuming the iterator
            assert torch.is_grad_enabled()
            assert_no_grad_tracking(synthesis_batch.outputs, f"batch {i} outputs")
            assert_no_grad_tracking(synthesis_batch.batch, f"batch {i} batch")
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
            for synthesis_batch in text_to_spec_synthesis(
                model, deepcopy(CHUNKED_DATA), torch.device("cpu"), batch_size=2
            )
            for length in synthesis_batch.outputs["tgt_lens"]
        ]
        spec_a = torch.load(spec_dir / files[0])
        spec_b = torch.load(spec_dir / files[1])
        assert spec_a.size(-1) == sum(chunk_lens[:3])
        assert spec_b.size(-1) == chunk_lens[3]

    def test_writers_driven_directly(self, model, tmp_path):
        """write_utterance is called once per text, finalize once at the end"""
        spec_writer_class = prediction_writers.SpecPredictionWriter
        with (
            patch.object(spec_writer_class, "write_utterance") as mock_write,
            patch.object(spec_writer_class, "finalize") as mock_finalize,
        ):
            self.run_helper(model, tmp_path, filelist_data=deepcopy(CHUNKED_DATA))
        # CHUNKED_DATA has two texts: uttA, in three chunks, and uttB
        assert [call.args[0].basename for call in mock_write.call_args_list] == [
            "uttA",
            "uttB",
        ]
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
CHUNKING: dict[str, Any] = dict(split_text=True, desired_length=10, max_length=25)


class TestApiTextToSpec:
    def test_yields_synthesis_batches(self, api_model):
        texts = [SHORT_TEXT, "a second text", "and a third one"]
        results = list(api.text_to_spec(api_model, texts, batch_size=2))
        assert all(isinstance(r, SynthesisBatch) for r in results)
        assert [len(r) for r in results] == [2, 1]
        for i, r in enumerate(results):
            assert r.output_key == api_model.output_key
            assert r.outputs[r.output_key].size(0) == len(r)
            assert r.outputs[r.output_key].device == api_model.device
            assert (
                r.space_token == api_model.text_processor.encode_string_tokens([" "])[0]
            )
            assert_no_grad_tracking(r.outputs, f"batch {i} outputs")
        assert_no_param_grads(api_model, "fs2 model")

    def test_single_string(self, api_model):
        results = list(api.text_to_spec(api_model, SHORT_TEXT))
        assert len(results) == 1
        assert results[0].batch["raw_text"] == [SHORT_TEXT]

    def test_no_texts(self, api_model):
        assert list(api.text_to_spec(api_model, [])) == []

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_texts_are_rejected(self, api_model, blank):
        with pytest.raises(ValueError, match=r"indices \[1\] are empty or blank"):
            next(api.text_to_spec(api_model, [SHORT_TEXT, blank, "another one"]))

    @pytest.mark.parametrize("split_text, n_chunks", [(True, 3), (False, 1), (None, 1)])
    def test_split_text(self, api_model, split_text, n_chunks):
        api_model.config.text.split_text = False
        chunking = CHUNKING | dict(split_text=split_text)
        results = list(api.text_to_spec(api_model, LONG_TEXT, **chunking))
        assert sum(len(r) for r in results) == n_chunks


class TestApiUnbatch:
    def test_utterances(self, api_model):
        texts = [SHORT_TEXT, LONG_TEXT, "another one"]
        batches = list(api.text_to_spec(api_model, texts, batch_size=2, **CHUNKING))
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
        (batch,) = api.text_to_spec(api_model, texts, batch_size=2)
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


# Space token id for hand-made Utterances
SPACE = 99


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
        space_token=SPACE,
    )


class TestApiCombineChunks:
    def test_combines_chunks(self, api_model):
        texts = [SHORT_TEXT, LONG_TEXT, "another one"]
        batches = api.text_to_spec(api_model, texts, batch_size=2, **CHUNKING)
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
        # tokens and durations: the chunks', with a 0-frame space between them
        (space,) = api_model.text_processor.encode_string_tokens([" "])
        assert long.space_token == space
        a, b, c = long_chunks
        assert long.tokens.tolist() == (
            a.tokens.tolist()
            + [space]
            + b.tokens.tolist()
            + [space]
            + c.tokens.tolist()
        )
        assert long.durations.tolist() == (
            a.durations.tolist()
            + [0]
            + b.durations.tolist()
            + [0]
            + c.durations.tolist()
        )
        assert int(long.durations.sum()) == long.spec.size(-1)

        # ... so that the tokens read like the text
        def decode(tokens: torch.Tensor) -> str:
            return "".join(
                api_model.text_processor.decode_tokens(tokens.tolist(), None)
            )

        assert decode(long.tokens) == " ".join(decode(c.tokens) for c in long_chunks)

    def test_shared_basename_is_kept(self):
        chunks = [
            make_utterance(0, 0, False, "utt1"),
            make_utterance(0, 1, True, "utt1"),
        ]
        (combined,) = api.combine_chunks(chunks)
        assert combined.basename == "utt1"
        assert combined.text == "text 0 chunk 0 text 0 chunk 1"
        assert combined.tokens.tolist() == [0, 0, SPACE, 0, 1]
        assert combined.durations.tolist() == [2, 1, 0, 2, 1]
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


@pytest.fixture(scope="module")
def vocoder(stubbed_vocoder):
    """The stubbed vocoder, loaded for synthesis as a user would"""
    from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl import api as hfgl_api

    _vocoder, vocoder_path = stubbed_vocoder
    return hfgl_api.load_model(vocoder_path, accelerator="cpu")


class TestApiSpecToWav:
    def test_adds_wavs_to_batches(self, api_model, vocoder):
        texts = [SHORT_TEXT, "a much longer second text, to get some padding"]
        hop_size = get_output_hop_size(vocoder.config)
        batches = list(api.text_to_spec(api_model, texts, batch_size=2))
        with_wavs = list(api.spec_to_wav(batches, vocoder))

        assert len(with_wavs) == len(batches) == 1
        (batch,), (with_wav,) = batches, with_wavs
        assert batch.wavs is None
        assert with_wav.outputs is batch.outputs
        assert with_wav.wavs is not None and with_wav.wav_lens is not None
        assert with_wav.wavs.shape[:2] == (2, 1)
        assert with_wav.wavs.device == torch.device("cpu")
        assert torch.equal(with_wav.wav_lens, batch.outputs["tgt_lens"] * hop_size)
        assert with_wav.wavs.size(-1) >= int(with_wav.wav_lens.max())
        assert (
            with_wav.sample_rate
            == vocoder.config.preprocessing.audio.output_sampling_rate
        )
        assert_no_grad_tracking(with_wav.wavs, "wavs")
        assert_no_param_grads(vocoder, "vocoder")

    def test_unbatch_trims_wavs(self, api_model, vocoder):
        texts = [SHORT_TEXT, "a much longer second text, to get some padding"]
        hop_size = get_output_hop_size(vocoder.config)
        (batch,) = api.spec_to_wav(
            api.text_to_spec(api_model, texts, batch_size=2), vocoder
        )
        utterances = list(api.unbatch([batch]))
        assert batch.wavs is not None
        for i, u in enumerate(utterances):
            assert u.wav is not None
            assert u.wav.size(-1) == u.spec.size(-1) * hop_size
            assert torch.equal(u.wav, batch.wavs[i, :, : u.wav.size(-1)])
            assert u.sample_rate == batch.sample_rate
        assert utterances[0].wav.size(-1) < batch.wavs.size(-1)

    def test_combine_joins_wavs(self, api_model, vocoder):
        hop_size = get_output_hop_size(vocoder.config)
        batches = api.text_to_spec(api_model, LONG_TEXT, batch_size=2, **CHUNKING)
        chunks = list(api.unbatch(api.spec_to_wav(batches, vocoder)))
        assert len(chunks) == 3
        (combined,) = api.combine_chunks(chunks)
        assert combined.wav is not None
        assert torch.equal(
            combined.wav,
            torch.cat([c.wav for c in chunks], dim=-1),  # type: ignore[misc]
        )
        assert combined.wav.size(-1) == combined.spec.size(-1) * hop_size

    def test_no_wav_without_spec_to_wav(self, api_model):
        (utterance,) = api.unbatch(api.text_to_spec(api_model, SHORT_TEXT))
        assert utterance.wav is None
        assert utterance.sample_rate is None


class TestApiWriteToFiles:
    def utterances(self, api_model, texts) -> list[Utterance]:
        batches = api.text_to_spec(api_model, texts, batch_size=2, **CHUNKING)
        return list(api.combine_chunks(api.unbatch(batches)))

    def test_writes_whole_texts(self, api_model, tmp_path):
        utterances = self.utterances(api_model, [SHORT_TEXT, LONG_TEXT])
        written = list(
            api.write_to_files(utterances, api_model, tmp_path, ["spec", "textgrid"])
        )
        assert written == utterances  # passed through unchanged
        assert len(list((tmp_path / "synthesized_spec").glob("*.pt"))) == 2
        textgrids = [p.name for p in (tmp_path / "textgrids").glob("*")]
        assert len(textgrids) == 2
        short_basename = truncate_basename(slugify(SHORT_TEXT))
        assert any(name.startswith(short_basename + "--") for name in textgrids)

    def test_rejects_chunks(self, api_model, tmp_path):
        batches = api.text_to_spec(api_model, LONG_TEXT, **CHUNKING)
        chunks = api.unbatch(batches)
        with pytest.raises(ValueError, match="combine"):
            list(api.write_to_files(chunks, api_model, tmp_path, ["spec"]))

    def test_finalizes_only_when_exhausted(self, api_model, tmp_path):
        utterances = self.utterances(api_model, [SHORT_TEXT, "another one"])
        spec_writer_class = prediction_writers.SpecPredictionWriter
        with patch.object(spec_writer_class, "finalize") as mock_finalize:
            written = api.write_to_files(utterances, api_model, tmp_path, ["spec"])
            next(written)
            mock_finalize.assert_not_called()
            list(written)
        mock_finalize.assert_called_once_with()

    def test_wav_requires_vocoder(self, api_model, tmp_path):
        utterances = self.utterances(api_model, [SHORT_TEXT])
        with pytest.raises(ValueError, match="vocoder"):
            list(api.write_to_files(utterances, api_model, tmp_path, ["wav"]))


class TestApiSynthesizeToFiles:
    def test_writes_files(self, api_model, vocoder, tmp_path):
        result = api.synthesize_to_files(
            api_model,
            [SHORT_TEXT, LONG_TEXT],
            tmp_path,
            ["spec", "wav", "textgrid", "readalong-html"],
            vocoder=vocoder,
            batch_size=2,
            **CHUNKING,
        )
        assert result is None
        for subdir, pattern in [
            ("synthesized_spec", "*.pt"),
            ("wav", "*.wav"),
            ("textgrids", "*.TextGrid"),
            ("readalongs", "*.html"),
        ]:
            assert len(list((tmp_path / subdir).glob(pattern))) == 2, subdir
        wav_names = [p.name for p in (tmp_path / "wav").glob("*.wav")]
        assert all(f"v_ckpt={vocoder.global_step_at_load}" in n for n in wav_names)
