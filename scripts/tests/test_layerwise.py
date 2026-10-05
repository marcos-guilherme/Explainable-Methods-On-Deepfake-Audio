"""Testes da extração conjunta de hidden states sem carregar checkpoints reais."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from brspeech_xai import layerwise_paths as layerwise_paths_module
from brspeech_xai.config import AudioConfig
from brspeech_xai.encoders import LayerwiseAudioEmbedder
from brspeech_xai.encoders.hf_ssl import HFSSLEmbedder
from brspeech_xai.layerwise import validate_layer_embeddings
from brspeech_xai.layerwise_paths import (
    LayerwiseSuitePaths,
    build_embedding_cache_metadata,
    embedding_cache_key,
    load_valid_embedding_cache,
    sample_id_catalog_hash,
    write_embedding_cache,
)


class _FakeProcessor:
    def __call__(self, wavs, **_kwargs):
        max_len = max(len(wav) for wav in wavs)
        values = np.zeros((len(wavs), max_len), dtype=np.float32)
        mask = np.zeros((len(wavs), max_len), dtype=np.int64)
        for index, wav in enumerate(wavs):
            values[index, :len(wav)] = wav
            mask[index, :len(wav)] = 1
        return {
            "input_values": torch.from_numpy(values),
            "attention_mask": torch.from_numpy(mask),
        }


class _FakeProcessorWithoutMask(_FakeProcessor):
    def __call__(self, wavs, **kwargs):
        inputs = super().__call__(wavs, **kwargs)
        inputs.pop("attention_mask")
        return inputs


class _FakeModel:
    def __init__(
        self,
        n_hidden_states: int = 13,
        hidden_size: int = 3,
        hidden_sizes_by_call: tuple[int, ...] | None = None,
        non_finite: float | None = None,
    ) -> None:
        self.config = SimpleNamespace(
            num_hidden_layers=n_hidden_states - 1,
            hidden_size=hidden_size,
        )
        self.n_hidden_states = n_hidden_states
        self.hidden_size = hidden_size
        self.hidden_sizes_by_call = hidden_sizes_by_call
        self.non_finite = non_finite
        self.forward_calls = 0

    def _get_feat_extract_output_lengths(self, lengths):
        return lengths

    def __call__(
        self,
        input_values,
        attention_mask=None,
        output_hidden_states=False,
    ):
        assert output_hidden_states is True
        self.forward_calls += 1
        hidden_size = (
            self.hidden_sizes_by_call[self.forward_calls - 1]
            if self.hidden_sizes_by_call
            else self.hidden_size
        )
        base = input_values.unsqueeze(-1).expand(-1, -1, hidden_size)
        hidden_states = tuple(
            base + float(layer_index)
            for layer_index in range(self.n_hidden_states)
        )
        if self.non_finite is not None:
            hidden_states = list(hidden_states)
            hidden_states[-1] = hidden_states[-1].clone()
            hidden_states[-1][0, 0, 0] = self.non_finite
            hidden_states = tuple(hidden_states)
        return SimpleNamespace(hidden_states=hidden_states)


def _make_embedder(
    *,
    layer: int = -1,
    n_hidden_states: int = 13,
    hidden_size: int = 3,
    hidden_sizes_by_call: tuple[int, ...] | None = None,
    non_finite: float | None = None,
) -> HFSSLEmbedder:
    embedder = HFSSLEmbedder.__new__(HFSSLEmbedder)
    embedder.name = "hf_ssl:fake"
    embedder.checkpoint = "fake"
    embedder.layer = layer
    embedder.pooling = "mean"
    embedder.device = "cpu"
    embedder.num_samples = 64600
    embedder._processor = _FakeProcessor()
    embedder._model = _FakeModel(
        n_hidden_states=n_hidden_states,
        hidden_size=hidden_size,
        hidden_sizes_by_call=hidden_sizes_by_call,
        non_finite=non_finite,
    )
    embedder.dim = hidden_size
    embedder.n_transformer_layers = n_hidden_states - 1
    embedder._to_16k_mono = lambda audio, _sr: np.asarray(audio, dtype=np.float32)
    return embedder


def test_extracts_all_layers_with_one_forward_per_batch():
    embedder = _make_embedder()

    result = embedder.extract_all_layer_embeddings(
        [np.zeros(320), np.ones(320)],
        [16000, 16000],
        batch_size=2,
    )

    assert result.shape == (2, 13, embedder.dim)
    assert result.dtype == np.float32
    assert embedder._model.forward_calls == 1


def test_masked_pooling_ignores_padded_frames():
    embedder = _make_embedder()

    result = embedder.extract_all_layer_embeddings(
        [np.array([1.0, 3.0]), np.array([2.0])],
        [16000, 16000],
        batch_size=2,
    )

    np.testing.assert_allclose(result[0, 0], 2.0)
    np.testing.assert_allclose(result[1, 0], 2.0)


def test_real_preprocessing_uses_canonical_fixed_length_waveform():
    embedder = _make_embedder()
    del embedder._to_16k_mono
    embedder.num_samples = 5

    result = embedder.extract_all_layer_embeddings(
        [np.zeros(5), np.array([1.0, 3.0])],
        [16000, 16000],
        batch_size=2,
    )

    np.testing.assert_allclose(result[1, 0], 1.8)


def test_canonical_hf_preprocessing_mono_resamples_and_fixes_exact_length():
    embedder = _make_embedder()
    del embedder._to_16k_mono
    embedder.num_samples = 5

    long_stereo = np.stack(
        [np.arange(8, dtype=np.float32), np.arange(8, dtype=np.float32) + 2]
    )
    long_result = embedder.preprocess_waveform(long_stereo, 16000)
    short_result = embedder.preprocess_waveform(
        np.asarray([1.0, 3.0], dtype=np.float32), 16000
    )

    np.testing.assert_array_equal(long_result, np.arange(5, dtype=np.float32) + 1)
    np.testing.assert_array_equal(
        short_result, np.asarray([1.0, 3.0, 1.0, 3.0, 1.0], dtype=np.float32)
    )


def test_builds_attention_mask_when_processor_omits_it():
    embedder = _make_embedder()
    embedder._processor = _FakeProcessorWithoutMask()

    result = embedder.extract_all_layer_embeddings(
        [np.array([1.0, 3.0]), np.array([2.0])],
        [16000, 16000],
        batch_size=2,
    )

    np.testing.assert_allclose(result[1, 0], 2.0)


def test_masked_pooling_fails_if_feature_lengths_cannot_be_derived():
    embedder = _make_embedder()

    def fail_length_conversion(_lengths):
        raise RuntimeError("unsupported convolution")

    embedder._model._get_feat_extract_output_lengths = fail_length_conversion

    with pytest.raises(RuntimeError, match="feature lengths"):
        embedder.extract_all_layer_embeddings(
            [np.zeros(4), np.zeros(2)],
            [16000, 16000],
        )


@pytest.mark.parametrize("non_finite", [np.nan, np.inf])
def test_rejects_non_finite_embeddings(non_finite):
    embedder = _make_embedder(non_finite=non_finite)

    with pytest.raises(ValueError, match="finite"):
        embedder.extract_all_layer_embeddings([np.zeros(4)], [16000])


def test_rejects_wrong_number_of_hidden_states():
    embedder = _make_embedder()
    embedder._model.n_hidden_states = 12

    with pytest.raises(ValueError, match="layers"):
        embedder.extract_all_layer_embeddings([np.zeros(4)], [16000])


def test_empty_batch_returns_typed_empty_array():
    embedder = _make_embedder()

    result = embedder.extract_all_layer_embeddings([], [])

    assert result.shape == (0, 13, embedder.dim)
    assert result.dtype == np.float32
    assert embedder._model.forward_calls == 0


def test_rejects_different_audio_and_sample_rate_lengths():
    embedder = _make_embedder()

    with pytest.raises(ValueError, match="same length"):
        embedder.extract_all_layer_embeddings(
            [np.zeros(4), np.zeros(4)],
            [16000],
        )


def test_rejects_inconsistent_dimensions_between_batches():
    embedder = _make_embedder(hidden_sizes_by_call=(3, 4))

    with pytest.raises(ValueError, match="dimension"):
        embedder.extract_all_layer_embeddings(
            [np.zeros(4), np.zeros(4)],
            [16000, 16000],
            batch_size=1,
        )


@pytest.mark.parametrize(("layer", "expected"), [(-1, 12.0), (-13, 0.0), (2, 2.0)])
def test_extract_embeddings_selects_layer_from_common_result(layer, expected):
    embedder = _make_embedder(layer=layer)

    result = embedder.extract_embeddings([np.zeros(4)], [16000])

    assert result.shape == (1, embedder.dim)
    np.testing.assert_allclose(result, expected)
    assert embedder._model.forward_calls == 1


@pytest.mark.parametrize("layer", [-14, 13])
def test_extract_embeddings_rejects_layer_out_of_range(layer):
    embedder = _make_embedder(layer=layer)

    with pytest.raises(IndexError, match="layer"):
        embedder.extract_embeddings([np.zeros(4)], [16000])


def test_validate_layer_embeddings_checks_shape_and_finiteness():
    valid = np.zeros((2, 13, 3), dtype=np.float32)
    validate_layer_embeddings(valid, expected_samples=2, expected_layers=12)

    with pytest.raises(ValueError, match="shape"):
        validate_layer_embeddings(valid[:, :-1], expected_samples=2, expected_layers=12)
    with pytest.raises(ValueError, match="finite"):
        invalid = valid.copy()
        invalid[0, 0, 0] = np.nan
        validate_layer_embeddings(invalid, expected_samples=2, expected_layers=12)


def test_hf_ssl_satisfies_layerwise_protocol():
    assert isinstance(_make_embedder(), LayerwiseAudioEmbedder)


def test_layerwise_paths_are_unambiguous(tmp_path):
    paths = LayerwiseSuitePaths(tmp_path)

    assert paths.embeddings("hubert_base", "eng", "train") == (
        tmp_path / "hubert_base/embeddings/eng/train.npy"
    )
    assert paths.embedding_metadata("hubert_base", "eng", "train").name == (
        "train.metadata.json"
    )
    assert paths.probe("hubert_base", 7, "por").name == "d_ad.joblib"
    assert paths.thresholds("hubert_base", 7, "por").name == "thresholds.json"
    assert paths.cell("hubert_base", 7, "por", "zho").name == "por_to_zho"
    assert paths.layer_xai("hubert_base", 7, "por", "zho").name == "por_to_zho"
    assert paths.final_decision_trace("hubert_base").name == "final_decision_trace"
    assert paths.suite_aggregate("layerwise_performance.csv") == (
        tmp_path / "aggregates/layerwise_performance.csv"
    )


def test_layerwise_paths_reject_reserved_profile_namespace(tmp_path):
    paths = LayerwiseSuitePaths(tmp_path)

    for profile_id in ("aggregates", "Aggregates"):
        with pytest.raises(ValueError, match="reserved root namespace"):
            paths.profile(profile_id)


@pytest.mark.parametrize(
    "call",
    [
        lambda paths: paths.embeddings("../escape", "eng", "train"),
        lambda paths: paths.embeddings("hubert/base", "eng", "train"),
        lambda paths: paths.embeddings(r"hubert\base", "eng", "train"),
        lambda paths: paths.embeddings("C:escape", "eng", "train"),
        lambda paths: paths.embeddings("", "eng", "train"),
        lambda paths: paths.embeddings("hubert_base", "spa", "train"),
        lambda paths: paths.embeddings("hubert_base", "eng", ".."),
        lambda paths: paths.cell("hubert_base", 0, "eng", "por"),
        lambda paths: paths.cell("hubert_base", 13, "eng", "por"),
        lambda paths: paths.suite_aggregate("/absolute.csv"),
    ],
)
def test_layerwise_paths_reject_unsafe_identifiers(tmp_path, call):
    with pytest.raises(ValueError):
        call(LayerwiseSuitePaths(tmp_path))


def test_embedding_cache_key_is_canonical_and_sensitive():
    audio = AudioConfig(sample_rate=16000, num_samples=64600)
    arguments = {
        "profile_id": "hubert_base",
        "checkpoint": "facebook/hubert-base-ls960",
        "manifest_sha256": "a" * 64,
        "language": "eng",
        "role": "train",
        "seed": 42,
        "audio_config": audio,
    }

    first = embedding_cache_key(**arguments)
    second = embedding_cache_key(**dict(reversed(list(arguments.items()))))

    assert first == second
    assert len(first) == 64
    for field, changed in (
        ("profile_id", "wavlm_base_plus"),
        ("checkpoint", "other/checkpoint"),
        ("manifest_sha256", "b" * 64),
        ("language", "por"),
        ("role", "test"),
        ("seed", 43),
        ("audio_config", AudioConfig(sample_rate=8000, num_samples=64600)),
    ):
        modified = {**arguments, field: changed}
        assert embedding_cache_key(**modified) != first


@pytest.mark.parametrize("manifest_sha256", ["", "abc", "g" * 64, "a" * 63])
def test_embedding_cache_key_rejects_invalid_manifest_hash(manifest_sha256):
    with pytest.raises(ValueError, match="manifest"):
        embedding_cache_key(
            profile_id="hubert_base",
            checkpoint="facebook/hubert-base-ls960",
            manifest_sha256=manifest_sha256,
            language="eng",
            role="train",
            seed=42,
            audio_config=AudioConfig(),
        )


def _cache_fixture(tmp_path):
    paths = LayerwiseSuitePaths(tmp_path)
    embedding_path = paths.embeddings("hubert_base", "eng", "train")
    metadata_path = paths.embedding_metadata("hubert_base", "eng", "train")
    sample_ids = ["sample-a", "sample-b"]
    array = np.arange(2 * 13 * 3, dtype=np.float32).reshape(2, 13, 3)
    cache_key = embedding_cache_key(
        profile_id="hubert_base",
        checkpoint="facebook/hubert-base-ls960",
        manifest_sha256="a" * 64,
        language="eng",
        role="train",
        seed=42,
        audio_config=AudioConfig(),
    )
    metadata = build_embedding_cache_metadata(
        cache_key=cache_key,
        profile_id="hubert_base",
        checkpoint="facebook/hubert-base-ls960",
        manifest_sha256="a" * 64,
        language="eng",
        role="train",
        array=array,
        sample_ids=sample_ids,
    )
    return embedding_path, metadata_path, array, sample_ids, cache_key, metadata


def test_embedding_cache_accepts_exact_metadata_array_and_catalog(tmp_path):
    embedding_path, metadata_path, array, sample_ids, cache_key, metadata = (
        _cache_fixture(tmp_path)
    )
    write_embedding_cache(embedding_path, metadata_path, array, metadata)

    loaded = load_valid_embedding_cache(
        embedding_path,
        metadata_path,
        expected_cache_key=cache_key,
        expected_profile_id="hubert_base",
        expected_checkpoint="facebook/hubert-base-ls960",
        expected_manifest_sha256="a" * 64,
        expected_language="eng",
        expected_role="train",
        expected_sample_ids=sample_ids,
    )

    np.testing.assert_array_equal(loaded, array)
    assert sample_id_catalog_hash(sample_ids) == metadata["sample_ids_sha256"]


@pytest.mark.parametrize(
    "invalid_array",
    [
        np.zeros((2, 12, 3), dtype=np.float32),
        np.zeros((2, 13), dtype=np.float32),
        np.zeros((2, 13, 3), dtype=np.float64),
        np.full((2, 13, 3), np.nan, dtype=np.float32),
        np.full((2, 13, 3), np.inf, dtype=np.float32),
    ],
)
def test_embedding_cache_rejects_invalid_array(tmp_path, invalid_array):
    embedding_path, metadata_path, _, sample_ids, cache_key, metadata = (
        _cache_fixture(tmp_path)
    )
    embedding_path.parent.mkdir(parents=True)
    np.save(embedding_path, invalid_array)
    metadata_path.write_text(json.dumps({**metadata, "shape": list(invalid_array.shape)}))

    assert load_valid_embedding_cache(
        embedding_path,
        metadata_path,
        expected_cache_key=cache_key,
        expected_profile_id="hubert_base",
        expected_checkpoint="facebook/hubert-base-ls960",
        expected_manifest_sha256="a" * 64,
        expected_language="eng",
        expected_role="train",
        expected_sample_ids=sample_ids,
    ) is None


def test_embedding_cache_rejects_duplicate_or_reordered_sample_ids(tmp_path):
    embedding_path, metadata_path, array, sample_ids, cache_key, metadata = (
        _cache_fixture(tmp_path)
    )
    write_embedding_cache(embedding_path, metadata_path, array, metadata)

    for expected_ids in (["sample-a", "sample-a"], list(reversed(sample_ids))):
        assert load_valid_embedding_cache(
            embedding_path,
            metadata_path,
            expected_cache_key=cache_key,
            expected_profile_id="hubert_base",
            expected_checkpoint="facebook/hubert-base-ls960",
            expected_manifest_sha256="a" * 64,
            expected_language="eng",
            expected_role="train",
            expected_sample_ids=expected_ids,
        ) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 999),
        ("cache_key", "b" * 64),
        ("profile", "wavlm_base_plus"),
        ("checkpoint", "other/checkpoint"),
        ("manifest_sha256", "b" * 64),
        ("language", "por"),
        ("role", "test"),
        ("sample_ids_sha256", "b" * 64),
        ("unexpected_field", "tampered"),
    ],
)
def test_embedding_cache_rejects_tampered_metadata(tmp_path, field, value):
    embedding_path, metadata_path, array, sample_ids, cache_key, metadata = (
        _cache_fixture(tmp_path)
    )
    write_embedding_cache(embedding_path, metadata_path, array, metadata)
    metadata_path.write_text(json.dumps({**metadata, field: value}))

    assert load_valid_embedding_cache(
        embedding_path,
        metadata_path,
        expected_cache_key=cache_key,
        expected_profile_id="hubert_base",
        expected_checkpoint="facebook/hubert-base-ls960",
        expected_manifest_sha256="a" * 64,
        expected_language="eng",
        expected_role="train",
        expected_sample_ids=sample_ids,
    ) is None


def test_embedding_cache_rejects_unreadable_metadata_or_missing_array(tmp_path):
    embedding_path, metadata_path, _, sample_ids, cache_key, _ = _cache_fixture(
        tmp_path
    )
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text("{not-json")

    assert load_valid_embedding_cache(
        embedding_path,
        metadata_path,
        expected_cache_key=cache_key,
        expected_profile_id="hubert_base",
        expected_checkpoint="facebook/hubert-base-ls960",
        expected_manifest_sha256="a" * 64,
        expected_language="eng",
        expected_role="train",
        expected_sample_ids=sample_ids,
    ) is None


@pytest.mark.parametrize(
    ("metadata_field", "expected_field", "invalid_value"),
    [
        ("cache_key", "expected_cache_key", "not-a-sha256"),
        ("profile", "expected_profile_id", "../escape"),
        ("language", "expected_language", "spa"),
        ("role", "expected_role", ".."),
    ],
)
def test_embedding_cache_rejects_matching_malformed_context(
    tmp_path,
    metadata_field,
    expected_field,
    invalid_value,
):
    embedding_path, metadata_path, array, sample_ids, cache_key, metadata = (
        _cache_fixture(tmp_path)
    )
    write_embedding_cache(embedding_path, metadata_path, array, metadata)
    metadata_path.write_text(
        json.dumps({**metadata, metadata_field: invalid_value}),
        encoding="utf-8",
    )
    expected = {
        "expected_cache_key": cache_key,
        "expected_profile_id": "hubert_base",
        "expected_checkpoint": "facebook/hubert-base-ls960",
        "expected_manifest_sha256": "a" * 64,
        "expected_language": "eng",
        "expected_role": "train",
        "expected_sample_ids": sample_ids,
    }
    expected[expected_field] = invalid_value

    assert load_valid_embedding_cache(
        embedding_path,
        metadata_path,
        **expected,
    ) is None


def test_embedding_cache_rejects_matching_unsafe_sample_ids(tmp_path):
    embedding_path, metadata_path, array, _, cache_key, metadata = _cache_fixture(
        tmp_path
    )
    write_embedding_cache(embedding_path, metadata_path, array, metadata)
    unsafe_ids = ["sample-a", "../sample-b"]
    encoded = json.dumps(
        unsafe_ids,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                **metadata,
                "sample_ids": unsafe_ids,
                "sample_ids_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    assert load_valid_embedding_cache(
        embedding_path,
        metadata_path,
        expected_cache_key=cache_key,
        expected_profile_id="hubert_base",
        expected_checkpoint="facebook/hubert-base-ls960",
        expected_manifest_sha256="a" * 64,
        expected_language="eng",
        expected_role="train",
        expected_sample_ids=unsafe_ids,
    ) is None


@pytest.mark.parametrize("failed_replace", [1, 2])
def test_failed_cache_publication_preserves_previous_valid_pair(
    tmp_path,
    monkeypatch,
    failed_replace,
):
    embedding_path, metadata_path, old_array, sample_ids, cache_key, metadata = (
        _cache_fixture(tmp_path)
    )
    write_embedding_cache(embedding_path, metadata_path, old_array, metadata)
    new_array = old_array + np.float32(1000.0)
    real_replace = layerwise_paths_module.os.replace
    calls = 0

    def fail_once(source, destination):
        nonlocal calls
        calls += 1
        if calls == failed_replace:
            raise OSError("injected replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(layerwise_paths_module.os, "replace", fail_once)

    with pytest.raises(OSError, match="injected replace failure"):
        write_embedding_cache(
            embedding_path,
            metadata_path,
            new_array,
            metadata,
        )

    loaded = load_valid_embedding_cache(
        embedding_path,
        metadata_path,
        expected_cache_key=cache_key,
        expected_profile_id="hubert_base",
        expected_checkpoint="facebook/hubert-base-ls960",
        expected_manifest_sha256="a" * 64,
        expected_language="eng",
        expected_role="train",
        expected_sample_ids=sample_ids,
    )
    np.testing.assert_array_equal(loaded, old_array)
