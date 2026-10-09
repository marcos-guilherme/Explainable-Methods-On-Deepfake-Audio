import pytest

from brspeech_xai.xai_registry import (
    EncoderExperimentSpec,
    get_encoder_spec,
    get_method_spec,
    resolve_experiment,
)


def test_initial_registry_has_three_layerwise_encoders():
    assert get_encoder_spec("hubert_base").checkpoint == "facebook/hubert-base-ls960"
    assert get_encoder_spec("wavlm_base").n_transformer_layers == 12
    assert get_encoder_spec("wav2vec2_base").family == "wav2vec2"
    assert get_encoder_spec("hubert_base").xai_dtype == "float32"
    assert get_encoder_spec("wavlm_base").xai_dtype == "float32"
    assert get_encoder_spec("wav2vec2_base").xai_dtype == "float64"


def test_encoder_profiles_have_exact_checkpoints():
    assert get_encoder_spec("hubert_base").checkpoint == "facebook/hubert-base-ls960"
    assert get_encoder_spec("wavlm_base").checkpoint == "microsoft/wavlm-base"
    assert get_encoder_spec("wav2vec2_base").checkpoint == "facebook/wav2vec2-base-960h"
    with pytest.raises(ValueError, match="not registered"):
        get_encoder_spec("wavlm_base_plus")


def test_encoder_profiles_share_layerwise_capabilities():
    expected = frozenset(
        {
            "all_hidden_states",
            "layerwise_probe",
            "attnlrp_cp",
            "dft_lrp",
            "layer_relevance_hooks",
        }
    )
    for profile_id in ("hubert_base", "wavlm_base", "wav2vec2_base"):
        spec = get_encoder_spec(profile_id)
        assert spec.layer_indices == tuple(range(1, 13))
        assert spec.capabilities == expected


def test_primary_methods_are_required_and_dependency_ordered():
    resolved = resolve_experiment(
        "hubert_base",
        ["layerwise_linear_probe", "attnlrp_time", "dft_lrp_frequency"],
    )
    assert [method.method_id for method in resolved.methods] == [
        "layerwise_linear_probe",
        "attnlrp_time",
        "dft_lrp_frequency",
    ]


def test_resolve_experiment_topologically_sorts_requested_methods():
    resolved = resolve_experiment(
        "wavlm_base",
        ["dft_lrp_frequency", "layerwise_linear_probe", "attnlrp_time"],
    )
    assert [method.method_id for method in resolved.methods] == [
        "layerwise_linear_probe",
        "attnlrp_time",
        "dft_lrp_frequency",
    ]


def test_method_specs_declare_dependencies_and_primary_flag():
    assert get_method_spec("attnlrp_time").dependencies == ("layerwise_linear_probe",)
    assert get_method_spec("dft_lrp_frequency").dependencies == ("attnlrp_time",)
    assert get_method_spec("stdft_lrp_time_frequency").dependencies == ("attnlrp_time",)
    assert get_method_spec("final_decision_layer_trace").dependencies == ("attnlrp_time",)
    assert get_method_spec("classical_band_audit").primary is False
    assert get_method_spec("layerwise_linear_probe").primary is True


def test_registry_rejects_unknown_method_in_resolve_experiment():
    with pytest.raises(
        ValueError,
        match=r"profile 'hubert_base'.*method 'unknown_method'.*not registered",
    ):
        resolve_experiment("hubert_base", ["unknown_method"])


def test_registry_rejects_missing_capability(monkeypatch):
    hubert = get_encoder_spec("hubert_base")
    reduced_capabilities = frozenset(
        capability for capability in hubert.capabilities if capability != "dft_lrp"
    )
    monkeypatch.setattr(
        "brspeech_xai.xai_registry._ENCODER_REGISTRY",
        {
            "hubert_base": EncoderExperimentSpec(
                profile_id=hubert.profile_id,
                encoder=hubert.encoder,
                checkpoint=hubert.checkpoint,
                family=hubert.family,
                n_transformer_layers=hubert.n_transformer_layers,
                layer_indices=hubert.layer_indices,
                capabilities=reduced_capabilities,
                attention_rule=hubert.attention_rule,
                xai_dtype=hubert.xai_dtype,
            )
        },
    )
    with pytest.raises(
        ValueError,
        match=r"profile 'hubert_base' lacks capability 'dft_lrp' for method 'dft_lrp_frequency'",
    ):
        resolve_experiment(
            "hubert_base",
            ["layerwise_linear_probe", "attnlrp_time", "dft_lrp_frequency"],
        )


def test_registry_rejects_unknown_encoder_profile():
    with pytest.raises(ValueError, match="profile"):
        resolve_experiment("not_a_profile", ["layerwise_linear_probe"])


def test_registry_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        get_method_spec("not_a_method")


def test_registry_rejects_missing_dependency():
    with pytest.raises(ValueError, match="dependency"):
        resolve_experiment("hubert_base", ["dft_lrp_frequency"])
