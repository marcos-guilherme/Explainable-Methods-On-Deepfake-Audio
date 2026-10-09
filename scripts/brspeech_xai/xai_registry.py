"""Registries declarativos de perfis de encoder e métodos XAI layer-wise."""
from __future__ import annotations

from dataclasses import dataclass

_LAYERWISE_CAPABILITIES = frozenset(
    {
        "all_hidden_states",
        "layerwise_probe",
        "attnlrp_cp",
        "dft_lrp",
        "layer_relevance_hooks",
    }
)

_ENCODER_REGISTRY: dict[str, EncoderExperimentSpec] = {}
_METHOD_REGISTRY: dict[str, XaiMethodSpec] = {}


@dataclass(frozen=True)
class EncoderExperimentSpec:
    profile_id: str
    encoder: str
    checkpoint: str
    family: str
    n_transformer_layers: int
    layer_indices: tuple[int, ...]
    capabilities: frozenset[str]
    attention_rule: str
    xai_dtype: str


@dataclass(frozen=True)
class XaiMethodSpec:
    method_id: str
    required_capabilities: frozenset[str]
    dependencies: tuple[str, ...]
    primary: bool


@dataclass(frozen=True)
class ResolvedExperiment:
    encoder: EncoderExperimentSpec
    methods: tuple[XaiMethodSpec, ...]


def _register_encoder(spec: EncoderExperimentSpec) -> None:
    _ENCODER_REGISTRY[spec.profile_id] = spec


def _register_method(spec: XaiMethodSpec) -> None:
    _METHOD_REGISTRY[spec.method_id] = spec


def _build_registries() -> None:
    if _ENCODER_REGISTRY:
        return

    layer_indices = tuple(range(1, 13))
    for profile_id, checkpoint, family, attention_rule, xai_dtype in (
        (
            "hubert_base",
            "facebook/hubert-base-ls960",
            "hubert",
            "cp_lrp",
            "float32",
        ),
        (
            "wavlm_base",
            "microsoft/wavlm-base",
            "wavlm",
            "cp_lrp",
            "float32",
        ),
        (
            "wav2vec2_base",
            "facebook/wav2vec2-base",
            "wav2vec2",
            "cp_lrp",
            "float64",
        ),
    ):
        _register_encoder(
            EncoderExperimentSpec(
                profile_id=profile_id,
                encoder="hf_ssl",
                checkpoint=checkpoint,
                family=family,
                n_transformer_layers=12,
                layer_indices=layer_indices,
                capabilities=_LAYERWISE_CAPABILITIES,
                attention_rule=attention_rule,
                xai_dtype=xai_dtype,
            )
        )

    for method_id, required_capabilities, dependencies, primary in (
        ("layerwise_linear_probe", frozenset({"layerwise_probe"}), (), True),
        ("attnlrp_time", frozenset({"attnlrp_cp"}), ("layerwise_linear_probe",), True),
        ("dft_lrp_frequency", frozenset({"dft_lrp"}), ("attnlrp_time",), True),
        ("stdft_lrp_time_frequency", frozenset({"dft_lrp"}), ("attnlrp_time",), True),
        (
            "final_decision_layer_trace",
            frozenset({"layer_relevance_hooks"}),
            ("attnlrp_time",),
            True,
        ),
        ("classical_band_audit", frozenset(), (), False),
    ):
        _register_method(
            XaiMethodSpec(
                method_id=method_id,
                required_capabilities=required_capabilities,
                dependencies=dependencies,
                primary=primary,
            )
        )


def get_encoder_spec(profile_id: str) -> EncoderExperimentSpec:
    _build_registries()
    try:
        return _ENCODER_REGISTRY[profile_id]
    except KeyError as exc:
        raise ValueError(
            f"encoder profile {profile_id!r} is not registered"
        ) from exc


def get_method_spec(method_id: str) -> XaiMethodSpec:
    _build_registries()
    try:
        return _METHOD_REGISTRY[method_id]
    except KeyError as exc:
        raise ValueError(f"method {method_id!r} is not registered") from exc


def _topological_order(
    method_ids: tuple[str, ...],
    specs: dict[str, XaiMethodSpec],
) -> tuple[str, ...]:
    requested = set(method_ids)
    indegree = {method_id: 0 for method_id in method_ids}
    dependents: dict[str, list[str]] = {method_id: [] for method_id in method_ids}

    for method_id in method_ids:
        for dependency in specs[method_id].dependencies:
            if dependency not in requested:
                continue
            indegree[method_id] += 1
            dependents[dependency].append(method_id)

    ready = sorted(method_id for method_id, degree in indegree.items() if degree == 0)
    ordered: list[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for dependent in sorted(dependents[current]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
                ready.sort()

    if len(ordered) != len(method_ids):
        raise ValueError("method dependency cycle detected")
    return tuple(ordered)


def resolve_experiment(
    profile_id: str,
    method_ids: list[str],
) -> ResolvedExperiment:
    _build_registries()
    encoder = get_encoder_spec(profile_id)
    if not method_ids:
        raise ValueError(
            f"profile {profile_id!r} requires at least one method"
        )

    specs: dict[str, XaiMethodSpec] = {}
    for method_id in method_ids:
        if method_id not in _METHOD_REGISTRY:
            raise ValueError(
                f"profile {profile_id!r} does not support method "
                f"{method_id!r}: method not registered"
            )
        specs[method_id] = get_method_spec(method_id)

    requested = tuple(dict.fromkeys(method_ids))
    requested_set = set(requested)

    for method_id in requested:
        spec = specs[method_id]
        missing = spec.required_capabilities - encoder.capabilities
        if missing:
            missing_name = sorted(missing)[0]
            raise ValueError(
                f"profile {profile_id!r} lacks capability {missing_name!r} "
                f"for method {method_id!r}"
            )
        for dependency in spec.dependencies:
            if dependency not in requested_set:
                raise ValueError(
                    f"method {method_id!r} requires dependency {dependency!r} "
                    f"for profile {profile_id!r}"
                )

    ordered = _topological_order(requested, specs)
    return ResolvedExperiment(
        encoder=encoder,
        methods=tuple(specs[method_id] for method_id in ordered),
    )
