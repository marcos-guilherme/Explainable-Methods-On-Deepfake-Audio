"""AttnLRP (Achtibat et al., 2024) escrito à mão para encoders SSL da HuggingFace:
wav2vec2 e hubert (atenção padrão) e wavlm (atenção com viés de posição relativa e gating).

Propaga a relevância do logit de "spoof" do D_ad até a forma de onda de entrada de modo
CONSERVATIVO: em cada não-linearidade do Transformer aplicamos a regra do AttnLRP, para a
soma da relevância ser preservada camada a camada. Isso corrige o Gradient×Input cru, que
não conserva ao atravessar softmax/normalização e por isso não é fiel.

Por que à mão: a biblioteca de referência (LXT) não importa com ``transformers`` 5.x (usa um
símbolo removido de ``transformers.pytorch_utils``) e o zoo dela não cobre encoders de áudio
SSL. As três regras primitivas do AttnLRP, porém, são pequenas e só dependem de
``torch.autograd``; então as reimplementamos aqui e escrevemos o "patch" do wav2vec2.

Regras (formulação eficiente Input×Gradient, Achtibat et al., 2024):
  * ativação elementar (GELU): regra da identidade (Eq. 9). O Jacobiano local f'(x) é trocado
    por f(x)/x, então a relevância passa proporcionalmente, sem criar nem destruir relevância.
  * normalização (LayerNorm/GroupNorm): regra da identidade. O desvio-padrão (denominador) é
    destacado do grafo (stop-gradient), tratando a normalização como escala constante.
  * atenção: usamos CP-LRP (Ali et al., 2022) por padrão, que destaca Q e K do grafo. Assim o
    mapa de atenção vira uma constante e a atenção fica linear em V, conservando a relevância
    exatamente. Isso evita "explicar o softmax", que é a parte não-conservativa do AttnLRP puro
    (o próprio LXT admite isso e usa CP-LRP como padrão no GPT-2). A relevância flui pelos
    valores (V) e pelo resíduo. A regra uniforme do AttnLRP (Eq. 7) fica disponível para
    comparação, mas não conserva exatamente (ver testes).
  * linear/conv/matmul/média: autograd puro já é a regra ε (com ε→0).

Conservação exata vale para uma rede SEM viés (funções positivamente homogêneas de grau 1, em
que Euler dá ``sum_i x_i * dOut/dx_i == Out``). Com viés, a diferença é a relevância absorvida
pelos vieses (um "sumidouro" pequeno). O portão de conservação do runner/teste mede isso.

Créditos: as três primitivas abaixo reproduzem ``lxt/efficient/rules.py`` de
rachtibat/LRP-eXplains-Transformers (licença Clear BSD), dos autores do AttnLRP.
"""
from __future__ import annotations

import types
from typing import Callable, Collection, Mapping

import torch
from torch.autograd import Function


class _IdentityRule(Function):
    """Regra da identidade (AttnLRP, Eq. 9) para uma não-linearidade elementar ``fn``.

    forward: y = fn(x) e guarda o "Jacobiano efetivo" fn(x)/x.
    backward: usa fn(x)/x no lugar de fn'(x), de modo que a relevância R = x·grad conserva.
    """

    @staticmethod
    def forward(ctx, fn, x, eps=1e-10):
        y = fn(x)
        if x.requires_grad:
            ctx.save_for_backward(y / (x + eps))
        return y

    @staticmethod
    def backward(ctx, grad_out):
        (jac,) = ctx.saved_tensors
        return None, jac * grad_out, None


def identity_rule(fn, x: torch.Tensor) -> torch.Tensor:
    """Aplica a regra da identidade a uma função elementar ``fn`` avaliada em ``x``."""
    return _IdentityRule.apply(fn, x)


class _DivideGradient(Function):
    """Regra uniforme (AttnLRP, Eq. 7): identidade no forward; no backward divide a
    relevância por ``factor``, repartindo-a entre os fatores de um produto bilinear."""

    @staticmethod
    def forward(ctx, x, factor):
        ctx.factor = factor
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out / ctx.factor, None


def divide_gradient(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Divide o gradiente/relevância que passa por ``x`` por ``factor`` (regra uniforme)."""
    return _DivideGradient.apply(x, factor)


def stop_gradient(x: torch.Tensor) -> torch.Tensor:
    """Interrompe o fluxo de gradiente por ``x`` (usado no denominador da normalização)."""
    return x.detach()


def _preserve_forward_value(
    reference: torch.Tensor,
    custom_gradient: torch.Tensor,
) -> torch.Tensor:
    """Use the exact reference value while keeping the custom backward graph."""
    return reference.detach() + (custom_gradient - custom_gradient.detach())


class _ScaleGradient(Function):
    """Escala SÓ o backward por ``factor`` (>0), mantendo o forward idêntico.

    Usada para estabilizar a norma no LRP: pomos um piso no desvio-padrão apenas na passagem de
    relevância (backward), sem alterar o valor normalizado (forward)."""

    @staticmethod
    def forward(ctx, x, factor):
        ctx.save_for_backward(factor)
        return x

    @staticmethod
    def backward(ctx, grad_out):
        (factor,) = ctx.saved_tensors
        return grad_out * factor, None


def scale_gradient(x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """Multiplica o gradiente/relevância que passa por ``x`` por ``factor`` (forward inalterado)."""
    return _ScaleGradient.apply(x, factor)


def _norm_damping(numerator: torch.Tensor, var: torch.Tensor, std: torch.Tensor,
                  eps: float, kappa: float) -> torch.Tensor:
    """Piso relativo no ``std`` do backward da norma (estilo LRP-ε aplicado à normalização).

    O forward continua exato (``numerator/std``). No backward, o gradiente enxerga ``1/std_floored``
    com ``std_floored = sqrt(var + kappa*var_ref)``, onde ``var_ref`` é a variância média das
    unidades normalizadas (linha do LayerNorm, grupo do GroupNorm). Canais com ``var << var_ref``
    deixam de amplificar a relevância via ``1/std``. ``kappa=0`` recupera a norma pura (identidade).
    """
    if kappa <= 0:
        return numerator
    var_ref = var.mean()
    std_floored = (var + kappa * var_ref + eps).sqrt()
    factor = (std / std_floored).detach()
    return scale_gradient(numerator, factor)


def _identity_layer_norm_forward(self, x: torch.Tensor) -> torch.Tensor:
    """LayerNorm com regra da identidade: normaliza pela última dimensão com o desvio-padrão
    destacado do grafo, então o gradiente enxerga uma escala constante (relevância conserva).

    Se ``self._lrp_norm_stab > 0``, aplica um piso relativo no ``std`` do backward (denoising de
    canais de baixa variância), sem mudar o forward."""
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    std = (var + self.eps).sqrt()
    numerator = _norm_damping(x - mean, var, std, self.eps, getattr(self, "_lrp_norm_stab", 0.0))
    y = numerator / stop_gradient(std)
    if self.weight is not None:
        y = y * self.weight
    if self.bias is not None:
        y = y + self.bias
    original_forward = getattr(self, "_attnlrp_original_forward", None)
    if original_forward is not None:
        with torch.no_grad():
            reference = original_forward(x)
        y = _preserve_forward_value(reference, y)
    return y


def _identity_group_norm_forward(self, x: torch.Tensor) -> torch.Tensor:
    """GroupNorm com regra da identidade (usada na 1ª conv do extrator do wav2vec2).

    Normaliza dentro de cada grupo (canais do grupo + dimensões espaciais) com o desvio-padrão
    destacado do grafo. Suporta entrada ``(N, C, *espacial)``. Se ``self._lrp_norm_stab > 0``,
    aplica o piso relativo no ``std`` do backward (denoising), sem mudar o forward."""
    n, c = x.shape[0], x.shape[1]
    g = self.num_groups
    x_grouped = x.view(n, g, c // g, *x.shape[2:])
    dims = tuple(range(2, x_grouped.dim()))
    mean = x_grouped.mean(dim=dims, keepdim=True)
    var = ((x_grouped - mean) ** 2).mean(dim=dims, keepdim=True)
    std = (var + self.eps).sqrt()
    numerator = _norm_damping(x_grouped - mean, var, std, self.eps,
                              getattr(self, "_lrp_norm_stab", 0.0))
    x_grouped = numerator / stop_gradient(std)
    y = x_grouped.reshape_as(x)
    if self.affine:
        shape = (1, c) + (1,) * (x.dim() - 2)
        y = y * self.weight.view(shape) + self.bias.view(shape)
    original_forward = getattr(self, "_attnlrp_original_forward", None)
    if original_forward is not None:
        with torch.no_grad():
            reference = original_forward(x)
        y = _preserve_forward_value(reference, y)
    return y


def _project_qkv(self, hidden_states):
    """Projeções Q/K/V do wav2vec2 no formato (B, cabeças, T, head_dim)."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    return input_shape, query, key, value


def _finish_attention(self, input_shape, attn_weights, value):
    """Segundo matmul + projeção de saída, comum às duas variantes de atenção."""
    attn_output = torch.matmul(attn_weights, value).transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.out_proj(attn_output), attn_weights


def _attention_forward_cp(self, hidden_states, key_value_states=None, attention_mask=None,
                          output_attentions=False, **kwargs):
    """Atenção CP-LRP (Ali et al., 2022): destaca Q e K, então o mapa de atenção vira uma
    CONSTANTE e a atenção fica linear em V. Assim a relevância conserva exatamente pela
    atenção (a menos dos vieses), sem depender de "explicar" o softmax (a parte não-conservativa
    do AttnLRP puro). A relevância é atribuída pelo caminho dos valores (V) e pelo resíduo.
    Reproduz ``Wav2Vec2Attention.forward`` (sem dropout, pois rodamos em eval).
    """
    input_shape, query, key, value = _project_qkv(self, hidden_states)
    query = stop_gradient(query)
    key = stop_gradient(key)
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * self.scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = torch.softmax(attn_weights, dim=-1)
    out, weights = _finish_attention(self, input_shape, attn_weights, value)
    original_forward = getattr(self, "_attnlrp_original_forward", None)
    if original_forward is not None:
        with torch.no_grad():
            reference = original_forward(
                hidden_states,
                key_value_states=key_value_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                **kwargs,
            )
        out = _preserve_forward_value(reference[0], out)
        weights = reference[1]
    return out, weights, None


def _attention_forward_uniform(self, hidden_states, key_value_states=None, attention_mask=None,
                               output_attentions=False, **kwargs):
    """Atenção com a regra uniforme do AttnLRP (Eq. 7) em Q (/4), K (/4) e V (/2), deixando o
    softmax com o gradiente padrão. Mantida para COMPARAÇÃO: como o softmax não é conservativo
    sob gradiente padrão, esta variante não conserva exatamente (ver testes)."""
    input_shape, query, key, value = _project_qkv(self, hidden_states)
    query = divide_gradient(query, 4)
    key = divide_gradient(key, 4)
    value = divide_gradient(value, 2)
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * self.scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = torch.softmax(attn_weights, dim=-1)
    out, weights = _finish_attention(self, input_shape, attn_weights, value)
    original_forward = getattr(self, "_attnlrp_original_forward", None)
    if original_forward is not None:
        with torch.no_grad():
            reference = original_forward(
                hidden_states,
                key_value_states=key_value_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                **kwargs,
            )
        out = _preserve_forward_value(reference[0], out)
        weights = reference[1]
    return out, weights, None


def _attention_forward_cp_wavlm(self, hidden_states, attention_mask=None, position_bias=None,
                                output_attentions=False, index=0):
    """CP-LRP para a atenção do WavLM (viés de posição relativa com gating).

    Reproduz ``WavLMAttention.forward`` + ``torch_multi_head_self_attention`` À MÃO (sem a op
    fundida ``F.multi_head_attention_forward``), para inserir os stop-gradients do CP-LRP: destaca
    Q, K e o ``gated_position_bias`` do grafo, então o mapa de atenção vira uma CONSTANTE e a saída
    fica linear em V (conserva a relevância, a menos dos vieses). O forward é numericamente igual ao
    original (o detach só afeta o backward), então as predições não mudam. Roda em eval (sem
    dropout). Devolve ``position_bias`` para a camada repassar às seguintes.
    """
    bsz, tgt_len, _ = hidden_states.size()

    # Viés de posição relativa (idêntico ao original): criado na 1ª camada e repassado adiante.
    if position_bias is None:
        position_bias = self.compute_bias(tgt_len, tgt_len)
        position_bias = (position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1)
                         .view(bsz * self.num_heads, tgt_len, tgt_len))

    # Gate do viés a partir do hidden_states (cópia verbatim do original).
    gated_hidden_states = hidden_states.view(hidden_states.shape[:-1] + (self.num_heads, -1))
    gated_hidden_states = gated_hidden_states.permute(0, 2, 1, 3)
    relative_position_proj = self.gru_rel_pos_linear(gated_hidden_states)
    relative_position_proj = relative_position_proj.view(
        gated_hidden_states.shape[:-1] + (2, 4)).sum(-1)
    gate_a, gate_b = torch.sigmoid(relative_position_proj).chunk(2, dim=-1)
    gate_output = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0
    gated_position_bias = gate_output.view(bsz * self.num_heads, -1, 1) * position_bias
    gated_position_bias = gated_position_bias.view(bsz, self.num_heads, tgt_len, tgt_len)

    # Atenção multi-cabeça à mão. q=k=v=hidden_states (auto-atenção).
    def _heads(x):
        return x.view(bsz, tgt_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    q = _heads(torch.nn.functional.linear(hidden_states, self.q_proj.weight, self.q_proj.bias))
    k = _heads(torch.nn.functional.linear(hidden_states, self.k_proj.weight, self.k_proj.bias))
    v = _heads(torch.nn.functional.linear(hidden_states, self.v_proj.weight, self.v_proj.bias))

    # CP-LRP: Q, K e o viés de posição saem do grafo -> mapa de atenção constante, linear em V.
    attn = torch.matmul(stop_gradient(q), stop_gradient(k).transpose(-2, -1)) * self.scaling
    attn = attn + stop_gradient(gated_position_bias)
    if attention_mask is not None:                       # padding de chave (não ocorre em clipes fixos)
        attn = attn.masked_fill((attention_mask == 0).view(bsz, 1, 1, tgt_len), float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(bsz, tgt_len, self.embed_dim)
    out = torch.nn.functional.linear(out, self.out_proj.weight, self.out_proj.bias)
    weights = attn if output_attentions else None
    original_forward = getattr(self, "_attnlrp_original_forward", None)
    if original_forward is not None:
        with torch.no_grad():
            reference = original_forward(
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                output_attentions=output_attentions,
                index=index,
            )
        out = _preserve_forward_value(reference[0], out)
        weights = reference[1]
        position_bias = reference[2]
    return out, weights, position_bias


def _in_feature_extractor(name: str) -> bool:
    """A norma pertence ao front-end convolucional (extrator ou projeção de features)?

    No wav2vec2-base (``feat_extract_norm="group"``) só a conv0 tem norma (GroupNorm) e a
    ``feature_projection.layer_norm`` fica logo após o extrator. São exatamente as normas de
    baixa variância que amplificam a relevância; as 26 LayerNorms do transformer ficam de fora."""
    return name.startswith("feature_extractor") or name.startswith("feature_projection")


# Classes de atenção que compartilham a mesma estrutura padrão (q/k/v/out, head_dim, scaling)
# e, por isso, aceitam o mesmo forward CP-LRP. O WavLM tem forward próprio (viés de posição
# relativa com gating), tratado por ``_attention_forward_cp_wavlm``.
_STANDARD_ATTENTION_CLASSES = {"Wav2Vec2Attention", "HubertAttention"}
_PATCH_CATEGORIES = ("layer_norm", "group_norm", "gelu", "attention")


def _attnlrp_module_category(module: torch.nn.Module) -> str | None:
    if isinstance(module, torch.nn.LayerNorm):
        return "layer_norm"
    if isinstance(module, torch.nn.GroupNorm):
        return "group_norm"
    if type(module).__name__ == "GELUActivation":
        return "gelu"
    if (
        type(module).__name__ in _STANDARD_ATTENTION_CLASSES
        or type(module).__name__ == "WavLMAttention"
    ):
        return "attention"
    return None


def _attnlrp_modules(model: torch.nn.Module):
    for name, module in model.named_modules():
        category = _attnlrp_module_category(module)
        if category is not None:
            yield name, module, category


def attnlrp_patch_coverage(model: torch.nn.Module) -> dict[str, int]:
    """Conta exatamente os módulos que ``patch_ssl_encoder_for_attnlrp`` descobre."""
    counts = dict.fromkeys(_PATCH_CATEGORIES, 0)
    for _, _, category in _attnlrp_modules(model):
        counts[category] += 1
    return counts


def patch_ssl_encoder_for_attnlrp(model: torch.nn.Module,
                                  attention: str = "cp",
                                  norm_stabilizer: float = 0.0,
                                  norm_stab_scope: str = "extractor") -> dict[str, int]:
    """Reescreve, in place, os forwards das não-linearidades do encoder SSL com as regras do LRP.

    Cobre wav2vec2 e hubert (atenção padrão q/k/v/out, forward ``attention``) e wavlm (atenção com
    viés de posição relativa e gating, forward CP-LRP dedicado que destaca Q, K e o viés). Regras:
    GELU (identidade), LayerNorm/GroupNorm (identidade), atenção (``attention``):
      * "cp"      : CP-LRP, conservativo por construção (padrão);
      * "uniform" : regra uniforme do AttnLRP no softmax (para comparação, não conserva exato).

    ``norm_stabilizer`` (κ ≥ 0): piso relativo no ``std`` do backward das normas. κ=0 mantém a
    regra da identidade pura (conservativa); κ>0 amortece canais de baixa variância (denoising),
    ao custo de relaxar um pouco a conservação exata.

    ``norm_stab_scope``: onde κ vale. "extractor" (padrão) aplica só ao front-end convolucional
    (GroupNorm da conv0 + ``feature_projection.layer_norm``), que é onde estão as normas de baixa
    variância; o transformer, que já conserva bem, fica com κ=0. "all" aplica a todas as normas
    (amortecimento composto nas LayerNorms => colapsa a relevância; mantido só para comparação).

    Deve ser chamado UMA vez sobre o encoder (``AutoModel``) antes do backward. O patch é por
    instância (``types.MethodType``), então é cirúrgico e não afeta outros modelos.

    Returns:
        Contagem por tipo de módulo patcheado (se algum tipo vier zerado, o patch não pegou
        e o portão de conservação vai denunciar).
    """
    if attention not in ("cp", "uniform"):
        raise ValueError(f"attention={attention!r} inválido; use 'cp' ou 'uniform'.")
    if norm_stab_scope not in ("extractor", "all"):
        raise ValueError(f"norm_stab_scope={norm_stab_scope!r} inválido; use 'extractor' ou 'all'.")
    attn_forward = _attention_forward_cp if attention == "cp" else _attention_forward_uniform

    def _kappa_for(name: str) -> float:
        if norm_stab_scope == "all" or _in_feature_extractor(name):
            return norm_stabilizer
        return 0.0

    counts = dict.fromkeys(_PATCH_CATEGORIES, 0)
    for name, module, category in _attnlrp_modules(model):
        if type(module).forward is not torch.nn.Module.forward:
            module._attnlrp_original_forward = module.forward
        if category == "layer_norm":
            module._lrp_norm_stab = _kappa_for(name)
            module.forward = types.MethodType(_identity_layer_norm_forward, module)
            counts["layer_norm"] += 1
        elif category == "group_norm":
            module._lrp_norm_stab = _kappa_for(name)
            module.forward = types.MethodType(_identity_group_norm_forward, module)
            counts["group_norm"] += 1
        elif category == "gelu":
            original_forward = module.forward
            module.forward = types.MethodType(
                lambda self, x, _fn=original_forward: identity_rule(_fn, x), module)
            counts["gelu"] += 1
        elif category == "attention":
            module.forward = types.MethodType(
                _attention_forward_cp_wavlm
                if type(module).__name__ == "WavLMAttention"
                else attn_forward,
                module,
            )
            counts["attention"] += 1
    return counts


def ensure_ssl_encoder_attnlrp(
    encoder: torch.nn.Module,
    *,
    attention_rule: str,
    capabilities: Collection[str],
    patch_fn: Callable | None = patch_ssl_encoder_for_attnlrp,
) -> Mapping[str, object]:
    """Aplica ou valida um patch AttnLRP autenticado pelo contrato compartilhado."""
    rules = {
        "cp_lrp": ("cp", "attnlrp_cp"),
        "uniform_lrp": ("uniform", "attnlrp_uniform"),
    }
    if attention_rule not in rules:
        raise ValueError(f"attention_rule inválida: {attention_rule!r}")
    attention, capability = rules[attention_rule]
    if capability not in frozenset(capabilities):
        raise ValueError(
            f"attention_rule {attention_rule!r} exige capability {capability!r}"
        )
    expected_counts = attnlrp_patch_coverage(encoder)
    missing_categories = [
        name for name in _PATCH_CATEGORIES if expected_counts[name] <= 0
    ]
    if missing_categories:
        raise ValueError(
            "AttnLRP patch coverage is incomplete; architecture has zero "
            f"modules for {missing_categories}"
        )

    def exact_counts(raw_counts: object) -> dict[str, int]:
        if (
            not isinstance(raw_counts, Mapping)
            or set(raw_counts) != set(_PATCH_CATEGORIES)
            or any(type(raw_counts[name]) is not int for name in _PATCH_CATEGORIES)
        ):
            raise ValueError(
                "AttnLRP patch coverage deve conter exatamente "
                f"{list(_PATCH_CATEGORIES)}"
            )
        actual = {name: raw_counts[name] for name in _PATCH_CATEGORIES}
        if actual != expected_counts:
            mismatch = {
                name: {"expected": expected_counts[name], "actual": actual[name]}
                for name in _PATCH_CATEGORIES
                if actual[name] != expected_counts[name]
            }
            raise ValueError(f"AttnLRP patch coverage diverge da arquitetura: {mismatch}")
        return actual

    marker = getattr(encoder, "_brspeech_attnlrp_patch", None)
    if marker is None:
        if patch_fn is None:
            raise ValueError("encoder não possui marker de patch AttnLRP validado")
        counts = patch_fn(encoder, attention=attention)
        counts = exact_counts(counts)
        marker = {
            "schema_version": 1,
            "managed_by": "ensure_ssl_encoder_attnlrp",
            "attention_rule": attention_rule,
            "attention": attention,
            "capability": capability,
            "counts": counts,
        }
        try:
            setattr(encoder, "_brspeech_attnlrp_patch", marker)
        except (AttributeError, TypeError) as exc:
            raise ValueError("encoder não aceita marker AttnLRP validado") from exc
    expected_keys = {
        "schema_version",
        "managed_by",
        "attention_rule",
        "attention",
        "capability",
        "counts",
    }
    if (
        not isinstance(marker, Mapping)
        or set(marker) != expected_keys
        or marker.get("schema_version") != 1
        or marker.get("managed_by") != "ensure_ssl_encoder_attnlrp"
        or marker.get("attention_rule") != attention_rule
        or marker.get("attention") != attention
        or marker.get("capability") != capability
    ):
        raise ValueError("encoder contém marker AttnLRP arbitrário ou incompatível")
    exact_counts(marker.get("counts"))
    return marker


# Alias de compatibilidade: o nome antigo cobria só o wav2vec2, mas a função é a mesma.
patch_wav2vec2_for_attnlrp = patch_ssl_encoder_for_attnlrp
