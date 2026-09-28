"""Testes de conservação das regras do AttnLRP (precisam de torch; rodam no container).

Ideia central: numa rede SEM viés, cada regra do AttnLRP é positivamente homogênea de grau 1,
então pelo teorema de Euler a relevância Input×Gradient conserva exatamente:
    sum_i x_i * dOut/dx_i == Out.
Cada teste monta um bloco minúsculo sem viés e confere esse balanço. É o portão que valida a
matemática das regras, independente do modelo real (onde os vieses causam um resíduo pequeno).
"""
import torch

from brspeech_xai.attnlrp import (_attention_forward_cp, _attention_forward_uniform,
                                  _identity_group_norm_forward, _identity_layer_norm_forward,
                                  divide_gradient, identity_rule,
                                  patch_ssl_encoder_for_attnlrp, patch_wav2vec2_for_attnlrp)


def _conservation_residual(x: torch.Tensor, out: torch.Tensor) -> float:
    """|sum_i x_i * dOut/dx_i - Out| para um Out escalar (a relevância total conservada)."""
    (grad,) = torch.autograd.grad(out, x)
    return float((x.detach() * grad).sum() - out.detach())


def test_identity_rule_conserves_through_activation():
    torch.manual_seed(0)
    d = 32
    x = torch.randn(d, dtype=torch.float64, requires_grad=True)
    w_in = torch.randn(d, d, dtype=torch.float64)
    u_out = torch.randn(d, dtype=torch.float64)
    z = x @ w_in.T                                   # linear sem viés
    y = identity_rule(torch.nn.functional.gelu, z)   # regra da identidade na GELU
    out = (u_out * y).sum()                           # leitura linear sem viés
    assert abs(_conservation_residual(x, out)) < 1e-9


def test_divide_gradient_is_identity_in_forward():
    x = torch.randn(10, dtype=torch.float64)
    assert torch.allclose(divide_gradient(x, 4), x)


def test_layer_norm_identity_rule_conserves():
    torch.manual_seed(1)
    d = 48
    x = torch.randn(4, d, dtype=torch.float64, requires_grad=True)
    ln = torch.nn.LayerNorm(d, elementwise_affine=True).double()
    torch.nn.init.normal_(ln.weight)
    torch.nn.init.zeros_(ln.bias)                    # sem viés => conservação exata
    ln.forward = _identity_layer_norm_forward.__get__(ln, type(ln))
    u_out = torch.randn(4, d, dtype=torch.float64)
    out = (u_out * ln(x)).sum()
    assert abs(_conservation_residual(x, out)) < 1e-8


def test_group_norm_identity_rule_conserves():
    torch.manual_seed(2)
    c, t = 16, 20
    x = torch.randn(2, c, t, dtype=torch.float64, requires_grad=True)
    gn = torch.nn.GroupNorm(num_groups=c, num_channels=c, affine=True).double()
    torch.nn.init.normal_(gn.weight)
    torch.nn.init.zeros_(gn.bias)                    # sem viés => conservação exata
    gn.forward = _identity_group_norm_forward.__get__(gn, type(gn))
    u_out = torch.randn(2, c, t, dtype=torch.float64)
    out = (u_out * gn(x)).sum()
    assert abs(_conservation_residual(x, out)) < 1e-8


class _BareAttention(torch.nn.Module):
    """Bloco de atenção multi-cabeça sem viés (o forward é definido por teste)."""

    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(d, d, bias=False)
        self.k_proj = torch.nn.Linear(d, d, bias=False)
        self.v_proj = torch.nn.Linear(d, d, bias=False)
        self.out_proj = torch.nn.Linear(d, d, bias=False)
        self.num_heads = heads
        self.head_dim = d // heads
        self.scaling = self.head_dim ** -0.5


def _attn_residual(forward_fn) -> float:
    """Resíduo de conservação de um bloco de atenção sem viés com o ``forward_fn`` dado."""
    torch.manual_seed(3)
    d, heads, t = 24, 4, 7
    attn = _BareAttention(d, heads).double()
    attn.forward = forward_fn.__get__(attn, type(attn))
    x = torch.randn(1, t, d, dtype=torch.float64, requires_grad=True)
    u_out = torch.randn(1, t, d, dtype=torch.float64)
    out_tensor, _, _ = attn(x)
    out = (u_out * out_tensor).sum()
    return abs(_conservation_residual(x, out))


def test_attention_cp_conserves():
    """Portão principal: a atenção CP-LRP (destaca Q/K) conserva exatamente sem viés."""
    assert _attn_residual(_attention_forward_cp) < 1e-8


def test_attention_uniform_does_not_conserve_exactly():
    """A regra uniforme com softmax padrão NÃO conserva (motivo de usarmos CP-LRP)."""
    assert _attn_residual(_attention_forward_uniform) > 1e-3


class _Wav2Vec2Attention(_BareAttention):
    """Nome de classe = o que o transformers usa; o patch casa por nome."""


class _HubertAttention(_BareAttention):
    """HuBERT reusa a atenção padrão; o patch deve casá-la igual ao wav2vec2."""


def test_patch_matches_standard_attention_by_class_name():
    """O patch generalizado troca o forward de atenções wav2vec2 E hubert (casadas por nome).

    Renomeamos as subclasses para os nomes exatos do transformers, montamos um mini-encoder e
    conferimos que ambas foram patcheadas (contagem=2) sem precisar baixar um modelo real."""
    _Wav2Vec2Attention.__name__ = "Wav2Vec2Attention"
    _HubertAttention.__name__ = "HubertAttention"
    encoder = torch.nn.Module()
    encoder.w2v2_attn = _Wav2Vec2Attention(24, 4)
    encoder.hubert_attn = _HubertAttention(24, 4)
    counts = patch_ssl_encoder_for_attnlrp(encoder, attention="cp")
    assert counts["attention"] == 2

    x = torch.randn(1, 7, 24, dtype=torch.float32, requires_grad=True)
    out, weights, _ = encoder.hubert_attn(x)              # forward CP-LRP já ativo
    assert out.shape == x.shape and weights is not None


class _WavLMAttention(torch.nn.Module):
    """Nome exato da classe do transformers; o patch tem regra CP-LRP dedicada p/ o WavLM."""


def test_patch_matches_wavlm_attention_by_class_name():
    """O WavLM tem forward próprio (viés de posição com gating); o patch deve casá-lo por nome.

    Só validamos o DISPATCH (contagem=1): o forward do WavLM depende de atributos reais do modelo
    (compute_bias, gru_rel_pos_linear, ...), então a conservação é aferida em runtime pelo
    certificado, não aqui."""
    _WavLMAttention.__name__ = "WavLMAttention"
    encoder = torch.nn.Module()
    encoder.wavlm_attn = _WavLMAttention()
    counts = patch_ssl_encoder_for_attnlrp(encoder, attention="cp")
    assert counts["attention"] == 1
    assert encoder.wavlm_attn.forward.__func__.__name__ == "_attention_forward_cp_wavlm"


def test_backward_compat_alias_points_to_generalized_patch():
    """O nome antigo (`patch_wav2vec2_for_attnlrp`) segue válido como alias."""
    assert patch_wav2vec2_for_attnlrp is patch_ssl_encoder_for_attnlrp
