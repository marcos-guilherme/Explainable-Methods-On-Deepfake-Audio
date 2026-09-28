"""Gating de D_zs: detectores presentes derivam das colunas da tabela mestra."""
import pandas as pd

from brspeech_xai.stages import _present_detectors


def test_present_detectors_full():
    master = pd.DataFrame({"p_spoof_zs": [0.1], "p_spoof_ad": [0.2]})
    assert _present_detectors(master) == ["zs", "ad"]


def test_present_detectors_ad_only():
    # Encoder só-extrator (has_zero_shot=False) não gera D_zs.
    master = pd.DataFrame({"p_spoof_ad": [0.2]})
    assert _present_detectors(master) == ["ad"]
