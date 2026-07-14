import numpy as np
import pandas as pd

from brspeech_xai.stats import confirmatory_tests


def test_confirmatory_runs_on_top_features():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "mfcc2_mean": rng.normal(size=n),
        "p_spoof_zs": rng.uniform(size=n),
        "quadrant_zs": rng.choice(["TP", "TN", "FP", "FN"], size=n),
    })
    out = confirmatory_tests(df, "zs", "quadrant_zs", ["mfcc2_mean"])
    assert {"detector", "feature", "test", "statistic", "p_value", "q_value_fdr"}.issubset(out.columns)
