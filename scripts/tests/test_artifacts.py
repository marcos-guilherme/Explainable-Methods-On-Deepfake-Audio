import numpy as np
import pandas as pd

from brspeech_xai.artifacts import RunPaths, save_npy, load_npy, save_table, load_table, \
    write_done, is_done


def test_npy_roundtrip(tmp_path):
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    p = tmp_path / "a.npy"
    save_npy(arr, p)
    np.testing.assert_array_equal(load_npy(p), arr)


def test_table_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    p = tmp_path / "t.parquet"
    save_table(df, p)
    pd.testing.assert_frame_equal(load_table(p), df)


def test_done_marker(tmp_path):
    rp = RunPaths(root=tmp_path)
    assert not is_done(rp, "collect", "hash123")
    write_done(rp, "collect", "hash123", meta={"n": 40})
    assert is_done(rp, "collect", "hash123")
    # hash diferente invalida o marcador
    assert not is_done(rp, "collect", "hashXYZ")
