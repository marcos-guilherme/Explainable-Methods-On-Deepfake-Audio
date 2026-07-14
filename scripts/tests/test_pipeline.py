from brspeech_xai import pipeline as P
from brspeech_xai.artifacts import RunPaths, is_done


def test_pipeline_skips_done_stages(tmp_path, monkeypatch):
    calls = []

    def fake(name):
        def _f(ctx):
            calls.append(name)
        return _f

    monkeypatch.setattr(P, "STAGES", [("a", fake("a")), ("b", fake("b")), ("c", fake("c"))])

    class Cfg:
        def config_hash(self):
            return "h1"

    paths = RunPaths(root=tmp_path)
    P.run_stages(Cfg(), paths, logger=None, ctx_extra={}, start=None, only=None, force=False)
    assert calls == ["a", "b", "c"]
    assert is_done(paths, "a", "h1")

    # segunda execução: tudo done -> não chama nada
    calls.clear()
    P.run_stages(Cfg(), paths, logger=None, ctx_extra={}, start=None, only=None, force=False)
    assert calls == []

    # only='b' força só b
    calls.clear()
    P.run_stages(Cfg(), paths, logger=None, ctx_extra={}, start=None, only="b", force=True)
    assert calls == ["b"]
