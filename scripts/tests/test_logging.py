from brspeech_xai.logging_utils import format_duration, get_logger, progress


def test_format_duration_scales():
    assert format_duration(12.34) == "12.3s"
    assert format_duration(90) == "1m30s"
    assert format_duration(3723) == "1h02m03s"


def test_progress_is_transparent_iterator():
    assert list(progress([1, 2, 3], desc="x")) == [1, 2, 3]
    assert list(progress(range(3), total=3)) == [0, 1, 2]


def test_get_logger_is_shared_and_writes_file(tmp_path):
    logfile = tmp_path / "run.log"
    logger = get_logger(logfile=logfile)
    assert get_logger() is logger  # loguru é um logger global compartilhado
    logger.info("marcador-de-teste")
    assert "marcador-de-teste" in logfile.read_text()
