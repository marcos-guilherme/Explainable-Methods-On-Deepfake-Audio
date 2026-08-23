"""Logging estruturado (loguru) e barras de progresso (tqdm) com estimativa de tempo.

Um único logger loguru é compartilhado pelo processo: o console é sempre ligado; o
arquivo do run é anexado quando o diretório existe. As barras (``progress``) escrevem
ETA e, em execução headless (log redirecionado), imprimem linhas periódicas com o tempo
restante, então dá para acompanhar um run em 2º plano só olhando o arquivo de log.
"""
from __future__ import annotations

import sys
from pathlib import Path

from tqdm.auto import tqdm

# Console: hora curta + nível colorido. Arquivo: data completa, sem cor.
_CONSOLE_FMT = "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}"
_FILE_FMT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}"

_console_ready = False
_file_sinks: set[str] = set()


def _ensure_console() -> None:
    """Liga o sink de console uma única vez, escrevendo via ``tqdm.write``.

    Passar pelo ``tqdm.write`` evita que as barras de progresso sejam quebradas pelas
    linhas de log quando as duas coisas disputam o terminal.
    """
    global _console_ready
    if _console_ready:
        return
    from loguru import logger
    logger.remove()  # tira o handler default do loguru (stderr cru)
    # Cor só em terminal interativo; em log redirecionado sai texto limpo (sem ANSI).
    logger.add(lambda m: tqdm.write(m, end=""), format=_CONSOLE_FMT, level="INFO",
               colorize=sys.stderr.isatty())
    _console_ready = True


def get_logger(name: str = "brspeech_xai", logfile: str | Path | None = None):
    """Logger loguru compartilhado. Console sempre; arquivo do run quando informado.

    Idempotente: chamar de novo não duplica sinks. O ``name`` é mantido por
    compatibilidade de assinatura (loguru usa um logger global).
    """
    _ensure_console()
    from loguru import logger
    if logfile is not None and str(logfile) not in _file_sinks:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        logger.add(str(logfile), format=_FILE_FMT, level="INFO", encoding="utf-8")
        _file_sinks.add(str(logfile))
    return logger


def progress(iterable=None, *, total=None, desc=None, unit="it"):
    """Barra de progresso com ETA, amigável a execução headless.

    ``mininterval`` alto limita a frequência de atualização: no terminal a barra é
    fluida; no log redirecionado saem linhas espaçadas com percentual e tempo restante.
    """
    return tqdm(iterable, total=total, desc=desc, unit=unit,
                dynamic_ncols=True, mininterval=5.0, smoothing=0.1)


def format_duration(seconds: float) -> str:
    """Duração legível: ``12.3s``, ``4m07s`` ou ``1h02m03s``."""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"
