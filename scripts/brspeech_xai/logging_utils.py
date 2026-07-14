"""Logger padronizado: console + arquivo no diretório do run."""
from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str = "brspeech_xai", logfile: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    # Anexa o handler de arquivo mesmo se o logger já tiver um console handler (ex.: quando
    # algum módulo chamou get_logger() no import antes de o run ter um diretório).
    if logfile is not None and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
