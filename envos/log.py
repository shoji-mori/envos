"""envos logging (P2-C): logger, setup(), StandardFormatter, update_logfile() shim."""
import logging
import warnings
from pathlib import Path


def _to_level(name: str) -> int:
    return logging.getLevelName(name.upper())


class StandardFormatter(logging.Formatter):
    """Per-level format strings for the envos logger."""

    _fmtdict = {
        logging.DEBUG: "(Debug) %(message)s",
        logging.INFO: "%(message)s",
        logging.WARNING: "Warning! -- %(message)s",
        logging.ERROR: "!!ERROR!! %(message)s",
    }

    def __init__(self):
        super().__init__(fmt="[%(filename)s] %(levelname)s: %(message)s")

    def format(self, record):
        fmt_orig = self._style._fmt
        self._style._fmt = self._fmtdict.get(record.levelno, fmt_orig)
        result = logging.Formatter.format(self, record)
        self._style._fmt = fmt_orig
        return result


def setup(level: str = "INFO", logfile=None, file_level: str = None) -> None:
    """(Re-)configure the envos logger.  Safe to call multiple times.

    Clears existing handlers and attaches a StreamHandler at *level* and,
    if *logfile* is given, a FileHandler at *file_level* (defaults to *level*).
    """
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)

    logger.setLevel(logging.DEBUG)  # handlers control granularity

    sh = logging.StreamHandler()
    sh.setFormatter(StandardFormatter())
    sh.setLevel(_to_level(level))
    logger.addHandler(sh)

    if logfile is not None:
        logfile = Path(logfile)
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, mode="a", encoding="utf-8")
        fh.setFormatter(StandardFormatter())
        fh.setLevel(_to_level(file_level or level))
        logger.addHandler(fh)


def update_logfile(**kw) -> None:
    """Deprecated: use ``setup()`` instead.  Forwards to :func:`setup`."""
    warnings.warn(
        "envos.log.update_logfile() is deprecated; use envos.log.setup() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    logfile = kw.get("filepath", None)
    file_level = kw.get("level", None)
    stream_level = "INFO"
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            stream_level = logging.getLevelName(h.level)
            break
    setup(level=stream_level, logfile=logfile, file_level=file_level)


# ---------------------------------------------------------------------------
# Module-level initialisation
# ---------------------------------------------------------------------------
logger = logging.getLogger("envos")
logger.propagate = False
setup(level="INFO")
