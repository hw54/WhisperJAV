#!/usr/bin/env python3
"""Simple logging setup for WhisperJAV."""

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, TextIO

try:  # pragma: no cover - best-effort import for color support
    from colorama import Fore, Style, init as colorama_init

    colorama_init()  # ensure ANSI works on Windows terminals
    _USE_COLOR = True
    _COLOR_MAP = {
        "blue": Fore.BLUE,
        "green": Fore.GREEN,
        "yellow": Fore.YELLOW,
        "red": Fore.RED,
        "cyan": Fore.CYAN,
        "magenta": Fore.MAGENTA,
    }
    _COLOR_RESET = Style.RESET_ALL
except Exception:  # pragma: no cover - color fallback when colorama missing
    _USE_COLOR = sys.stdout.isatty()
    _COLOR_MAP = {
        "blue": "\033[34m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "red": "\033[31m",
        "cyan": "\033[36m",
        "magenta": "\033[35m",
    }
    _COLOR_RESET = "\033[0m"


def color_text(message: str, color: str) -> str:
    """Return message wrapped with ANSI color codes when supported."""

    if not _USE_COLOR:
        return message

    prefix = _COLOR_MAP.get(color)
    if not prefix:
        return message

    return f"{prefix}{message}{_COLOR_RESET}"


class ColorFormatter(logging.Formatter):
    """Formatter that applies ANSI colors based on record metadata."""

    def __init__(self, *args, **kwargs):
        self.enable_color = kwargs.pop("enable_color", True)
        super().__init__(*args, **kwargs)

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        if not self.enable_color:
            return formatted

        color = getattr(record, "color", None)
        if not color:
            return formatted

        return color_text(formatted, color)


class TqdmLoggingHandler(logging.StreamHandler):
    """Console logging handler that writes without corrupting tqdm bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from tqdm import tqdm

            msg = self.format(record)
            tqdm.write(msg, file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


@contextmanager
def tqdm_console_logging(
    name: str = "whisperjav",
    *,
    file: TextIO | None = None,
) -> Iterator[None]:
    """Route console logs through tqdm.write while progress bars are active."""

    logger = logging.getLogger(name)
    original_handlers = list(logger.handlers)
    replacement_handlers: list[logging.Handler] = []

    for handler in original_handlers:
        if _is_console_stream_handler(handler):
            tqdm_handler = TqdmLoggingHandler(file or handler.stream)
            tqdm_handler.setLevel(handler.level)
            tqdm_handler.setFormatter(handler.formatter)
            for log_filter in handler.filters:
                tqdm_handler.addFilter(log_filter)
            replacement_handlers.append(tqdm_handler)
        else:
            replacement_handlers.append(handler)

    logger.handlers = replacement_handlers
    try:
        yield
    finally:
        logger.handlers = original_handlers


def _is_console_stream_handler(handler: logging.Handler) -> bool:
    return (
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
    )


def setup_logger(name: str = "whisperjav",
                log_level: str = "INFO",
                log_file: Optional[str] = None) -> logging.Logger:
    """Setup logger with console and optional file output."""

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers and prevent propagation to root logger.
    # Without this, messages duplicate when any dependency triggers
    # logging.basicConfig() — the root handler re-emits our messages
    # through stderr, doubling subprocess pipe traffic and risking
    # pipe buffer deadlocks on Windows GUI subprocesses.
    logger.handlers = []
    logger.propagate = False

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = ColorFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        enable_color=_USE_COLOR,
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


# Global logger instance
logger = setup_logger()
