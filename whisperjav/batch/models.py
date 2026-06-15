from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

BatchStatus = Literal[
    "skip_external_subtitle",
    "skip_translated",
    "translate_existing_japanese",
    "transcribe_then_translate",
    "failed_asr",
    "failed_translation",
    "failed_precondition",
    "cancelled",
    "completed",
    "completed_translation_only",
]


@dataclass(frozen=True)
class BatchOptions:
    root: Path
    report_dir: Path | None = None
    include_audio: bool = False
    dry_run: bool = False
    force: bool = False
    force_translate: bool = False
    translate_api_key: str | None = None
    translate_workers: int = 1
    translation_queue_size: int = 2
    stream: bool = False
    debug: bool = False
    accept_cpu_mode: bool = False
    no_nfo: bool = False
    actress: str | None = None


@dataclass(frozen=True)
class ClassifiedVideo:
    video_path: Path
    status: BatchStatus
    reason: str | None = None
    external_subtitle: Path | None = None
    japanese_srt: Path | None = None
    chinese_srt: Path | None = None
    nfo_path: Path | None = None
    actresses: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProcessResult:
    seconds: float = 0.0
    return_code: int | None = None
    command_redacted: list[str] = field(default_factory=list)
    api_key_source: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass(frozen=True)
class VideoResult:
    video_path: Path
    status: BatchStatus
    reason: str | None = None
    nfo_path: Path | None = None
    actresses: tuple[str, ...] = ()
    japanese_srt: Path | None = None
    chinese_srt: Path | None = None
    asr: ProcessResult | None = None
    translation: ProcessResult | None = None
    error: str | None = None
    warnings: tuple[str, ...] = ()
