from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

AsrMode = Literal["subprocess", "staged"]

BatchStatus = Literal[
    "skip_external_subtitle",
    "skip_duration_limit",
    "skip_translated",
    "translate_existing_japanese",
    "transcribe_then_translate",
    "failed_asr",
    "failed_translation",
    "failed_precondition",
    "cancelled",
    "deferred_time_limit",
    "deferred_gpu_error_limit",
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
    asr_retries: int = 1
    asr_mode: AsrMode = "subprocess"
    asr_cpu_workers: int = 1
    translation_retries: int = 2
    max_video_minutes: float = 230
    run_minutes: float | None = None
    max_consecutive_gpu_errors: int = 3
    stream: bool = False
    no_progress: bool = False
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
    movie_title: str | None = None
    movie_plot: str | None = None
    warnings: tuple[str, ...] = ()
    duration_seconds: float | None = None
    duration_limit_minutes: float | None = None


@dataclass(frozen=True)
class ProcessResult:
    seconds: float = 0.0
    return_code: int | None = None
    command_redacted: tuple[str, ...] = field(default_factory=tuple)
    api_key_source: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_redacted", tuple(self.command_redacted))


@dataclass(frozen=True)
class VideoResult:
    video_path: Path
    status: BatchStatus
    reason: str | None = None
    nfo_path: Path | None = None
    actresses: tuple[str, ...] = ()
    movie_title: str | None = None
    movie_plot: str | None = None
    japanese_srt: Path | None = None
    chinese_srt: Path | None = None
    summary_json: Path | None = None
    asr: ProcessResult | None = None
    translation: ProcessResult | None = None
    error: str | None = None
    warnings: tuple[str, ...] = ()
    duration_seconds: float | None = None
    duration_limit_minutes: float | None = None
