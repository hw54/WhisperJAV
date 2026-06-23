from __future__ import annotations

import hashlib
import logging
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, TypeVar

from .models import BatchOptions, ClassifiedVideo, ProcessResult

PreparedAsr = TypeVar("PreparedAsr")


@dataclass(frozen=True)
class AsrExecutionResult:
    process: ProcessResult
    japanese_srt: Path | None
    error: str | None = None


class StagedAsrPipeline(Protocol[PreparedAsr]):
    def prepare(self, item: ClassifiedVideo) -> PreparedAsr:
        ...

    def transcribe(self, prepared: PreparedAsr) -> AsrExecutionResult:
        ...


@dataclass(frozen=True)
class PreparedQwenAsr:
    item: ClassifiedVideo
    qwen_prepared: object
    temp_dir: Path
    prepare_seconds: float


class QwenStagedAsrPipeline:
    """Batch ASR pipeline with CPU/IO prepare separated from GPU transcription."""

    def __init__(self, options: BatchOptions) -> None:
        self.options = options

    def prepare(self, item: ClassifiedVideo) -> PreparedQwenAsr:
        with _worker_log_context(self.options):
            started = time.perf_counter()
            pipeline = self._build_pipeline(item)
            prepared = pipeline.prepare(_media_info(item.video_path))
            return PreparedQwenAsr(
                item=item,
                qwen_prepared=prepared,
                temp_dir=pipeline.temp_dir,
                prepare_seconds=time.perf_counter() - started,
            )

    def transcribe(self, prepared: PreparedQwenAsr) -> AsrExecutionResult:
        with _worker_log_context(self.options):
            started = time.perf_counter()
            pipeline = self._build_pipeline(prepared.item, temp_dir=prepared.temp_dir)
            try:
                metadata = pipeline.transcribe_prepared(prepared.qwen_prepared)
                final_srt = Path(metadata.get("output_files", {}).get("final_srt", ""))
                pass_output = _pass1_output_path(prepared.item.video_path)
                if not final_srt.exists():
                    return AsrExecutionResult(
                        process=ProcessResult(
                            return_code=1,
                            seconds=prepared.prepare_seconds + time.perf_counter() - started,
                            command_redacted=_command_label(prepared.item.video_path),
                            stderr_tail=f"staged ASR final SRT missing: {final_srt}",
                        ),
                        japanese_srt=None,
                        error=f"staged ASR final SRT missing: {final_srt}",
                    )
                if pass_output.exists():
                    pass_output.unlink()
                shutil.move(str(final_srt), str(pass_output))
                return AsrExecutionResult(
                    process=ProcessResult(
                        return_code=0,
                        seconds=prepared.prepare_seconds + time.perf_counter() - started,
                        command_redacted=_command_label(prepared.item.video_path),
                    ),
                    japanese_srt=pass_output,
                )
            except Exception as exc:
                return AsrExecutionResult(
                    process=ProcessResult(
                        return_code=1,
                        seconds=prepared.prepare_seconds + time.perf_counter() - started,
                        command_redacted=_command_label(prepared.item.video_path),
                        stderr_tail=str(exc),
                    ),
                    japanese_srt=None,
                    error=f"staged_asr_failed: {exc}",
                )
            finally:
                pipeline.cleanup()

    def _build_pipeline(self, item: ClassifiedVideo, *, temp_dir: Path | None = None):
        from whisperjav.pipelines.qwen_pipeline import QwenPipeline

        return QwenPipeline(
            output_dir=str(item.video_path.parent),
            temp_dir=str(temp_dir or _staged_temp_dir(item.video_path, self.options)),
            keep_temp_files=False,
            save_metadata_json=self.options.debug,
            progress_display=None,
            qwen_framer="vad-grouped",
            generator_backend="anime-whisper",
            timestamp_mode="vad_only",
            assembly_cleaner=False,
            stepdown_enabled=False,
            scene_detector="semantic",
            speech_segmenter="whisperseg",
            model_id="litagin/anime-whisper",
            language="Japanese",
            segmenter_chunk_threshold=0.5,
            segmenter_max_group_duration=5.0,
            segmenter_config={"force_cpu": True},
            regroup_mode="standard",
            subs_language="native",
            batch_status_prefix=_batch_status_prefix(item, self.options),
        )


def _media_info(video_path: Path) -> dict[str, object]:
    return {
        "path": video_path,
        "basename": video_path.stem,
        "type": "video",
    }


def _staged_temp_dir(video_path: Path, options: BatchOptions) -> Path:
    report_dir = options.report_dir or (options.root / ".whisperjav_batch")
    digest = hashlib.sha1(str(video_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return report_dir / "asr_staged" / f"{video_path.stem}-{digest}"


def _pass1_output_path(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}.ja.pass1.srt")


def _command_label(video_path: Path) -> tuple[str, str]:
    return ("staged-qwen-asr", str(video_path))


def _batch_status_prefix(item: ClassifiedVideo, options: BatchOptions) -> str | None:
    if _quiet_worker_logs(options):
        return None
    return f"ASR状态：{item.video_path.name}"


@contextmanager
def _worker_log_context(options: BatchOptions) -> Iterator[None]:
    if not _quiet_worker_logs(options):
        yield
        return

    logger = logging.getLogger("whisperjav")
    previous_logger_level = logger.level
    previous_handler_levels = [
        (handler, handler.level)
        for handler in logger.handlers
        if _is_console_stream_handler(handler)
    ]
    logger.setLevel(logging.WARNING)
    for handler, _level in previous_handler_levels:
        handler.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(previous_logger_level)
        for handler, level in previous_handler_levels:
            handler.setLevel(level)


def _quiet_worker_logs(options: BatchOptions) -> bool:
    return not (options.debug or options.no_progress or options.stream)


def _is_console_stream_handler(handler: logging.Handler) -> bool:
    return (
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
    )
