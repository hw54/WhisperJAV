from __future__ import annotations

import os
import threading
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

from .commands import (
    build_asr_command,
    build_translation_command,
    build_translation_env,
    expected_translation_path,
)
from .discovery import is_valid_srt
from .models import BatchOptions, BatchStatus, ClassifiedVideo, ProcessResult, VideoResult
from .progress import BatchProgressEvent, BatchProgressReporter, BatchProgressTotals
from .runners import SubprocessRunner
from .scene_summary import SceneSummaryExportError, export_scene_summary

TERMINAL_SKIP_STATUSES = {
    "skip_external_subtitle",
    "skip_duration_limit",
    "skip_translated",
    "failed_precondition",
}
INTERRUPTED_RETURN_CODES = {130, 143, -2, -15}


class BatchScheduler:
    def __init__(
        self,
        options: BatchOptions,
        *,
        runner: SubprocessRunner | None = None,
        progress: BatchProgressReporter | None = None,
    ) -> None:
        if options.translate_workers < 1:
            raise ValueError("translate_workers must be at least 1")
        if options.translation_queue_size < 1:
            raise ValueError("translation_queue_size must be at least 1")
        if options.asr_retries < 0:
            raise ValueError("asr_retries must be at least 0")
        if options.translation_retries < 0:
            raise ValueError("translation_retries must be at least 0")
        self.options = options
        self.runner = runner or SubprocessRunner(stream=options.stream)
        self.progress = progress

    def run(self, items: Iterable[ClassifiedVideo]) -> list[VideoResult]:
        item_list = list(items)
        results: list[VideoResult | None] = [None] * len(item_list)
        pending: dict[Future[VideoResult], int] = {}
        slots = threading.Semaphore(self.options.translation_queue_size)
        executor = ThreadPoolExecutor(max_workers=self.options.translate_workers)

        try:
            if self.progress is not None:
                self.progress.start(_progress_totals(item_list))
            for index, item in enumerate(item_list):
                _collect_completed(
                    pending,
                    results,
                    items=item_list,
                    progress=self.progress,
                    block=False,
                )
                if self.options.dry_run or item.status in TERMINAL_SKIP_STATUSES:
                    self._set_result(results, index, item, _from_classified(item))
                    continue
                if item.status == "translate_existing_japanese":
                    self._submit_existing_translation(executor, slots, pending, results, index, item)
                    continue
                if item.status == "transcribe_then_translate":
                    self._run_asr_then_submit_translation(executor, slots, pending, results, index, item)
                    continue
                self._set_result(results, index, item, _from_classified(item))

            while pending:
                _collect_completed(
                    pending,
                    results,
                    items=item_list,
                    progress=self.progress,
                    block=True,
                )
        except KeyboardInterrupt:
            _terminate_runner(self.runner)
            for future in pending:
                future.cancel()
            _mark_unfinished_cancelled(item_list, results, progress=self.progress)
        finally:
            if self.progress is not None:
                self.progress.close()
            executor.shutdown(wait=True, cancel_futures=True)

        return _finished_results(results)

    def _submit_existing_translation(
        self,
        executor: ThreadPoolExecutor,
        slots: threading.Semaphore,
        pending: dict[Future[VideoResult], int],
        results: list[VideoResult | None],
        index: int,
        item: ClassifiedVideo,
    ) -> None:
        japanese = item.japanese_srt
        if japanese is None or not _is_valid_existing_srt(japanese):
            self._set_result(
                results,
                index,
                item,
                _failed_precondition(item, "missing_japanese_srt"),
            )
            return
        self._submit_translation(executor, slots, pending, index, item, None, japanese)

    def _run_asr_then_submit_translation(
        self,
        executor: ThreadPoolExecutor,
        slots: threading.Semaphore,
        pending: dict[Future[VideoResult], int],
        results: list[VideoResult | None],
        index: int,
        item: ClassifiedVideo,
    ) -> None:
        _wait_for_translation_capacity(slots)
        asr, japanese, error = self._run_asr(item)
        if japanese is None:
            self._set_result(
                results,
                index,
                item,
                _from_classified(
                    item,
                    status="failed_asr",
                    asr=asr,
                    error=error,
                ),
            )
            return

        self._submit_translation(executor, slots, pending, index, item, asr, japanese)

    def _set_result(
        self,
        results: list[VideoResult | None],
        index: int,
        item: ClassifiedVideo,
        result: VideoResult,
    ) -> None:
        results[index] = result
        _advance_progress(self.progress, item, result)

    def _run_asr(self, item: ClassifiedVideo) -> tuple[ProcessResult, Path | None, str | None]:
        command = build_asr_command(item.video_path, self.options)
        expected_srt = _expected_japanese_srt(item.video_path)
        last_result: ProcessResult | None = None
        last_error: str | None = None

        for attempt in range(self.options.asr_retries + 1):
            if attempt == 0:
                _progress_message(self.progress, f"正在处理ASR：{item.video_path.name}")
            result = self.runner.run(
                command,
                heartbeat=lambda elapsed: _progress_message(
                    self.progress,
                    f"ASR仍在处理：{item.video_path.name}（已耗时 {_format_elapsed(elapsed)}）",
                ),
            )
            last_result = result
            if result.return_code != 0:
                last_error = result.stderr_tail or "asr failed"
            elif _is_valid_existing_srt(expected_srt):
                _progress_message(self.progress, f"ASR完成：{item.video_path.name}")
                _advance_asr_progress(self.progress, item, "ok")
                return result, expected_srt, None
            else:
                last_error = f"expected Japanese SRT missing or invalid: {expected_srt}"

            if _is_interrupted_return_code(result.return_code):
                break
            if attempt < self.options.asr_retries:
                continue

        if last_result is None:
            raise RuntimeError("ASR was not attempted")
        _progress_message(self.progress, f"ASR失败：{item.video_path.name}")
        return last_result, None, last_error or "asr failed"

    def _submit_translation(
        self,
        executor: ThreadPoolExecutor,
        slots: threading.Semaphore,
        pending: dict[Future[VideoResult], int],
        index: int,
        item: ClassifiedVideo,
        asr: ProcessResult | None,
        japanese_srt: Path,
    ) -> None:
        slots.acquire()
        future = executor.submit(
            self._translation_result,
            slots,
            item,
            asr,
            japanese_srt,
        )
        pending[future] = index

    def _translation_result(
        self,
        slots: threading.Semaphore,
        item: ClassifiedVideo,
        asr: ProcessResult | None,
        japanese_srt: Path,
    ) -> VideoResult:
        try:
            _progress_message(self.progress, f"正在处理翻译：{item.video_path.name}")
            translation = self._run_translation(japanese_srt, item)
            if translation.return_code != 0:
                _progress_message(self.progress, f"翻译失败：{item.video_path.name}")
                _advance_translation_progress(self.progress, item, "failed")
                return _from_classified(
                    item,
                    status="failed_translation",
                    japanese_srt=japanese_srt,
                    asr=asr,
                    translation=translation,
                    error=translation.stderr_tail or "translation failed",
                )
            chinese = expected_translation_path(japanese_srt)
            summary_json, summary_warnings = self._export_scene_summary(japanese_srt, chinese)
            status: BatchStatus = "completed_translation_only" if asr is None else "completed"
            _progress_message(self.progress, f"翻译完成：{item.video_path.name}")
            _advance_translation_progress(self.progress, item, "translated")
            return _from_classified(
                item,
                status=status,
                japanese_srt=japanese_srt,
                chinese_srt=chinese if chinese.exists() else None,
                summary_json=summary_json,
                asr=asr,
                translation=translation,
                warnings=(*item.warnings, *summary_warnings),
            )
        finally:
            slots.release()

    def _export_scene_summary(
        self,
        japanese_srt: Path,
        chinese_srt: Path,
    ) -> tuple[Path | None, tuple[str, ...]]:
        if not chinese_srt.exists():
            return None, ()
        try:
            result = export_scene_summary(
                source_srt=japanese_srt,
                translated_srt=chinese_srt,
            )
        except SceneSummaryExportError as exc:
            return None, (f"scene_summary_export_failed: {exc}",)
        warnings = tuple(f"scene_summary_warning: {warning}" for warning in result.warnings)
        return result.path, warnings

    def _run_translation(
        self,
        japanese_srt: Path,
        item: ClassifiedVideo,
    ) -> ProcessResult:
        command = build_translation_command(
            japanese_srt,
            self.options,
            actresses=item.actresses,
            movie_title=item.movie_title,
            movie_plot=item.movie_plot,
        )
        env = build_translation_env(self.options, base_env=os.environ)
        result = _run_process_with_retries(
            runner=self.runner,
            command=command,
            retries=self.options.translation_retries,
            env=env,
            heartbeat=lambda elapsed: _progress_message(
                self.progress,
                f"翻译仍在处理：{item.video_path.name}（已耗时 {_format_elapsed(elapsed)}）",
            ),
        )
        api_key_source = "env" if self.options.translate_api_key else result.api_key_source
        return replace(result, api_key_source=api_key_source)


def _from_classified(
    item: ClassifiedVideo,
    *,
    status: BatchStatus | None = None,
    reason: str | None = None,
    japanese_srt: Path | None = None,
    chinese_srt: Path | None = None,
    summary_json: Path | None = None,
    asr: ProcessResult | None = None,
    translation: ProcessResult | None = None,
    error: str | None = None,
    warnings: tuple[str, ...] | None = None,
) -> VideoResult:
    return VideoResult(
        video_path=item.video_path,
        status=status or item.status,
        reason=reason if reason is not None else item.reason,
        nfo_path=item.nfo_path,
        actresses=item.actresses,
        movie_title=item.movie_title,
        movie_plot=item.movie_plot,
        japanese_srt=japanese_srt or item.japanese_srt,
        chinese_srt=chinese_srt or item.chinese_srt,
        summary_json=summary_json,
        asr=asr,
        translation=translation,
        error=error,
        warnings=warnings if warnings is not None else item.warnings,
        duration_seconds=item.duration_seconds,
        duration_limit_minutes=item.duration_limit_minutes,
    )


def _failed_precondition(item: ClassifiedVideo, reason: str) -> VideoResult:
    return _from_classified(item, status="failed_precondition", reason=reason)


def _collect_completed(
    pending: dict[Future[VideoResult], int],
    results: list[VideoResult | None],
    *,
    items: list[ClassifiedVideo],
    progress: BatchProgressReporter | None,
    block: bool,
) -> None:
    if not pending:
        return
    if block:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
    else:
        done = {future for future in pending if future.done()}
    for future in done:
        index = pending.pop(future)
        result = future.result()
        results[index] = result
        _advance_progress(progress, items[index], result)


def _wait_for_translation_capacity(slots: threading.Semaphore) -> None:
    slots.acquire()
    slots.release()


def _run_process_with_retries(
    *,
    runner: SubprocessRunner,
    command: list[str],
    retries: int,
    env: dict[str, str] | None = None,
    heartbeat=None,
) -> ProcessResult:
    last_result: ProcessResult | None = None
    for attempt in range(retries + 1):
        result = runner.run(command, env=env, heartbeat=heartbeat)
        last_result = result
        if result.return_code == 0:
            return result
        if _is_interrupted_return_code(result.return_code):
            return result
        if attempt < retries:
            continue
    if last_result is None:
        raise RuntimeError("process was not attempted")
    return last_result


def _is_interrupted_return_code(return_code: int | None) -> bool:
    return return_code in INTERRUPTED_RETURN_CODES


def _terminate_runner(runner: object) -> None:
    terminate_all = getattr(runner, "terminate_all", None)
    if terminate_all is not None:
        terminate_all()


def _mark_unfinished_cancelled(
    items: list[ClassifiedVideo],
    results: list[VideoResult | None],
    *,
    progress: BatchProgressReporter | None = None,
) -> None:
    for index, result in enumerate(results):
        if result is None:
            cancelled = _from_classified(
                items[index],
                status="cancelled",
                reason="interrupted",
            )
            results[index] = cancelled
            _advance_progress(progress, items[index], cancelled)


def _finished_results(results: list[VideoResult | None]) -> list[VideoResult]:
    return [result for result in results if result is not None]


def _expected_japanese_srt(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}.ja.pass1.srt")


def _is_valid_existing_srt(path: Path) -> bool:
    return path.exists() and is_valid_srt(path)


def _progress_totals(items: list[ClassifiedVideo]) -> BatchProgressTotals:
    return BatchProgressTotals(
        files=len(items),
        asr=sum(1 for item in items if item.status == "transcribe_then_translate"),
        translation=sum(
            1
            for item in items
            if item.status in {"transcribe_then_translate", "translate_existing_japanese"}
        ),
    )


def _advance_progress(
    progress: BatchProgressReporter | None,
    item: ClassifiedVideo,
    result: VideoResult,
) -> None:
    if progress is None:
        return
    if item.status == "transcribe_then_translate" and result.status in {"failed_asr", "cancelled"}:
        _advance_asr_progress(progress, item, _asr_progress_outcome(result))
    if (
        item.status in {"transcribe_then_translate", "translate_existing_japanese"}
        and result.status in {"failed_asr", "failed_precondition", "cancelled"}
    ):
        _advance_translation_progress(progress, item, _translation_progress_outcome(result))
    progress.advance(BatchProgressEvent("files", result.status, result.video_path))


def _progress_message(progress: BatchProgressReporter | None, text: str) -> None:
    if progress is not None:
        progress.message(text)


def _advance_asr_progress(
    progress: BatchProgressReporter | None,
    item: ClassifiedVideo,
    outcome: str,
) -> None:
    if progress is not None:
        progress.advance(BatchProgressEvent("asr", outcome, item.video_path))


def _advance_translation_progress(
    progress: BatchProgressReporter | None,
    item: ClassifiedVideo,
    outcome: str,
) -> None:
    if progress is not None:
        progress.advance(BatchProgressEvent("translation", outcome, item.video_path))


def _asr_progress_outcome(result: VideoResult) -> str:
    if result.status == "cancelled":
        return "cancelled"
    if result.asr is None:
        return "blocked"
    if result.status == "failed_asr" or result.asr.return_code != 0:
        return "failed"
    return "ok"


def _translation_progress_outcome(result: VideoResult) -> str:
    if result.status == "cancelled":
        return "cancelled"
    if result.status == "failed_asr":
        return "blocked_asr"
    if result.status == "failed_precondition":
        return "blocked_precondition"
    if result.translation is None:
        return "blocked"
    if result.status == "failed_translation" or result.translation.return_code != 0:
        return "failed"
    return "translated"


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds_part:02d}s"
    if minutes:
        return f"{minutes}m{seconds_part:02d}s"
    return f"{seconds_part}s"
