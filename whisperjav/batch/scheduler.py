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

    def run(self, items: Iterable[ClassifiedVideo]) -> list[VideoResult]:
        item_list = list(items)
        results: list[VideoResult | None] = [None] * len(item_list)
        pending: dict[Future[VideoResult], int] = {}
        slots = threading.Semaphore(self.options.translation_queue_size)
        executor = ThreadPoolExecutor(max_workers=self.options.translate_workers)

        try:
            for index, item in enumerate(item_list):
                _collect_completed(pending, results, block=False)
                if self.options.dry_run or item.status in TERMINAL_SKIP_STATUSES:
                    results[index] = _from_classified(item)
                    continue
                if item.status == "translate_existing_japanese":
                    self._submit_existing_translation(executor, slots, pending, results, index, item)
                    continue
                if item.status == "transcribe_then_translate":
                    self._run_asr_then_submit_translation(executor, slots, pending, results, index, item)
                    continue
                results[index] = _from_classified(item)

            while pending:
                _collect_completed(pending, results, block=True)
        except KeyboardInterrupt:
            _terminate_runner(self.runner)
            for future in pending:
                future.cancel()
            _mark_unfinished_cancelled(item_list, results)
        finally:
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
            results[index] = _failed_precondition(item, "missing_japanese_srt")
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
            results[index] = _from_classified(
                item,
                status="failed_asr",
                asr=asr,
                error=error,
            )
            return

        self._submit_translation(executor, slots, pending, index, item, asr, japanese)

    def _run_asr(self, item: ClassifiedVideo) -> tuple[ProcessResult, Path | None, str | None]:
        command = build_asr_command(item.video_path, self.options)
        expected_srt = _expected_japanese_srt(item.video_path)
        last_result: ProcessResult | None = None
        last_error: str | None = None

        for attempt in range(self.options.asr_retries + 1):
            result = self.runner.run(command)
            last_result = result
            if result.return_code != 0:
                last_error = result.stderr_tail or "asr failed"
            elif _is_valid_existing_srt(expected_srt):
                return result, expected_srt, None
            else:
                last_error = f"expected Japanese SRT missing or invalid: {expected_srt}"

            if _is_interrupted_return_code(result.return_code):
                break
            if attempt < self.options.asr_retries:
                continue

        if last_result is None:
            raise RuntimeError("ASR was not attempted")
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
            translation = self._run_translation(japanese_srt, item)
            if translation.return_code != 0:
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
        results[index] = future.result()


def _wait_for_translation_capacity(slots: threading.Semaphore) -> None:
    slots.acquire()
    slots.release()


def _run_process_with_retries(
    *,
    runner: SubprocessRunner,
    command: list[str],
    retries: int,
    env: dict[str, str] | None = None,
) -> ProcessResult:
    last_result: ProcessResult | None = None
    for attempt in range(retries + 1):
        result = runner.run(command, env=env)
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
) -> None:
    for index, result in enumerate(results):
        if result is None:
            results[index] = _from_classified(
                items[index],
                status="cancelled",
                reason="interrupted",
            )


def _finished_results(results: list[VideoResult | None]) -> list[VideoResult]:
    return [result for result in results if result is not None]


def _expected_japanese_srt(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}.ja.pass1.srt")


def _is_valid_existing_srt(path: Path) -> bool:
    return path.exists() and is_valid_srt(path)
