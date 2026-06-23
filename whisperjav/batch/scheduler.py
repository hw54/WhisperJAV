from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Iterable
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import replace
from pathlib import Path

from .asr import AsrExecutionResult, QwenStagedAsrPipeline, StagedAsrPipeline
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
PREPARE_ASR_ERROR_RETURN_CODE = 1
RUN_TIME_LIMIT_REASON = "run_time_limit_reached"
GPU_ERROR_LIMIT_REASON = "consecutive_gpu_asr_errors"
SECONDS_PER_MINUTE = 60
CLEANUP_START_MESSAGE = "正在清理产物……"
CLEANUP_COMPLETE_MESSAGE = "清理产物完成。"
GPU_FATAL_ASR_ERROR_PATTERNS = (
    "hip error",
    "unspecified launch failure",
    "device-side assert",
)


class BatchScheduler:
    def __init__(
        self,
        options: BatchOptions,
        *,
        runner: SubprocessRunner | None = None,
        progress: BatchProgressReporter | None = None,
        asr_pipeline: StagedAsrPipeline | None = None,
    ) -> None:
        if options.translate_workers < 1:
            raise ValueError("translate_workers must be at least 1")
        if options.translation_queue_size < 1:
            raise ValueError("translation_queue_size must be at least 1")
        if options.asr_retries < 0:
            raise ValueError("asr_retries must be at least 0")
        if options.asr_cpu_workers < 1:
            raise ValueError("asr_cpu_workers must be at least 1")
        if options.asr_mode not in {"subprocess", "staged"}:
            raise ValueError("asr_mode must be 'subprocess' or 'staged'")
        if options.translation_retries < 0:
            raise ValueError("translation_retries must be at least 0")
        if options.run_minutes is not None and options.run_minutes < 0:
            raise ValueError("run_minutes must be at least 0")
        if options.max_consecutive_gpu_errors < 0:
            raise ValueError("max_consecutive_gpu_errors must be at least 0")
        self.options = options
        self.runner = runner or SubprocessRunner(stream=options.stream)
        self.progress = progress
        self.asr_pipeline = (
            asr_pipeline
            if asr_pipeline is not None
            else QwenStagedAsrPipeline(options) if options.asr_mode == "staged"
            else None
        )

    def run(self, items: Iterable[ClassifiedVideo]) -> list[VideoResult]:
        item_list = list(items)
        if self.options.asr_mode == "staged":
            return self._run_staged(item_list)

        return self._run_subprocess(item_list)

    def _run_subprocess(self, item_list: list[ClassifiedVideo]) -> list[VideoResult]:
        results: list[VideoResult | None] = [None] * len(item_list)
        pending: dict[Future[VideoResult], int] = {}
        slots = threading.Semaphore(self.options.translation_queue_size)
        executor = ThreadPoolExecutor(max_workers=self.options.translate_workers)
        deadline = _run_deadline(self.options)
        interrupted_cleanup = False
        consecutive_gpu_errors = 0

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
                if _time_limit_reached(deadline):
                    self._set_result(results, index, item, _deferred_time_limit(item))
                    continue
                if item.status == "translate_existing_japanese":
                    self._submit_existing_translation(executor, slots, pending, results, index, item)
                    continue
                if item.status == "transcribe_then_translate":
                    if _gpu_error_limit_reached(consecutive_gpu_errors, self.options):
                        self._set_result(results, index, item, _deferred_gpu_error_limit(item))
                        continue
                    asr_failure = self._run_asr_then_submit_translation(
                        executor, slots, pending, results, index, item
                    )
                    consecutive_gpu_errors = _next_consecutive_gpu_errors(
                        consecutive_gpu_errors,
                        asr_failure,
                    )
                    if _gpu_error_limit_reached(consecutive_gpu_errors, self.options):
                        _announce_gpu_error_limit(self.progress, self.options)
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
            interrupted_cleanup = True
            _begin_interrupt_cleanup(
                self.runner,
                pending,
                items=item_list,
                results=results,
                progress=self.progress,
            )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            if interrupted_cleanup:
                _complete_interrupt_cleanup(self.progress)
            if self.progress is not None:
                self.progress.close()

        return _finished_results(results)

    def _run_staged(self, item_list: list[ClassifiedVideo]) -> list[VideoResult]:
        deadline = _run_deadline(self.options)
        if deadline is not None:
            return self._run_staged_limited(item_list, deadline)

        results: list[VideoResult | None] = [None] * len(item_list)
        pending: dict[Future[VideoResult], int] = {}
        prepare_futures: dict[int, Future] = {}
        slots = threading.Semaphore(self.options.translation_queue_size)
        translation_executor = ThreadPoolExecutor(max_workers=self.options.translate_workers)
        prepare_executor = _create_prepare_executor(
            self.asr_pipeline,
            max_workers=self.options.asr_cpu_workers,
        )
        transcribe_executor = _create_transcribe_executor(self.asr_pipeline)
        interrupted_cleanup = False
        consecutive_gpu_errors = 0

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
                    self._submit_existing_translation(
                        translation_executor, slots, pending, results, index, item
                    )
                    continue
                if item.status == "transcribe_then_translate":
                    _progress_message(self.progress, f"正在准备ASR：{item.video_path.name}")
                    prepare_futures[index] = prepare_executor.submit(
                        self.asr_pipeline.prepare,
                        item,
                    )
                    continue
                self._set_result(results, index, item, _from_classified(item))

            for index, item in enumerate(item_list):
                _collect_completed(
                    pending,
                    results,
                    items=item_list,
                    progress=self.progress,
                    block=False,
                )
                future = prepare_futures.get(index)
                if future is None:
                    continue
                if _gpu_error_limit_reached(consecutive_gpu_errors, self.options):
                    future.cancel()
                    self._set_result(results, index, item, _deferred_gpu_error_limit(item))
                    continue
                asr_result, transcribe_executor = self._run_prepared_asr(
                    future,
                    item,
                    transcribe_executor,
                )
                if asr_result.japanese_srt is None:
                    failed_result = _from_classified(
                        item,
                        status="failed_asr",
                        asr=asr_result.process,
                        error=asr_result.error,
                    )
                    self._set_result(results, index, item, failed_result)
                    consecutive_gpu_errors = _next_consecutive_gpu_errors(
                        consecutive_gpu_errors,
                        failed_result,
                    )
                    if _gpu_error_limit_reached(consecutive_gpu_errors, self.options):
                        _announce_gpu_error_limit(self.progress, self.options)
                    continue
                consecutive_gpu_errors = 0
                self._submit_translation(
                    translation_executor,
                    slots,
                    pending,
                    index,
                    item,
                    asr_result.process,
                    asr_result.japanese_srt,
                )

            while pending:
                _collect_completed(
                    pending,
                    results,
                    items=item_list,
                    progress=self.progress,
                    block=True,
                )
        except KeyboardInterrupt:
            interrupted_cleanup = True
            _begin_interrupt_cleanup(
                self.runner,
                pending,
                items=item_list,
                results=results,
                progress=self.progress,
                prepare_futures=prepare_futures.values(),
            )
        finally:
            _shutdown_executor(transcribe_executor)
            prepare_executor.shutdown(wait=True, cancel_futures=True)
            translation_executor.shutdown(wait=True, cancel_futures=True)
            if interrupted_cleanup:
                _complete_interrupt_cleanup(self.progress)
            if self.progress is not None:
                self.progress.close()

        return _finished_results(results)

    def _run_staged_limited(
        self,
        item_list: list[ClassifiedVideo],
        deadline: float,
    ) -> list[VideoResult]:
        results: list[VideoResult | None] = [None] * len(item_list)
        pending: dict[Future[VideoResult], int] = {}
        prepare_futures: dict[int, Future] = {}
        slots = threading.Semaphore(self.options.translation_queue_size)
        translation_executor = ThreadPoolExecutor(max_workers=self.options.translate_workers)
        prepare_executor = _create_prepare_executor(
            self.asr_pipeline,
            max_workers=self.options.asr_cpu_workers,
        )
        transcribe_executor = _create_transcribe_executor(self.asr_pipeline)
        prepare_scan_index = 0
        interrupted_cleanup = False
        consecutive_gpu_errors = 0

        def schedule_prepares() -> None:
            nonlocal prepare_scan_index
            while (
                len(prepare_futures) < self.options.asr_cpu_workers
                and prepare_scan_index < len(item_list)
                and not _time_limit_reached(deadline)
                and not _gpu_error_limit_reached(consecutive_gpu_errors, self.options)
            ):
                candidate_index = prepare_scan_index
                candidate = item_list[candidate_index]
                prepare_scan_index += 1
                if candidate.status != "transcribe_then_translate":
                    continue
                if self.options.dry_run or candidate.status in TERMINAL_SKIP_STATUSES:
                    continue
                _progress_message(self.progress, f"正在准备ASR：{candidate.video_path.name}")
                prepare_futures[candidate_index] = prepare_executor.submit(
                    self.asr_pipeline.prepare,
                    candidate,
                )

        try:
            if self.progress is not None:
                self.progress.start(_progress_totals(item_list))

            schedule_prepares()
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
                    if _time_limit_reached(deadline):
                        self._set_result(results, index, item, _deferred_time_limit(item))
                        continue
                    self._submit_existing_translation(
                        translation_executor, slots, pending, results, index, item
                    )
                    continue
                if item.status != "transcribe_then_translate":
                    self._set_result(results, index, item, _from_classified(item))
                    continue

                if _gpu_error_limit_reached(consecutive_gpu_errors, self.options):
                    future = prepare_futures.pop(index, None)
                    if future is not None:
                        future.cancel()
                    self._set_result(results, index, item, _deferred_gpu_error_limit(item))
                    continue

                future = prepare_futures.get(index)
                if future is None:
                    if _time_limit_reached(deadline):
                        self._set_result(results, index, item, _deferred_time_limit(item))
                        continue
                    _progress_message(self.progress, f"正在准备ASR：{item.video_path.name}")
                    future = prepare_executor.submit(self.asr_pipeline.prepare, item)
                    prepare_futures[index] = future
                    prepare_scan_index = max(prepare_scan_index, index + 1)

                prepared_result = self._wait_for_prepared_asr(future, item)
                prepare_futures.pop(index, None)
                schedule_prepares()
                if isinstance(prepared_result, AsrExecutionResult):
                    failed_result = _from_classified(
                        item,
                        status="failed_asr",
                        asr=prepared_result.process,
                        error=prepared_result.error,
                    )
                    self._set_result(results, index, item, failed_result)
                    consecutive_gpu_errors = _next_consecutive_gpu_errors(
                        consecutive_gpu_errors,
                        failed_result,
                    )
                    if _gpu_error_limit_reached(consecutive_gpu_errors, self.options):
                        _announce_gpu_error_limit(self.progress, self.options)
                    continue

                asr_result, transcribe_executor = self._run_transcribe_prepared(
                    prepared_result,
                    item,
                    transcribe_executor,
                )
                if asr_result.japanese_srt is None:
                    failed_result = _from_classified(
                        item,
                        status="failed_asr",
                        asr=asr_result.process,
                        error=asr_result.error,
                    )
                    self._set_result(results, index, item, failed_result)
                    consecutive_gpu_errors = _next_consecutive_gpu_errors(
                        consecutive_gpu_errors,
                        failed_result,
                    )
                    if _gpu_error_limit_reached(consecutive_gpu_errors, self.options):
                        _announce_gpu_error_limit(self.progress, self.options)
                    continue
                consecutive_gpu_errors = 0
                self._submit_translation(
                    translation_executor,
                    slots,
                    pending,
                    index,
                    item,
                    asr_result.process,
                    asr_result.japanese_srt,
                )

            while pending:
                _collect_completed(
                    pending,
                    results,
                    items=item_list,
                    progress=self.progress,
                    block=True,
                )
        except KeyboardInterrupt:
            interrupted_cleanup = True
            _begin_interrupt_cleanup(
                self.runner,
                pending,
                items=item_list,
                results=results,
                progress=self.progress,
                prepare_futures=prepare_futures.values(),
            )
        finally:
            _shutdown_executor(transcribe_executor)
            prepare_executor.shutdown(wait=True, cancel_futures=True)
            translation_executor.shutdown(wait=True, cancel_futures=True)
            if interrupted_cleanup:
                _complete_interrupt_cleanup(self.progress)
            if self.progress is not None:
                self.progress.close()

        return _finished_results(results)

    def _run_prepared_asr(
        self,
        future: Future,
        item: ClassifiedVideo,
        transcribe_executor,
    ) -> tuple[AsrExecutionResult, object | None]:
        try:
            prepared = future.result()
        except Exception as exc:
            return _failed_prepare_asr_result(self.progress, item, exc), transcribe_executor

        return self._run_transcribe_prepared(prepared, item, transcribe_executor)

    def _wait_for_prepared_asr(
        self,
        future: Future,
        item: ClassifiedVideo,
    ):
        try:
            return future.result()
        except Exception as exc:
            return _failed_prepare_asr_result(self.progress, item, exc)

    def _run_transcribe_prepared(
        self,
        prepared,
        item: ClassifiedVideo,
        transcribe_executor,
    ) -> tuple[AsrExecutionResult, object | None]:
        _progress_message(self.progress, f"正在处理ASR：{item.video_path.name}")
        worker_failed = False
        try:
            result = _transcribe_prepared(self.asr_pipeline, prepared, transcribe_executor)
        except Exception as exc:
            worker_failed = True
            result = _failed_transcribe_asr_result(item, exc)
        if result.japanese_srt is not None:
            _progress_message(self.progress, f"ASR完成：{item.video_path.name}")
            _advance_asr_progress(self.progress, item, "ok")
        else:
            suffix = f"（{result.error}）" if result.error else ""
            _progress_message(self.progress, f"ASR失败：{item.video_path.name}{suffix}")
        if worker_failed:
            _progress_message(self.progress, "检测到ASR worker异常，重启ASR worker")
            transcribe_executor = _restart_transcribe_executor(
                self.asr_pipeline,
                transcribe_executor,
            )
        elif _is_fatal_gpu_asr_error(result.error or result.process.stderr_tail):
            _progress_message(self.progress, "检测到GPU运行时错误，重启ASR worker")
            transcribe_executor = _restart_transcribe_executor(
                self.asr_pipeline,
                transcribe_executor,
            )
        return result, transcribe_executor

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
    ) -> VideoResult | None:
        _wait_for_translation_capacity(slots)
        asr, japanese, error = self._run_asr(item)
        if japanese is None:
            result = _from_classified(
                item,
                status="failed_asr",
                asr=asr,
                error=error,
            )
            self._set_result(results, index, item, result)
            return result

        self._submit_translation(executor, slots, pending, index, item, asr, japanese)
        return None

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


def _deferred_time_limit(item: ClassifiedVideo) -> VideoResult:
    return _from_classified(
        item,
        status="deferred_time_limit",
        reason=RUN_TIME_LIMIT_REASON,
    )


def _deferred_gpu_error_limit(item: ClassifiedVideo) -> VideoResult:
    return _from_classified(
        item,
        status="deferred_gpu_error_limit",
        reason=GPU_ERROR_LIMIT_REASON,
    )


def _next_consecutive_gpu_errors(
    current: int,
    result: VideoResult | None,
) -> int:
    if result is None:
        return 0
    if result.status != "failed_asr":
        return 0
    error = result.error or (result.asr.stderr_tail if result.asr is not None else None)
    if _is_fatal_gpu_asr_error(error):
        return current + 1
    return 0


def _gpu_error_limit_reached(count: int, options: BatchOptions) -> bool:
    return options.max_consecutive_gpu_errors > 0 and count >= options.max_consecutive_gpu_errors


def _announce_gpu_error_limit(
    progress: BatchProgressReporter | None,
    options: BatchOptions,
) -> None:
    message = f"连续GPU运行时错误达到{options.max_consecutive_gpu_errors}次，停止安排新的ASR任务。"
    if progress is not None:
        progress.message(message)
        return
    print(message, file=sys.stderr, flush=True)


def _failed_prepare_asr_result(
    progress: BatchProgressReporter | None,
    item: ClassifiedVideo,
    exc: Exception,
) -> AsrExecutionResult:
    error = str(exc) or exc.__class__.__name__
    _progress_message(progress, f"ASR准备失败：{item.video_path.name}（{error}）")
    return AsrExecutionResult(
        process=ProcessResult(
            return_code=PREPARE_ASR_ERROR_RETURN_CODE,
            stderr_tail=error,
        ),
        japanese_srt=None,
        error=error,
    )


def _failed_transcribe_asr_result(item: ClassifiedVideo, exc: Exception) -> AsrExecutionResult:
    error = str(exc) or exc.__class__.__name__
    return AsrExecutionResult(
        process=ProcessResult(
            return_code=PREPARE_ASR_ERROR_RETURN_CODE,
            stderr_tail=error,
            command_redacted=("staged-qwen-asr-worker", str(item.video_path)),
        ),
        japanese_srt=None,
        error=f"staged_asr_worker_failed: {error}",
    )


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


def _create_prepare_executor(
    asr_pipeline: StagedAsrPipeline | None,
    *,
    max_workers: int,
):
    if isinstance(asr_pipeline, QwenStagedAsrPipeline):
        return ProcessPoolExecutor(max_workers=max_workers)
    return ThreadPoolExecutor(max_workers=max_workers)


def _create_transcribe_executor(asr_pipeline: StagedAsrPipeline | None):
    if isinstance(asr_pipeline, QwenStagedAsrPipeline):
        return ProcessPoolExecutor(max_workers=1)
    return None


def _transcribe_prepared(
    asr_pipeline: StagedAsrPipeline,
    prepared,
    transcribe_executor,
) -> AsrExecutionResult:
    if transcribe_executor is None:
        return asr_pipeline.transcribe(prepared)
    future = transcribe_executor.submit(asr_pipeline.transcribe, prepared)
    return future.result()


def _restart_transcribe_executor(
    asr_pipeline: StagedAsrPipeline | None,
    transcribe_executor,
):
    if transcribe_executor is None:
        return None
    _shutdown_executor(transcribe_executor)
    return _create_transcribe_executor(asr_pipeline)


def _shutdown_executor(executor) -> None:
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)


def _is_fatal_gpu_asr_error(error: str | None) -> bool:
    if not error:
        return False
    normalized = error.lower()
    return any(pattern in normalized for pattern in GPU_FATAL_ASR_ERROR_PATTERNS)


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


def _run_deadline(options: BatchOptions) -> float | None:
    if options.run_minutes is None:
        return None
    return time.perf_counter() + options.run_minutes * SECONDS_PER_MINUTE


def _time_limit_reached(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


def _begin_interrupt_cleanup(
    runner: object,
    pending: dict[Future[VideoResult], int],
    *,
    items: list[ClassifiedVideo],
    results: list[VideoResult | None],
    progress: BatchProgressReporter | None,
    prepare_futures: Iterable[Future] = (),
) -> None:
    _cleanup_message(progress, CLEANUP_START_MESSAGE)
    _terminate_runner(runner)
    for future in pending:
        future.cancel()
    for future in prepare_futures:
        future.cancel()
    _mark_unfinished_cancelled(items, results, progress=progress)


def _complete_interrupt_cleanup(progress: BatchProgressReporter | None) -> None:
    _cleanup_message(progress, CLEANUP_COMPLETE_MESSAGE)


def _cleanup_message(progress: BatchProgressReporter | None, text: str) -> None:
    if progress is not None:
        progress.message(text)
        return
    print(text, file=sys.stderr, flush=True)


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
    if (
        item.status == "transcribe_then_translate"
        and result.status
        in {"failed_asr", "cancelled", "deferred_time_limit", "deferred_gpu_error_limit"}
    ):
        _advance_asr_progress(progress, item, _asr_progress_outcome(result))
    if (
        item.status in {"transcribe_then_translate", "translate_existing_japanese"}
        and result.status
        in {
            "failed_asr",
            "failed_precondition",
            "cancelled",
            "deferred_time_limit",
            "deferred_gpu_error_limit",
        }
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
    if result.status == "deferred_time_limit":
        return "deferred"
    if result.status == "deferred_gpu_error_limit":
        return "deferred"
    if result.status == "cancelled":
        return "cancelled"
    if result.asr is None:
        return "blocked"
    if result.status == "failed_asr" or result.asr.return_code != 0:
        return "failed"
    return "ok"


def _translation_progress_outcome(result: VideoResult) -> str:
    if result.status == "deferred_time_limit":
        return "deferred"
    if result.status == "deferred_gpu_error_limit":
        return "deferred"
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
