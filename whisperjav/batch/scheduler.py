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

TERMINAL_SKIP_STATUSES = {
    "skip_external_subtitle",
    "skip_translated",
    "failed_precondition",
}


class BatchScheduler:
    def __init__(
        self,
        options: BatchOptions,
        *,
        runner: SubprocessRunner | None = None,
    ) -> None:
        if options.translate_workers < 1 or options.translate_workers > 4:
            raise ValueError("translate_workers must be between 1 and 4")
        if options.translation_queue_size < 1:
            raise ValueError("translation_queue_size must be at least 1")
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
        asr = self.runner.run(build_asr_command(item.video_path, self.options))
        if asr.return_code != 0:
            results[index] = _from_classified(
                item,
                status="failed_asr",
                asr=asr,
                error=asr.stderr_tail or "asr failed",
            )
            return

        japanese = _expected_japanese_srt(item.video_path)
        if not _is_valid_existing_srt(japanese):
            results[index] = _from_classified(
                item,
                status="failed_asr",
                asr=asr,
                error=f"expected Japanese SRT missing or invalid: {japanese}",
            )
            return
        self._submit_translation(executor, slots, pending, index, item, asr, japanese)

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
            translation = self._run_translation(japanese_srt, item.actresses)
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
            status: BatchStatus = "completed_translation_only" if asr is None else "completed"
            return _from_classified(
                item,
                status=status,
                japanese_srt=japanese_srt,
                chinese_srt=chinese if chinese.exists() else None,
                asr=asr,
                translation=translation,
            )
        finally:
            slots.release()

    def _run_translation(
        self,
        japanese_srt: Path,
        actresses: tuple[str, ...],
    ) -> ProcessResult:
        command = build_translation_command(
            japanese_srt,
            self.options,
            actresses=actresses,
        )
        env = build_translation_env(self.options, base_env=os.environ)
        result = self.runner.run(command, env=env)
        api_key_source = "env" if self.options.translate_api_key else result.api_key_source
        return replace(result, api_key_source=api_key_source)


def _from_classified(
    item: ClassifiedVideo,
    *,
    status: BatchStatus | None = None,
    reason: str | None = None,
    japanese_srt: Path | None = None,
    chinese_srt: Path | None = None,
    asr: ProcessResult | None = None,
    translation: ProcessResult | None = None,
    error: str | None = None,
) -> VideoResult:
    return VideoResult(
        video_path=item.video_path,
        status=status or item.status,
        reason=reason if reason is not None else item.reason,
        nfo_path=item.nfo_path,
        actresses=item.actresses,
        japanese_srt=japanese_srt or item.japanese_srt,
        chinese_srt=chinese_srt or item.chinese_srt,
        asr=asr,
        translation=translation,
        error=error,
        warnings=item.warnings,
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
