from __future__ import annotations

import threading
import time
from pathlib import Path

import whisperjav.batch.scheduler as scheduler_module
from whisperjav.batch.asr import AsrExecutionResult, QwenStagedAsrPipeline
from whisperjav.batch.models import BatchOptions, ClassifiedVideo, ProcessResult
from whisperjav.batch.scheduler import BatchScheduler

from tests.test_batch_scheduler import FakeRunner, RecordingProgress, write_valid_srt


class FakeStagedAsrPipeline:
    def __init__(self) -> None:
        self.timeline: list[str] = []
        self.second_prepared = threading.Event()

    def prepare(self, item: ClassifiedVideo):
        self.timeline.append(f"prepare:{item.video_path.name}")
        if item.video_path.name == "SECOND.mp4":
            self.second_prepared.set()
        time.sleep(0.01)
        return item

    def transcribe(self, prepared: ClassifiedVideo) -> AsrExecutionResult:
        self.timeline.append(f"transcribe:{prepared.video_path.name}")
        if prepared.video_path.name == "FIRST.mp4":
            assert self.second_prepared.wait(timeout=1.0)
        japanese_srt = prepared.video_path.with_name(f"{prepared.video_path.stem}.ja.pass1.srt")
        write_valid_srt(japanese_srt)
        return AsrExecutionResult(
            process=ProcessResult(return_code=0, seconds=5.0),
            japanese_srt=japanese_srt,
            error=None,
        )


class FailingStagedAsrPipeline:
    def prepare(self, item: ClassifiedVideo):
        return item

    def transcribe(self, prepared: ClassifiedVideo) -> AsrExecutionResult:
        return AsrExecutionResult(
            process=ProcessResult(return_code=1, stderr_tail="hip launch failed"),
            japanese_srt=None,
            error="staged_asr_failed: hip launch failed",
        )


class HipThenSuccessfulStagedAsrPipeline:
    def prepare(self, item: ClassifiedVideo):
        return item

    def transcribe(self, prepared: ClassifiedVideo) -> AsrExecutionResult:
        if prepared.video_path.name == "FIRST.mp4":
            return AsrExecutionResult(
                process=ProcessResult(
                    return_code=1,
                    stderr_tail="HIP error: unspecified launch failure",
                ),
                japanese_srt=None,
                error="staged_asr_failed: HIP error: unspecified launch failure",
            )
        japanese_srt = prepared.video_path.with_name(f"{prepared.video_path.stem}.ja.pass1.srt")
        write_valid_srt(japanese_srt)
        return AsrExecutionResult(
            process=ProcessResult(return_code=0, seconds=5.0),
            japanese_srt=japanese_srt,
            error=None,
        )


class AlwaysHipStagedAsrPipeline:
    def prepare(self, item: ClassifiedVideo):
        return item

    def transcribe(self, prepared: ClassifiedVideo) -> AsrExecutionResult:
        return AsrExecutionResult(
            process=ProcessResult(
                return_code=1,
                stderr_tail="HIP error: unspecified launch failure",
            ),
            japanese_srt=None,
            error="staged_asr_failed: HIP error: unspecified launch failure",
        )


class ImmediateFuture:
    def __init__(self, result) -> None:
        self._result = result

    def result(self):
        return self._result


class RecordingTranscribeExecutor:
    def __init__(self) -> None:
        self.shutdown_calls = 0
        self.submissions = 0

    def submit(self, fn, prepared):
        self.submissions += 1
        return ImmediateFuture(fn(prepared))

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        self.shutdown_calls += 1


class RaisingFuture:
    def result(self):
        raise RuntimeError("process pool worker terminated abruptly")


class RaisingTranscribeExecutor(RecordingTranscribeExecutor):
    def submit(self, fn, prepared):
        self.submissions += 1
        return RaisingFuture()


def test_staged_asr_prepares_next_file_before_current_transcription_finishes(tmp_path: Path) -> None:
    first = tmp_path / "FIRST.mp4"
    second = tmp_path / "SECOND.mp4"
    first.write_text("video", encoding="utf-8")
    second.write_text("video", encoding="utf-8")
    items = [
        ClassifiedVideo(video_path=first, status="transcribe_then_translate"),
        ClassifiedVideo(video_path=second, status="transcribe_then_translate"),
    ]
    staged_asr = FakeStagedAsrPipeline()
    progress = RecordingProgress()

    results = BatchScheduler(
        BatchOptions(root=tmp_path, asr_mode="staged", asr_cpu_workers=2),
        runner=FakeRunner(),
        progress=progress,
        asr_pipeline=staged_asr,
    ).run(items)

    assert [result.status for result in results] == ["completed", "completed"]
    assert progress.messages[:2] == [
        "正在准备ASR：FIRST.mp4",
        "正在准备ASR：SECOND.mp4",
    ]
    assert staged_asr.timeline.index("prepare:SECOND.mp4") < staged_asr.timeline.index(
        "transcribe:FIRST.mp4"
    )


def test_staged_asr_failure_advances_asr_progress_once(tmp_path: Path) -> None:
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    item = ClassifiedVideo(video_path=video, status="transcribe_then_translate")
    progress = RecordingProgress()

    results = BatchScheduler(
        BatchOptions(root=tmp_path, asr_mode="staged"),
        runner=FakeRunner(),
        progress=progress,
        asr_pipeline=FailingStagedAsrPipeline(),
    ).run([item])

    assert results[0].status == "failed_asr"
    assert progress.events.count(("asr", "failed", "ABC-123.mp4")) == 1
    assert progress.events.count(("translation", "blocked_asr", "ABC-123.mp4")) == 1
    assert progress.events.count(("files", "failed_asr", "ABC-123.mp4")) == 1


def test_staged_asr_restarts_transcribe_executor_after_hip_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = tmp_path / "FIRST.mp4"
    second = tmp_path / "SECOND.mp4"
    first.write_text("video", encoding="utf-8")
    second.write_text("video", encoding="utf-8")
    items = [
        ClassifiedVideo(video_path=first, status="transcribe_then_translate"),
        ClassifiedVideo(video_path=second, status="transcribe_then_translate"),
    ]
    progress = RecordingProgress()
    executors: list[RecordingTranscribeExecutor] = []

    def create_transcribe_executor(_asr_pipeline):
        executor = RecordingTranscribeExecutor()
        executors.append(executor)
        return executor

    monkeypatch.setattr(
        scheduler_module,
        "_create_transcribe_executor",
        create_transcribe_executor,
        raising=False,
    )

    results = BatchScheduler(
        BatchOptions(root=tmp_path, asr_mode="staged"),
        runner=FakeRunner(),
        progress=progress,
        asr_pipeline=HipThenSuccessfulStagedAsrPipeline(),
    ).run(items)

    assert [result.status for result in results] == ["failed_asr", "completed"]
    assert len(executors) == 2
    assert executors[0].shutdown_calls == 1
    assert executors[1].submissions == 1
    assert "检测到GPU运行时错误，重启ASR worker" in progress.messages


def test_staged_asr_worker_exception_fails_one_file_and_restarts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = tmp_path / "FIRST.mp4"
    second = tmp_path / "SECOND.mp4"
    first.write_text("video", encoding="utf-8")
    second.write_text("video", encoding="utf-8")
    items = [
        ClassifiedVideo(video_path=first, status="transcribe_then_translate"),
        ClassifiedVideo(video_path=second, status="transcribe_then_translate"),
    ]
    progress = RecordingProgress()
    executors = [RaisingTranscribeExecutor(), RecordingTranscribeExecutor()]

    def create_transcribe_executor(_asr_pipeline):
        return executors.pop(0)

    monkeypatch.setattr(
        scheduler_module,
        "_create_transcribe_executor",
        create_transcribe_executor,
        raising=False,
    )

    results = BatchScheduler(
        BatchOptions(root=tmp_path, asr_mode="staged"),
        runner=FakeRunner(),
        progress=progress,
        asr_pipeline=HipThenSuccessfulStagedAsrPipeline(),
    ).run(items)

    assert [result.status for result in results] == ["failed_asr", "completed"]
    assert (
        "staged_asr_worker_failed: process pool worker terminated abruptly"
        in results[0].error
    )
    assert progress.events.count(("files", "failed_asr", "FIRST.mp4")) == 1
    assert progress.events.count(("files", "completed", "SECOND.mp4")) == 1
    assert "检测到ASR worker异常，重启ASR worker" in progress.messages


def test_staged_asr_stops_scheduling_after_consecutive_gpu_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = tmp_path / "FIRST.mp4"
    second = tmp_path / "SECOND.mp4"
    third = tmp_path / "THIRD.mp4"
    for video in (first, second, third):
        video.write_text("video", encoding="utf-8")
    items = [
        ClassifiedVideo(video_path=first, status="transcribe_then_translate"),
        ClassifiedVideo(video_path=second, status="transcribe_then_translate"),
        ClassifiedVideo(video_path=third, status="transcribe_then_translate"),
    ]
    progress = RecordingProgress()
    executors: list[RecordingTranscribeExecutor] = []

    def create_transcribe_executor(_asr_pipeline):
        executor = RecordingTranscribeExecutor()
        executors.append(executor)
        return executor

    monkeypatch.setattr(
        scheduler_module,
        "_create_transcribe_executor",
        create_transcribe_executor,
        raising=False,
    )

    results = BatchScheduler(
        BatchOptions(
            root=tmp_path,
            asr_mode="staged",
            max_consecutive_gpu_errors=2,
        ),
        runner=FakeRunner(),
        progress=progress,
        asr_pipeline=AlwaysHipStagedAsrPipeline(),
    ).run(items)

    assert [result.status for result in results] == [
        "failed_asr",
        "failed_asr",
        "deferred_gpu_error_limit",
    ]
    assert results[2].reason == "consecutive_gpu_asr_errors"
    assert executors[-1].submissions == 0
    assert "连续GPU运行时错误达到2次，停止安排新的ASR任务。" in progress.messages
    assert progress.events.count(("asr", "deferred", "THIRD.mp4")) == 1
    assert progress.events.count(("translation", "deferred", "THIRD.mp4")) == 1
    assert progress.events.count(("files", "deferred_gpu_error_limit", "THIRD.mp4")) == 1


def test_qwen_staged_asr_suppresses_child_status_prefix_in_progress_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_kwargs = []

    class FakeQwenPipeline:
        def __init__(self, **kwargs) -> None:
            captured_kwargs.append(kwargs)
            self.temp_dir = Path(kwargs["temp_dir"])

    import whisperjav.pipelines.qwen_pipeline as qwen_module

    monkeypatch.setattr(qwen_module, "QwenPipeline", FakeQwenPipeline)
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    item = ClassifiedVideo(video_path=video, status="transcribe_then_translate")

    QwenStagedAsrPipeline(BatchOptions(root=tmp_path))._build_pipeline(item)

    assert captured_kwargs[0]["batch_status_prefix"] is None


def test_qwen_staged_asr_keeps_child_status_prefix_without_progress(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_kwargs = []

    class FakeQwenPipeline:
        def __init__(self, **kwargs) -> None:
            captured_kwargs.append(kwargs)
            self.temp_dir = Path(kwargs["temp_dir"])

    import whisperjav.pipelines.qwen_pipeline as qwen_module

    monkeypatch.setattr(qwen_module, "QwenPipeline", FakeQwenPipeline)
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    item = ClassifiedVideo(video_path=video, status="transcribe_then_translate")

    QwenStagedAsrPipeline(BatchOptions(root=tmp_path, no_progress=True))._build_pipeline(item)

    assert captured_kwargs[0]["batch_status_prefix"] == "ASR状态：ABC-123.mp4"


def test_staged_asr_defers_unscheduled_items_after_time_limit(tmp_path: Path) -> None:
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    item = ClassifiedVideo(video_path=video, status="transcribe_then_translate")
    staged_asr = FakeStagedAsrPipeline()
    progress = RecordingProgress()

    results = BatchScheduler(
        BatchOptions(root=tmp_path, asr_mode="staged", run_minutes=0),
        runner=FakeRunner(),
        progress=progress,
        asr_pipeline=staged_asr,
    ).run([item])

    assert [result.status for result in results] == ["deferred_time_limit"]
    assert results[0].reason == "run_time_limit_reached"
    assert staged_asr.timeline == []
    assert progress.events == [
        ("asr", "deferred", "ABC-123.mp4"),
        ("translation", "deferred", "ABC-123.mp4"),
        ("files", "deferred_time_limit", "ABC-123.mp4"),
    ]
