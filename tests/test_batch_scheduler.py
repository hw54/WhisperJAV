from __future__ import annotations

import threading
import time
import json
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from whisperjav.batch.models import BatchOptions, ClassifiedVideo, ProcessResult
from whisperjav.batch.scheduler import BatchScheduler


def write_valid_srt(path: Path, text: str = "はい") -> None:
    path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n"
        f"{text}\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n"
        f"{text}\n",
        encoding="utf-8",
    )


def write_valid_subtrans(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "scenes": [
                    {
                        "scene": 1,
                        "context": {"summary": "Scene summary"},
                        "batches": [
                            {
                                "batch": 1,
                                "summary": "Batch summary",
                                "originals": [{"index": 1}, {"index": 2}],
                            }
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class FakeRunner:
    def __init__(
        self,
        *,
        fail_asr: bool = False,
        fail_translation: bool = False,
    ) -> None:
        self.fail_asr = fail_asr
        self.fail_translation = fail_translation
        self.commands: list[list[str]] = []
        self.envs: list[Mapping[str, str] | None] = []

    def run(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        heartbeat=None,
        heartbeat_interval_seconds: float | None = None,
    ) -> ProcessResult:
        command_list = list(command)
        self.commands.append(command_list)
        self.envs.append(env)
        if "whisperjav.main" in command_list:
            return self._run_asr(command_list)
        return self._run_translation(command_list)

    def _run_asr(self, command: list[str]) -> ProcessResult:
        if self.fail_asr:
            return ProcessResult(
                return_code=9,
                command_redacted=command,
                stderr_tail="asr failed",
            )
        video = Path(command[4])
        write_valid_srt(video.with_name(f"{video.stem}.ja.pass1.srt"))
        return ProcessResult(return_code=0, command_redacted=command, seconds=2.0)

    def _run_translation(self, command: list[str]) -> ProcessResult:
        if self.fail_translation:
            return ProcessResult(
                return_code=8,
                command_redacted=command,
                stderr_tail="translation failed",
            )
        srt = Path(command[command.index("-i") + 1])
        write_valid_srt(srt.with_name(f"{srt.stem}.chinese.srt"), "中文")
        write_valid_subtrans(srt.with_suffix(".subtrans"))
        return ProcessResult(return_code=0, command_redacted=command, seconds=3.0)


class RecordingProgress:
    def __init__(self) -> None:
        self.totals = None
        self.events: list[tuple[str, str, str]] = []
        self.messages: list[str] = []
        self.timeline: list[tuple[str, str]] = []
        self.closed = False

    def start(self, totals) -> None:
        self.totals = totals

    def advance(self, event) -> None:
        self.events.append((event.phase, event.outcome, event.video_path.name))
        self.timeline.append(("event", f"{event.phase}:{event.outcome}:{event.video_path.name}"))

    def message(self, text: str) -> None:
        self.messages.append(text)
        self.timeline.append(("message", text))

    def close(self) -> None:
        self.closed = True


def test_existing_japanese_srt_translation_completes(tmp_path: Path) -> None:
    video = tmp_path / "ABC-123.mp4"
    japanese = tmp_path / "ABC-123.ja.pass1.srt"
    nfo = tmp_path / "ABC-123.nfo"
    video.write_text("video", encoding="utf-8")
    nfo.write_text("<movie />", encoding="utf-8")
    write_valid_srt(japanese)
    item = ClassifiedVideo(
        video_path=video,
        status="translate_existing_japanese",
        japanese_srt=japanese,
        nfo_path=nfo,
        actresses=("Name1", "Name2"),
        movie_title="Movie Title",
        movie_plot="Movie Plot",
        warnings=("nfo_warning",),
    )
    runner = FakeRunner()

    results = BatchScheduler(BatchOptions(root=tmp_path), runner=runner).run([item])

    assert results[0].status == "completed_translation_only"
    assert results[0].translation is not None
    assert results[0].translation.return_code == 0
    assert results[0].nfo_path == nfo
    assert results[0].actresses == ("Name1", "Name2")
    assert results[0].movie_title == "Movie Title"
    assert results[0].movie_plot == "Movie Plot"
    assert results[0].warnings == ("nfo_warning",)
    assert results[0].japanese_srt == japanese
    assert results[0].chinese_srt == tmp_path / "ABC-123.ja.pass1.chinese.srt"
    assert results[0].summary_json == tmp_path / "ABC-123.ja.pass1.chinese.summary.json"
    assert results[0].summary_json.exists()
    assert any("whisperjav.translate.cli" in command for command in runner.commands)
    assert not any("whisperjav.main" in command for command in runner.commands)
    translation_command = next(command for command in runner.commands if "whisperjav.translate.cli" in command)
    assert translation_command[translation_command.index("--movie-title") + 1] == "Movie Title"
    assert translation_command[translation_command.index("--movie-plot") + 1] == "Movie Plot"


def test_transcribe_then_translate_runs_asr_before_translation(tmp_path: Path) -> None:
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    item = ClassifiedVideo(video_path=video, status="transcribe_then_translate")
    runner = FakeRunner()

    results = BatchScheduler(BatchOptions(root=tmp_path), runner=runner).run([item])

    assert results[0].status == "completed"
    assert results[0].japanese_srt == tmp_path / "ABC-123.ja.pass1.srt"
    assert results[0].chinese_srt == tmp_path / "ABC-123.ja.pass1.chinese.srt"
    assert results[0].summary_json == tmp_path / "ABC-123.ja.pass1.chinese.summary.json"
    assert [command[3] for command in runner.commands] == [
        "whisperjav.main",
        "whisperjav.translate.cli",
    ]


def test_scheduler_reports_file_asr_and_translation_progress(tmp_path: Path) -> None:
    skipped_video = tmp_path / "SKIP-001.mp4"
    existing_video = tmp_path / "EXIST-001.mp4"
    asr_video = tmp_path / "ASR-001.mp4"
    existing_japanese = tmp_path / "EXIST-001.ja.pass1.srt"
    existing_chinese = tmp_path / "SKIP-001.chinese.srt"
    skipped_video.write_text("video", encoding="utf-8")
    existing_video.write_text("video", encoding="utf-8")
    asr_video.write_text("video", encoding="utf-8")
    write_valid_srt(existing_japanese)
    write_valid_srt(existing_chinese, "中文")
    items = [
        ClassifiedVideo(
            video_path=skipped_video,
            status="skip_translated",
            chinese_srt=existing_chinese,
        ),
        ClassifiedVideo(
            video_path=existing_video,
            status="translate_existing_japanese",
            japanese_srt=existing_japanese,
        ),
        ClassifiedVideo(video_path=asr_video, status="transcribe_then_translate"),
    ]
    runner = FakeRunner(fail_asr=True)
    progress = RecordingProgress()

    results = BatchScheduler(BatchOptions(root=tmp_path), runner=runner, progress=progress).run(items)

    assert [result.status for result in results] == [
        "skip_translated",
        "completed_translation_only",
        "failed_asr",
    ]
    assert progress.totals.files == 3
    assert progress.totals.asr == 1
    assert progress.totals.translation == 2
    assert progress.closed
    assert progress.events.count(("files", "skip_translated", "SKIP-001.mp4")) == 1
    assert progress.events.count(("files", "completed_translation_only", "EXIST-001.mp4")) == 1
    assert progress.events.count(("files", "failed_asr", "ASR-001.mp4")) == 1
    assert progress.events.count(("translation", "translated", "EXIST-001.mp4")) == 1
    assert progress.events.count(("asr", "failed", "ASR-001.mp4")) == 1
    assert progress.events.count(("translation", "blocked_asr", "ASR-001.mp4")) == 1


def test_scheduler_reports_asr_and_translation_activity_messages(tmp_path: Path) -> None:
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    item = ClassifiedVideo(video_path=video, status="transcribe_then_translate")
    runner = FakeRunner()
    progress = RecordingProgress()

    results = BatchScheduler(BatchOptions(root=tmp_path), runner=runner, progress=progress).run([item])

    assert results[0].status == "completed"
    assert progress.messages == [
        "正在处理ASR：ABC-123.mp4",
        "ASR完成：ABC-123.mp4",
        "正在处理翻译：ABC-123.mp4",
        "翻译完成：ABC-123.mp4",
    ]


def test_scheduler_advances_asr_progress_before_translation_starts(tmp_path: Path) -> None:
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    item = ClassifiedVideo(video_path=video, status="transcribe_then_translate")
    progress = RecordingProgress()

    results = BatchScheduler(BatchOptions(root=tmp_path), runner=FakeRunner(), progress=progress).run([item])

    assert results[0].status == "completed"
    assert ("asr", "ok", "ABC-123.mp4") in progress.events
    assert progress.timeline.index(("event", "asr:ok:ABC-123.mp4")) < progress.timeline.index(
        ("message", "正在处理翻译：ABC-123.mp4")
    )


class HeartbeatRunner(FakeRunner):
    def run(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        heartbeat=None,
        heartbeat_interval_seconds: float | None = None,
    ) -> ProcessResult:
        if heartbeat is not None:
            if "whisperjav.main" in command:
                heartbeat(65.0)
            else:
                heartbeat(125.0)
        return super().run(
            command,
            env=env,
            heartbeat=heartbeat,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        )


def test_scheduler_reports_long_running_subprocess_heartbeats(tmp_path: Path) -> None:
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    item = ClassifiedVideo(video_path=video, status="transcribe_then_translate")
    progress = RecordingProgress()

    results = BatchScheduler(BatchOptions(root=tmp_path), runner=HeartbeatRunner(), progress=progress).run([item])

    assert results[0].status == "completed"
    assert "ASR仍在处理：ABC-123.mp4（已耗时 1m05s）" in progress.messages
    assert "翻译仍在处理：ABC-123.mp4（已耗时 2m05s）" in progress.messages


class TranslationCompletesDuringNextAsrRunner(FakeRunner):
    def __init__(self) -> None:
        super().__init__()
        self.translation_started = threading.Event()
        self.allow_translation_finish = threading.Event()

    def run(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        heartbeat=None,
        heartbeat_interval_seconds: float | None = None,
    ) -> ProcessResult:
        command_list = list(command)
        self.commands.append(command_list)
        self.envs.append(env)
        if "whisperjav.main" not in command_list:
            return self._run_translation(command_list)

        video = Path(command_list[4])
        if video.name == "SECOND.mp4":
            assert self.translation_started.wait(timeout=1.0)
            self.allow_translation_finish.set()
            time.sleep(0.05)
            if heartbeat is not None:
                heartbeat(65.0)
        return self._run_asr(command_list)

    def _run_translation(self, command: list[str]) -> ProcessResult:
        srt = Path(command[command.index("-i") + 1])
        if srt.name == "FIRST.ja.pass1.srt":
            self.translation_started.set()
            assert self.allow_translation_finish.wait(timeout=1.0)
        return super()._run_translation(command)


def test_scheduler_advances_translation_progress_while_next_asr_is_running(tmp_path: Path) -> None:
    first = tmp_path / "FIRST.mp4"
    second = tmp_path / "SECOND.mp4"
    first.write_text("video", encoding="utf-8")
    second.write_text("video", encoding="utf-8")
    items = [
        ClassifiedVideo(video_path=first, status="transcribe_then_translate"),
        ClassifiedVideo(video_path=second, status="transcribe_then_translate"),
    ]
    progress = RecordingProgress()

    results = BatchScheduler(
        BatchOptions(root=tmp_path),
        runner=TranslationCompletesDuringNextAsrRunner(),
        progress=progress,
    ).run(items)

    assert [result.status for result in results] == ["completed", "completed"]
    assert progress.timeline.index(("event", "translation:translated:FIRST.mp4")) < progress.timeline.index(
        ("message", "ASR仍在处理：SECOND.mp4（已耗时 1m05s）")
    )


class MissingSubtransRunner(FakeRunner):
    def _run_translation(self, command: list[str]) -> ProcessResult:
        srt = Path(command[command.index("-i") + 1])
        write_valid_srt(srt.with_name(f"{srt.stem}.chinese.srt"), "中文")
        return ProcessResult(return_code=0, command_redacted=command, seconds=3.0)


def test_summary_export_failure_warns_without_failing_translation(tmp_path: Path) -> None:
    video = tmp_path / "ABC-123.mp4"
    japanese = tmp_path / "ABC-123.ja.pass1.srt"
    video.write_text("video", encoding="utf-8")
    write_valid_srt(japanese)
    item = ClassifiedVideo(
        video_path=video,
        status="translate_existing_japanese",
        japanese_srt=japanese,
    )
    runner = MissingSubtransRunner()

    results = BatchScheduler(BatchOptions(root=tmp_path), runner=runner).run([item])

    assert results[0].status == "completed_translation_only"
    assert results[0].summary_json is None
    assert len(results[0].warnings) == 1
    assert results[0].warnings[0].startswith("scene_summary_export_failed:")


def test_asr_failure_does_not_enqueue_translation(tmp_path: Path) -> None:
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    item = ClassifiedVideo(video_path=video, status="transcribe_then_translate")
    runner = FakeRunner(fail_asr=True)

    results = BatchScheduler(BatchOptions(root=tmp_path), runner=runner).run([item])

    assert results[0].status == "failed_asr"
    assert results[0].error == "asr failed"
    assert len(runner.commands) == 2


class HipFailRunner(FakeRunner):
    def _run_asr(self, command: list[str]) -> ProcessResult:
        return ProcessResult(
            return_code=9,
            command_redacted=command,
            stderr_tail="HIP error: unspecified launch failure",
        )


def test_subprocess_asr_stops_scheduling_after_consecutive_gpu_errors(
    tmp_path: Path,
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
    runner = HipFailRunner()
    progress = RecordingProgress()

    results = BatchScheduler(
        BatchOptions(
            root=tmp_path,
            asr_retries=0,
            max_consecutive_gpu_errors=2,
        ),
        runner=runner,
        progress=progress,
    ).run(items)

    assert [result.status for result in results] == [
        "failed_asr",
        "failed_asr",
        "deferred_gpu_error_limit",
    ]
    assert results[2].reason == "consecutive_gpu_asr_errors"
    assert [Path(command[4]).name for command in runner.commands] == [
        "FIRST.mp4",
        "SECOND.mp4",
    ]
    assert "连续GPU运行时错误达到2次，停止安排新的ASR任务。" in progress.messages
    assert progress.events.count(("asr", "deferred", "THIRD.mp4")) == 1
    assert progress.events.count(("translation", "deferred", "THIRD.mp4")) == 1
    assert progress.events.count(("files", "deferred_gpu_error_limit", "THIRD.mp4")) == 1


class FlakyAsrRunner(FakeRunner):
    def __init__(self) -> None:
        super().__init__()
        self.asr_attempts = 0

    def _run_asr(self, command: list[str]) -> ProcessResult:
        self.asr_attempts += 1
        if self.asr_attempts == 1:
            return ProcessResult(
                return_code=9,
                command_redacted=command,
                stderr_tail="temporary asr failure",
            )
        return super()._run_asr(command)


def test_asr_retries_before_translation(tmp_path: Path) -> None:
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    item = ClassifiedVideo(video_path=video, status="transcribe_then_translate")
    runner = FlakyAsrRunner()

    results = BatchScheduler(BatchOptions(root=tmp_path, asr_retries=1), runner=runner).run([item])

    assert results[0].status == "completed"
    assert runner.asr_attempts == 2
    assert [command[3] for command in runner.commands] == [
        "whisperjav.main",
        "whisperjav.main",
        "whisperjav.translate.cli",
    ]


class InterruptedAsrReturnRunner(FakeRunner):
    def __init__(self) -> None:
        super().__init__()
        self.asr_attempts = 0

    def _run_asr(self, command: list[str]) -> ProcessResult:
        self.asr_attempts += 1
        return ProcessResult(
            return_code=130,
            command_redacted=command,
            stderr_tail="asr interrupted",
        )


def test_asr_interrupt_return_code_is_not_retried(tmp_path: Path) -> None:
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    item = ClassifiedVideo(video_path=video, status="transcribe_then_translate")
    runner = InterruptedAsrReturnRunner()

    results = BatchScheduler(BatchOptions(root=tmp_path, asr_retries=2), runner=runner).run([item])

    assert results[0].status == "failed_asr"
    assert results[0].error == "asr interrupted"
    assert runner.asr_attempts == 1


def test_translation_failure_keeps_processing_later_items(tmp_path: Path) -> None:
    first = tmp_path / "ABC-123.mp4"
    second = tmp_path / "DEF-456.mp4"
    translated = tmp_path / "DEF-456.chinese.srt"
    first.write_text("video", encoding="utf-8")
    second.write_text("video", encoding="utf-8")
    write_valid_srt(translated, "中文")
    runner = FakeRunner(fail_translation=True)
    items = [
        ClassifiedVideo(video_path=first, status="transcribe_then_translate"),
        ClassifiedVideo(
            video_path=second,
            status="skip_translated",
            chinese_srt=translated,
        ),
    ]

    results = BatchScheduler(BatchOptions(root=tmp_path), runner=runner).run(items)

    assert [result.status for result in results] == [
        "failed_translation",
        "skip_translated",
    ]
    assert results[0].error == "translation failed"
    assert results[1].chinese_srt == translated


class FlakyTranslationRunner(FakeRunner):
    def __init__(self) -> None:
        super().__init__()
        self.translation_attempts = 0

    def _run_translation(self, command: list[str]) -> ProcessResult:
        self.translation_attempts += 1
        if self.translation_attempts == 1:
            return ProcessResult(
                return_code=8,
                command_redacted=command,
                stderr_tail="temporary translation failure",
            )
        return super()._run_translation(command)


def test_translation_retries_before_successful_result(tmp_path: Path) -> None:
    video = tmp_path / "ABC-123.mp4"
    japanese = tmp_path / "ABC-123.ja.pass1.srt"
    video.write_text("video", encoding="utf-8")
    write_valid_srt(japanese)
    item = ClassifiedVideo(
        video_path=video,
        status="translate_existing_japanese",
        japanese_srt=japanese,
    )
    runner = FlakyTranslationRunner()

    results = BatchScheduler(
        BatchOptions(root=tmp_path, translation_retries=1),
        runner=runner,
    ).run([item])

    assert results[0].status == "completed_translation_only"
    assert runner.translation_attempts == 2
    assert [command[3] for command in runner.commands] == [
        "whisperjav.translate.cli",
        "whisperjav.translate.cli",
    ]


class InterruptedTranslationReturnRunner(FakeRunner):
    def __init__(self) -> None:
        super().__init__()
        self.translation_attempts = 0

    def _run_translation(self, command: list[str]) -> ProcessResult:
        self.translation_attempts += 1
        return ProcessResult(
            return_code=130,
            command_redacted=command,
            stderr_tail="translation interrupted",
        )


def test_translation_interrupt_return_code_is_not_retried(tmp_path: Path) -> None:
    video = tmp_path / "ABC-123.mp4"
    japanese = tmp_path / "ABC-123.ja.pass1.srt"
    video.write_text("video", encoding="utf-8")
    write_valid_srt(japanese)
    item = ClassifiedVideo(
        video_path=video,
        status="translate_existing_japanese",
        japanese_srt=japanese,
    )
    runner = InterruptedTranslationReturnRunner()

    results = BatchScheduler(
        BatchOptions(root=tmp_path, translation_retries=2),
        runner=runner,
    ).run([item])

    assert results[0].status == "failed_translation"
    assert results[0].error == "translation interrupted"
    assert runner.translation_attempts == 1


def test_dry_run_returns_classification_without_subprocesses(tmp_path: Path) -> None:
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    item = ClassifiedVideo(video_path=video, status="transcribe_then_translate")
    runner = FakeRunner()

    results = BatchScheduler(BatchOptions(root=tmp_path, dry_run=True), runner=runner).run([item])

    assert results[0].status == "transcribe_then_translate"
    assert runner.commands == []


def test_translation_api_key_uses_env_not_argv(tmp_path: Path) -> None:
    video = tmp_path / "ABC-123.mp4"
    japanese = tmp_path / "ABC-123.ja.pass1.srt"
    video.write_text("video", encoding="utf-8")
    write_valid_srt(japanese)
    item = ClassifiedVideo(
        video_path=video,
        status="translate_existing_japanese",
        japanese_srt=japanese,
    )
    runner = FakeRunner()

    results = BatchScheduler(
        BatchOptions(root=tmp_path, translate_api_key="secret-key"),
        runner=runner,
    ).run([item])

    assert results[0].translation is not None
    assert results[0].translation.api_key_source == "env"
    assert runner.envs[0] is not None
    assert runner.envs[0]["DEEPSEEK_API_KEY"] == "secret-key"
    assert "secret-key" not in runner.commands[0]


class OverlapRunner(FakeRunner):
    def __init__(self) -> None:
        super().__init__()
        self.first_translation_started = threading.Event()
        self.first_translation_done = threading.Event()
        self.second_asr_started_during_translation = threading.Event()
        self.allow_first_translation_finish = threading.Event()

    def _run_asr(self, command: list[str]) -> ProcessResult:
        video = Path(command[4])
        if video.stem == "DEF-456":
            if self.first_translation_started.is_set() and not self.first_translation_done.is_set():
                self.second_asr_started_during_translation.set()
                self.allow_first_translation_finish.set()
        write_valid_srt(video.with_name(f"{video.stem}.ja.pass1.srt"))
        return ProcessResult(return_code=0, command_redacted=command, seconds=2.0)

    def _run_translation(self, command: list[str]) -> ProcessResult:
        srt = Path(command[command.index("-i") + 1])
        if srt.stem.startswith("ABC-123"):
            self.first_translation_started.set()
            self.allow_first_translation_finish.wait(timeout=0.5)
            self.first_translation_done.set()
        write_valid_srt(srt.with_name(f"{srt.stem}.chinese.srt"), "中文")
        return ProcessResult(return_code=0, command_redacted=command, seconds=3.0)


def test_asr_overlaps_previous_translation(tmp_path: Path) -> None:
    first = tmp_path / "ABC-123.mp4"
    second = tmp_path / "DEF-456.mp4"
    first.write_text("video", encoding="utf-8")
    second.write_text("video", encoding="utf-8")
    runner = OverlapRunner()
    items = [
        ClassifiedVideo(video_path=first, status="transcribe_then_translate"),
        ClassifiedVideo(video_path=second, status="transcribe_then_translate"),
    ]

    results = BatchScheduler(
        BatchOptions(root=tmp_path, translate_workers=1, translation_queue_size=2),
        runner=runner,
    ).run(items)

    assert [result.status for result in results] == ["completed", "completed"]
    assert runner.second_asr_started_during_translation.is_set()


class BackpressureRunner(FakeRunner):
    def __init__(self) -> None:
        super().__init__()
        self.first_translation_started = threading.Event()
        self.allow_first_translation_finish = threading.Event()
        self.second_asr_started_while_queue_full = threading.Event()

    def _run_asr(self, command: list[str]) -> ProcessResult:
        video = Path(command[4])
        if video.stem == "DEF-456" and not self.allow_first_translation_finish.is_set():
            self.second_asr_started_while_queue_full.set()
        write_valid_srt(video.with_name(f"{video.stem}.ja.pass1.srt"))
        return ProcessResult(return_code=0, command_redacted=command, seconds=2.0)

    def _run_translation(self, command: list[str]) -> ProcessResult:
        srt = Path(command[command.index("-i") + 1])
        if srt.stem.startswith("ABC-123"):
            self.first_translation_started.set()
            self.allow_first_translation_finish.wait(timeout=2.0)
        write_valid_srt(srt.with_name(f"{srt.stem}.chinese.srt"), "中文")
        return ProcessResult(return_code=0, command_redacted=command, seconds=3.0)


def test_full_translation_queue_blocks_before_next_asr(tmp_path: Path) -> None:
    first = tmp_path / "ABC-123.mp4"
    second = tmp_path / "DEF-456.mp4"
    first.write_text("video", encoding="utf-8")
    second.write_text("video", encoding="utf-8")
    runner = BackpressureRunner()
    items = [
        ClassifiedVideo(video_path=first, status="transcribe_then_translate"),
        ClassifiedVideo(video_path=second, status="transcribe_then_translate"),
    ]
    results: list[list[str]] = []

    def run_scheduler() -> None:
        batch_results = BatchScheduler(
            BatchOptions(root=tmp_path, translate_workers=1, translation_queue_size=1),
            runner=runner,
        ).run(items)
        results.append([result.status for result in batch_results])

    worker = threading.Thread(target=run_scheduler)
    worker.start()
    assert runner.first_translation_started.wait(timeout=1.0)
    time.sleep(0.05)
    assert not runner.second_asr_started_while_queue_full.is_set()
    runner.allow_first_translation_finish.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert results == [["completed", "completed"]]


@pytest.mark.parametrize("translate_workers", [0])
def test_translate_workers_range_is_validated(
    tmp_path: Path,
    translate_workers: int,
) -> None:
    with pytest.raises(ValueError, match="translate_workers"):
        BatchScheduler(BatchOptions(root=tmp_path, translate_workers=translate_workers))


def test_translate_workers_accepts_api_parallelism(tmp_path: Path) -> None:
    scheduler = BatchScheduler(BatchOptions(root=tmp_path, translate_workers=20))

    assert scheduler.options.translate_workers == 20


def test_translation_queue_size_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="translation_queue_size"):
        BatchScheduler(BatchOptions(root=tmp_path, translation_queue_size=0))


@pytest.mark.parametrize("field", ["asr_retries", "translation_retries"])
def test_retry_counts_must_not_be_negative(tmp_path: Path, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        BatchScheduler(BatchOptions(root=tmp_path, **{field: -1}))


class MissingSrtRunner(FakeRunner):
    def _run_asr(self, command: list[str]) -> ProcessResult:
        return ProcessResult(return_code=0, command_redacted=command, seconds=2.0)


def test_successful_asr_with_missing_expected_srt_fails_asr(tmp_path: Path) -> None:
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    item = ClassifiedVideo(video_path=video, status="transcribe_then_translate")
    runner = MissingSrtRunner()

    results = BatchScheduler(BatchOptions(root=tmp_path), runner=runner).run([item])

    assert results[0].status == "failed_asr"
    assert "expected Japanese SRT missing or invalid" in results[0].error
    assert len(runner.commands) == 2


class InterruptingRunner(FakeRunner):
    def __init__(self) -> None:
        super().__init__()
        self.terminated = False

    def run(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        heartbeat=None,
        heartbeat_interval_seconds: float | None = None,
    ) -> ProcessResult:
        raise KeyboardInterrupt

    def terminate_all(self) -> None:
        self.terminated = True


class TimelineInterruptingRunner(FakeRunner):
    def __init__(self, timeline: list[tuple[str, str]]) -> None:
        super().__init__()
        self.timeline = timeline

    def run(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        heartbeat=None,
        heartbeat_interval_seconds: float | None = None,
    ) -> ProcessResult:
        raise KeyboardInterrupt

    def terminate_all(self) -> None:
        self.timeline.append(("runner", "terminated"))


def test_keyboard_interrupt_marks_unfinished_items_cancelled(tmp_path: Path) -> None:
    first = tmp_path / "ABC-123.mp4"
    second = tmp_path / "DEF-456.mp4"
    first.write_text("video", encoding="utf-8")
    second.write_text("video", encoding="utf-8")
    runner = InterruptingRunner()
    items = [
        ClassifiedVideo(video_path=first, status="transcribe_then_translate"),
        ClassifiedVideo(video_path=second, status="transcribe_then_translate"),
    ]

    results = BatchScheduler(BatchOptions(root=tmp_path), runner=runner).run(items)

    assert [result.status for result in results] == ["cancelled", "cancelled"]
    assert [result.reason for result in results] == ["interrupted", "interrupted"]
    assert runner.terminated


def test_keyboard_interrupt_logs_cleanup_start_before_termination_and_completion(
    tmp_path: Path,
) -> None:
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    item = ClassifiedVideo(video_path=video, status="transcribe_then_translate")
    progress = RecordingProgress()
    runner = TimelineInterruptingRunner(progress.timeline)

    results = BatchScheduler(
        BatchOptions(root=tmp_path),
        runner=runner,
        progress=progress,
    ).run([item])

    assert results[0].status == "cancelled"
    assert "正在清理产物……" in progress.messages
    assert "清理产物完成。" in progress.messages
    assert progress.timeline.index(("message", "正在清理产物……")) < progress.timeline.index(
        ("runner", "terminated")
    )
    assert progress.timeline[-1] == ("message", "清理产物完成。")


def test_keyboard_interrupt_logs_cleanup_to_stderr_without_progress(
    tmp_path: Path,
    capsys,
) -> None:
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    item = ClassifiedVideo(video_path=video, status="transcribe_then_translate")

    results = BatchScheduler(
        BatchOptions(root=tmp_path),
        runner=InterruptingRunner(),
    ).run([item])

    assert results[0].status == "cancelled"
    captured = capsys.readouterr()
    assert "正在清理产物……" in captured.err
    assert "清理产物完成。" in captured.err
    assert captured.err.index("正在清理产物……") < captured.err.index("清理产物完成。")
