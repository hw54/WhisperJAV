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
    ) -> ProcessResult:
        raise KeyboardInterrupt

    def terminate_all(self) -> None:
        self.terminated = True


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
