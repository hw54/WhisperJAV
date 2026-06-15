import json
import sys
import threading
import time
from pathlib import Path

from whisperjav.batch.models import ProcessResult, VideoResult
from whisperjav.batch.reports import BatchReportWriter, summarize_results
from whisperjav.batch.runners import SubprocessRunner, completed_process_result


def test_batch_report_writer_writes_jsonl_and_summary_without_secrets(tmp_path):
    report_dir = tmp_path / "reports"
    input_root = tmp_path / "input"
    video = input_root / "ABC-123.mp4"
    japanese = input_root / "ABC-123.ja.pass1.srt"
    chinese = input_root / "ABC-123.ja.pass1.chinese.srt"
    nfo = input_root / "ABC-123.nfo"
    writer = BatchReportWriter(report_dir, input_root)
    results = [
        VideoResult(
            video_path=video,
            status="completed",
            nfo_path=nfo,
            actresses=("Actor One",),
            japanese_srt=japanese,
            chinese_srt=chinese,
            asr=ProcessResult(seconds=1.25, return_code=0, command_redacted=("asr", str(video))),
            translation=ProcessResult(
                seconds=2.5,
                return_code=0,
                command_redacted=("translate", "--model", "deepseek-v4-flash"),
                api_key_source="env",
            ),
        ),
        VideoResult(
            video_path=input_root / "BAD-999.mp4",
            status="failed_translation",
            reason="translation_failed",
            translation=ProcessResult(
                seconds=0.5,
                return_code=1,
                command_redacted=("translate", "--model", "deepseek-v4-flash"),
                api_key_source="env",
                stderr_tail="failed",
            ),
            error="translation failed",
        ),
    ]

    paths = writer.write(results, started_at="2026-06-16T00:00:00Z", ended_at="2026-06-16T00:00:03Z")

    assert paths.report_file == report_dir / "batch_results.jsonl"
    assert paths.summary_file == report_dir / "batch_summary.json"
    records = [json.loads(line) for line in paths.report_file.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(paths.summary_file.read_text(encoding="utf-8"))
    serialized = json.dumps({"records": records, "summary": summary})
    assert "fake-secret" not in serialized
    assert "--api-key" not in records[0]["translation"]["command_redacted"]
    assert records[0]["video_path"] == str(video)
    assert records[0]["nfo_path"] == str(nfo)
    assert records[0]["japanese_srt"] == str(japanese)
    assert records[0]["chinese_srt"] == str(chinese)
    assert records[0]["status"] == "completed"
    assert records[0]["translation"]["api_key_source"] == "env"
    assert summary["input_root"] == str(input_root)
    assert summary["started_at"] == "2026-06-16T00:00:00Z"
    assert summary["ended_at"] == "2026-06-16T00:00:03Z"
    assert summary["report_file"] == str(paths.report_file)
    assert summary["counts"] == {"completed": 1, "failed_translation": 1}
    assert summary["failed"] == [
        {
            "video_path": str(input_root / "BAD-999.mp4"),
            "status": "failed_translation",
            "reason": "translation_failed",
            "error": "translation failed",
        }
    ]


def test_summarize_results_totals_seconds_and_counts_statuses(tmp_path):
    results = [
        VideoResult(
            video_path=tmp_path / "ok.mp4",
            status="completed",
            asr=ProcessResult(seconds=1.5),
            translation=ProcessResult(seconds=2.25),
        ),
        VideoResult(
            video_path=tmp_path / "skip.mp4",
            status="skip_translated",
            translation=ProcessResult(seconds=0.75),
        ),
        VideoResult(
            video_path=tmp_path / "failed.mp4",
            status="failed_asr",
            reason="asr_failed",
            error="boom",
            asr=ProcessResult(seconds=3.0),
        ),
    ]

    summary = summarize_results(results, input_root=tmp_path, report_file=tmp_path / "batch_results.jsonl")

    assert summary["counts"] == {"completed": 1, "failed_asr": 1, "skip_translated": 1}
    assert summary["total"] == 3
    assert summary["asr_seconds"] == 4.5
    assert summary["translation_seconds"] == 3.0
    assert summary["failed"] == [
        {
            "video_path": str(tmp_path / "failed.mp4"),
            "status": "failed_asr",
            "reason": "asr_failed",
            "error": "boom",
        }
    ]


def test_completed_process_result_redacts_command_and_keeps_bounded_tails():
    stdout_lines = [f"stdout {index}" for index in range(85)]
    stderr_lines = [f"stderr {index}" for index in range(85)]

    result = completed_process_result(
        ["cmd", "--api-key", "fake-secret", "--model", "x"],
        return_code=7,
        seconds=1.25,
        stdout_lines=stdout_lines,
        stderr_lines=stderr_lines,
        api_key_source="env",
    )

    assert result.command_redacted == ("cmd", "--api-key", "<redacted>", "--model", "x")
    assert isinstance(result.command_redacted, tuple)
    assert result.return_code == 7
    assert result.seconds == 1.25
    assert result.api_key_source == "env"
    assert "stdout 0" not in result.stdout_tail
    assert "stdout 5" in result.stdout_tail
    assert "stderr 0" not in result.stderr_tail
    assert "stderr 5" in result.stderr_tail
    assert "fake-secret" not in result.stdout_tail
    assert "fake-secret" not in result.stderr_tail


def test_subprocess_runner_streams_output_and_preserves_tails(capsys):
    runner = SubprocessRunner(stream=True)
    command = [
        sys.executable,
        "-c",
        "import os, sys; print('out:' + os.environ['BATCH_TEST_VALUE']); "
        "print('err:' + os.environ['BATCH_TEST_VALUE'], file=sys.stderr)",
    ]

    result = runner.run(command, env={"BATCH_TEST_VALUE": "from-env"})

    captured = capsys.readouterr()
    assert result.return_code == 0
    assert "out:from-env" in captured.out
    assert "err:from-env" in captured.err
    assert "out:from-env" in result.stdout_tail
    assert "err:from-env" in result.stderr_tail
    assert result.command_redacted == tuple(command)


def test_subprocess_runner_terminate_all_stops_active_process(tmp_path):
    runner = SubprocessRunner(stream=False)
    marker = tmp_path / "subprocess-started"
    result_holder = []
    thread = threading.Thread(
        target=lambda: result_holder.append(
            runner.run(
                [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; import time; Path({str(marker)!r}).write_text('1'); time.sleep(30)",
                ]
            )
        )
    )
    thread.start()
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    runner.terminate_all()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert result_holder
    assert result_holder[0].return_code != 0
