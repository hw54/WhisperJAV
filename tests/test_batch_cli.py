import json

import pytest

from whisperjav.batch import cli
from whisperjav.batch import discovery
from whisperjav.batch.models import VideoResult


def write_valid_srt(path, text="ja text"):
    path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n"
        f"{text}\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n"
        f"{text}\n",
        encoding="utf-8",
    )


def read_single_jsonl_row(report_dir):
    jsonl = next(report_dir.glob("*.jsonl"))
    return json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])


def read_single_summary(report_dir):
    summary = next(report_dir.glob("*.summary.json"))
    return json.loads(summary.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def short_media_duration(monkeypatch):
    monkeypatch.setattr(discovery, "_probe_duration_seconds", lambda _path: 60.0, raising=False)


def test_parse_args_rejects_force_conflict(tmp_path):
    with pytest.raises(SystemExit):
        cli.parse_args([str(tmp_path), "--force", "--force-translate"])


def test_parse_args_rejects_missing_root(tmp_path):
    with pytest.raises(SystemExit):
        cli.parse_args([str(tmp_path / "missing")])


def test_parse_args_rejects_file_root(tmp_path):
    file_root = tmp_path / "not-a-directory"
    file_root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(SystemExit):
        cli.parse_args([str(file_root)])


def test_parse_args_accepts_multiple_roots(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    args = cli.parse_args([str(first), str(second)])

    assert args.roots == [first, second]


def test_parse_args_rejects_any_missing_root(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(SystemExit):
        cli.parse_args([str(existing), str(tmp_path / "missing")])


@pytest.mark.parametrize("workers", ["0"])
def test_parse_args_rejects_invalid_worker_count(tmp_path, workers):
    with pytest.raises(SystemExit):
        cli.parse_args([str(tmp_path), "--translate-workers", workers])


def test_parse_args_accepts_high_translation_worker_count(tmp_path):
    args = cli.parse_args([str(tmp_path), "--translate-workers", "20"])

    assert args.translate_workers == 20


def test_parse_args_rejects_invalid_translation_queue_size(tmp_path):
    with pytest.raises(SystemExit):
        cli.parse_args([str(tmp_path), "--translation-queue-size", "0"])


@pytest.mark.parametrize("option", ["--asr-retries", "--translation-retries"])
def test_parse_args_rejects_negative_retry_count(tmp_path, option):
    with pytest.raises(SystemExit):
        cli.parse_args([str(tmp_path), option, "-1"])


def test_parse_args_defaults_duration_limit_to_230_minutes(tmp_path):
    args = cli.parse_args([str(tmp_path)])

    assert args.max_video_minutes == 230


def test_parse_args_accepts_zero_duration_limit_as_disabled(tmp_path):
    args = cli.parse_args([str(tmp_path), "--max-video-minutes", "0"])

    assert args.max_video_minutes == 0


def test_parse_args_rejects_negative_duration_limit(tmp_path):
    with pytest.raises(SystemExit):
        cli.parse_args([str(tmp_path), "--max-video-minutes", "-1"])


def test_dry_run_writes_reports_and_returns_zero(tmp_path, capsys):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    report_dir = tmp_path / "reports"

    code = cli.main([str(tmp_path), "--dry-run", "--report-dir", str(report_dir)])

    assert code == 0
    summary = read_single_summary(report_dir)
    assert summary["counts_by_status"] == {"transcribe_then_translate": 1}
    assert summary["failed"] == []
    captured = capsys.readouterr()
    assert "WHISPERJAV BATCH SUMMARY" in captured.out
    assert "transcribe_then_translate: 1" in captured.out
    assert str(report_dir) in captured.out


def test_dry_run_reports_duration_limited_skips(tmp_path, monkeypatch):
    video = tmp_path / "LONG-001.mp4"
    video.write_text("video", encoding="utf-8")
    report_dir = tmp_path / "reports"
    monkeypatch.setattr(discovery, "_probe_duration_seconds", lambda _path: 231 * 60, raising=False)

    code = cli.main([str(tmp_path), "--dry-run", "--report-dir", str(report_dir)])

    assert code == 0
    row = read_single_jsonl_row(report_dir)
    summary = read_single_summary(report_dir)
    assert row["status"] == "skip_duration_limit"
    assert row["reason"] == "duration_exceeds_limit"
    assert row["duration_seconds"] == 231 * 60
    assert row["duration_limit_minutes"] == 230
    assert summary["counts_by_status"] == {"skip_duration_limit": 1}


def test_dry_run_uses_default_report_dir(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")

    code = cli.main([str(tmp_path), "--dry-run"])

    report_dir = tmp_path / ".whisperjav_batch"
    assert code == 0
    assert len(list(report_dir.glob("*.jsonl"))) == 1
    assert len(list(report_dir.glob("*.summary.json"))) == 1


def test_dry_run_processes_multiple_roots_with_default_report_dirs(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "ABC-123.mp4").write_text("video", encoding="utf-8")
    (second / "DEF-456.mp4").write_text("video", encoding="utf-8")

    code = cli.main([str(first), str(second), "--dry-run"])

    assert code == 0
    first_summary = read_single_summary(first / ".whisperjav_batch")
    second_summary = read_single_summary(second / ".whisperjav_batch")
    assert first_summary["total_videos"] == 1
    assert first_summary["input_root"] == str(first)
    assert second_summary["total_videos"] == 1
    assert second_summary["input_root"] == str(second)


def test_multi_root_uses_one_scheduler_run_to_keep_pipeline_filled(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    first_video = first / "ABC-123.mp4"
    second_video = second / "DEF-456.mp4"
    first_video.write_text("video", encoding="utf-8")
    second_video.write_text("video", encoding="utf-8")
    scheduler_runs = []

    class RecordingScheduler:
        def __init__(self, options):
            self.options = options

        def run(self, classified):
            items = list(classified)
            scheduler_runs.append([item.video_path for item in items])
            return [VideoResult(video_path=item.video_path, status=item.status) for item in items]

    monkeypatch.setattr(cli, "BatchScheduler", RecordingScheduler)

    code = cli.main([str(first), str(second), "--dry-run"])

    assert code == 0
    assert scheduler_runs == [[first_video, second_video]]
    first_summary = read_single_summary(first / ".whisperjav_batch")
    second_summary = read_single_summary(second / ".whisperjav_batch")
    assert first_summary["total_videos"] == 1
    assert second_summary["total_videos"] == 1


def test_multi_root_explicit_report_dir_uses_root_subdirectories(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "ABC-123.mp4").write_text("video", encoding="utf-8")
    (second / "DEF-456.mp4").write_text("video", encoding="utf-8")
    report_dir = tmp_path / "reports"

    code = cli.main([str(first), str(second), "--dry-run", "--report-dir", str(report_dir)])

    assert code == 0
    summaries = list(report_dir.glob("*/*.summary.json"))
    assert len(summaries) == 2
    roots = {json.loads(path.read_text(encoding="utf-8"))["input_root"] for path in summaries}
    assert roots == {str(first), str(second)}
    assert not list(report_dir.glob("*.summary.json"))


def test_cli_includes_nfo_actress_context_in_dry_run_report(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    write_valid_srt(tmp_path / "ABC-123.ja.pass1.srt")
    (tmp_path / "ABC-123.nfo").write_text(
        "<movie><actor><name>Name1</name></actor></movie>",
        encoding="utf-8",
    )
    report_dir = tmp_path / "reports"

    code = cli.main([str(tmp_path), "--dry-run", "--report-dir", str(report_dir)])

    assert code == 0
    row = read_single_jsonl_row(report_dir)
    assert row["actresses"] == ["Name1"]
    assert row["nfo_path"] == str(tmp_path / "ABC-123.nfo")


def test_cli_includes_nfo_title_and_plot_context_in_dry_run_report(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    write_valid_srt(tmp_path / "ABC-123.ja.pass1.srt")
    (tmp_path / "ABC-123.nfo").write_text(
        "<movie>"
        "<title>ABC-123-Title</title>"
        "<plot>Plot line one\nPlot line two</plot>"
        "<actor><name>Name1</name></actor>"
        "</movie>",
        encoding="utf-8",
    )
    report_dir = tmp_path / "reports"

    code = cli.main([str(tmp_path), "--dry-run", "--report-dir", str(report_dir)])

    assert code == 0
    row = read_single_jsonl_row(report_dir)
    assert row["movie_title"] == "ABC-123-Title"
    assert row["movie_plot"] == "Plot line one Plot line two"


def test_cli_manual_actress_context_overrides_nfo_in_dry_run_report(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    write_valid_srt(tmp_path / "ABC-123.ja.pass1.srt")
    (tmp_path / "ABC-123.nfo").write_text(
        "<movie><title>Nfo Title</title><actor><name>Nfo Name</name></actor></movie>",
        encoding="utf-8",
    )
    report_dir = tmp_path / "reports"

    code = cli.main(
        [
            str(tmp_path),
            "--dry-run",
            "--report-dir",
            str(report_dir),
            "--actress",
            "Manual One, Manual Two, ",
        ]
    )

    assert code == 0
    row = read_single_jsonl_row(report_dir)
    assert row["actresses"] == ["Manual One", "Manual Two"]
    assert row["movie_title"] == "Nfo Title"
    assert row["nfo_path"] == str(tmp_path / "ABC-123.nfo")
    assert row["warnings"] == []


def test_cli_no_nfo_disables_nfo_lookup_and_parsing(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    write_valid_srt(tmp_path / "ABC-123.ja.pass1.srt")
    (tmp_path / "ABC-123.nfo").write_text(
        "<movie><actor><name>Name1</name></actor>",
        encoding="utf-8",
    )
    report_dir = tmp_path / "reports"

    code = cli.main([str(tmp_path), "--dry-run", "--no-nfo", "--report-dir", str(report_dir)])

    assert code == 0
    row = read_single_jsonl_row(report_dir)
    assert row["actresses"] == []
    assert row["movie_title"] is None
    assert row["movie_plot"] is None
    assert row["nfo_path"] is None
    assert row["warnings"] == []


def test_cli_records_nfo_parse_errors_as_warnings(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    write_valid_srt(tmp_path / "ABC-123.ja.pass1.srt")
    (tmp_path / "ABC-123.nfo").write_text("<movie>", encoding="utf-8")
    report_dir = tmp_path / "reports"

    code = cli.main([str(tmp_path), "--dry-run", "--report-dir", str(report_dir)])

    assert code == 0
    row = read_single_jsonl_row(report_dir)
    assert row["actresses"] == []
    assert row["nfo_path"] == str(tmp_path / "ABC-123.nfo")
    assert row["warnings"]
    assert "no element found" in row["warnings"][0]


def test_cli_records_nfo_lookup_errors_as_warnings(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    write_valid_srt(tmp_path / "ABC-123.ja.pass1.srt")
    (tmp_path / "one.nfo").write_text("<movie />", encoding="utf-8")
    (tmp_path / "two.nfo").write_text("<movie />", encoding="utf-8")
    report_dir = tmp_path / "reports"

    code = cli.main([str(tmp_path), "--dry-run", "--report-dir", str(report_dir)])

    assert code == 0
    row = read_single_jsonl_row(report_dir)
    assert row["actresses"] == []
    assert row["nfo_path"] is None
    assert row["warnings"] == ["nfo_ambiguous"]


def test_cli_returns_nonzero_when_summary_has_failures(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    report_dir = tmp_path / "reports"

    code = cli.main([str(tmp_path), "--dry-run", "--force-translate", "--report-dir", str(report_dir)])

    assert code == 1
    summary = read_single_summary(report_dir)
    assert summary["counts_by_status"] == {"failed_precondition": 1}
    assert summary["failed"] == [str(video.resolve())]
