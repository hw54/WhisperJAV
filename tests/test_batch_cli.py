import json

import pytest

from whisperjav.batch import cli


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


@pytest.mark.parametrize("workers", ["0", "5"])
def test_parse_args_rejects_invalid_worker_count(tmp_path, workers):
    with pytest.raises(SystemExit):
        cli.parse_args([str(tmp_path), "--translate-workers", workers])


def test_parse_args_rejects_invalid_translation_queue_size(tmp_path):
    with pytest.raises(SystemExit):
        cli.parse_args([str(tmp_path), "--translation-queue-size", "0"])


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


def test_dry_run_uses_default_report_dir(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")

    code = cli.main([str(tmp_path), "--dry-run"])

    report_dir = tmp_path / ".whisperjav_batch"
    assert code == 0
    assert len(list(report_dir.glob("*.jsonl"))) == 1
    assert len(list(report_dir.glob("*.summary.json"))) == 1


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


def test_cli_manual_actress_context_overrides_nfo_in_dry_run_report(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video", encoding="utf-8")
    write_valid_srt(tmp_path / "ABC-123.ja.pass1.srt")
    (tmp_path / "ABC-123.nfo").write_text(
        "<movie><actor><name>Nfo Name</name></actor></movie>",
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
    assert row["nfo_path"] is None
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
