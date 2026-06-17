from __future__ import annotations

import json
from pathlib import Path

import pytest

from whisperjav.batch.scene_summary import (
    SceneSummaryExportError,
    export_scene_summary,
)


def write_srt(path: Path) -> None:
    path.write_text(
        "1\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "一\n\n"
        "2\n"
        "00:00:03,500 --> 00:00:04,000\n"
        "二\n\n"
        "3\n"
        "00:00:05,000 --> 00:00:06,250\n"
        "三\n",
        encoding="utf-8",
    )


def write_subtrans(
    path: Path,
    *,
    second_batch_originals: list[dict[str, int]] | None = None,
) -> None:
    originals = second_batch_originals or [{"index": 3}]
    path.write_text(
        json.dumps(
            {
                "scenes": [
                    {
                        "scene": 1,
                        "linecount": 3,
                        "context": {"summary": "Scene summary"},
                        "batches": [
                            {
                                "batch": 1,
                                "size": 2,
                                "summary": "Batch one summary",
                                "originals": [{"index": 1}, {"index": 2}],
                            },
                            {
                                "batch": 2,
                                "size": len(originals),
                                "summary": "Batch two summary",
                                "originals": originals,
                            },
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_export_scene_summary_writes_scene_and_batch_time_ranges(tmp_path: Path) -> None:
    source_srt = tmp_path / "ABC-123.ja.pass1.srt"
    translated_srt = tmp_path / "ABC-123.ja.pass1.chinese.srt"
    subtrans = tmp_path / "ABC-123.ja.pass1.subtrans"
    write_srt(source_srt)
    translated_srt.write_text("translated", encoding="utf-8")
    write_subtrans(subtrans)

    result = export_scene_summary(
        source_srt=source_srt,
        translated_srt=translated_srt,
        subtrans_path=subtrans,
    )

    payload = json.loads(result.path.read_text(encoding="utf-8"))
    assert result.warnings == ()
    assert result.path == tmp_path / "ABC-123.ja.pass1.chinese.summary.json"
    assert payload["source_srt"] == str(source_srt)
    assert payload["translated_srt"] == str(translated_srt)
    assert payload["subtrans"] == str(subtrans)
    assert payload["warnings"] == []
    scene = payload["scenes"][0]
    assert scene["scene"] == 1
    assert scene["line_start"] == 1
    assert scene["line_end"] == 3
    assert scene["start"] == "00:00:01,000"
    assert scene["end"] == "00:00:06,250"
    assert scene["summary"] == "Scene summary"
    assert scene["batches"][0] == {
        "batch": 1,
        "line_start": 1,
        "line_end": 2,
        "start": "00:00:01,000",
        "end": "00:00:04,000",
        "summary": "Batch one summary",
    }


def test_export_scene_summary_records_warning_when_srt_line_is_missing(tmp_path: Path) -> None:
    source_srt = tmp_path / "ABC-123.ja.pass1.srt"
    translated_srt = tmp_path / "ABC-123.ja.pass1.chinese.srt"
    subtrans = tmp_path / "ABC-123.ja.pass1.subtrans"
    write_srt(source_srt)
    translated_srt.write_text("translated", encoding="utf-8")
    write_subtrans(subtrans, second_batch_originals=[{"index": 99}])

    result = export_scene_summary(
        source_srt=source_srt,
        translated_srt=translated_srt,
        subtrans_path=subtrans,
    )

    payload = json.loads(result.path.read_text(encoding="utf-8"))
    assert result.warnings == ("missing_srt_time: line 99",)
    scene = payload["scenes"][0]
    assert scene["start"] is None
    assert scene["end"] is None
    assert scene["batches"][1]["line_start"] == 99
    assert scene["batches"][1]["line_end"] == 99
    assert scene["batches"][1]["start"] is None
    assert scene["batches"][1]["end"] is None
    assert payload["warnings"] == ["missing_srt_time: line 99"]


def test_export_scene_summary_compresses_missing_line_warnings(tmp_path: Path) -> None:
    source_srt = tmp_path / "ABC-123.ja.pass1.srt"
    translated_srt = tmp_path / "ABC-123.ja.pass1.chinese.srt"
    subtrans = tmp_path / "ABC-123.ja.pass1.subtrans"
    write_srt(source_srt)
    translated_srt.write_text("translated", encoding="utf-8")
    write_subtrans(
        subtrans,
        second_batch_originals=[
            {"index": 5},
            {"index": 6},
            {"index": 8},
        ],
    )

    result = export_scene_summary(
        source_srt=source_srt,
        translated_srt=translated_srt,
        subtrans_path=subtrans,
    )

    payload = json.loads(result.path.read_text(encoding="utf-8"))
    assert result.warnings == (
        "missing_srt_time: lines 5-6",
        "missing_srt_time: line 8",
    )
    assert payload["warnings"] == [
        "missing_srt_time: lines 5-6",
        "missing_srt_time: line 8",
    ]


def test_export_scene_summary_raises_clear_error_for_missing_subtrans(tmp_path: Path) -> None:
    source_srt = tmp_path / "ABC-123.ja.pass1.srt"
    translated_srt = tmp_path / "ABC-123.ja.pass1.chinese.srt"
    write_srt(source_srt)
    translated_srt.write_text("translated", encoding="utf-8")

    with pytest.raises(SceneSummaryExportError, match="subtrans file not found"):
        export_scene_summary(source_srt=source_srt, translated_srt=translated_srt)
