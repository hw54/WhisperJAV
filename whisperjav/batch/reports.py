from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .commands import SECRET_FLAGS
from .models import ProcessResult, VideoResult


@dataclass(frozen=True)
class ReportPaths:
    report_file: Path
    summary_file: Path


class BatchReportWriter:
    def __init__(self, report_dir: Path, input_root: Path) -> None:
        self.report_dir = report_dir
        self.input_root = input_root

    def write(
        self,
        results: Iterable[VideoResult],
        *,
        started_at: str | None = None,
        ended_at: str | None = None,
    ) -> ReportPaths:
        result_list = list(results)
        start_time = started_at or _utc_timestamp()
        paths = ReportPaths(
            report_file=self.report_dir / "batch_results.jsonl",
            summary_file=self.report_dir / "batch_summary.json",
        )

        self.report_dir.mkdir(parents=True, exist_ok=True)
        with paths.report_file.open("w", encoding="utf-8") as report_file:
            for result in result_list:
                line = json.dumps(_serialize_video_result(result), ensure_ascii=False, sort_keys=True)
                report_file.write(f"{line}\n")

        end_time = ended_at or _utc_timestamp()
        summary = summarize_results(
            result_list,
            input_root=self.input_root,
            report_file=paths.report_file,
            started_at=start_time,
            ended_at=end_time,
        )
        paths.summary_file.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return paths


def summarize_results(
    results: Iterable[VideoResult],
    *,
    input_root: Path,
    report_file: Path,
    started_at: str | None = None,
    ended_at: str | None = None,
) -> dict[str, Any]:
    result_list = list(results)
    counts = Counter(result.status for result in result_list)
    return {
        "input_root": str(input_root),
        "started_at": started_at,
        "ended_at": ended_at,
        "report_file": str(report_file),
        "total": len(result_list),
        "counts": dict(sorted(counts.items())),
        "asr_seconds": sum(result.asr.seconds for result in result_list if result.asr is not None),
        "translation_seconds": sum(
            result.translation.seconds for result in result_list if result.translation is not None
        ),
        "failed": [_failed_entry(result) for result in result_list if _is_failed(result)],
    }


def _serialize_video_result(result: VideoResult) -> dict[str, Any]:
    return {
        "video_path": str(result.video_path),
        "status": result.status,
        "reason": result.reason,
        "nfo_path": _path_or_none(result.nfo_path),
        "actresses": list(result.actresses),
        "japanese_srt": _path_or_none(result.japanese_srt),
        "chinese_srt": _path_or_none(result.chinese_srt),
        "asr": _serialize_process_result(result.asr),
        "translation": _serialize_process_result(result.translation),
        "error": result.error,
        "warnings": list(result.warnings),
    }


def _serialize_process_result(result: ProcessResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "seconds": result.seconds,
        "return_code": result.return_code,
        "command_redacted": _command_without_secret_flags(result.command_redacted),
        "api_key_source": result.api_key_source,
        "stdout_tail": result.stdout_tail,
        "stderr_tail": result.stderr_tail,
    }


def _command_without_secret_flags(command: Iterable[str]) -> list[str]:
    redacted: list[str] = []
    skip_next = False
    for part in command:
        if skip_next:
            skip_next = False
            continue
        if part in SECRET_FLAGS:
            skip_next = True
            continue
        redacted.append(part)
    return redacted


def _path_or_none(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _is_failed(result: VideoResult) -> bool:
    return result.status.startswith("failed_") or result.status == "cancelled"


def _failed_entry(result: VideoResult) -> dict[str, str | None]:
    return {
        "video_path": str(result.video_path),
        "status": result.status,
        "reason": result.reason,
        "error": result.error,
    }


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
