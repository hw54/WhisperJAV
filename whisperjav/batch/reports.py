from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .commands import redact_command
from .models import ProcessResult, VideoResult


@dataclass(frozen=True)
class ReportPaths:
    jsonl: Path
    summary: Path


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
        wall_seconds: float | None = None,
    ) -> ReportPaths:
        result_list = list(results)
        start_time = started_at or _utc_timestamp()
        report_id = _report_id(start_time)
        paths = ReportPaths(
            jsonl=self.report_dir / f"run-{report_id}.jsonl",
            summary=self.report_dir / f"run-{report_id}.summary.json",
        )

        self.report_dir.mkdir(parents=True, exist_ok=True)
        with paths.jsonl.open("w", encoding="utf-8") as report_file:
            for result in result_list:
                line = json.dumps(_serialize_video_result(result), ensure_ascii=False, sort_keys=True)
                report_file.write(f"{line}\n")

        end_time = ended_at or _utc_timestamp()
        summary = summarize_results(result_list, wall_seconds=wall_seconds)
        summary["input_root"] = str(self.input_root)
        summary["started_at"] = start_time
        summary["ended_at"] = end_time
        summary["report_file"] = str(paths.jsonl)
        paths.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return paths


def summarize_results(
    results: Iterable[VideoResult],
    *,
    wall_seconds: float | None = None,
) -> dict[str, Any]:
    result_list = list(results)
    counts = Counter(result.status for result in result_list)
    total_asr_seconds = sum(
        result.asr.seconds for result in result_list if result.asr is not None
    )
    total_translation_seconds = sum(
        result.translation.seconds for result in result_list if result.translation is not None
    )
    return {
        "total_videos": len(result_list),
        "counts_by_status": dict(sorted(counts.items())),
        "total_asr_seconds": total_asr_seconds,
        "total_translation_seconds": total_translation_seconds,
        "performance": _performance_summary(
            result_list,
            total_asr_seconds=total_asr_seconds,
            total_translation_seconds=total_translation_seconds,
            wall_seconds=wall_seconds,
        ),
        "failed": [str(result.video_path) for result in result_list if _is_failed(result)],
    }


def _performance_summary(
    results: list[VideoResult],
    *,
    total_asr_seconds: float,
    total_translation_seconds: float,
    wall_seconds: float | None,
) -> dict[str, Any]:
    asr_count = sum(1 for result in results if result.asr is not None)
    translation_count = sum(1 for result in results if result.translation is not None)
    processed = [result for result in results if result.asr is not None or result.translation is not None]
    duration_included = [
        result
        for result in processed
        if result.duration_seconds is not None and result.duration_seconds > 0
    ]
    total_video_duration = sum(result.duration_seconds or 0.0 for result in duration_included)
    total_processing_seconds = sum(
        (result.asr.seconds if result.asr is not None else 0.0)
        + (result.translation.seconds if result.translation is not None else 0.0)
        for result in processed
    )
    processing_to_duration_ratio = _safe_divide(wall_seconds, total_video_duration)

    return {
        "asr_count": asr_count,
        "translation_count": translation_count,
        "processed_video_count": len(processed),
        "duration_included_count": len(duration_included),
        "unknown_duration_count": sum(
            1 for result in processed if result.duration_seconds is None
        ),
        "duration_excluded_count": len(results) - len(duration_included),
        "total_video_duration_seconds": total_video_duration,
        "total_wall_seconds": wall_seconds,
        "average_asr_seconds": _safe_divide(total_asr_seconds, asr_count),
        "average_translation_seconds": _safe_divide(total_translation_seconds, translation_count),
        "average_total_processing_seconds": _safe_divide(total_processing_seconds, len(processed)),
        "processing_to_duration_ratio": processing_to_duration_ratio,
        "wall_to_duration_ratio": processing_to_duration_ratio,
        "throughput_ratio": _safe_divide(total_video_duration, wall_seconds),
    }


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _serialize_video_result(result: VideoResult) -> dict[str, Any]:
    return {
        "video_path": str(result.video_path),
        "status": result.status,
        "reason": result.reason,
        "nfo_path": _path_or_none(result.nfo_path),
        "actresses": list(result.actresses),
        "movie_title": result.movie_title,
        "movie_plot": result.movie_plot,
        "japanese_srt": _path_or_none(result.japanese_srt),
        "chinese_srt": _path_or_none(result.chinese_srt),
        "summary_json": _path_or_none(result.summary_json),
        "asr": _serialize_process_result(result.asr),
        "translation": _serialize_process_result(result.translation),
        "error": result.error,
        "warnings": list(result.warnings),
        "duration_seconds": result.duration_seconds,
        "duration_limit_minutes": result.duration_limit_minutes,
    }


def _serialize_process_result(result: ProcessResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "seconds": result.seconds,
        "return_code": result.return_code,
        "command_redacted": redact_command(list(result.command_redacted)),
        "api_key_source": result.api_key_source,
        "stdout_tail": result.stdout_tail,
        "stderr_tail": result.stderr_tail,
    }


def _path_or_none(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _is_failed(result: VideoResult) -> bool:
    return result.status.startswith("failed_") or result.status == "cancelled"


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _report_id(timestamp: str) -> str:
    normalized = timestamp.removesuffix("Z") + "+00:00" if timestamp.endswith("Z") else timestamp
    return datetime.fromisoformat(normalized).strftime("%Y%m%d-%H%M%S")
