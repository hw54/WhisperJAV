from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SRT_TIMING_RE = re.compile(
    r"(?m)^\s*(?P<index>\d+)\s*\n"
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*"
    r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)


class SceneSummaryExportError(RuntimeError):
    """Raised when scene summary sidecar generation cannot start."""


@dataclass(frozen=True)
class SceneSummaryExportResult:
    path: Path
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SrtTimeRange:
    start: str
    end: str


def export_scene_summary(
    *,
    source_srt: Path,
    translated_srt: Path,
    subtrans_path: Path | None = None,
    output_path: Path | None = None,
) -> SceneSummaryExportResult:
    subtrans = subtrans_path or _default_subtrans_path(source_srt)
    output = output_path or _default_output_path(translated_srt)
    if subtrans is None or not subtrans.exists():
        raise SceneSummaryExportError(f"subtrans file not found for {source_srt}")
    if not source_srt.exists():
        raise SceneSummaryExportError(f"source SRT not found: {source_srt}")

    data = _load_subtrans(subtrans)
    srt_times = _parse_srt_times(source_srt)
    missing_srt_lines: set[int] = set()
    scenes = [
        _build_scene_payload(scene, srt_times=srt_times, missing_srt_lines=missing_srt_lines)
        for scene in data.get("scenes", [])
    ]
    payload = {
        "source_srt": str(source_srt),
        "translated_srt": str(translated_srt),
        "subtrans": str(subtrans),
        "generated_at": _utc_timestamp(),
        "warnings": _format_missing_srt_warnings(missing_srt_lines),
        "scenes": scenes,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SceneSummaryExportResult(path=output, warnings=tuple(payload["warnings"]))


def _default_subtrans_path(source_srt: Path) -> Path | None:
    canonical = source_srt.with_suffix(".subtrans")
    legacy = Path(f"{source_srt}.subtrans")
    for candidate in (canonical, legacy):
        if candidate.exists():
            return candidate
    return canonical


def _default_output_path(translated_srt: Path) -> Path:
    return translated_srt.with_name(f"{translated_srt.stem}.summary.json")


def _load_subtrans(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SceneSummaryExportError(f"failed to read subtrans file: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SceneSummaryExportError(f"invalid subtrans payload: {path}")
    return data


def _parse_srt_times(path: Path) -> dict[int, SrtTimeRange]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise SceneSummaryExportError(f"failed to read source SRT: {path}: {exc}") from exc
    return {
        int(match.group("index")): SrtTimeRange(
            start=match.group("start"),
            end=match.group("end"),
        )
        for match in SRT_TIMING_RE.finditer(text)
    }


def _build_scene_payload(
    scene: dict[str, Any],
    *,
    srt_times: dict[int, SrtTimeRange],
    missing_srt_lines: set[int],
) -> dict[str, Any]:
    batches = [
        _build_batch_payload(batch, srt_times=srt_times, missing_srt_lines=missing_srt_lines)
        for batch in scene.get("batches", [])
    ]
    line_numbers = [
        line
        for batch in batches
        for line in (batch["line_start"], batch["line_end"])
        if line is not None
    ]
    start, end = _resolve_time_range(
        line_numbers,
        srt_times=srt_times,
        missing_srt_lines=missing_srt_lines,
    )
    return {
        "scene": scene.get("scene"),
        "line_start": min(line_numbers) if line_numbers else None,
        "line_end": max(line_numbers) if line_numbers else None,
        "start": start,
        "end": end,
        "summary": _scene_summary(scene),
        "batches": batches,
    }


def _build_batch_payload(
    batch: dict[str, Any],
    *,
    srt_times: dict[int, SrtTimeRange],
    missing_srt_lines: set[int],
) -> dict[str, Any]:
    line_numbers = _original_line_numbers(batch)
    start, end = _resolve_time_range(
        line_numbers,
        srt_times=srt_times,
        missing_srt_lines=missing_srt_lines,
    )
    return {
        "batch": batch.get("batch"),
        "line_start": min(line_numbers) if line_numbers else None,
        "line_end": max(line_numbers) if line_numbers else None,
        "start": start,
        "end": end,
        "summary": batch.get("summary"),
    }


def _original_line_numbers(batch: dict[str, Any]) -> list[int]:
    numbers: list[int] = []
    for original in batch.get("originals", []):
        index = original.get("index") if isinstance(original, dict) else None
        if isinstance(index, int):
            numbers.append(index)
    return numbers


def _resolve_time_range(
    line_numbers: list[int],
    *,
    srt_times: dict[int, SrtTimeRange],
    missing_srt_lines: set[int],
) -> tuple[str | None, str | None]:
    if not line_numbers:
        return None, None
    missing = [line for line in line_numbers if line not in srt_times]
    missing_srt_lines.update(missing)
    if missing:
        return None, None
    start_line = min(line_numbers)
    end_line = max(line_numbers)
    return srt_times[start_line].start, srt_times[end_line].end


def _scene_summary(scene: dict[str, Any]) -> str | None:
    context = scene.get("context")
    if isinstance(context, dict) and context.get("summary"):
        return context["summary"]
    summary = scene.get("summary")
    return summary if isinstance(summary, str) and summary else None


def _format_missing_srt_warnings(lines: set[int]) -> list[str]:
    return [
        f"missing_srt_time: line {start}"
        if start == end
        else f"missing_srt_time: lines {start}-{end}"
        for start, end in _line_ranges(lines)
    ]


def _line_ranges(values: set[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    sorted_values = sorted(values)
    if not sorted_values:
        return ranges
    start = previous = sorted_values[0]
    for value in sorted_values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append((start, previous))
        start = previous = value
    ranges.append((start, previous))
    return ranges


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
