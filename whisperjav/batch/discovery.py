from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .commands import expected_translation_path
from .models import BatchOptions, ClassifiedVideo

VIDEO_EXTENSION_PRIORITY = (".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg")
AUDIO_EXTENSION_PRIORITY = (".m4a", ".m4b", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".opus")
VIDEO_EXTENSIONS = set(VIDEO_EXTENSION_PRIORITY)
AUDIO_EXTENSIONS = set(AUDIO_EXTENSION_PRIORITY)
SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".ass", ".ssa", ".sub"}
WHISPERJAV_STEM_MARKERS = (".ja.pass1", ".ja.whisperjav", ".ja.merged.whisperjav")
WHISPERJAV_SOURCE_SUFFIXES = tuple(f"{marker}.srt" for marker in WHISPERJAV_STEM_MARKERS)
TRANSLATION_LANGUAGE_SUFFIXES = (".chinese.srt", ".zh.srt", ".cn.srt")
TRANSLATION_END_TOLERANCE_SECONDS = 5.0
SRT_TIMING_PATTERN = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)


def discover_media_files(root: Path, *, include_audio: bool = False) -> list[Path]:
    extensions = set(VIDEO_EXTENSIONS)
    if include_audio:
        extensions.update(AUDIO_EXTENSIONS)

    seen: set[Path] = set()
    selected: dict[tuple[Path, str], Path] = {}
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file() or candidate.suffix.lower() not in extensions:
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        key = (candidate.parent.resolve(), candidate.stem)
        current = selected.get(key)
        if current is None or _media_priority(candidate) < _media_priority(current):
            selected[key] = candidate
    return [path.resolve() for path in sorted(selected.values())]


def is_valid_srt(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return False
    blocks = [block for block in text.split("\n\n") if " --> " in block]
    return len(blocks) >= 2


def is_complete_translation_srt(translation: Path, source: Path | None) -> bool:
    if not is_valid_srt(translation):
        return False
    if source is None:
        return True
    source_end = _last_srt_end_seconds(source)
    translation_end = _last_srt_end_seconds(translation)
    if source_end is None or translation_end is None:
        return False
    return translation_end + TRANSLATION_END_TOLERANCE_SECONDS >= source_end


def _media_priority(path: Path) -> tuple[int, int, str]:
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return (0, VIDEO_EXTENSION_PRIORITY.index(suffix), path.name)
    return (1, AUDIO_EXTENSION_PRIORITY.index(suffix), path.name)


def classify_video(video_path: Path, options: BatchOptions) -> ClassifiedVideo:
    external = _find_external_subtitle(video_path)
    if external is not None:
        return ClassifiedVideo(
            video_path=video_path,
            status="skip_external_subtitle",
            external_subtitle=external,
        )

    japanese = None if options.force else _find_japanese_srt(video_path)
    chinese = None if (options.force or options.force_translate) else _find_chinese_srt(video_path, japanese)
    invalid_translation = chinese is not None and not is_complete_translation_srt(chinese, japanese)

    if invalid_translation:
        chinese = None

    if chinese is not None:
        return ClassifiedVideo(
            video_path=video_path,
            status="skip_translated",
            japanese_srt=japanese,
            chinese_srt=chinese,
        )

    duration_guard = _classify_duration_limit(video_path, options)
    if duration_guard is not None:
        return duration_guard

    if options.force:
        return ClassifiedVideo(video_path=video_path, status="transcribe_then_translate")

    if options.force_translate and japanese is None:
        return ClassifiedVideo(
            video_path=video_path,
            status="failed_precondition",
            reason="missing_japanese_srt",
        )

    if options.force_translate and japanese is not None:
        return ClassifiedVideo(
            video_path=video_path,
            status="translate_existing_japanese",
            japanese_srt=japanese,
        )

    if japanese is not None:
        return ClassifiedVideo(
            video_path=video_path,
            status="translate_existing_japanese",
            reason="invalid_existing_translation" if invalid_translation else None,
            japanese_srt=japanese,
        )

    if options.force_translate:
        return ClassifiedVideo(
            video_path=video_path,
            status="failed_precondition",
            reason="missing_japanese_srt",
        )

    return ClassifiedVideo(
        video_path=video_path,
        status="transcribe_then_translate",
        reason="invalid_existing_translation" if invalid_translation else None,
    )


def _classify_duration_limit(video_path: Path, options: BatchOptions) -> ClassifiedVideo | None:
    if options.max_video_minutes <= 0:
        return None
    try:
        duration_seconds = _probe_duration_seconds(video_path)
    except RuntimeError as exc:
        return ClassifiedVideo(
            video_path=video_path,
            status="failed_precondition",
            reason="duration_unavailable",
            warnings=(str(exc),),
        )

    limit_seconds = options.max_video_minutes * 60
    if duration_seconds <= limit_seconds:
        return None
    return ClassifiedVideo(
        video_path=video_path,
        status="skip_duration_limit",
        reason="duration_exceeds_limit",
        duration_seconds=duration_seconds,
        duration_limit_minutes=options.max_video_minutes,
    )


def _probe_duration_seconds(video_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise RuntimeError(f"ffprobe failed: {exc}") from exc
    if result.returncode != 0:
        message = result.stderr.strip() or f"ffprobe exited with code {result.returncode}"
        raise RuntimeError(message)
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("ffprobe returned invalid duration") from exc
    if duration < 0:
        raise RuntimeError("ffprobe returned negative duration")
    return duration


def _last_srt_end_seconds(path: Path) -> float | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(SRT_TIMING_PATTERN.finditer(text))
    if not matches:
        return None
    return _srt_timestamp_seconds(matches[-1].group("end"))


def _srt_timestamp_seconds(value: str) -> float:
    hours, minutes, seconds_ms = value.split(":")
    seconds, milliseconds = seconds_ms.split(",")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds) / 1000
    )


def _find_external_subtitle(video_path: Path) -> Path | None:
    for subtitle in sorted(video_path.parent.iterdir()):
        if not subtitle.is_file() or subtitle.suffix.lower() not in SUBTITLE_EXTENSIONS:
            continue
        if not _matches_video_basename(video_path.stem, subtitle):
            continue
        if _is_whisperjav_subtitle(video_path.stem, subtitle):
            continue
        return subtitle
    return None


def _find_japanese_srt(video_path: Path) -> Path | None:
    for subtitle in sorted(video_path.parent.iterdir()):
        if not subtitle.is_file() or subtitle.suffix.lower() != ".srt":
            continue
        if _is_whisperjav_source(video_path.stem, subtitle) and is_valid_srt(subtitle):
            return subtitle
    return None


def _find_chinese_srt(video_path: Path, japanese_srt: Path | None) -> Path | None:
    if japanese_srt is not None:
        expected = expected_translation_path(japanese_srt)
        if expected.exists():
            return expected
    for subtitle in sorted(video_path.parent.iterdir()):
        if not subtitle.is_file() or subtitle.suffix.lower() != ".srt":
            continue
        if _is_whisperjav_translation(video_path.stem, subtitle):
            return subtitle
    return None


def _matches_video_basename(video_stem: str, subtitle: Path) -> bool:
    return subtitle.stem == video_stem or subtitle.name.startswith(f"{video_stem}.")


def _is_whisperjav_subtitle(video_stem: str, subtitle: Path) -> bool:
    return _is_whisperjav_source(video_stem, subtitle) or _is_whisperjav_translation(
        video_stem,
        subtitle,
    )


def _is_whisperjav_source(video_stem: str, subtitle: Path) -> bool:
    if not subtitle.name.startswith(f"{video_stem}."):
        return False
    rest = subtitle.name[len(video_stem) :].lower()
    return rest in WHISPERJAV_SOURCE_SUFFIXES


def _is_whisperjav_translation(video_stem: str, subtitle: Path) -> bool:
    if not subtitle.name.startswith(f"{video_stem}."):
        return False
    rest = subtitle.name[len(video_stem) :].lower()
    return any(
        rest.startswith(marker) and rest.endswith(suffix)
        for marker in WHISPERJAV_STEM_MARKERS
        for suffix in TRANSLATION_LANGUAGE_SUFFIXES
    )
