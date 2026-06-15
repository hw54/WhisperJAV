from __future__ import annotations

from pathlib import Path

from .commands import expected_translation_path
from .models import BatchOptions, ClassifiedVideo

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a", ".m4b", ".opus"}
SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".ass", ".ssa", ".sub"}
WHISPERJAV_STEM_MARKERS = (".ja.pass1", ".ja.whisperjav", ".ja.merged.whisperjav")
WHISPERJAV_SOURCE_SUFFIXES = tuple(f"{marker}.srt" for marker in WHISPERJAV_STEM_MARKERS)
TRANSLATION_LANGUAGE_SUFFIXES = (".chinese.srt", ".zh.srt", ".cn.srt")


def discover_media_files(root: Path, *, include_audio: bool = False) -> list[Path]:
    extensions = set(VIDEO_EXTENSIONS)
    if include_audio:
        extensions.update(AUDIO_EXTENSIONS)

    seen: set[Path] = set()
    found: list[Path] = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file() or candidate.suffix.lower() not in extensions:
            continue
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            found.append(resolved)
    return found


def is_valid_srt(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return False
    blocks = [block for block in text.split("\n\n") if " --> " in block]
    return len(blocks) >= 2


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
    invalid_translation = chinese is not None and not is_valid_srt(chinese)

    if invalid_translation:
        chinese = None

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

    if chinese is not None:
        return ClassifiedVideo(
            video_path=video_path,
            status="skip_translated",
            japanese_srt=japanese,
            chinese_srt=chinese,
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
