# WhisperJAV Batch CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `whisperjav-batch`, a directory-oriented CLI that transcribes untranslated JAV videos to Japanese subtitles and translates them to Chinese Adult/Explicit subtitles with resumable reports.

**Architecture:** Add a focused `whisperjav.batch` package with pure classification, NFO parsing, command construction, reporting, and scheduler modules. The batch CLI remains a coordinator around existing `whisperjav.main` and `whisperjav.translate.cli` subprocesses, keeping ROCm/ONNX/LLM failures visible and isolated.

**Tech Stack:** Python 3.10+, argparse, dataclasses, pathlib, queue/threading, subprocess, pytest, existing WhisperJAV CLI modules.

---

## Reference Spec

- `docs/superpowers/specs/2026-06-15-whisperjav-batch-cli-design.md`
- Branch: `codex/rocm-onnx-provider`
- Existing entry points live in `pyproject.toml:[project.scripts]`.
- Existing media extensions live in `whisperjav.modules.media_discovery.MediaDiscovery`.
- Existing translation output naming lives in `whisperjav.translate.cli.generate_output_path`.

## File Structure

- Create `whisperjav/batch/__init__.py`: package marker and public version surface.
- Create `whisperjav/batch/models.py`: immutable dataclasses, status literals, config objects, result objects.
- Create `whisperjav/batch/commands.py`: ASR/translation command builders, expected translation path helper, redaction helper.
- Create `whisperjav/batch/discovery.py`: recursive media discovery, same-basename subtitle classification, reusable SRT validation.
- Create `whisperjav/batch/nfo.py`: deterministic XML NFO lookup and actress extraction.
- Create `whisperjav/batch/reports.py`: JSONL and summary report writer with redacted command records.
- Create `whisperjav/batch/runners.py`: subprocess runner and fake-runner-friendly protocol.
- Create `whisperjav/batch/scheduler.py`: one-ASR-at-a-time scheduler with bounded translation queue and cancellation handling.
- Create `whisperjav/batch/cli.py`: argparse entry point and terminal summary.
- Modify `pyproject.toml`: add `whisperjav-batch = "whisperjav.batch.cli:main"`.
- Create `tests/test_batch_commands.py`.
- Create `tests/test_batch_discovery.py`.
- Create `tests/test_batch_nfo.py`.
- Create `tests/test_batch_reports.py`.
- Create `tests/test_batch_scheduler.py`.
- Create `tests/test_batch_cli.py`.

## Implementation Rules

- TDD is mandatory: write a focused failing test, run it and confirm the expected failure, then implement.
- Use `timeout 60s pytest ...` for backend unit tests.
- Do not call real ASR, ROCm, DeepSeek, or PySubtrans in tests.
- Do not store raw API keys in reports, debug output, snapshots, fixtures, or commits.
- Subprocess commands must use `sys.executable` and `-u`.
- Keep `install_log_rocm_attempt.txt` untouched.

### Task 1: Command Builders and Entry Point Metadata

**Files:**
- Create: `whisperjav/batch/__init__.py`
- Create: `whisperjav/batch/models.py`
- Create: `whisperjav/batch/commands.py`
- Modify: `pyproject.toml`
- Test: `tests/test_batch_commands.py`

- [ ] **Step 1: Write failing command-builder tests**

Create `tests/test_batch_commands.py`:

```python
import json
import sys
from pathlib import Path

from whisperjav.batch.commands import (
    build_asr_command,
    build_translation_command,
    expected_translation_path,
    redact_command,
)
from whisperjav.batch.models import BatchOptions


def test_asr_command_uses_recommended_anime_whisper_settings(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("fake")
    options = BatchOptions(root=tmp_path)

    command = build_asr_command(video, options)

    assert command[:3] == [sys.executable, "-u", "-m"]
    assert command[3] == "whisperjav.main"
    assert str(video) in command
    assert "--ensemble" in command
    assert "--ensemble-serial" in command
    assert "--pass1-pipeline" in command
    assert command[command.index("--pass1-pipeline") + 1] == "qwen"
    assert "--pass1-scene-detector" in command
    assert command[command.index("--pass1-scene-detector") + 1] == "semantic"
    assert "--pass1-speech-segmenter" in command
    assert command[command.index("--pass1-speech-segmenter") + 1] == "whisperseg"
    assert "--pass1-model" in command
    assert command[command.index("--pass1-model") + 1] == "litagin/anime-whisper"
    assert "--translate" not in command
    qwen_params = command[command.index("--pass1-qwen-params") + 1]
    assert json.loads(qwen_params) == {
        "framer": "vad-grouped",
        "generator_backend": "anime-whisper",
        "timestamp_mode": "vad_only",
        "assembly_cleaner": "passthrough",
        "stepdown": False,
    }


def test_translation_command_uses_standalone_translate_cli_and_context(tmp_path):
    srt = tmp_path / "ABC-123.ja.pass1.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nはい\n")
    options = BatchOptions(
        root=tmp_path,
        translate_api_key="secret-key",
        actress="Name1, Name2",
    )

    command = build_translation_command(srt, options, actresses=["Ignored"])

    assert command[:4] == [sys.executable, "-u", "-m", "whisperjav.translate.cli"]
    assert "-i" in command
    assert command[command.index("-i") + 1] == str(srt)
    assert "--provider" in command
    assert command[command.index("--provider") + 1] == "deepseek"
    assert "--source" in command
    assert command[command.index("--source") + 1] == "japanese"
    assert "--target" in command
    assert command[command.index("--target") + 1] == "chinese"
    assert "--tone" in command
    assert command[command.index("--tone") + 1] == "pornify"
    assert "--model" in command
    assert command[command.index("--model") + 1] == "deepseek-v4-flash"
    assert "--api-key" in command
    assert command[command.index("--api-key") + 1] == "secret-key"
    assert "--actress" in command
    assert command[command.index("--actress") + 1] == "Name1, Name2"
    assert "--translate-provider" not in command


def test_translation_command_uses_nfo_actresses_when_no_manual_override(tmp_path):
    srt = tmp_path / "ABC-123.ja.pass1.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nはい\n")
    options = BatchOptions(root=tmp_path)

    command = build_translation_command(srt, options, actresses=["Name1", "Name2"])

    assert command[command.index("--actress") + 1] == "Name1, Name2"
    assert "--api-key" not in command


def test_expected_translation_path_matches_translate_cli_rule(tmp_path):
    assert expected_translation_path(tmp_path / "ABC-123.ja.pass1.srt") == tmp_path / "ABC-123.ja.pass1.chinese.srt"
    assert expected_translation_path(tmp_path / "ABC-123.ja.srt") == tmp_path / "ABC-123.chinese.srt"


def test_redact_command_masks_api_key_values():
    command = ["cmd", "--api-key", "secret", "--translate-api-key", "other", "--model", "x"]

    assert redact_command(command) == [
        "cmd",
        "--api-key",
        "<redacted>",
        "--translate-api-key",
        "<redacted>",
        "--model",
        "x",
    ]


def test_pyproject_exposes_batch_entry_point():
    text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'whisperjav-batch = "whisperjav.batch.cli:main"' in text
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
timeout 60s pytest tests/test_batch_commands.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'whisperjav.batch'` or pyproject entry missing.

- [ ] **Step 3: Implement models and command builders**

Create `whisperjav/batch/__init__.py`:

```python
"""Batch directory workflow for WhisperJAV."""
```

Create `whisperjav/batch/models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

BatchStatus = Literal[
    "skip_external_subtitle",
    "skip_translated",
    "translate_existing_japanese",
    "transcribe_then_translate",
    "failed_asr",
    "failed_translation",
    "failed_precondition",
    "cancelled",
    "completed",
    "completed_translation_only",
]


@dataclass(frozen=True)
class BatchOptions:
    root: Path
    report_dir: Path | None = None
    include_audio: bool = False
    dry_run: bool = False
    force: bool = False
    force_translate: bool = False
    translate_api_key: str | None = None
    translate_workers: int = 1
    translation_queue_size: int = 2
    stream: bool = False
    debug: bool = False
    accept_cpu_mode: bool = False
    no_nfo: bool = False
    actress: str | None = None


@dataclass(frozen=True)
class ClassifiedVideo:
    video_path: Path
    status: BatchStatus
    reason: str | None = None
    external_subtitle: Path | None = None
    japanese_srt: Path | None = None
    chinese_srt: Path | None = None
    nfo_path: Path | None = None
    actresses: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProcessResult:
    seconds: float = 0.0
    return_code: int | None = None
    command_redacted: list[str] = field(default_factory=list)
    api_key_source: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass(frozen=True)
class VideoResult:
    video_path: Path
    status: BatchStatus
    reason: str | None = None
    nfo_path: Path | None = None
    actresses: tuple[str, ...] = ()
    japanese_srt: Path | None = None
    chinese_srt: Path | None = None
    asr: ProcessResult | None = None
    translation: ProcessResult | None = None
    error: str | None = None
    warnings: tuple[str, ...] = ()
```

Create `whisperjav/batch/commands.py`:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

from .models import BatchOptions

SECRET_FLAGS = {"--api-key", "--translate-api-key"}

QWEN_PARAMS = {
    "framer": "vad-grouped",
    "generator_backend": "anime-whisper",
    "timestamp_mode": "vad_only",
    "assembly_cleaner": "passthrough",
    "stepdown": False,
}


def build_asr_command(video_path: Path, options: BatchOptions) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "whisperjav.main",
        str(video_path),
        "--ensemble",
        "--ensemble-serial",
        "--pass1-pipeline",
        "qwen",
        "--pass1-sensitivity",
        "balanced",
        "--pass1-qwen-params",
        json.dumps(QWEN_PARAMS, separators=(",", ":")),
        "--pass1-scene-detector",
        "semantic",
        "--pass1-speech-segmenter",
        "whisperseg",
        "--pass1-model",
        "litagin/anime-whisper",
        "--merge-strategy",
        "pass1_primary",
        "--output-dir",
        "source",
        "--subs-language",
        "native",
        "--language",
        "japanese",
    ]
    if options.stream:
        command.append("--stream")
    if options.debug:
        command.append("--debug")
    if options.accept_cpu_mode:
        command.append("--accept-cpu-mode")
    return command


def build_translation_command(
    japanese_srt: Path,
    options: BatchOptions,
    *,
    actresses: list[str] | tuple[str, ...] = (),
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "whisperjav.translate.cli",
        "-i",
        str(japanese_srt),
        "--provider",
        "deepseek",
        "--source",
        "japanese",
        "--target",
        "chinese",
        "--tone",
        "pornify",
        "--model",
        "deepseek-v4-flash",
    ]
    if options.translate_api_key:
        command.extend(["--api-key", options.translate_api_key])
    actress_context = options.actress or ", ".join(actresses)
    if actress_context:
        command.extend(["--actress", actress_context])
    if options.stream:
        command.append("--stream")
    if options.debug:
        command.append("--debug")
    return command


def expected_translation_path(japanese_srt: Path) -> Path:
    stem = japanese_srt.stem
    parts = stem.split(".")
    if len(parts) > 1 and parts[-1] in {"japanese", "english", "ja", "en", "jp"}:
        stem = ".".join(parts[:-1])
    return japanese_srt.with_name(f"{stem}.chinese.srt")


def redact_command(command: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for part in command:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        redacted.append(part)
        if part in SECRET_FLAGS:
            redact_next = True
    return redacted
```

Modify `pyproject.toml` under `[project.scripts]`:

```toml
whisperjav-batch = "whisperjav.batch.cli:main"
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
timeout 60s pytest tests/test_batch_commands.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add pyproject.toml whisperjav/batch/__init__.py whisperjav/batch/models.py whisperjav/batch/commands.py tests/test_batch_commands.py
git commit -m "feat: add batch command builders"
```

### Task 2: Media Discovery and Subtitle Classification

**Files:**
- Modify: `whisperjav/batch/models.py`
- Create: `whisperjav/batch/discovery.py`
- Test: `tests/test_batch_discovery.py`

- [ ] **Step 1: Write failing discovery/classification tests**

Create `tests/test_batch_discovery.py`:

```python
from pathlib import Path

from whisperjav.batch.discovery import classify_video, discover_media_files, is_valid_srt
from whisperjav.batch.models import BatchOptions


def write_valid_srt(path: Path, text: str = "你好") -> None:
    path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n"
        f"{text}\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n"
        f"{text}\n",
        encoding="utf-8",
    )


def test_discovery_finds_videos_recursively_and_skips_audio_by_default(tmp_path):
    (tmp_path / "A").mkdir()
    video = tmp_path / "A" / "ABC-123.mp4"
    audio = tmp_path / "A" / "ABC-123.mp3"
    video.write_text("video")
    audio.write_text("audio")

    found = discover_media_files(tmp_path, include_audio=False)

    assert found == [video.resolve()]


def test_discovery_can_include_audio(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    audio = tmp_path / "ABC-123.mp3"
    video.write_text("video")
    audio.write_text("audio")

    found = discover_media_files(tmp_path, include_audio=True)

    assert set(found) == {video.resolve(), audio.resolve()}


def test_external_same_basename_subtitle_blocks_video(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    subtitle = tmp_path / "ABC-123.zh.srt"
    video.write_text("video")
    write_valid_srt(subtitle)

    item = classify_video(video, BatchOptions(root=tmp_path))

    assert item.status == "skip_external_subtitle"
    assert item.external_subtitle == subtitle


def test_whisperjav_japanese_srt_is_reused_for_translation(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    japanese = tmp_path / "ABC-123.ja.pass1.srt"
    video.write_text("video")
    write_valid_srt(japanese, "はい")

    item = classify_video(video, BatchOptions(root=tmp_path))

    assert item.status == "translate_existing_japanese"
    assert item.japanese_srt == japanese
    assert item.external_subtitle is None


def test_valid_chinese_translation_is_skipped(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    japanese = tmp_path / "ABC-123.ja.pass1.srt"
    chinese = tmp_path / "ABC-123.ja.pass1.chinese.srt"
    video.write_text("video")
    write_valid_srt(japanese, "はい")
    write_valid_srt(chinese, "中文")

    item = classify_video(video, BatchOptions(root=tmp_path))

    assert item.status == "skip_translated"
    assert item.chinese_srt == chinese


def test_invalid_existing_translation_is_retranslated(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    japanese = tmp_path / "ABC-123.ja.pass1.srt"
    chinese = tmp_path / "ABC-123.ja.pass1.chinese.srt"
    video.write_text("video")
    write_valid_srt(japanese, "はい")
    chinese.write_text("not enough subtitle blocks", encoding="utf-8")

    item = classify_video(video, BatchOptions(root=tmp_path))

    assert item.status == "translate_existing_japanese"
    assert item.reason == "invalid_existing_translation"
    assert item.chinese_srt is None


def test_no_reusable_srt_transcribes_then_translates(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video")

    item = classify_video(video, BatchOptions(root=tmp_path))

    assert item.status == "transcribe_then_translate"


def test_force_ignores_whisperjav_outputs_but_not_external_subtitles(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    japanese = tmp_path / "ABC-123.ja.pass1.srt"
    chinese = tmp_path / "ABC-123.ja.pass1.chinese.srt"
    video.write_text("video")
    write_valid_srt(japanese, "はい")
    write_valid_srt(chinese, "中文")

    item = classify_video(video, BatchOptions(root=tmp_path, force=True))

    assert item.status == "transcribe_then_translate"
    assert item.japanese_srt is None
    assert item.chinese_srt is None


def test_force_still_respects_external_subtitle_blocker(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    external = tmp_path / "ABC-123.zh.srt"
    japanese = tmp_path / "ABC-123.ja.pass1.srt"
    video.write_text("video")
    write_valid_srt(external, "中文")
    write_valid_srt(japanese, "はい")

    item = classify_video(video, BatchOptions(root=tmp_path, force=True))

    assert item.status == "skip_external_subtitle"
    assert item.external_subtitle == external


def test_force_translate_ignores_existing_chinese_translation(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    japanese = tmp_path / "ABC-123.ja.pass1.srt"
    chinese = tmp_path / "ABC-123.ja.pass1.chinese.srt"
    video.write_text("video")
    write_valid_srt(japanese, "はい")
    write_valid_srt(chinese, "中文")

    item = classify_video(video, BatchOptions(root=tmp_path, force_translate=True))

    assert item.status == "translate_existing_japanese"
    assert item.japanese_srt == japanese
    assert item.chinese_srt is None


def test_plain_zh_srt_is_external_not_whisperjav_translation(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    external = tmp_path / "ABC-123.zh.srt"
    video.write_text("video")
    write_valid_srt(external, "中文")

    item = classify_video(video, BatchOptions(root=tmp_path))

    assert item.status == "skip_external_subtitle"
    assert item.external_subtitle == external


def test_whisperjav_style_zh_translation_is_reusable(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    japanese = tmp_path / "ABC-123.ja.pass1.srt"
    chinese = tmp_path / "ABC-123.ja.pass1.zh.srt"
    video.write_text("video")
    write_valid_srt(japanese, "はい")
    write_valid_srt(chinese, "中文")

    item = classify_video(video, BatchOptions(root=tmp_path))

    assert item.status == "skip_translated"
    assert item.chinese_srt == chinese


def test_force_translate_requires_existing_japanese_srt(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video")

    item = classify_video(video, BatchOptions(root=tmp_path, force_translate=True))

    assert item.status == "failed_precondition"
    assert item.reason == "missing_japanese_srt"


def test_is_valid_srt_requires_two_timecoded_blocks(tmp_path):
    one = tmp_path / "one.srt"
    two = tmp_path / "two.srt"
    one.write_text("1\n00:00:00,000 --> 00:00:01,000\nA\n", encoding="utf-8")
    write_valid_srt(two)

    assert not is_valid_srt(one)
    assert is_valid_srt(two)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
timeout 60s pytest tests/test_batch_discovery.py -q
```

Expected: FAIL because `whisperjav.batch.discovery` does not exist.

- [ ] **Step 3: Implement discovery and classification**

Create `whisperjav/batch/discovery.py`:

```python
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
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return False
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
    for suffix in WHISPERJAV_SOURCE_SUFFIXES:
        candidate = video_path.with_name(f"{video_path.stem}{suffix}")
        if candidate.exists() and is_valid_srt(candidate):
            return candidate
    return None


def _find_chinese_srt(video_path: Path, japanese_srt: Path | None) -> Path | None:
    if japanese_srt is not None:
        expected = expected_translation_path(japanese_srt)
        if expected.exists():
            return expected
    for subtitle in sorted(video_path.parent.glob(f"{video_path.stem}*.srt")):
        if _is_whisperjav_translation(video_path.stem, subtitle):
            return subtitle
    return None


def _matches_video_basename(video_stem: str, subtitle: Path) -> bool:
    return subtitle.stem == video_stem or subtitle.name.startswith(f"{video_stem}.")


def _is_whisperjav_subtitle(video_stem: str, subtitle: Path) -> bool:
    if not subtitle.name.startswith(f"{video_stem}."):
        return False
    source = any(subtitle.name.endswith(suffix) for suffix in WHISPERJAV_SOURCE_SUFFIXES)
    return source or _is_whisperjav_translation(video_stem, subtitle)


def _is_whisperjav_translation(video_stem: str, subtitle: Path) -> bool:
    if not subtitle.name.startswith(f"{video_stem}."):
        return False
    rest = subtitle.name[len(video_stem) :]
    return any(
        rest.startswith(marker) and rest.endswith(suffix)
        for marker in WHISPERJAV_STEM_MARKERS
        for suffix in TRANSLATION_LANGUAGE_SUFFIXES
    )
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
timeout 60s pytest tests/test_batch_discovery.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add whisperjav/batch/discovery.py whisperjav/batch/models.py tests/test_batch_discovery.py
git commit -m "feat: classify batch media inputs"
```

### Task 3: NFO Actress Extraction

**Files:**
- Modify: `whisperjav/batch/models.py`
- Create: `whisperjav/batch/nfo.py`
- Test: `tests/test_batch_nfo.py`

- [ ] **Step 1: Write failing NFO tests**

Create `tests/test_batch_nfo.py`:

```python
from whisperjav.batch.nfo import extract_actresses_from_nfo, find_nfo_for_video


def test_basename_nfo_wins_over_directory_nfo(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video")
    exact = tmp_path / "ABC-123.nfo"
    other = tmp_path / "movie.nfo"
    exact.write_text("<movie><actress>Exact</actress></movie>", encoding="utf-8")
    other.write_text("<movie><actress>Other</actress></movie>", encoding="utf-8")

    result = find_nfo_for_video(video)

    assert result.path == exact
    assert result.reason is None


def test_unique_directory_nfo_is_used(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video")
    nfo = tmp_path / "movie.nfo"
    nfo.write_text("<movie><actress>Name</actress></movie>", encoding="utf-8")

    result = find_nfo_for_video(video)

    assert result.path == nfo
    assert result.reason is None


def test_multiple_unmatched_nfos_are_ambiguous(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video")
    (tmp_path / "one.nfo").write_text("<movie />", encoding="utf-8")
    (tmp_path / "two.nfo").write_text("<movie />", encoding="utf-8")

    result = find_nfo_for_video(video)

    assert result.path is None
    assert result.reason == "nfo_ambiguous"


def test_actor_name_priority_and_case_insensitive_local_names(tmp_path):
    nfo = tmp_path / "ABC-123.nfo"
    nfo.write_text(
        "<movie><Actor><Name>Actor Name</Name>Ignored Direct</Actor></movie>",
        encoding="utf-8",
    )

    result = extract_actresses_from_nfo(nfo)

    assert result.actresses == ("Actor Name",)
    assert result.error is None


def test_direct_actor_text_when_no_child_name(tmp_path):
    nfo = tmp_path / "ABC-123.nfo"
    nfo.write_text("<movie><actor>Direct Actor</actor></movie>", encoding="utf-8")

    result = extract_actresses_from_nfo(nfo)

    assert result.actresses == ("Direct Actor",)


def test_cast_fields_split_and_deduplicate_in_order(tmp_path):
    nfo = tmp_path / "ABC-123.nfo"
    nfo.write_text(
        "<movie><actress>A、B</actress><cast>B; C\nD</cast><performer>A</performer></movie>",
        encoding="utf-8",
    )

    result = extract_actresses_from_nfo(nfo)

    assert result.actresses == ("A", "B", "C", "D")


def test_parse_error_is_reported(tmp_path):
    nfo = tmp_path / "broken.nfo"
    nfo.write_text("<movie><actor>", encoding="utf-8")

    result = extract_actresses_from_nfo(nfo)

    assert result.actresses == ()
    assert result.error is not None
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
timeout 60s pytest tests/test_batch_nfo.py -q
```

Expected: FAIL because `whisperjav.batch.nfo` does not exist.

- [ ] **Step 3: Implement deterministic NFO extraction**

Create `whisperjav/batch/nfo.py`:

```python
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

SPLIT_RE = re.compile(r"[,;、，\n\r]+")


@dataclass(frozen=True)
class NfoLookupResult:
    path: Path | None
    reason: str | None = None


@dataclass(frozen=True)
class NfoActressResult:
    actresses: tuple[str, ...]
    error: str | None = None


def find_nfo_for_video(video_path: Path) -> NfoLookupResult:
    exact = video_path.with_suffix(".nfo")
    if exact.exists():
        return NfoLookupResult(path=exact)
    nfos = sorted(video_path.parent.glob("*.nfo"))
    if len(nfos) == 1:
        return NfoLookupResult(path=nfos[0])
    if len(nfos) > 1:
        return NfoLookupResult(path=None, reason="nfo_ambiguous")
    return NfoLookupResult(path=None)


def extract_actresses_from_nfo(nfo_path: Path) -> NfoActressResult:
    try:
        root = ET.parse(nfo_path).getroot()
    except (OSError, ET.ParseError) as exc:
        return NfoActressResult(actresses=(), error=str(exc))

    values: list[str] = []
    for element in root.iter():
        name = _local_name(element.tag)
        if name == "actor":
            child_name = _first_child_text(element, "name")
            if child_name:
                values.extend(_split_names(child_name))
            elif element.text:
                values.extend(_split_names(element.text))
        elif name in {"actress", "cast", "performer"} and element.text:
            values.extend(_split_names(element.text))
    return NfoActressResult(actresses=tuple(_dedupe(values)))


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _first_child_text(element: ET.Element, child_name: str) -> str | None:
    for child in list(element):
        if _local_name(child.tag) == child_name and child.text:
            return child.text
    return None


def _split_names(value: str) -> list[str]:
    return [_normalize(part) for part in SPLIT_RE.split(value) if _normalize(part)]


def _normalize(value: str) -> str:
    return " ".join(value.split())


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
timeout 60s pytest tests/test_batch_nfo.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add whisperjav/batch/nfo.py tests/test_batch_nfo.py
git commit -m "feat: extract batch actress context from nfo"
```

### Task 4: Reports and Subprocess Runner Boundaries

**Files:**
- Create: `whisperjav/batch/reports.py`
- Create: `whisperjav/batch/runners.py`
- Test: `tests/test_batch_reports.py`

- [ ] **Step 1: Write failing report/runner tests**

Create `tests/test_batch_reports.py`:

```python
import json
import sys

from whisperjav.batch.models import ProcessResult, VideoResult
from whisperjav.batch.reports import BatchReportWriter, summarize_results
from whisperjav.batch.runners import SubprocessRunner, completed_process_result


def test_report_writer_outputs_jsonl_and_summary_without_secrets(tmp_path):
    writer = BatchReportWriter(report_dir=tmp_path, input_root=tmp_path)
    result = VideoResult(
        video_path=tmp_path / "ABC-123.mp4",
        status="failed_translation",
        reason="provider_error",
        japanese_srt=tmp_path / "ABC-123.ja.pass1.srt",
        translation=ProcessResult(
            seconds=1.2,
            return_code=1,
            command_redacted=["cmd", "--api-key", "<redacted>"],
            api_key_source="cli",
            stderr_tail="bad request",
        ),
        error="translation failed",
    )

    paths = writer.write([result])

    row = json.loads(paths.jsonl.read_text(encoding="utf-8").splitlines()[0])
    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
    assert row["status"] == "failed_translation"
    assert row["translation"]["command_redacted"] == ["cmd", "--api-key", "<redacted>"]
    assert row["translation"]["api_key_source"] == "cli"
    assert "secret" not in paths.jsonl.read_text(encoding="utf-8")
    assert summary["counts_by_status"] == {"failed_translation": 1}
    assert summary["input_root"] == str(tmp_path)
    assert summary["started_at"]
    assert summary["ended_at"]
    assert summary["report_file"] == str(paths.jsonl)
    assert summary["failed"] == [str(tmp_path / "ABC-123.mp4")]


def test_summarize_results_totals_asr_and_translation_seconds(tmp_path):
    results = [
        VideoResult(
            video_path=tmp_path / "one.mp4",
            status="completed",
            asr=ProcessResult(seconds=3.0),
            translation=ProcessResult(seconds=4.0),
        ),
        VideoResult(video_path=tmp_path / "two.mp4", status="skip_translated"),
    ]

    summary = summarize_results(results)

    assert summary["total_videos"] == 2
    assert summary["counts_by_status"] == {"completed": 1, "skip_translated": 1}
    assert summary["total_asr_seconds"] == 3.0
    assert summary["total_translation_seconds"] == 4.0


def test_completed_process_result_keeps_tail_and_redacted_command():
    result = completed_process_result(
        command=["cmd", "--api-key", "secret"],
        return_code=7,
        seconds=1.0,
        stdout="line1\nline2",
        stderr="err1\nerr2",
    )

    assert result.return_code == 7
    assert result.command_redacted == ["cmd", "--api-key", "<redacted>"]
    assert result.stdout_tail == "line1\nline2"
    assert result.stderr_tail == "err1\nerr2"


def test_subprocess_runner_streams_and_preserves_tail(capsys):
    runner = SubprocessRunner(stream=True)

    result = runner.run(
        [
            sys.executable,
            "-c",
            "import sys; print('out-line'); print('err-line', file=sys.stderr)",
        ]
    )

    captured = capsys.readouterr()
    assert result.return_code == 0
    assert "out-line" in captured.out
    assert "err-line" in captured.err
    assert result.stdout_tail == "out-line"
    assert result.stderr_tail == "err-line"
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
timeout 60s pytest tests/test_batch_reports.py -q
```

Expected: FAIL because report/runner modules do not exist.

- [ ] **Step 3: Implement reports and runner result helpers**

Create `whisperjav/batch/reports.py`:

```python
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .models import ProcessResult, VideoResult


@dataclass(frozen=True)
class ReportPaths:
    jsonl: Path
    summary: Path


class BatchReportWriter:
    def __init__(self, report_dir: Path, input_root: Path) -> None:
        self.report_dir = report_dir
        self.input_root = input_root

    def write(self, results: Iterable[VideoResult]) -> ReportPaths:
        result_list = list(results)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        started_at = datetime.now().isoformat(timespec="seconds")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        jsonl = self.report_dir / f"run-{timestamp}.jsonl"
        summary_path = self.report_dir / f"run-{timestamp}.summary.json"

        with jsonl.open("w", encoding="utf-8") as handle:
            for result in result_list:
                handle.write(json.dumps(_result_to_dict(result), ensure_ascii=False) + "\n")
        ended_at = datetime.now().isoformat(timespec="seconds")
        summary = summarize_results(result_list)
        summary["input_root"] = str(self.input_root)
        summary["started_at"] = started_at
        summary["ended_at"] = ended_at
        summary["report_file"] = str(jsonl)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return ReportPaths(jsonl=jsonl, summary=summary_path)


def summarize_results(results: Iterable[VideoResult]) -> dict:
    result_list = list(results)
    counts = Counter(result.status for result in result_list)
    failed_statuses = {"failed_asr", "failed_translation", "failed_precondition", "cancelled"}
    return {
        "total_videos": len(result_list),
        "counts_by_status": dict(counts),
        "total_asr_seconds": sum((result.asr.seconds if result.asr else 0.0) for result in result_list),
        "total_translation_seconds": sum(
            (result.translation.seconds if result.translation else 0.0) for result in result_list
        ),
        "failed": [str(result.video_path) for result in result_list if result.status in failed_statuses],
    }


def _result_to_dict(result: VideoResult) -> dict:
    data = asdict(result)
    for key in ("video_path", "nfo_path", "japanese_srt", "chinese_srt"):
        if data[key] is not None:
            data[key] = str(data[key])
    return data
```

Create `whisperjav/batch/runners.py`:

```python
from __future__ import annotations

import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

from .commands import redact_command
from .models import ProcessResult

TAIL_LINES = 80


@dataclass(frozen=True)
class SubprocessRunner:
    stream: bool = False
    _processes: list[subprocess.Popen] = field(default_factory=list, init=False, repr=False)

    def run(self, command: Sequence[str]) -> ProcessResult:
        started = time.monotonic()
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self._processes.append(process)
        stdout, stderr = self._communicate(process)
        self._processes.remove(process)
        seconds = time.monotonic() - started
        return completed_process_result(
            command=list(command),
            return_code=process.returncode,
            seconds=seconds,
            stdout=stdout,
            stderr=stderr,
        )

    def terminate_all(self) -> None:
        for process in list(self._processes):
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 5.0
        for process in list(self._processes):
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()

    def _communicate(self, process: subprocess.Popen) -> tuple[str, str]:
        stdout_tail: deque[str] = deque(maxlen=TAIL_LINES)
        stderr_tail: deque[str] = deque(maxlen=TAIL_LINES)
        stdout_thread = threading.Thread(
            target=_read_pipe,
            args=(process.stdout, stdout_tail, sys.stdout if self.stream else None),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_read_pipe,
            args=(process.stderr, stderr_tail, sys.stderr if self.stream else None),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        process.wait()
        stdout_thread.join()
        stderr_thread.join()
        return "\n".join(stdout_tail), "\n".join(stderr_tail)


def completed_process_result(
    *,
    command: list[str],
    return_code: int,
    seconds: float,
    stdout: str,
    stderr: str,
) -> ProcessResult:
    return ProcessResult(
        seconds=seconds,
        return_code=return_code,
        command_redacted=redact_command(command),
        stdout_tail=_tail(stdout),
        stderr_tail=_tail(stderr),
    )


def _tail(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-TAIL_LINES:])


def _read_pipe(pipe, tail: deque[str], mirror) -> None:
    if pipe is None:
        return
    for line in pipe:
        clean = line.rstrip("\n")
        tail.append(clean)
        if mirror is not None:
            print(clean, file=mirror)
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
timeout 60s pytest tests/test_batch_reports.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add whisperjav/batch/reports.py whisperjav/batch/runners.py tests/test_batch_reports.py
git commit -m "feat: write batch reports"
```

### Task 5: Scheduler With Overlapped Translation Queue

**Files:**
- Create: `whisperjav/batch/scheduler.py`
- Modify: `whisperjav/batch/discovery.py`
- Test: `tests/test_batch_scheduler.py`

- [ ] **Step 1: Write failing scheduler tests**

Create `tests/test_batch_scheduler.py`:

```python
from pathlib import Path

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


class FakeRunner:
    def __init__(self, tmp_path, *, fail_asr=False, fail_translation=False):
        self.tmp_path = tmp_path
        self.fail_asr = fail_asr
        self.fail_translation = fail_translation
        self.commands = []

    def run(self, command):
        self.commands.append(list(command))
        if "whisperjav.main" in command:
            if self.fail_asr:
                return ProcessResult(return_code=9, command_redacted=list(command), stderr_tail="asr failed")
            video = Path(command[4])
            write_valid_srt(video.with_name(f"{video.stem}.ja.pass1.srt"))
            return ProcessResult(return_code=0, command_redacted=list(command), seconds=2.0)
        if self.fail_translation:
            return ProcessResult(return_code=8, command_redacted=list(command), stderr_tail="translation failed")
        srt = Path(command[command.index("-i") + 1])
        write_valid_srt(srt.with_name(f"{srt.stem}.chinese.srt"), "中文")
        return ProcessResult(return_code=0, command_redacted=list(command), seconds=3.0)


def test_existing_japanese_srt_translation_completes(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    japanese = tmp_path / "ABC-123.ja.pass1.srt"
    video.write_text("video")
    write_valid_srt(japanese)
    item = ClassifiedVideo(video_path=video, status="translate_existing_japanese", japanese_srt=japanese)
    runner = FakeRunner(tmp_path)

    results = BatchScheduler(BatchOptions(root=tmp_path), runner=runner).run([item])

    assert results[0].status == "completed_translation_only"
    assert results[0].translation.return_code == 0
    assert any("whisperjav.translate.cli" in command for command in runner.commands)
    assert not any("whisperjav.main" in command for command in runner.commands)


def test_transcribe_then_translate_runs_asr_before_translation(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video")
    item = ClassifiedVideo(video_path=video, status="transcribe_then_translate")
    runner = FakeRunner(tmp_path)

    results = BatchScheduler(BatchOptions(root=tmp_path), runner=runner).run([item])

    assert results[0].status == "completed"
    assert [command[3] for command in runner.commands] == ["whisperjav.main", "whisperjav.translate.cli"]


def test_asr_failure_does_not_enqueue_translation(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video")
    item = ClassifiedVideo(video_path=video, status="transcribe_then_translate")
    runner = FakeRunner(tmp_path, fail_asr=True)

    results = BatchScheduler(BatchOptions(root=tmp_path), runner=runner).run([item])

    assert results[0].status == "failed_asr"
    assert len(runner.commands) == 1


def test_translation_failure_keeps_processing_later_items(tmp_path):
    first = tmp_path / "ABC-123.mp4"
    second = tmp_path / "DEF-456.mp4"
    first.write_text("video")
    second.write_text("video")
    runner = FakeRunner(tmp_path, fail_translation=True)
    items = [
        ClassifiedVideo(video_path=first, status="transcribe_then_translate"),
        ClassifiedVideo(video_path=second, status="skip_translated", chinese_srt=tmp_path / "DEF-456.chinese.srt"),
    ]

    results = BatchScheduler(BatchOptions(root=tmp_path), runner=runner).run(items)

    assert [result.status for result in results] == ["failed_translation", "skip_translated"]


def test_dry_run_returns_classification_without_subprocesses(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video")
    item = ClassifiedVideo(video_path=video, status="transcribe_then_translate")
    runner = FakeRunner(tmp_path)

    results = BatchScheduler(BatchOptions(root=tmp_path, dry_run=True), runner=runner).run([item])

    assert results[0].status == "transcribe_then_translate"
    assert runner.commands == []
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
timeout 60s pytest tests/test_batch_scheduler.py -q
```

Expected: FAIL because `whisperjav.batch.scheduler` does not exist.

- [ ] **Step 3: Implement scheduler**

Create `whisperjav/batch/scheduler.py`:

```python
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from .commands import build_asr_command, build_translation_command, expected_translation_path
from .discovery import is_valid_srt
from .models import BatchOptions, ClassifiedVideo, ProcessResult, VideoResult
from .runners import SubprocessRunner

TERMINAL_SKIP_STATUSES = {"skip_external_subtitle", "skip_translated", "failed_precondition"}


class BatchScheduler:
    def __init__(self, options: BatchOptions, *, runner: SubprocessRunner | None = None) -> None:
        self.options = options
        self.runner = runner or SubprocessRunner(stream=options.stream)

    def run(self, items: Iterable[ClassifiedVideo]) -> list[VideoResult]:
        results: list[VideoResult] = []
        for item in items:
            if self.options.dry_run or item.status in TERMINAL_SKIP_STATUSES:
                results.append(_from_classified(item))
                continue
            if item.status == "translate_existing_japanese":
                results.append(self._translate_existing(item))
                continue
            if item.status == "transcribe_then_translate":
                results.append(self._transcribe_then_translate(item))
                continue
            results.append(_from_classified(item))
        return results

    def _translate_existing(self, item: ClassifiedVideo) -> VideoResult:
        if item.japanese_srt is None:
            return _failed_precondition(item, "missing_japanese_srt")
        translation = self._run_translation(item.japanese_srt, item.actresses)
        if translation.return_code != 0:
            return _from_classified(
                item,
                status="failed_translation",
                translation=translation,
                error=translation.stderr_tail or "translation failed",
            )
        chinese = expected_translation_path(item.japanese_srt)
        return _from_classified(
            item,
            status="completed_translation_only",
            chinese_srt=chinese if chinese.exists() else None,
            translation=translation,
        )

    def _transcribe_then_translate(self, item: ClassifiedVideo) -> VideoResult:
        asr = self.runner.run(build_asr_command(item.video_path, self.options))
        if asr.return_code != 0:
            return _from_classified(
                item,
                status="failed_asr",
                asr=asr,
                error=asr.stderr_tail or "asr failed",
            )
        japanese = _expected_japanese_srt(item.video_path)
        if not is_valid_srt(japanese):
            return _from_classified(
                item,
                status="failed_asr",
                asr=asr,
                error=f"expected Japanese SRT missing or invalid: {japanese}",
            )
        translation = self._run_translation(japanese, item.actresses)
        if translation.return_code != 0:
            return _from_classified(
                item,
                status="failed_translation",
                japanese_srt=japanese,
                asr=asr,
                translation=translation,
                error=translation.stderr_tail or "translation failed",
            )
        chinese = expected_translation_path(japanese)
        return _from_classified(
            item,
            status="completed",
            japanese_srt=japanese,
            chinese_srt=chinese if chinese.exists() else None,
            asr=asr,
            translation=translation,
        )

    def _run_translation(self, japanese_srt: Path, actresses: tuple[str, ...]) -> ProcessResult:
        result = self.runner.run(build_translation_command(japanese_srt, self.options, actresses=actresses))
        api_key_source = "cli" if self.options.translate_api_key else "env"
        return replace(result, api_key_source=api_key_source)


def _from_classified(
    item: ClassifiedVideo,
    *,
    status: str | None = None,
    reason: str | None = None,
    japanese_srt: Path | None = None,
    chinese_srt: Path | None = None,
    asr: ProcessResult | None = None,
    translation: ProcessResult | None = None,
    error: str | None = None,
) -> VideoResult:
    return VideoResult(
        video_path=item.video_path,
        status=status or item.status,
        reason=reason if reason is not None else item.reason,
        nfo_path=item.nfo_path,
        actresses=item.actresses,
        japanese_srt=japanese_srt or item.japanese_srt,
        chinese_srt=chinese_srt or item.chinese_srt,
        asr=asr,
        translation=translation,
        error=error,
        warnings=item.warnings,
    )


def _failed_precondition(item: ClassifiedVideo, reason: str) -> VideoResult:
    return _from_classified(item, status="failed_precondition", reason=reason)


def _expected_japanese_srt(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}.ja.pass1.srt")
```

This first scheduler implementation is deliberately serial only for the initial red/green slice. The next step must replace the translation path with bounded worker execution so ASR for the next video can proceed while translation for the previous video is still running.

- [ ] **Step 4: Add overlapped translation worker tests**

Append to `tests/test_batch_scheduler.py`:

```python
import threading


class OverlapRunner(FakeRunner):
    def __init__(self, tmp_path):
        super().__init__(tmp_path)
        self.first_translation_started = threading.Event()
        self.first_translation_done = threading.Event()
        self.second_asr_started_after_translation = threading.Event()
        self.allow_first_translation_finish = threading.Event()

    def run(self, command):
        if "whisperjav.main" in command:
            video = Path(command[4])
            if video.stem == "DEF-456":
                if self.first_translation_started.is_set() and not self.first_translation_done.is_set():
                    self.second_asr_started_after_translation.set()
                    self.allow_first_translation_finish.set()
            write_valid_srt(video.with_name(f"{video.stem}.ja.pass1.srt"))
            self.commands.append(list(command))
            return ProcessResult(return_code=0, command_redacted=list(command), seconds=2.0)
        srt = Path(command[command.index("-i") + 1])
        self.commands.append(list(command))
        if srt.stem.startswith("ABC-123"):
            self.first_translation_started.set()
            self.allow_first_translation_finish.wait(timeout=5.0)
            self.first_translation_done.set()
        write_valid_srt(srt.with_name(f"{srt.stem}.chinese.srt"), "中文")
        return ProcessResult(return_code=0, command_redacted=list(command), seconds=3.0)


def test_asr_overlaps_previous_translation(tmp_path):
    first = tmp_path / "ABC-123.mp4"
    second = tmp_path / "DEF-456.mp4"
    first.write_text("video")
    second.write_text("video")
    runner = OverlapRunner(tmp_path)
    items = [
        ClassifiedVideo(video_path=first, status="transcribe_then_translate"),
        ClassifiedVideo(video_path=second, status="transcribe_then_translate"),
    ]

    results = BatchScheduler(
        BatchOptions(root=tmp_path, translate_workers=1, translation_queue_size=2),
        runner=runner,
    ).run(items)

    assert [result.status for result in results] == ["completed", "completed"]
    assert runner.second_asr_started_after_translation.is_set()


def test_translation_queue_size_is_validated(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video")

    try:
        BatchScheduler(BatchOptions(root=tmp_path, translation_queue_size=0))
    except ValueError as exc:
        assert "translation_queue_size" in str(exc)
    else:
        raise AssertionError("expected ValueError")


class InterruptingRunner(FakeRunner):
    def __init__(self, tmp_path):
        super().__init__(tmp_path)
        self.terminated = False

    def run(self, command):
        raise KeyboardInterrupt

    def terminate_all(self):
        self.terminated = True


def test_keyboard_interrupt_marks_unfinished_items_cancelled(tmp_path):
    first = tmp_path / "ABC-123.mp4"
    second = tmp_path / "DEF-456.mp4"
    first.write_text("video")
    second.write_text("video")
    runner = InterruptingRunner(tmp_path)
    items = [
        ClassifiedVideo(video_path=first, status="transcribe_then_translate"),
        ClassifiedVideo(video_path=second, status="transcribe_then_translate"),
    ]

    results = BatchScheduler(BatchOptions(root=tmp_path), runner=runner).run(items)

    assert [result.status for result in results] == ["cancelled", "cancelled"]
    assert runner.terminated
```

Run:

```bash
timeout 60s pytest tests/test_batch_scheduler.py::test_asr_overlaps_previous_translation -q
```

Expected: FAIL because the serial scheduler waits for the first translation to finish before starting the second ASR.

- [ ] **Step 5: Replace serial translation with bounded worker execution**

Update `whisperjav/batch/scheduler.py` imports:

```python
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import threading
```

Update `BatchScheduler.__init__` and `run`:

```python
    def __init__(self, options: BatchOptions, *, runner: SubprocessRunner | None = None) -> None:
        if options.translate_workers < 1 or options.translate_workers > 4:
            raise ValueError("translate_workers must be between 1 and 4")
        if options.translation_queue_size < 1:
            raise ValueError("translation_queue_size must be at least 1")
        self.options = options
        self.runner = runner or SubprocessRunner(stream=options.stream)

    def run(self, items: Iterable[ClassifiedVideo]) -> list[VideoResult]:
        item_list = list(items)
        results: list[VideoResult | None] = [None] * len(item_list)
        pending: dict[Future[VideoResult], int] = {}
        slots = threading.Semaphore(self.options.translation_queue_size)

        executor = ThreadPoolExecutor(max_workers=self.options.translate_workers)
        try:
            for index, item in enumerate(item_list):
                _collect_completed(pending, results, block=False)
                if self.options.dry_run or item.status in TERMINAL_SKIP_STATUSES:
                    results[index] = _from_classified(item)
                    continue
                if item.status == "translate_existing_japanese":
                    self._submit_translation(executor, slots, pending, index, item, None, item.japanese_srt)
                    continue
                if item.status == "transcribe_then_translate":
                    asr = self.runner.run(build_asr_command(item.video_path, self.options))
                    if asr.return_code != 0:
                        results[index] = _from_classified(
                            item,
                            status="failed_asr",
                            asr=asr,
                            error=asr.stderr_tail or "asr failed",
                        )
                        continue
                    japanese = _expected_japanese_srt(item.video_path)
                    if not is_valid_srt(japanese):
                        results[index] = _from_classified(
                            item,
                            status="failed_asr",
                            asr=asr,
                            error=f"expected Japanese SRT missing or invalid: {japanese}",
                        )
                        continue
                    self._submit_translation(executor, slots, pending, index, item, asr, japanese)
                    continue
                results[index] = _from_classified(item)

            while pending:
                _collect_completed(pending, results, block=True)
        except KeyboardInterrupt:
            terminate_all = getattr(self.runner, "terminate_all", None)
            if terminate_all is not None:
                terminate_all()
            for future in pending:
                future.cancel()
            for index, result in enumerate(results):
                if result is None:
                    results[index] = _from_classified(
                        item_list[index],
                        status="cancelled",
                        reason="interrupted",
                    )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        return [result for result in results if result is not None]

    def _submit_translation(
        self,
        executor: ThreadPoolExecutor,
        slots: threading.Semaphore,
        pending: dict[Future[VideoResult], int],
        index: int,
        item: ClassifiedVideo,
        asr: ProcessResult | None,
        japanese_srt: Path | None,
    ) -> None:
        slots.acquire()
        future = executor.submit(self._translation_result, slots, item, asr, japanese_srt)
        pending[future] = index

    def _translation_result(
        self,
        slots: threading.Semaphore,
        item: ClassifiedVideo,
        asr: ProcessResult | None,
        japanese_srt: Path | None,
    ) -> VideoResult:
        try:
            if japanese_srt is None:
                return _failed_precondition(item, "missing_japanese_srt")
            translation = self._run_translation(japanese_srt, item.actresses)
            if translation.return_code != 0:
                return _from_classified(
                    item,
                    status="failed_translation",
                    japanese_srt=japanese_srt,
                    asr=asr,
                    translation=translation,
                    error=translation.stderr_tail or "translation failed",
                )
            chinese = expected_translation_path(japanese_srt)
            status = "completed_translation_only" if asr is None else "completed"
            return _from_classified(
                item,
                status=status,
                japanese_srt=japanese_srt,
                chinese_srt=chinese if chinese.exists() else None,
                asr=asr,
                translation=translation,
            )
        finally:
            slots.release()
```

Add this helper below `BatchScheduler`:

```python
def _collect_completed(
    pending: dict[Future[VideoResult], int],
    results: list[VideoResult | None],
    *,
    block: bool,
) -> None:
    if not pending:
        return
    if block:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
    else:
        done = {future for future in pending if future.done()}
    for future in done:
        index = pending.pop(future)
        results[index] = future.result()
```

Run:

```bash
timeout 60s pytest tests/test_batch_scheduler.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

Run:

```bash
git add whisperjav/batch/scheduler.py tests/test_batch_scheduler.py
git commit -m "feat: schedule batch transcription and translation"
```

### Task 6: CLI Integration

**Files:**
- Create: `whisperjav/batch/cli.py`
- Modify: `whisperjav/batch/discovery.py`
- Test: `tests/test_batch_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_batch_cli.py`:

```python
import json

import pytest

from whisperjav.batch import cli


def write_valid_srt(path, text="はい"):
    path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n"
        f"{text}\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n"
        f"{text}\n",
        encoding="utf-8",
    )


def test_parse_args_rejects_force_conflict(tmp_path):
    with pytest.raises(SystemExit):
        cli.parse_args([str(tmp_path), "--force", "--force-translate"])


def test_parse_args_rejects_missing_root(tmp_path):
    missing = tmp_path / "missing"

    with pytest.raises(SystemExit):
        cli.parse_args([str(missing)])


def test_parse_args_rejects_invalid_worker_count(tmp_path):
    with pytest.raises(SystemExit):
        cli.parse_args([str(tmp_path), "--translate-workers", "5"])


def test_dry_run_writes_reports_and_returns_zero(tmp_path, monkeypatch):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video")
    report_dir = tmp_path / "reports"

    code = cli.main([str(tmp_path), "--dry-run", "--report-dir", str(report_dir)])

    assert code == 0
    summaries = list(report_dir.glob("*.summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["counts_by_status"] == {"transcribe_then_translate": 1}


def test_cli_includes_nfo_actress_context_in_dry_run_report(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video")
    write_valid_srt(tmp_path / "ABC-123.ja.pass1.srt")
    (tmp_path / "ABC-123.nfo").write_text("<movie><actor><name>Name1</name></actor></movie>", encoding="utf-8")
    report_dir = tmp_path / "reports"

    code = cli.main([str(tmp_path), "--dry-run", "--report-dir", str(report_dir)])

    assert code == 0
    jsonl = next(report_dir.glob("*.jsonl"))
    row = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert row["actresses"] == ["Name1"]


def test_cli_manual_actress_context_is_visible_in_dry_run_report(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video")
    write_valid_srt(tmp_path / "ABC-123.ja.pass1.srt")
    report_dir = tmp_path / "reports"

    code = cli.main([str(tmp_path), "--dry-run", "--report-dir", str(report_dir), "--actress", "Manual Name"])

    assert code == 0
    jsonl = next(report_dir.glob("*.jsonl"))
    row = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert row["actresses"] == ["Manual Name"]
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
timeout 60s pytest tests/test_batch_cli.py -q
```

Expected: FAIL because `whisperjav.batch.cli` does not exist.

- [ ] **Step 3: Implement CLI**

Create `whisperjav/batch/cli.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from .discovery import classify_video, discover_media_files
from .models import BatchOptions, ClassifiedVideo
from .nfo import extract_actresses_from_nfo, find_nfo_for_video
from .reports import BatchReportWriter, summarize_results
from .scheduler import BatchScheduler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch transcribe and translate JAV videos.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--translate-api-key")
    parser.add_argument("--translate-workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--translation-queue-size", type=int, default=2)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--include-audio", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-translate", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--accept-cpu-mode", action="store_true")
    parser.add_argument("--no-nfo", action="store_true")
    parser.add_argument("--actress")
    args = parser.parse_args(argv)
    if args.force and args.force_translate:
        parser.error("--force and --force-translate are mutually exclusive")
    if not args.root.exists() or not args.root.is_dir():
        parser.error("root must be an existing directory")
    if args.translation_queue_size < 1:
        parser.error("--translation-queue-size must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    options = BatchOptions(
        root=args.root,
        report_dir=args.report_dir,
        include_audio=args.include_audio,
        dry_run=args.dry_run,
        force=args.force,
        force_translate=args.force_translate,
        translate_api_key=args.translate_api_key,
        translate_workers=args.translate_workers,
        translation_queue_size=args.translation_queue_size,
        stream=args.stream,
        debug=args.debug,
        accept_cpu_mode=args.accept_cpu_mode,
        no_nfo=args.no_nfo,
        actress=args.actress,
    )
    report_dir = options.report_dir or options.root / ".whisperjav_batch"
    videos = discover_media_files(options.root, include_audio=options.include_audio)
    classified = [_attach_nfo(classify_video(video, options), options) for video in videos]
    results = BatchScheduler(options).run(classified)
    paths = BatchReportWriter(report_dir=report_dir, input_root=options.root).write(results)
    summary = summarize_results(results)
    _print_summary(summary, paths.jsonl, paths.summary)
    return 1 if summary["failed"] else 0


def _attach_nfo(item: ClassifiedVideo, options: BatchOptions) -> ClassifiedVideo:
    if options.actress:
        return _replace_context(item, actresses=tuple(part.strip() for part in options.actress.split(",") if part.strip()))
    if options.no_nfo:
        return item
    lookup = find_nfo_for_video(item.video_path)
    warnings = list(item.warnings)
    actresses: tuple[str, ...] = ()
    if lookup.reason:
        warnings.append(lookup.reason)
    if lookup.path is not None:
        parsed = extract_actresses_from_nfo(lookup.path)
        actresses = parsed.actresses
        if parsed.error:
            warnings.append(parsed.error)
    return _replace_context(
        item,
        nfo_path=lookup.path,
        actresses=actresses,
        warnings=tuple(warnings),
    )


def _replace_context(
    item: ClassifiedVideo,
    *,
    nfo_path: Path | None = None,
    actresses: tuple[str, ...] | None = None,
    warnings: tuple[str, ...] | None = None,
) -> ClassifiedVideo:
    return ClassifiedVideo(
        video_path=item.video_path,
        status=item.status,
        reason=item.reason,
        external_subtitle=item.external_subtitle,
        japanese_srt=item.japanese_srt,
        chinese_srt=item.chinese_srt,
        nfo_path=nfo_path if nfo_path is not None else item.nfo_path,
        actresses=actresses if actresses is not None else item.actresses,
        warnings=warnings if warnings is not None else item.warnings,
    )


def _print_summary(summary: dict, jsonl_path: Path, summary_path: Path) -> None:
    print("WHISPERJAV BATCH SUMMARY")
    for status, count in sorted(summary["counts_by_status"].items()):
        print(f"{status}: {count}")
    print(f"Report: {jsonl_path}")
    print(f"Summary: {summary_path}")
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
timeout 60s pytest tests/test_batch_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Run focused batch suite**

Run:

```bash
timeout 60s pytest \
  tests/test_batch_commands.py \
  tests/test_batch_discovery.py \
  tests/test_batch_nfo.py \
  tests/test_batch_reports.py \
  tests/test_batch_scheduler.py \
  tests/test_batch_cli.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 6**

Run:

```bash
git add whisperjav/batch/cli.py tests/test_batch_cli.py
git commit -m "feat: add whisperjav batch cli"
```

## Plan Review Notes

2026-06-15 reviews completed by Codex subagent `019ecbfa-0298-78d2-a110-d9421e2377d7` and Claude.

Accepted changes:

- Do not import `whisperjav.translate.cli.generate_output_path` from batch code; the plan now implements the small naming rule locally to avoid translate-extra import side effects.
- Treat plain `ABC-123.zh.srt` / `ABC-123.cn.srt` as external subtitle blockers; only WhisperJAV-style stems such as `ABC-123.ja.pass1.zh.srt` are reusable translated outputs.
- `--force-translate` now ignores existing Chinese output when a reusable Japanese WhisperJAV SRT exists.
- Discovery no longer instantiates `MediaDiscovery` just to read extension sets, avoiding the `ffprobe -version` side effect.
- Reports now include `input_root`, `started_at`, `ended_at`, and `report_file`.
- Runner plan now uses `subprocess.Popen`, `terminate_all()`, streamed output mirroring, and bounded stdout/stderr tails.
- Scheduler plan now has a `KeyboardInterrupt` path that terminates active subprocesses, cancels futures, and marks unfinished results as `cancelled`.
- CLI plan now validates root directory existence, worker range, and translation queue size with argparse errors.
- Tests now cover `--source japanese`, Python 3.10-safe pyproject entry checking, `--force` with external subtitles, manual actress context in reports, and a non-false-positive overlap assertion.

Rejected or deferred:

- Claude's concern that the initial serial scheduler slice is dead code is noted, but the plan keeps it as an explicit TDD red/green stepping stone before the overlap test forces the final worker implementation.
- ANSI stripping from captured subprocess tails is deferred; the plan preserves diagnostic bytes as emitted because the current spec requires visible failure context, not normalized log formatting.
