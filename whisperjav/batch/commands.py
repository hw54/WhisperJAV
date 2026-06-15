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
