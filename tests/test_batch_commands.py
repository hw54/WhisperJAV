import importlib
import json
import sys
from pathlib import Path

import whisperjav.batch.commands as command_builders
from whisperjav.batch.commands import (
    build_asr_command,
    build_translation_command,
    expected_translation_path,
    redact_command,
)
from whisperjav.batch.models import BatchOptions, ProcessResult


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
    assert "--api-key" not in command
    assert "--actress" in command
    assert command[command.index("--actress") + 1] == "Name1, Name2"
    assert "--translate-provider" not in command


def test_translation_env_injects_api_key_without_mutating_base_env(tmp_path):
    options = BatchOptions(root=tmp_path, translate_api_key="secret-key")
    base_env = {"EXISTING": "1"}

    env = command_builders.build_translation_env(options, base_env=base_env)

    assert env["EXISTING"] == "1"
    assert env["DEEPSEEK_API_KEY"] == "secret-key"
    assert base_env == {"EXISTING": "1"}


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
    command = [
        "cmd",
        "--api-key",
        "secret",
        "--translate-api-key=other",
        "--model",
        "x",
    ]

    assert redact_command(command) == [
        "cmd",
        "--api-key",
        "<redacted>",
        "--translate-api-key=<redacted>",
        "--model",
        "x",
    ]


def test_process_result_command_redacted_is_immutable_tuple():
    command = ["cmd", "--model", "x"]

    result = ProcessResult(command_redacted=command)
    command.append("--debug")

    assert result.command_redacted == ("cmd", "--model", "x")


def test_pyproject_exposes_batch_entry_point():
    text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'whisperjav-batch = "whisperjav.batch.cli:main"' in text


def test_batch_cli_entry_point_module_is_importable():
    module = importlib.import_module("whisperjav.batch.cli")

    assert callable(module.main)
