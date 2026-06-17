from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace


def test_contextual_tone_metadata_and_defaults() -> None:
    from whisperjav.translate.tones import TONE_CHOICES, get_tone_config

    assert TONE_CHOICES == ("standard", "contextual", "pornify")
    assert get_tone_config("standard").temperature == 0.5
    assert get_tone_config("contextual").temperature == 0.8
    assert get_tone_config("pornify").temperature == 1.2
    assert get_tone_config("contextual").top_p == 0.9


def test_cli_provider_options_use_contextual_default_temperature() -> None:
    from whisperjav.translate.cli import build_provider_options

    args = SimpleNamespace(
        temperature=None,
        top_p=None,
        rate_limit=None,
        max_retries=None,
        backoff_time=None,
    )

    options = build_provider_options(args, {}, "contextual")

    assert options["temperature"] == 0.8
    assert options["top_p"] == 0.9


def test_cli_provider_options_keep_settings_temperature_global_override() -> None:
    from whisperjav.translate.cli import build_provider_options

    args = SimpleNamespace(
        temperature=None,
        top_p=None,
        rate_limit=None,
        max_retries=None,
        backoff_time=None,
    )

    options = build_provider_options(args, {"temperature": 0.42}, "contextual")

    assert options["temperature"] == 0.42


def test_service_provider_options_use_contextual_default_temperature() -> None:
    from whisperjav.translate.service import _build_provider_options

    options = _build_provider_options(tone="contextual")

    assert options["temperature"] == 0.8
    assert options["top_p"] == 0.9


def test_contextual_instructions_are_bundled_not_gist_backed() -> None:
    from whisperjav.translate.instructions import (
        DEFAULT_INSTRUCTION_URLS,
        get_instruction_content,
        load_bundled_default,
    )

    assert "contextual" not in DEFAULT_INSTRUCTION_URLS
    bundled = load_bundled_default("contextual")
    assert bundled is not None
    assert "matches the explicitness of the original" in bundled
    assert get_instruction_content("contextual") == bundled


def test_interactive_configure_can_select_contextual_tone(monkeypatch, tmp_path) -> None:
    from whisperjav.translate import configure
    from whisperjav.translate.settings import DEFAULT_SETTINGS

    saved_settings: dict = {}
    inputs = iter([
        "",  # provider: keep deepseek
        "",  # API key: blank
        "",  # model: default
        "",  # target language: keep
        "2",  # tone: contextual
        "n",  # advanced settings
        "",  # save settings
    ])

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(configure, "load_settings", lambda: DEFAULT_SETTINGS.copy())
    monkeypatch.setattr(configure, "get_settings_path", lambda: tmp_path / "settings.json")
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(inputs))

    def fake_save(settings: dict) -> bool:
        saved_settings.update(settings)
        return True

    monkeypatch.setattr(configure, "save_settings", fake_save)

    configure.interactive_configure()

    assert saved_settings["tone"] == "contextual"

