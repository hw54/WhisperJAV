from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch


def _write_translate_settings(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _api_instance():
    with patch("webview.create_window"), patch("webview.windows", []):
        from whisperjav.webview_gui.api import WhisperJAVAPI

        return WhisperJAVAPI.__new__(WhisperJAVAPI)


def test_gui_translation_settings_do_not_materialize_temperature_defaults(
    tmp_path, monkeypatch
) -> None:
    from whisperjav.translate.settings import DEFAULT_SETTINGS

    settings_path = tmp_path / "settings.json"
    data = DEFAULT_SETTINGS.copy()
    data["model_params"] = {"temperature": None, "top_p": None}
    _write_translate_settings(settings_path, data)
    monkeypatch.setattr(
        "whisperjav.translate.settings.get_settings_path",
        lambda: settings_path,
    )

    result = _api_instance().get_translation_settings()

    assert result["success"]
    assert result["settings"]["temperature"] is None
    assert result["settings"]["topP"] is None


def test_gui_translation_settings_can_clear_temperature_override(
    tmp_path, monkeypatch
) -> None:
    from whisperjav.translate.settings import DEFAULT_SETTINGS

    settings_path = tmp_path / "settings.json"
    data = DEFAULT_SETTINGS.copy()
    data["model_params"] = {"temperature": 0.5, "top_p": 0.9}
    _write_translate_settings(settings_path, data)
    monkeypatch.setattr(
        "whisperjav.translate.settings.get_settings_path",
        lambda: settings_path,
    )

    result = _api_instance().save_translation_settings(
        {
            "tone": "contextual",
            "temperature": None,
            "topP": None,
        }
    )

    assert result["success"]
    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    assert saved["tone"] == "contextual"
    assert saved["model_params"]["temperature"] is None
    assert saved["model_params"]["top_p"] is None


def test_gui_exposes_contextual_tone_without_frontend_temperature_map() -> None:
    repo = Path(__file__).resolve().parents[1]
    index_html = (repo / "whisperjav/webview_gui/assets/index.html").read_text(
        encoding="utf-8"
    )
    app_js = (repo / "whisperjav/webview_gui/assets/app.js").read_text(
        encoding="utf-8"
    )

    assert 'value="contextual"' in index_html
    assert "toneTemperatureDefaults" not in app_js
    assert "contextual: 0.8" not in app_js
