from __future__ import annotations

import io
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class MockStderr(io.StringIO):
    encoding = "utf-8"


def test_translate_subtitle_fails_partial_translation(tmp_path) -> None:
    from whisperjav.translate.core import translate_subtitle

    input_srt = tmp_path / "input.srt"
    output_srt = tmp_path / "input.chinese.srt"
    input_srt.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nはい\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nいいえ\n",
        encoding="utf-8",
    )

    mock_project = MagicMock()
    mock_project.events.batch_translated.connect = MagicMock()
    mock_project.existing_project = False
    mock_project.subtitles = SimpleNamespace(
        outputpath=str(output_srt),
        linecount=2,
        any_translated=True,
        all_translated=False,
    )

    def save_translation(path=None):
        target = output_srt if path is None else path
        output_srt.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\n是\n",
            encoding="utf-8",
        )
        return str(target)

    mock_project.SaveTranslation.side_effect = save_translation
    mock_project.SaveProject = MagicMock()
    mock_project.TranslateSubtitles = MagicMock()

    mock_translator = MagicMock()
    mock_translator.events = MagicMock()
    mock_provider = MagicMock()
    mock_provider.ValidateSettings.return_value = True

    captured_stderr = MockStderr()
    old_stderr = sys.stderr
    try:
        sys.stderr = captured_stderr
        with patch("PySubtrans.init_options") as init_options, \
                patch("PySubtrans.init_translation_provider") as init_provider, \
                patch("PySubtrans.init_project") as init_project, \
                patch("PySubtrans.init_translator") as init_translator:
            init_options.return_value = {}
            init_provider.return_value = mock_provider
            init_project.return_value = mock_project
            init_translator.return_value = mock_translator

            result = translate_subtitle(
                input_path=str(input_srt),
                output_path=output_srt,
                provider_config={"pysubtrans_name": "DeepSeek"},
                model="deepseek-v4-flash",
                api_key="test-key",
                target_lang="chinese",
                emit_raw_output=False,
            )
    finally:
        sys.stderr = old_stderr

    assert result is None
    stderr = captured_stderr.getvalue()
    assert "All subtitles translated: NO" in stderr
    assert "TRANSLATION FAILED" in stderr
