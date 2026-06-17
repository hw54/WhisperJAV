from whisperjav.translate.core import _remove_markdown_code_fences_from_srt_text


def test_remove_markdown_code_fences_from_srt_text() -> None:
    raw = (
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "换工作吧。\n"
        "```\n"
        "\n"
        "2\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "好舒服。 ```\n"
    )

    cleaned, removed = _remove_markdown_code_fences_from_srt_text(raw)

    assert removed == 2
    assert "```" not in cleaned
    assert "换工作吧。" in cleaned
    assert "好舒服。" in cleaned
