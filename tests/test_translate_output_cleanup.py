from whisperjav.translate import core


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

    cleaned, removed = core._remove_markdown_code_fences_from_srt_text(raw)

    assert removed == 2
    assert "```" not in cleaned
    assert "换工作吧。" in cleaned
    assert "好舒服。" in cleaned


def test_remove_translator_notes_from_srt_text() -> None:
    raw = (
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "（注：此句疑似听写错误，根据上下文推测为“また行きたいね”）\n"
        "还想再去呢。\n\n"
        "2\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "稍微有点…（注：可能是“お暇”，有空）\n\n"
        "3\n"
        "00:00:02,000 --> 00:00:03,000\n"
        "（翻译修正为上一句的延续，或根据上下文判断为语气词）嗯……\n"
    )

    cleaned, removed = core._remove_translator_notes_from_srt_text(raw)

    assert removed == 3
    assert "注：" not in cleaned
    assert "听写错误" not in cleaned
    assert "翻译修正" not in cleaned
    assert "还想再去呢。" in cleaned
    assert "稍微有点…" in cleaned
    assert "嗯……" in cleaned
