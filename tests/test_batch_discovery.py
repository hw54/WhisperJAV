from pathlib import Path

import pytest

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


def test_discovery_deduplicates_canonical_paths(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    link = tmp_path / "ABC-123-link.mp4"
    video.write_text("video")
    try:
        link.symlink_to(video)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    found = discover_media_files(tmp_path, include_audio=False)

    assert found == [video.resolve()]


def test_external_same_basename_subtitle_blocks_video(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    subtitle = tmp_path / "ABC-123.zh.srt"
    video.write_text("video")
    write_valid_srt(subtitle)

    item = classify_video(video, BatchOptions(root=tmp_path))

    assert item.status == "skip_external_subtitle"
    assert item.external_subtitle == subtitle


def test_non_same_basename_subtitle_does_not_block_video(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    subtitle = tmp_path / "OTHER.zh.srt"
    video.write_text("video")
    write_valid_srt(subtitle)

    item = classify_video(video, BatchOptions(root=tmp_path))

    assert item.status == "transcribe_then_translate"
    assert item.external_subtitle is None


def test_whisperjav_japanese_srt_is_reused_for_translation(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    japanese = tmp_path / "ABC-123.ja.pass1.srt"
    video.write_text("video")
    write_valid_srt(japanese, "はい")

    item = classify_video(video, BatchOptions(root=tmp_path))

    assert item.status == "translate_existing_japanese"
    assert item.japanese_srt == japanese
    assert item.external_subtitle is None


def test_whisperjav_japanese_suffix_detection_is_case_insensitive(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    japanese = tmp_path / "ABC-123.ja.pass1.SRT"
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


def test_whisperjav_style_cn_translation_is_reusable(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    japanese = tmp_path / "ABC-123.ja.pass1.srt"
    chinese = tmp_path / "ABC-123.ja.pass1.cn.srt"
    video.write_text("video")
    write_valid_srt(japanese, "はい")
    write_valid_srt(chinese, "中文")

    item = classify_video(video, BatchOptions(root=tmp_path))

    assert item.status == "skip_translated"
    assert item.chinese_srt == chinese


def test_whisperjav_translation_with_glob_metacharacters_is_reusable(tmp_path):
    video = tmp_path / "ABC[123].mp4"
    chinese = tmp_path / "ABC[123].ja.pass1.chinese.srt"
    video.write_text("video")
    write_valid_srt(chinese, "中文")

    item = classify_video(video, BatchOptions(root=tmp_path))

    assert item.status == "skip_translated"
    assert item.chinese_srt == chinese


def test_whisperjav_translation_suffix_detection_is_case_insensitive(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    japanese = tmp_path / "ABC-123.ja.pass1.srt"
    chinese = tmp_path / "ABC-123.ja.pass1.CHINESE.SRT"
    video.write_text("video")
    write_valid_srt(japanese, "はい")
    write_valid_srt(chinese, "中文")

    item = classify_video(video, BatchOptions(root=tmp_path))

    assert item.status == "skip_translated"
    assert item.chinese_srt == chinese
    assert item.external_subtitle is None


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


def test_is_valid_srt_propagates_read_errors(tmp_path, monkeypatch):
    subtitle = tmp_path / "broken.srt"
    subtitle.write_text("", encoding="utf-8")

    def raise_os_error(self, *args, **kwargs):
        if self == subtitle:
            raise OSError("read failed")
        return original_read_text(self, *args, **kwargs)

    original_read_text = Path.read_text
    monkeypatch.setattr(Path, "read_text", raise_os_error)

    with pytest.raises(OSError, match="read failed"):
        is_valid_srt(subtitle)
