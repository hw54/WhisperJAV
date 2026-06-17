from pathlib import Path

import pytest

from whisperjav.batch import discovery
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


@pytest.fixture(autouse=True)
def short_media_duration(monkeypatch):
    monkeypatch.setattr(discovery, "_probe_duration_seconds", lambda _path: 60.0, raising=False)


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
    audio_only = tmp_path / "DEF-456.mp3"
    video.write_text("video")
    audio.write_text("audio")
    audio_only.write_text("audio")

    found = discover_media_files(tmp_path, include_audio=True)

    assert found == [video.resolve(), audio_only.resolve()]


def test_discovery_deduplicates_same_stem_media_in_same_directory(tmp_path):
    preferred = tmp_path / "ABC-123.mp4"
    duplicate = tmp_path / "ABC-123.mkv"
    preferred.write_text("video")
    duplicate.write_text("video")

    found = discover_media_files(tmp_path)

    assert found == [preferred.resolve()]


def test_discovery_keeps_different_stems_in_same_directory(tmp_path):
    first = tmp_path / "ABC-123.mp4"
    second = tmp_path / "DEF-456.mkv"
    first.write_text("video")
    second.write_text("video")

    found = discover_media_files(tmp_path)

    assert found == [first.resolve(), second.resolve()]


def test_discovery_keeps_same_stem_media_in_different_directories(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "ABC-123.mp4"
    second = second_dir / "ABC-123.mkv"
    first.write_text("video")
    second.write_text("video")

    found = discover_media_files(tmp_path)

    assert found == [first.resolve(), second.resolve()]


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


def test_extra_prefixed_japanese_suffix_is_external_subtitle(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    subtitle = tmp_path / "ABC-123.extra.ja.pass1.srt"
    video.write_text("video")
    write_valid_srt(subtitle, "はい")

    item = classify_video(video, BatchOptions(root=tmp_path))

    assert item.status == "skip_external_subtitle"
    assert item.external_subtitle == subtitle


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


def test_incomplete_existing_translation_is_retranslated(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    japanese = tmp_path / "ABC-123.ja.pass1.srt"
    chinese = tmp_path / "ABC-123.ja.pass1.chinese.srt"
    video.write_text("video")
    japanese.write_text(
        "1\n00:00:00,000 --> 00:00:30,000\nはい\n\n"
        "2\n00:01:50,000 --> 00:02:00,000\nはい\n",
        encoding="utf-8",
    )
    chinese.write_text(
        "1\n00:00:00,000 --> 00:00:30,000\n中文\n\n"
        "2\n00:00:50,000 --> 00:01:00,000\n中文\n",
        encoding="utf-8",
    )

    item = classify_video(video, BatchOptions(root=tmp_path))

    assert item.status == "translate_existing_japanese"
    assert item.reason == "invalid_existing_translation"
    assert item.chinese_srt is None


def test_no_reusable_srt_transcribes_then_translates(tmp_path):
    video = tmp_path / "ABC-123.mp4"
    video.write_text("video")

    item = classify_video(video, BatchOptions(root=tmp_path))

    assert item.status == "transcribe_then_translate"


def test_default_duration_limit_skips_videos_over_230_minutes(tmp_path, monkeypatch):
    video = tmp_path / "LONG-001.mp4"
    video.write_text("video")
    duration_seconds = 231 * 60
    monkeypatch.setattr(
        discovery,
        "_probe_duration_seconds",
        lambda _path: duration_seconds,
        raising=False,
    )

    item = classify_video(video, BatchOptions(root=tmp_path))

    assert item.status == "skip_duration_limit"
    assert item.reason == "duration_exceeds_limit"
    assert item.duration_seconds == duration_seconds
    assert item.duration_limit_minutes == 230


def test_zero_duration_limit_disables_duration_probe(tmp_path, monkeypatch):
    video = tmp_path / "LONG-001.mp4"
    video.write_text("video")
    calls = []

    def fail_if_called(_path):
        calls.append(_path)
        return 999 * 60

    monkeypatch.setattr(
        discovery,
        "_probe_duration_seconds",
        fail_if_called,
        raising=False,
    )

    item = classify_video(video, BatchOptions(root=tmp_path, max_video_minutes=0))

    assert item.status == "transcribe_then_translate"
    assert item.duration_seconds is None
    assert calls == []


def test_duration_probe_failure_is_failed_precondition(tmp_path, monkeypatch):
    video = tmp_path / "BROKEN-001.mp4"
    video.write_text("video")

    def raise_probe_error(_path):
        raise RuntimeError("ffprobe failed")

    monkeypatch.setattr(
        discovery,
        "_probe_duration_seconds",
        raise_probe_error,
        raising=False,
    )

    item = classify_video(video, BatchOptions(root=tmp_path))

    assert item.status == "failed_precondition"
    assert item.reason == "duration_unavailable"
    assert item.warnings == ("ffprobe failed",)


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
