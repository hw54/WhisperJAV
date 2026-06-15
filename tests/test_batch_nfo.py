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
