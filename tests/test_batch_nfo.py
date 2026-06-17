from whisperjav.batch.nfo import (
    MAX_MOVIE_PLOT_CONTEXT_CHARS,
    extract_actresses_from_nfo,
    extract_metadata_from_nfo,
    find_nfo_for_video,
)


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


def test_actor_with_empty_name_child_does_not_use_direct_text(tmp_path):
    nfo = tmp_path / "ABC-123.nfo"
    nfo.write_text("<movie><actor>Direct<name></name></actor></movie>", encoding="utf-8")

    result = extract_actresses_from_nfo(nfo)

    assert result.actresses == ()


def test_direct_actor_text_when_no_child_name(tmp_path):
    nfo = tmp_path / "ABC-123.nfo"
    nfo.write_text("<movie><actor>Direct Actor</actor></movie>", encoding="utf-8")

    result = extract_actresses_from_nfo(nfo)

    assert result.actresses == ("Direct Actor",)


def test_cast_fields_split_and_deduplicate_in_order(tmp_path):
    nfo = tmp_path / "ABC-123.nfo"
    nfo.write_text(
        "<movie><actress>A   One, B、C</actress>"
        "<cast>B; C，D\nE</cast>"
        "<performer>A One</performer></movie>",
        encoding="utf-8",
    )

    result = extract_actresses_from_nfo(nfo)

    assert result.actresses == ("A One", "B", "C", "D", "E")


def test_metadata_reads_title_and_plot_from_realistic_nfo(tmp_path):
    nfo = tmp_path / "ABC-123.nfo"
    nfo.write_text(
        "<movie>"
        "<title>ABC-123-制服美少女の誘惑</title>"
        "<originaltitle>Ignored Original</originaltitle>"
        "<plot>ABC-123#第一行\n第二行\t  第三行</plot>"
        "<outline>Ignored Outline</outline>"
        "<actor><name>Actor Name</name></actor>"
        "</movie>",
        encoding="utf-8",
    )

    result = extract_metadata_from_nfo(nfo)

    assert result.actresses == ("Actor Name",)
    assert result.movie_title == "ABC-123-制服美少女の誘惑"
    assert result.movie_plot == "ABC-123#第一行 第二行 第三行"
    assert result.error is None


def test_metadata_falls_back_to_originaltitle_and_outline(tmp_path):
    nfo = tmp_path / "ABC-123.nfo"
    nfo.write_text(
        "<movie>"
        "<title>   </title>"
        "<originaltitle>Original Title</originaltitle>"
        "<plot></plot>"
        "<outline>Outline text</outline>"
        "</movie>",
        encoding="utf-8",
    )

    result = extract_metadata_from_nfo(nfo)

    assert result.movie_title == "Original Title"
    assert result.movie_plot == "Outline text"


def test_metadata_truncates_long_plot(tmp_path):
    nfo = tmp_path / "ABC-123.nfo"
    long_plot = "あ" * (MAX_MOVIE_PLOT_CONTEXT_CHARS + 20)
    nfo.write_text(f"<movie><plot>{long_plot}</plot></movie>", encoding="utf-8")

    result = extract_metadata_from_nfo(nfo)

    assert result.movie_plot == "あ" * MAX_MOVIE_PLOT_CONTEXT_CHARS


def test_parse_error_is_reported(tmp_path):
    nfo = tmp_path / "broken.nfo"
    nfo.write_text("<movie><actor>", encoding="utf-8")

    result = extract_actresses_from_nfo(nfo)

    assert result.actresses == ()
    assert result.error is not None
