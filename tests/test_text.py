from rosetta.util.text import (
    format_duration,
    format_timestamp,
    paragraphs,
    slugify,
    unique_path,
    word_count,
    wrap_caption,
)


def test_format_timestamp_srt_and_vtt():
    assert format_timestamp(3661.5) == "01:01:01,500"
    assert format_timestamp(75.25, separator=".") == "00:01:15.250"
    assert format_timestamp(-4) == "00:00:00,000"


def test_format_timestamp_rounds_to_milliseconds():
    assert format_timestamp(1.0004) == "00:00:01,000"
    assert format_timestamp(1.9999) == "00:00:02,000"


def test_format_duration():
    assert format_duration(12.34) == "12.3s"
    assert format_duration(95) == "1m 35s"
    assert format_duration(3785) == "1h 03m"


def test_slugify_keeps_readable_names_and_strips_unsafe_characters():
    assert slugify("  My Great: Podcast/Ep 3 ") == "My-Great-Podcast-Ep-3"
    assert slugify("///") == "transcript"
    assert len(slugify("x" * 200)) <= 80


def test_unique_path_avoids_collisions(tmp_path):
    first = unique_path(tmp_path, "show", "txt")
    first.write_text("a")
    assert unique_path(tmp_path, "show", "txt").name == "show-2.txt"


def test_wrap_caption_respects_line_limits():
    wrapped = wrap_caption("the quick brown fox jumps over the lazy dog and keeps running")
    assert len(wrapped.splitlines()) <= 2
    assert all(len(line) <= 46 for line in wrapped.splitlines())


def test_paragraphs_break_on_sentence_ends():
    texts = ["One sentence here.", "Another one follows.", "A third arrives."]
    # A paragraph closes on the first sentence end *after* the length target.
    assert paragraphs(texts, max_chars=20) == [
        "One sentence here. Another one follows.",
        "A third arrives.",
    ]
    assert paragraphs(texts, max_chars=10_000) == [" ".join(texts)]


def test_word_count():
    assert word_count("  two  words ") == 2
    assert word_count("") == 0
