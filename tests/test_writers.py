import json

from rosetta.models import MediaInfo, OutputFormat, Segment, TranscriptionOptions, Word
from rosetta.writers import render, write_outputs


def test_txt_is_plain_prose(sample_result):
    text = render(OutputFormat.TXT, sample_result)
    assert "The first thing" in text
    assert "-->" not in text
    assert text.endswith("\n")


def test_srt_structure(sample_result):
    srt = render(OutputFormat.SRT, sample_result)
    blocks = [b for b in srt.split("\n\n") if b.strip()]
    assert len(blocks) == 3
    assert blocks[0].splitlines()[0] == "1"
    assert " --> " in blocks[0].splitlines()[1]
    assert "," in blocks[0].splitlines()[1]  # SRT uses a comma for milliseconds


def test_vtt_header_and_dot_separator(sample_result):
    vtt = render(OutputFormat.VTT, sample_result)
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> " in vtt


def test_long_segment_is_split_into_cues():
    segment = Segment(id=0, start=0.0, end=30.0, text=" ".join(["word"] * 90))
    result = _result([segment])
    cues = [b for b in render(OutputFormat.SRT, result).split("\n\n") if b.strip()]
    assert len(cues) > 1


def test_cue_split_uses_word_timings_when_available():
    words = [Word(start=i * 0.4, end=i * 0.4 + 0.4, word=f" word{i}") for i in range(60)]
    segment = Segment(id=0, start=0.0, end=24.0, text="".join(w.word for w in words), words=words)
    srt = render(OutputFormat.SRT, _result([segment]))
    assert "00:00:00,000 --> " in srt
    assert srt.count("-->") >= 3


def test_json_round_trips(sample_result):
    payload = json.loads(render(OutputFormat.JSON, sample_result, MediaInfo(source_url="u", title="T")))
    assert payload["transcript"]["language"] == "en"
    assert payload["media"]["title"] == "T"
    assert len(payload["segments"]) == 3
    assert payload["run"]["engine"] == "stub"


def test_markdown_has_front_matter_and_anchors(sample_result):
    md = render(OutputFormat.MD, sample_result, MediaInfo(source_url="u", title="Episode 4"))
    assert md.startswith("# Episode 4")
    assert "- **Model:** stub-1 (stub)" in md
    assert "**[00:00:00]**" in md


def test_tsv_is_tab_separated(sample_result):
    rows = render(OutputFormat.TSV, sample_result).strip().splitlines()
    assert rows[0] == "start\tend\ttext"
    assert rows[1].split("\t")[0] == "0"


def test_write_outputs_shares_one_stem(tmp_path, sample_result):
    options = TranscriptionOptions(formats=[OutputFormat.TXT, OutputFormat.SRT, OutputFormat.JSON])
    written = write_outputs(sample_result, None, options, tmp_path, "My-Show")
    assert set(written) == {"txt", "srt", "json"}
    assert {p.name for p in tmp_path.iterdir()} == {"My-Show.txt", "My-Show.srt", "My-Show.json"}

    again = write_outputs(sample_result, None, options, tmp_path, "My-Show")
    assert all("My-Show-2" in path for path in again.values())


def _result(segments):
    from rosetta.models import TranscriptResult

    return TranscriptResult(
        text=" ".join(s.text for s in segments), segments=segments, duration=30.0,
        engine="stub", model="stub-1", device="cpu",
    )


def test_cue_splitting_does_not_renumber_the_source_segments(sample_result):
    """SRT cue ids are independent of the transcript's own segment ids."""
    long_segment = Segment(id=0, start=0.0, end=40.0, text=" ".join(["word"] * 120))
    result = _result([long_segment, Segment(id=1, start=40.0, end=42.0, text="Short tail.")])
    render(OutputFormat.SRT, result)
    render(OutputFormat.VTT, result)
    assert [s.id for s in result.segments] == [0, 1]
    assert json.loads(render(OutputFormat.JSON, result))["segments"][1]["id"] == 1
