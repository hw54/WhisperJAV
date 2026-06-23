import pytest

from whisperjav.modules.subtitle_pipeline import orchestrator
from whisperjav.pipelines.qwen_pipeline import _segment_vad_scenes, _write_scene_srt


class _FailingSegmenter:
    def __init__(self):
        self.cleaned = False

    def segment(self, _scene_path, sample_rate):
        raise RuntimeError("provider did not activate")

    def cleanup(self):
        self.cleaned = True


def test_segment_vad_scenes_raises_instead_of_full_scene_fallback():
    segmenter = _FailingSegmenter()

    with pytest.raises(RuntimeError, match="provider did not activate"):
        _segment_vad_scenes(
            segmenter=segmenter,
            vad_scene_paths=[("scene.wav", 0.0, 1.0, 1.0)],
        )

    assert segmenter.cleaned is True


def test_write_scene_srt_suppresses_stable_whisper_saved_output(tmp_path, capsys):
    class NoisyResult:
        segments = [object()]

        def to_srt_vtt(self, filepath, **_kwargs):
            print(f"Saved: {filepath}")
            print(f"Saved err: {filepath}", file=__import__("sys").stderr)
            tmp_path.joinpath("scene.srt").write_text(
                "1\n00:00:00,000 --> 00:00:01,000\ntext\n",
                encoding="utf-8",
            )

    output = tmp_path / "scene.srt"

    _write_scene_srt(NoisyResult(), output)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert output.exists()


def test_scene_generation_progress_uses_debug_not_info(monkeypatch):
    events = []

    class RecordingLogger:
        def debug(self, message, *args):
            events.append(("debug", message % args))

        def info(self, message, *args):
            events.append(("info", message % args))

    monkeypatch.setattr(orchestrator, "logger", RecordingLogger())

    orchestrator._log_scene_generation_progress(6, 120, 42.2)

    assert events == [
        ("debug", "[DecoupledPipeline] Generating scene 7/120 (42.2s audio)...")
    ]
