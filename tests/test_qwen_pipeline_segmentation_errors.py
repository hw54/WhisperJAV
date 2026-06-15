import pytest

from whisperjav.pipelines.qwen_pipeline import _segment_vad_scenes


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
