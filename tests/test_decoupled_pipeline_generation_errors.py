import numpy as np
import pytest
import soundfile as sf
import logging

from whisperjav.modules.subtitle_pipeline.orchestrator import DecoupledSubtitlePipeline
from whisperjav.modules.subtitle_pipeline.types import (
    FramingResult,
    HardeningConfig,
    TemporalFrame,
    TimestampMode,
)


class _SingleFrameFramer:
    def __init__(self, events=None):
        self._events = events

    def frame(self, _audio, _sample_rate, **_kwargs):
        return FramingResult(
            frames=[TemporalFrame(start=0.0, end=1.0, source="test-framer")]
        )

    def cleanup(self):
        if self._events is not None:
            self._events.append("framer.cleanup")


class _FailingGenerator:
    def __init__(self):
        self.loaded = False
        self.unloaded = False

    def load(self):
        self.loaded = True

    def unload(self):
        self.unloaded = True

    def cleanup(self):
        self.unload()

    def generate(self, *_args, **_kwargs):
        raise AssertionError("per-frame fallback must not run")

    def generate_batch(self, *_args, **_kwargs):
        raise RuntimeError("HIP error: unspecified launch failure")


class _Cleaner:
    def clean(self, text, **_kwargs):
        return text

    def clean_batch(self, texts, **_kwargs):
        return texts


class _PrecomputedOnlyFramer:
    def __init__(self):
        self.frame_called = False

    def frame(self, *_args, **_kwargs):
        self.frame_called = True
        raise AssertionError("precomputed VAD groups should avoid framer.frame()")

    def cleanup(self):
        pass


class _CountingGenerator:
    def __init__(self):
        self.loaded = False
        self.audio_counts = []

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def cleanup(self):
        self.unload()

    def generate(self, *_args, **_kwargs):
        raise AssertionError("batch path should be used")

    def generate_batch(self, *, audio_paths, **_kwargs):
        from whisperjav.modules.subtitle_pipeline.types import TranscriptionResult

        self.audio_counts.append(len(audio_paths))
        return [
            TranscriptionResult(text=f"テスト{idx + 1}", language="ja")
            for idx, _path in enumerate(audio_paths)
        ]


def test_generation_failure_propagates_instead_of_empty_subtitles(tmp_path):
    audio_path = tmp_path / "scene.wav"
    sf.write(str(audio_path), np.zeros(16000, dtype=np.float32), 16000)

    generator = _FailingGenerator()
    pipeline = DecoupledSubtitlePipeline(
        framer=_SingleFrameFramer(),
        generator=generator,
        cleaner=_Cleaner(),
        aligner=None,
        hardening_config=HardeningConfig(timestamp_mode=TimestampMode.VAD_ONLY),
    )

    with pytest.raises(RuntimeError, match="HIP error"):
        pipeline.process_scenes([audio_path], [1.0])

    assert generator.loaded is True
    assert generator.unloaded is True


def test_precomputed_vad_groups_skip_framer_segmentation(tmp_path):
    audio_path = tmp_path / "scene.wav"
    sf.write(str(audio_path), np.zeros(32000, dtype=np.float32), 16000)

    framer = _PrecomputedOnlyFramer()
    generator = _CountingGenerator()
    pipeline = DecoupledSubtitlePipeline(
        framer=framer,
        generator=generator,
        cleaner=_Cleaner(),
        aligner=None,
        hardening_config=HardeningConfig(timestamp_mode=TimestampMode.VAD_ONLY),
    )

    result = pipeline.process_scenes(
        [audio_path],
        [2.0],
        scene_speech_regions=[[(0.1, 0.4), (1.0, 1.4)]],
        scene_speech_groups=[[[(0.1, 0.4)], [(1.0, 1.4)]]],
    )

    assert framer.frame_called is False
    assert generator.audio_counts == [2]
    assert len(result) == 1
    assert result[0][1]["frame_count"] == 2


class _SuccessfulGenerator:
    def __init__(self, events):
        self._events = events

    def load(self):
        self._events.append("generator.load")

    def unload(self):
        self._events.append("generator.unload")

    def cleanup(self):
        self.unload()

    def generate(self, *_args, **_kwargs):
        raise AssertionError("batch path should be used")

    def generate_batch(self, *_args, **_kwargs):
        from whisperjav.modules.subtitle_pipeline.types import TranscriptionResult

        return [TranscriptionResult(text="テスト", language="ja")]


def test_framer_resources_are_released_before_generator_load(tmp_path):
    audio_path = tmp_path / "scene.wav"
    sf.write(str(audio_path), np.zeros(16000, dtype=np.float32), 16000)
    events = []

    pipeline = DecoupledSubtitlePipeline(
        framer=_SingleFrameFramer(events),
        generator=_SuccessfulGenerator(events),
        cleaner=_Cleaner(),
        aligner=None,
        hardening_config=HardeningConfig(timestamp_mode=TimestampMode.VAD_ONLY),
    )

    pipeline.process_scenes([audio_path], [1.0])

    assert events.index("framer.cleanup") < events.index("generator.load")


def test_per_scene_reconstruction_progress_is_not_info_log(tmp_path, caplog):
    audio_path = tmp_path / "scene.wav"
    sf.write(str(audio_path), np.zeros(16000, dtype=np.float32), 16000)

    pipeline = DecoupledSubtitlePipeline(
        framer=_SingleFrameFramer(),
        generator=_SuccessfulGenerator([]),
        cleaner=_Cleaner(),
        aligner=None,
        hardening_config=HardeningConfig(timestamp_mode=TimestampMode.VAD_ONLY),
    )

    with caplog.at_level(logging.INFO, logger="whisperjav"):
        pipeline.process_scenes([audio_path], [1.0])

    info_messages = [
        record.message for record in caplog.records
        if record.name == "whisperjav" and record.levelno == logging.INFO
    ]
    assert not any(
        message.startswith("[DecoupledPipeline] Scene 1/1:")
        for message in info_messages
    )
