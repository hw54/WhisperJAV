from whisperjav.ensemble import orchestrator as orchestrator_module
from whisperjav.ensemble.orchestrator import (
    EnsembleOrchestrator,
    _pass_uses_whisperseg,
    _requires_rocm_process_isolation,
)


def _build_orchestrator(tmp_path, **kwargs):
    return EnsembleOrchestrator(
        output_dir=str(tmp_path / "out"),
        temp_dir=str(tmp_path / "tmp"),
        keep_temp_files=True,
        subs_language="native",
        **kwargs,
    )


def test_qwen_default_requires_whisperseg_isolation():
    assert _pass_uses_whisperseg({"pipeline": "qwen"})
    assert _pass_uses_whisperseg(
        {"pipeline": "qwen", "qwen_params": {"generator_backend": "anime-whisper"}}
    )
    assert not _pass_uses_whisperseg(
        {"pipeline": "qwen", "speech_segmenter": "silero-v6.2"}
    )
    assert not _pass_uses_whisperseg(
        {"pipeline": "qwen", "qwen_params": {"qwen_segmenter": "none"}}
    )
    assert _pass_uses_whisperseg(
        {"pipeline": "qwen", "qwen_params": {"segmenter": "none"}}
    )


def test_rocm_isolation_requires_rocm_runtime_and_onnx_provider(monkeypatch):
    pass_config = {"pipeline": "balanced", "speech_segmenter": "whisperseg"}

    monkeypatch.setattr(orchestrator_module, "_torch_rocm_runtime_available", lambda: True)
    monkeypatch.setattr(orchestrator_module, "_rocm_onnx_provider_available", lambda: True)
    assert _requires_rocm_process_isolation(pass_config, None)

    monkeypatch.setattr(orchestrator_module, "_rocm_onnx_provider_available", lambda: False)
    assert not _requires_rocm_process_isolation(pass_config, None)


def test_rocm_whisperseg_batch_routes_to_serial_process_isolation(tmp_path, monkeypatch):
    orchestrator = _build_orchestrator(tmp_path)
    media_files = [
        {"basename": "first", "path": str(tmp_path / "first.mp4")},
        {"basename": "second", "path": str(tmp_path / "second.mp4")},
    ]
    pass_config = {"pipeline": "qwen", "speech_segmenter": "whisperseg"}
    captured = {}

    monkeypatch.setattr(orchestrator_module, "_torch_rocm_runtime_available", lambda: True)
    monkeypatch.setattr(orchestrator_module, "_rocm_onnx_provider_available", lambda: True)

    def fake_serial(media_files_arg, pass1_config_arg, pass2_config_arg, merge_strategy_arg):
        captured["media_files"] = media_files_arg
        captured["pass1_config"] = pass1_config_arg
        captured["pass2_config"] = pass2_config_arg
        captured["merge_strategy"] = merge_strategy_arg
        return [{"status": "completed"}]

    monkeypatch.setattr(orchestrator, "_process_batch_serial", fake_serial)

    results = orchestrator.process_batch(
        media_files=media_files,
        pass1_config=pass_config,
        pass2_config=None,
        merge_strategy="pass1_primary",
    )

    assert results == [{"status": "completed"}]
    assert captured["media_files"] == media_files
    assert captured["pass1_config"] == pass_config
    assert captured["pass2_config"] is None
    assert captured["merge_strategy"] == "pass1_primary"


def test_rocm_batch_can_be_explicitly_allowed(tmp_path, monkeypatch):
    orchestrator = _build_orchestrator(tmp_path, allow_rocm_batch=True)
    media_files = [
        {"basename": "first", "path": str(tmp_path / "first.mp4")},
        {"basename": "second", "path": str(tmp_path / "second.mp4")},
    ]
    calls = []

    monkeypatch.setattr(orchestrator_module, "_torch_rocm_runtime_available", lambda: True)
    monkeypatch.setattr(orchestrator_module, "_rocm_onnx_provider_available", lambda: True)

    def fake_run_pass(*, pass_number, media_files, pass_config, language_code):
        calls.append((pass_number, list(media_files)))
        return {
            info["basename"]: {
                "status": "failed",
                "error": "stop before merge",
            }
            for info in media_files
        }

    monkeypatch.setattr(orchestrator, "_run_pass_in_subprocess", fake_run_pass)

    results = orchestrator.process_batch(
        media_files=media_files,
        pass1_config={"pipeline": "qwen", "speech_segmenter": "whisperseg"},
        pass2_config=None,
        merge_strategy="pass1_primary",
    )

    assert len(calls) == 1
    assert calls[0][0] == 1
    assert [item["basename"] for item in calls[0][1]] == ["first", "second"]
    assert [item["status"] for item in results] == ["failed", "failed"]
