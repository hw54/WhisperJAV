from pathlib import Path

import pytest

from whisperjav.batch import amd_cli


def test_configure_amd_rocm_environment_sets_required_values(tmp_path) -> None:
    cache_home = tmp_path / "cache"
    env = {
        "PATH": "/usr/bin",
        "HSA_OVERRIDE_GFX_VERSION": "11.0.3",
        "XDG_CACHE_HOME": str(cache_home),
    }

    amd_cli.configure_amd_rocm_environment(env, python_bin=Path("/opt/whisperjav/bin"))

    expected_cache_path = cache_home / "whisperjav" / "migraphx"
    assert env["HSA_OVERRIDE_GFX_VERSION"] == "11.0.0"
    assert env["PYTHONFAULTHANDLER"] == "1"
    assert env["PATH"].split(":")[:2] == ["/opt/whisperjav/bin", "/usr/bin"]
    assert env["ORT_MIGRAPHX_MODEL_CACHE_PATH"] == str(expected_cache_path)
    assert expected_cache_path.is_dir()


def test_configure_amd_rocm_environment_preserves_existing_migraphx_cache_path(tmp_path) -> None:
    custom_cache = tmp_path / "custom-migraphx-cache"
    env = {
        "PATH": "/usr/bin",
        "ORT_MIGRAPHX_MODEL_CACHE_PATH": str(custom_cache),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
    }

    amd_cli.configure_amd_rocm_environment(env, python_bin=Path("/opt/whisperjav/bin"))

    assert env["ORT_MIGRAPHX_MODEL_CACHE_PATH"] == str(custom_cache)
    assert custom_cache.is_dir()


def test_amd_batch_main_passes_one_directory_to_batch_cli(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-key")
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.3")
    monkeypatch.setattr(amd_cli, "reexec_with_render_group_if_needed", lambda _argv, *, env: None)
    monkeypatch.setattr(
        amd_cli.batch_cli,
        "main",
        lambda argv: calls.append(list(argv)) or 0,
    )

    code = amd_cli.main([str(tmp_path)])

    assert code == 0
    assert calls == [
        [
            str(tmp_path),
            "--asr-retries",
            "1",
            "--translation-retries",
            "2",
            "--translate-workers",
            "20",
            "--translation-queue-size",
            "21",
        ]
    ]
    assert all("secret-key" not in part for part in calls[0])


def test_amd_batch_main_passes_multiple_directories_to_batch_cli(tmp_path, monkeypatch) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    third = tmp_path / "third"
    first.mkdir()
    second.mkdir()
    third.mkdir()
    calls: list[list[str]] = []
    monkeypatch.setattr(amd_cli, "reexec_with_render_group_if_needed", lambda _argv, *, env: None)
    monkeypatch.setattr(
        amd_cli.batch_cli,
        "main",
        lambda argv: calls.append(list(argv)) or 0,
    )

    code = amd_cli.main([str(first), str(second), str(third)])

    assert code == 0
    assert calls == [
        [
            str(first),
            str(second),
            str(third),
            "--asr-retries",
            "1",
            "--translation-retries",
            "2",
            "--translate-workers",
            "20",
            "--translation-queue-size",
            "21",
        ]
    ]


def test_amd_batch_main_passes_custom_translate_workers_to_batch_cli(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(amd_cli, "reexec_with_render_group_if_needed", lambda _argv, *, env: None)
    monkeypatch.setattr(
        amd_cli.batch_cli,
        "main",
        lambda argv: calls.append(list(argv)) or 0,
    )

    code = amd_cli.main([str(tmp_path), "--translate-workers", "1"])

    assert code == 0
    assert calls == [
        [
            str(tmp_path),
            "--asr-retries",
            "1",
            "--translation-retries",
            "2",
            "--translate-workers",
            "1",
            "--translation-queue-size",
            "2",
        ]
    ]


def test_amd_batch_main_accepts_twenty_translate_workers(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(amd_cli, "reexec_with_render_group_if_needed", lambda _argv, *, env: None)
    monkeypatch.setattr(
        amd_cli.batch_cli,
        "main",
        lambda argv: calls.append(list(argv)) or 0,
    )

    code = amd_cli.main([str(tmp_path), "--translate-workers", "20"])

    assert code == 0
    assert calls[0][calls[0].index("--translate-workers") + 1] == "20"
    assert calls[0][calls[0].index("--translation-queue-size") + 1] == "21"


def test_amd_batch_main_accepts_hundred_translate_workers(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(amd_cli, "reexec_with_render_group_if_needed", lambda _argv, *, env: None)
    monkeypatch.setattr(
        amd_cli.batch_cli,
        "main",
        lambda argv: calls.append(list(argv)) or 0,
    )

    code = amd_cli.main([str(tmp_path), "--translate-workers", "100"])

    assert code == 0
    assert calls[0][calls[0].index("--translate-workers") + 1] == "100"
    assert calls[0][calls[0].index("--translation-queue-size") + 1] == "101"


def test_amd_batch_main_passes_custom_translation_queue_size_to_batch_cli(
    tmp_path, monkeypatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(amd_cli, "reexec_with_render_group_if_needed", lambda _argv, *, env: None)
    monkeypatch.setattr(
        amd_cli.batch_cli,
        "main",
        lambda argv: calls.append(list(argv)) or 0,
    )

    code = amd_cli.main(
        [str(tmp_path), "--translate-workers", "2", "--translation-queue-size", "6"]
    )

    assert code == 0
    assert calls == [
        [
            str(tmp_path),
            "--asr-retries",
            "1",
            "--translation-retries",
            "2",
            "--translate-workers",
            "2",
            "--translation-queue-size",
            "6",
        ]
    ]


def test_amd_batch_main_passes_custom_duration_limit_to_batch_cli(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(amd_cli, "reexec_with_render_group_if_needed", lambda _argv, *, env: None)
    monkeypatch.setattr(
        amd_cli.batch_cli,
        "main",
        lambda argv: calls.append(list(argv)) or 0,
    )

    code = amd_cli.main([str(tmp_path), "--max-video-minutes", "180"])

    assert code == 0
    assert calls == [
        [
            str(tmp_path),
            "--asr-retries",
            "1",
            "--translation-retries",
            "2",
            "--translate-workers",
            "20",
            "--translation-queue-size",
            "21",
            "--max-video-minutes",
            "180",
        ]
    ]


def test_amd_batch_main_passes_stream_to_batch_cli_when_requested(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(amd_cli, "reexec_with_render_group_if_needed", lambda _argv, *, env: None)
    monkeypatch.setattr(
        amd_cli.batch_cli,
        "main",
        lambda argv: calls.append(list(argv)) or 0,
    )

    code = amd_cli.main([str(tmp_path), "--stream"])

    assert code == 0
    assert "--stream" in calls[0]


def test_reexecs_through_render_group_when_kfd_is_inaccessible(tmp_path, monkeypatch) -> None:
    class ExecCalled(Exception):
        def __init__(self, file, args, env):
            super().__init__(file)
            self.file = file
            self.exec_args = args
            self.env = env

    def fake_execvpe(file, args, env):
        raise ExecCalled(file, args, env)

    env = {"PATH": "/usr/bin"}
    monkeypatch.setattr(amd_cli.os, "access", lambda _path, _mode: False)
    monkeypatch.setattr(amd_cli.shutil, "which", lambda name: "/usr/bin/sg" if name == "sg" else None)
    monkeypatch.setattr(amd_cli.os, "execvpe", fake_execvpe)
    monkeypatch.setattr(amd_cli.sys, "executable", "/opt/whisperjav/bin/python")

    with pytest.raises(ExecCalled) as exc_info:
        amd_cli.reexec_with_render_group_if_needed([str(tmp_path)], env=env)

    assert exc_info.value.file == "sg"
    assert exc_info.value.exec_args[:3] == ["sg", "render", "-c"]
    assert "/opt/whisperjav/bin/python -m whisperjav.batch.amd_cli" in exc_info.value.exec_args[3]
    assert str(tmp_path) in exc_info.value.exec_args[3]
    assert exc_info.value.env["HSA_OVERRIDE_GFX_VERSION"] == "11.0.0"
    assert exc_info.value.env["WHISPERJAV_AMD_BATCH_SG"] == "1"
