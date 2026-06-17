from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
from collections.abc import MutableMapping, Sequence
from pathlib import Path

from . import cli as batch_cli

AMD_GFX_OVERRIDE = "11.0.0"
MIGRAPHX_MODEL_CACHE_ENV = "ORT_MIGRAPHX_MODEL_CACHE_PATH"
MIGRAPHX_MODEL_CACHE_SUBDIR = Path("whisperjav") / "migraphx"
AMD_BATCH_ARGS = (
    "--asr-retries",
    "1",
    "--translation-retries",
    "2",
)
AMD_DEFAULT_TRANSLATE_WORKERS = 20
AMD_TRANSLATION_QUEUE_AHEAD = 1
_REEXEC_MARKER = "WHISPERJAV_AMD_BATCH_SG"
_KFD_DEVICE = "/dev/kfd"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AMD APU one-command batch transcription and translation."
    )
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--max-video-minutes", type=_nonnegative_minutes)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument(
        "--translate-workers",
        type=_translate_workers,
        default=AMD_DEFAULT_TRANSLATE_WORKERS,
    )
    parser.add_argument("--translation-queue-size", type=_positive_int)
    return parser.parse_args(argv)


def configure_amd_rocm_environment(
    env: MutableMapping[str, str],
    *,
    python_bin: Path | None = None,
) -> None:
    bin_dir = python_bin or Path(sys.executable).resolve().parent
    _prepend_path(env, str(bin_dir))
    env["HSA_OVERRIDE_GFX_VERSION"] = AMD_GFX_OVERRIDE
    env["PYTHONFAULTHANDLER"] = "1"
    cache_path = env.get(MIGRAPHX_MODEL_CACHE_ENV)
    if cache_path is None:
        cache_path = _default_migraphx_model_cache_path(env)
        env[MIGRAPHX_MODEL_CACHE_ENV] = cache_path
    if cache_path:
        Path(cache_path).expanduser().mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv if argv is not None else sys.argv[1:])
    reexec_with_render_group_if_needed(raw_argv, env=os.environ)
    args = parse_args(raw_argv)
    configure_amd_rocm_environment(os.environ)
    batch_args = [
        *(str(root) for root in args.roots),
        *AMD_BATCH_ARGS,
        "--translate-workers",
        str(args.translate_workers),
        "--translation-queue-size",
        str(
            args.translation_queue_size
            or args.translate_workers + AMD_TRANSLATION_QUEUE_AHEAD
        ),
    ]
    if args.max_video_minutes is not None:
        batch_args.extend(["--max-video-minutes", args.max_video_minutes])
    if args.stream:
        batch_args.append("--stream")
    return batch_cli.main(batch_args)


def _prepend_path(env: MutableMapping[str, str], path: str) -> None:
    parts = [part for part in env.get("PATH", "").split(os.pathsep) if part]
    env["PATH"] = os.pathsep.join([path, *[part for part in parts if part != path]])


def _nonnegative_minutes(value: str) -> str:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--max-video-minutes must be a number") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("--max-video-minutes must be at least 0")
    return value


def _translate_workers(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--translate-workers must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("--translate-workers must be at least 1")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--translation-queue-size must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("--translation-queue-size must be at least 1")
    return parsed


def _default_migraphx_model_cache_path(env: MutableMapping[str, str]) -> str:
    cache_home = env.get("XDG_CACHE_HOME")
    if cache_home:
        return str(Path(cache_home) / MIGRAPHX_MODEL_CACHE_SUBDIR)

    home = env.get("HOME")
    if home:
        return str(Path(home) / ".cache" / MIGRAPHX_MODEL_CACHE_SUBDIR)

    return str(Path.home() / ".cache" / MIGRAPHX_MODEL_CACHE_SUBDIR)


def reexec_with_render_group_if_needed(
    argv: Sequence[str],
    *,
    env: MutableMapping[str, str],
) -> None:
    if env.get(_REEXEC_MARKER) == "1":
        return
    if os.access(_KFD_DEVICE, os.R_OK | os.W_OK):
        return
    if shutil.which("sg") is None:
        raise RuntimeError(
            "/dev/kfd is not readable/writable and the sg command is unavailable. "
            "Run from a session with render group access."
        )

    reexec_env = dict(env)
    configure_amd_rocm_environment(reexec_env)
    reexec_env[_REEXEC_MARKER] = "1"
    command = shlex.join([sys.executable, "-m", "whisperjav.batch.amd_cli", *argv])
    os.execvpe("sg", ["sg", "render", "-c", command], reexec_env)


if __name__ == "__main__":
    raise SystemExit(main())
