from __future__ import annotations

import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from typing import TextIO

from .commands import redact_command
from .models import ProcessResult

TAIL_LINES = 80
TERMINATE_TIMEOUT_SECONDS = 2.0
KILL_TIMEOUT_SECONDS = 2.0


def completed_process_result(
    command: Sequence[str],
    *,
    return_code: int | None,
    seconds: float,
    stdout_lines: Iterable[str] = (),
    stderr_lines: Iterable[str] = (),
    api_key_source: str | None = None,
) -> ProcessResult:
    return ProcessResult(
        seconds=seconds,
        return_code=return_code,
        command_redacted=redact_command(list(command)),
        api_key_source=api_key_source,
        stdout_tail=_tail_text(stdout_lines),
        stderr_tail=_tail_text(stderr_lines),
    )


class SubprocessRunner:
    def __init__(self, *, stream: bool = False) -> None:
        self.stream = stream
        self._active_processes: set[subprocess.Popen[str]] = set()
        self._lock = threading.Lock()

    def run(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        command_list = list(command)
        stdout_tail: deque[str] = deque(maxlen=TAIL_LINES)
        stderr_tail: deque[str] = deque(maxlen=TAIL_LINES)
        started = time.perf_counter()
        process = subprocess.Popen(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        with self._lock:
            self._active_processes.add(process)

        stdout_thread = threading.Thread(
            target=_read_stream,
            args=(process.stdout, stdout_tail, sys.stdout if self.stream else None),
        )
        stderr_thread = threading.Thread(
            target=_read_stream,
            args=(process.stderr, stderr_tail, sys.stderr if self.stream else None),
        )
        stdout_thread.start()
        stderr_thread.start()
        try:
            return_code = process.wait()
            stdout_thread.join()
            stderr_thread.join()
        finally:
            with self._lock:
                self._active_processes.discard(process)

        return completed_process_result(
            command_list,
            return_code=return_code,
            seconds=time.perf_counter() - started,
            stdout_lines=stdout_tail,
            stderr_lines=stderr_tail,
        )

    def terminate_all(self) -> None:
        with self._lock:
            processes = tuple(self._active_processes)

        for process in processes:
            if process.poll() is None:
                process.terminate()

        deadline = time.monotonic() + TERMINATE_TIMEOUT_SECONDS
        for process in processes:
            if process.poll() is not None:
                continue
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass

        for process in processes:
            if process.poll() is None:
                process.kill()

        for process in processes:
            if process.poll() is None:
                process.wait(timeout=KILL_TIMEOUT_SECONDS)


def _read_stream(
    stream: TextIO | None,
    tail: deque[str],
    mirror: TextIO | None,
) -> None:
    if stream is None:
        return
    for line in iter(stream.readline, ""):
        tail.append(line.rstrip("\n"))
        if mirror is not None:
            mirror.write(line)
            mirror.flush()
    stream.close()


def _tail_text(lines: Iterable[str]) -> str:
    tail = deque((line.rstrip("\n") for line in lines), maxlen=TAIL_LINES)
    return "\n".join(tail)
