from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TextIO

from tqdm import tqdm

ProgressPhase = Literal["files", "asr", "translation"]


@dataclass(frozen=True)
class BatchProgressTotals:
    files: int
    asr: int
    translation: int


@dataclass(frozen=True)
class BatchProgressEvent:
    phase: ProgressPhase
    outcome: str
    video_path: Path


class BatchProgressReporter(Protocol):
    def start(self, totals: BatchProgressTotals) -> None:
        ...

    def advance(self, event: BatchProgressEvent) -> None:
        ...

    def message(self, text: str) -> None:
        ...

    def close(self) -> None:
        ...


class TqdmBatchProgress:
    def __init__(self, *, file: TextIO | None = None) -> None:
        self.file = file or sys.stderr
        self._bars: dict[ProgressPhase, tqdm] = {}
        self._counts: dict[ProgressPhase, Counter[str]] = {}

    def start(self, totals: BatchProgressTotals) -> None:
        specs: list[tuple[ProgressPhase, str, int]] = [
            ("files", "Files", totals.files),
            ("asr", "ASR", totals.asr),
            ("translation", "CN SRT", totals.translation),
        ]
        position = 0
        for phase, label, total in specs:
            if total <= 0:
                continue
            self._bars[phase] = tqdm(
                total=total,
                desc=label,
                unit="file",
                dynamic_ncols=True,
                leave=True,
                position=position,
                file=self.file,
            )
            self._counts[phase] = Counter()
            position += 1

    def advance(self, event: BatchProgressEvent) -> None:
        bar = self._bars.get(event.phase)
        if bar is None:
            return
        counts = self._counts[event.phase]
        counts[event.outcome] += 1
        bar.update(1)
        bar.set_postfix_str(_format_counts(counts), refresh=True)

    def message(self, text: str) -> None:
        tqdm.write(text, file=self.file)

    def close(self) -> None:
        for bar in reversed(tuple(self._bars.values())):
            bar.close()


def _format_counts(counts: Counter[str]) -> str:
    return " ".join(f"{key}={counts[key]}" for key in sorted(counts))
