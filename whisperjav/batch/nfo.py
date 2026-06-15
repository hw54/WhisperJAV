from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

SPLIT_RE = re.compile(r"[,;、，\n\r]+")


@dataclass(frozen=True)
class NfoLookupResult:
    path: Path | None
    reason: str | None = None


@dataclass(frozen=True)
class NfoActressResult:
    actresses: tuple[str, ...]
    error: str | None = None


def find_nfo_for_video(video_path: Path) -> NfoLookupResult:
    exact = video_path.with_suffix(".nfo")
    if exact.exists():
        return NfoLookupResult(path=exact)
    nfos = sorted(video_path.parent.glob("*.nfo"))
    if len(nfos) == 1:
        return NfoLookupResult(path=nfos[0])
    if len(nfos) > 1:
        return NfoLookupResult(path=None, reason="nfo_ambiguous")
    return NfoLookupResult(path=None)


def extract_actresses_from_nfo(nfo_path: Path) -> NfoActressResult:
    try:
        root = ET.parse(nfo_path).getroot()
    except (OSError, ET.ParseError) as exc:
        return NfoActressResult(actresses=(), error=str(exc))

    values: list[str] = []
    for element in root.iter():
        name = _local_name(element.tag)
        if name == "actor":
            name_child = _first_child(element, "name")
            if name_child is not None:
                if name_child.text:
                    values.extend(_split_names(name_child.text))
            elif element.text:
                values.extend(_split_names(element.text))
        elif name in {"actress", "cast", "performer"} and element.text:
            values.extend(_split_names(element.text))
    return NfoActressResult(actresses=tuple(_dedupe(values)))


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _first_child(element: ET.Element, child_name: str) -> ET.Element | None:
    for child in list(element):
        if _local_name(child.tag) == child_name:
            return child
    return None


def _split_names(value: str) -> list[str]:
    return [_normalize(part) for part in SPLIT_RE.split(value) if _normalize(part)]


def _normalize(value: str) -> str:
    return " ".join(value.split())


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
