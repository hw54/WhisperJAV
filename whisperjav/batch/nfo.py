from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

SPLIT_RE = re.compile(r"[,;、，\n\r]+")
MAX_MOVIE_PLOT_CONTEXT_CHARS = 500
TITLE_FIELDS = ("title", "originaltitle", "sorttitle")
PLOT_FIELDS = ("plot", "outline", "tagline")


@dataclass(frozen=True)
class NfoLookupResult:
    path: Path | None
    reason: str | None = None


@dataclass(frozen=True)
class NfoActressResult:
    actresses: tuple[str, ...]
    error: str | None = None


@dataclass(frozen=True)
class NfoMetadataResult:
    actresses: tuple[str, ...]
    movie_title: str | None = None
    movie_plot: str | None = None
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
    metadata = extract_metadata_from_nfo(nfo_path)
    return NfoActressResult(actresses=metadata.actresses, error=metadata.error)


def extract_metadata_from_nfo(nfo_path: Path) -> NfoMetadataResult:
    try:
        root = ET.parse(nfo_path).getroot()
    except (OSError, ET.ParseError) as exc:
        return NfoMetadataResult(actresses=(), error=str(exc))

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
    return NfoMetadataResult(
        actresses=tuple(_dedupe(values)),
        movie_title=_first_nonempty_text(root, TITLE_FIELDS),
        movie_plot=_truncated_context(_first_nonempty_text(root, PLOT_FIELDS)),
    )


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


def _first_nonempty_text(root: ET.Element, field_names: tuple[str, ...]) -> str | None:
    wanted = set(field_names)
    by_name = {field: [] for field in field_names}
    for element in root.iter():
        name = _local_name(element.tag)
        if name in wanted:
            by_name[name].append(_normalize_context_text(element.text or ""))
    for field in field_names:
        for value in by_name[field]:
            if value:
                return value
    return None


def _normalize_context_text(value: str) -> str:
    return " ".join(value.split())


def _truncated_context(value: str | None) -> str | None:
    if value is None:
        return None
    return value[:MAX_MOVIE_PLOT_CONTEXT_CHARS]
