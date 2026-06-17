from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToneConfig:
    key: str
    label: str
    description: str
    temperature: float
    top_p: float = 0.9


TONE_CONFIGS: dict[str, ToneConfig] = {
    "standard": ToneConfig(
        key="standard",
        label="Standard",
        description="faithful translation with adult context preserved",
        temperature=0.5,
    ),
    "contextual": ToneConfig(
        key="contextual",
        label="Adult (contextual)",
        description="explicit only where the original is explicit",
        temperature=0.8,
    ),
    "pornify": ToneConfig(
        key="pornify",
        label="Adult/Explicit",
        description="intentionally sexualized adult style",
        temperature=1.2,
    ),
}

TONE_CHOICES = tuple(TONE_CONFIGS)


def get_tone_config(tone: str | None) -> ToneConfig:
    return TONE_CONFIGS.get(tone or "standard", TONE_CONFIGS["standard"])


def get_tone_default_options(tone: str | None) -> dict[str, float]:
    config = get_tone_config(tone)
    return {
        "temperature": config.temperature,
        "top_p": config.top_p,
    }
