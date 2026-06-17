from whisperjav.translate.core import (
    _patch_deepseek_thinking_request,
    _resolve_deepseek_thinking,
)


def test_deepseek_v4_flash_defaults_to_disabled_thinking() -> None:
    mode = _resolve_deepseek_thinking(
        {"pysubtrans_name": "DeepSeek"},
        "deepseek-v4-flash",
        {},
    )

    assert mode == "disabled"


def test_deepseek_thinking_default_override_leaves_api_default() -> None:
    provider_options = {"deepseek_thinking": "default"}

    mode = _resolve_deepseek_thinking(
        {"pysubtrans_name": "DeepSeek"},
        "deepseek-v4-flash",
        provider_options,
    )

    assert mode is None
    assert provider_options == {}


def test_deepseek_thinking_enabled_override_is_respected() -> None:
    provider_options = {"deepseek_thinking": "enabled"}

    mode = _resolve_deepseek_thinking(
        {"pysubtrans_name": "DeepSeek"},
        "deepseek-v4-flash",
        provider_options,
    )

    assert mode == "enabled"
    assert provider_options == {}


def test_deepseek_thinking_patch_injects_request_body_field() -> None:
    class Client:
        def _generate_request_body(self, request, temperature):
            return {"model": "deepseek-v4-flash", "temperature": temperature}

    class Translator:
        client = Client()

    _patch_deepseek_thinking_request(Translator(), "disabled", debug=False)

    body = Translator.client._generate_request_body(object(), 0.5)

    assert body["thinking"] == {"type": "disabled"}
    assert body["temperature"] == 0.5
