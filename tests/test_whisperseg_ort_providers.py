from whisperjav.modules.speech_segmentation.backends.whisperseg import (
    _assert_requested_provider_active,
    _select_ort_providers,
)


def test_selects_cpu_when_forced_even_if_gpu_providers_are_available():
    providers, device = _select_ort_providers(
        ["CUDAExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"],
        force_cpu=True,
    )

    assert providers == ["CPUExecutionProvider"]
    assert device == "CPU"


def test_selects_cuda_provider_before_cpu():
    providers, device = _select_ort_providers(
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        force_cpu=False,
    )

    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert device == "GPU (CUDA)"


def test_selects_rocm_provider_before_cpu():
    providers, device = _select_ort_providers(
        ["ROCMExecutionProvider", "CPUExecutionProvider"],
        force_cpu=False,
    )

    assert providers == ["ROCMExecutionProvider", "CPUExecutionProvider"]
    assert device == "GPU (ROCm)"


def test_selects_migraphx_before_legacy_rocm_provider():
    providers, device = _select_ort_providers(
        [
            "MIGraphXExecutionProvider",
            "ROCMExecutionProvider",
            "CPUExecutionProvider",
        ],
        force_cpu=False,
    )

    assert providers == ["MIGraphXExecutionProvider", "CPUExecutionProvider"]
    assert device == "GPU (MIGraphX)"


def test_selects_cpu_provider_when_no_supported_gpu_provider_exists():
    providers, device = _select_ort_providers(["CPUExecutionProvider"], force_cpu=False)

    assert providers == ["CPUExecutionProvider"]
    assert device == "CPU"


def test_raises_when_requested_gpu_provider_falls_back_to_cpu():
    try:
        _assert_requested_provider_active(
            requested_providers=["ROCMExecutionProvider", "CPUExecutionProvider"],
            active_providers=["CPUExecutionProvider"],
            device_label="GPU (ROCm)",
        )
    except RuntimeError as exc:
        assert "ROCMExecutionProvider" in str(exc)
        assert "CPUExecutionProvider" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for silent ROCm-to-CPU fallback")


def test_accepts_cpu_when_cpu_was_requested():
    _assert_requested_provider_active(
        requested_providers=["CPUExecutionProvider"],
        active_providers=["CPUExecutionProvider"],
        device_label="CPU",
    )
