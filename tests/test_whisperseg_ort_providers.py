from whisperjav.modules.speech_segmentation.backends.whisperseg import (
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


def test_selects_cpu_provider_when_no_supported_gpu_provider_exists():
    providers, device = _select_ort_providers(["CPUExecutionProvider"], force_cpu=False)

    assert providers == ["CPUExecutionProvider"]
    assert device == "CPU"
