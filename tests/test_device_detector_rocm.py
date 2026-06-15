from types import SimpleNamespace

from whisperjav.utils import device_detector


class _CudaStub:
    def __init__(self, available=True, name="AMD Radeon Graphics", count=1):
        self._available = available
        self._name = name
        self._count = count

    def is_available(self):
        return self._available

    def get_device_name(self, index):
        return self._name

    def device_count(self):
        return self._count


def _torch_stub(*, hip, cuda, name):
    return SimpleNamespace(
        cuda=_CudaStub(name=name),
        version=SimpleNamespace(hip=hip, cuda=cuda),
    )


def test_rocm_torch_build_is_reported_as_rocm_not_nvidia_cuda(monkeypatch):
    torch = _torch_stub(hip="7.2.0", cuda=None, name="AMD Radeon Graphics")
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    monkeypatch.setattr(device_detector.platform, "system", lambda: "Linux")

    assert device_detector._check_cuda_available() == (False, None)
    assert device_detector._check_rocm_available() == (True, "AMD Radeon Graphics")
    assert device_detector.get_best_device() == "cuda"

    info = device_detector.get_device_info()
    assert info["cuda"]["available"] is False
    assert info["rocm"] == {"available": True, "name": "AMD Radeon Graphics"}


def test_nvidia_cuda_build_is_not_reported_as_rocm(monkeypatch):
    torch = _torch_stub(hip=None, cuda="12.9", name="NVIDIA GeForce RTX")
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    monkeypatch.setattr(device_detector.platform, "system", lambda: "Linux")

    assert device_detector._check_cuda_available() == (True, "NVIDIA GeForce RTX")
    assert device_detector._check_rocm_available() == (False, None)
