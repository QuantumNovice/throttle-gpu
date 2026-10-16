from throttle_gpu import sources


def test_default_source_falls_back_to_nvidia_smi_when_nvml_is_unavailable(monkeypatch):
    class FakeNvidiaSmiSource:
        pass

    def unavailable_nvml():
        raise ImportError("pynvml is not installed")

    monkeypatch.setattr(sources, "NvmlSource", unavailable_nvml)
    monkeypatch.setattr(sources, "NvidiaSmiSource", FakeNvidiaSmiSource)

    assert isinstance(sources.default_source(), FakeNvidiaSmiSource)
