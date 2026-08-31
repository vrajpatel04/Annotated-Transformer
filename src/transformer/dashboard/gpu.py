"""GPU telemetry helpers."""

from __future__ import annotations


def get_gpu_temperature() -> float | None:
    """Return GPU temperature in Celsius, or None if unavailable."""
    try:
        import GPUtil

        gpus = GPUtil.getGPUs()
        if gpus:
            return float(gpus[0].temperature)
    except Exception:
        pass

    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        temp = pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        )
        pynvml.nvmlShutdown()
        return float(temp)
    except Exception:
        return None


def get_gpu_utilization() -> float | None:
    try:
        import GPUtil

        gpus = GPUtil.getGPUs()
        if gpus:
            return float(gpus[0].load * 100)
    except Exception:
        return None
    return None
