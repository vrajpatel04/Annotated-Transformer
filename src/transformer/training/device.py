"""Device selection helpers for training and evaluation."""

from __future__ import annotations

import torch


def resolve_device(device: str) -> torch.device:
    """Resolve a device string into a torch.device."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but no compatible GPU is available. "
            "Install a CUDA-enabled PyTorch build or use --device cpu."
        )
    return resolved


def describe_device(device: torch.device) -> str:
    """Return a human-readable description of the active device."""
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        memory_gib = torch.cuda.get_device_properties(index).total_memory / (1024**3)
        return f"cuda:{index} ({name}, {memory_gib:.1f} GiB)"
    return str(device)


def log_device_info(device: torch.device) -> None:
    """Print the selected device and CUDA runtime details when available."""
    print(f"Using device: {describe_device(device)}")
    if device.type == "cuda":
        print(f"PyTorch CUDA version: {torch.version.cuda}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
