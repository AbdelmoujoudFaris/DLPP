import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a device string to a concrete torch.device.

    "auto" prefers CUDA, then Apple MPS, then CPU. An explicit request
    ("cpu", "cuda", "mps") is honored as-is (and will raise from torch
    if unavailable), so misconfiguration surfaces immediately.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
