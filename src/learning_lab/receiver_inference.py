"""Saved receiver inference for the declared independent, circular 64-symbol blocks."""

from pathlib import Path

import numpy as np
import torch

from .signals import Equalizer, constellation


@torch.no_grad()
def receive(checkpoint, observations, device="cpu", batch_size=256, allow_tf32=True):
    values = np.asarray(observations)
    if (
        values.ndim != 3
        or values.shape[1:] != (2, 64)
        or not len(values)
        or values.dtype.kind != "f"
        or not np.isfinite(values).all()
    ):
        raise ValueError("Finite floating N x 2 x 64 received blocks required")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Positive integer inference batch size required")
    if not isinstance(allow_tf32, bool):
        raise ValueError("allow_tf32 must be boolean")
    bundle = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    kind = bundle["kind"]
    if kind not in ["mlp", "cnn"]:
        raise ValueError("Unsupported receiver architecture")
    model = Equalizer(kind).to(device).eval()
    model.load_state_dict(bundle["state_dict"], strict=True)
    # Match the deterministic convolution setting used in the reference fitting.
    benchmark, deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    convolution_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    logits = []
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        for start in range(0, len(values), batch_size):
            x = torch.tensor(values[start : start + batch_size], device=device, dtype=torch.float32)
            if not torch.isfinite(x).all():
                raise ValueError("Received blocks exceed float32 numerical range")
            predicted = model(x)
            if not torch.isfinite(predicted).all():
                raise FloatingPointError("Saved receiver produced nonfinite logits")
            logits.append(predicted.cpu())
    finally:
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.allow_tf32 = convolution_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    logits = torch.cat(logits)
    symbols = logits.argmax(-1).numpy()
    _, labels = constellation()
    return {
        "symbol_classes": symbols,
        "gray_bits": labels[symbols].astype(np.uint8),
        "probabilities": logits.softmax(-1).numpy(),
    }
