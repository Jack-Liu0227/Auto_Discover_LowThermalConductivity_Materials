from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def setup_reproducibility(seed: int = 42, deterministic_torch: bool = True) -> dict[str, Any]:
    """Apply process-level reproducibility settings."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    # Required by some CUDA kernels for deterministic behavior.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    torch_applied = False
    torch_error: str | None = None
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch_applied = True
    except Exception as exc:  # pragma: no cover - optional dependency
        torch_error = str(exc)

    return {
        "seed": seed,
        "deterministic_torch": bool(deterministic_torch),
        "torch_applied": torch_applied,
        "torch_error": torch_error,
    }
