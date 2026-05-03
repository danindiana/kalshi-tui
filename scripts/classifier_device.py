"""
classifier_device.py — pick CUDA or CPU for classifier inference at runtime.

XGBoost booster allocation during unpickling reads `CUDA_VISIBLE_DEVICES`, so
this module must be called BEFORE the classifier pickle is loaded (i.e. before
xgboost is imported through the unpickle chain).

Why we need this: the classifier was trained with `device="cuda"`. On load,
XGBoost allocates a CUDA context and copies the booster to VRAM. If another
process is using the GPU (e.g. Ollama holding a large LLM), the load raises
`cudaErrorMemoryAllocation: out of memory` — as happened 2026-04-14 17:30
CDT when qwen2.5-coder:14b was resident.

Inference workload: ~15 per-strike `predict_proba` calls per 10-minute tick.
That is trivial on CPU (sub-millisecond each), so the CPU fallback is not a
performance compromise — it's strictly the safer default when VRAM is tight.
"""
from __future__ import annotations

import os
import subprocess


def pick_device(min_free_mib: int = 1500) -> str:
    """Return 'cuda' or 'cpu'. If no GPU has at least `min_free_mib` MiB free,
    set `CUDA_VISIBLE_DEVICES=""` so subsequent XGBoost imports bind to CPU.

    Safe to call repeatedly; a no-op if already set to ''."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return "cpu"

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=3,
        )
        frees = [int(x.strip()) for x in out.strip().split("\n") if x.strip()]
    except Exception:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"

    if not frees or max(frees) < min_free_mib:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    return "cuda"
