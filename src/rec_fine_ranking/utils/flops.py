"""Light wrapper over fvcore's FlopCountAnalysis with a graceful manual fallback.

`count_mflops` returns MFLOPs per sample, excluding embedding lookups (embedding
gathers are not matmuls and fvcore does not count them, which matches the budget
definition: FLOPs/sample excludes the embedding-table lookup).

The models are called with a single dict argument (``BaseRanker.forward(batch)``),
so fvcore is invoked with ``(batch,)`` as the positional input — fvcore forwards
that tuple straight into ``model.forward`` so the dict arrives intact. fvcore's
tracer cannot handle every op (e.g. some broadcasted einsum patterns raise an
``IndexError`` during graph construction), so any failure falls back to a manual
Linear-only FLOP count, guaranteeing a finite, positive return value.
"""
from __future__ import annotations

from typing import Dict

import torch

try:
    from fvcore.nn import FlopCountAnalysis

    _HAS_FVCORE = True
except Exception:  # pragma: no cover - exercised only when fvcore is absent
    _HAS_FVCORE = False


def _manual_linear_mflops(model: torch.nn.Module, batch_size: int) -> float:
    """Rough fallback: 2 * in * out FLOPs per nn.Linear, summed, per sample.

    Counts the whole-batch Linear cost then divides by the batch size so the
    returned value is per-sample and matches the fvcore path's normalisation.
    """
    flops = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            flops += 2 * m.in_features * m.out_features
    return flops * max(batch_size, 1) / 1e6 / max(batch_size, 1)


def count_mflops(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> float:
    """Return MFLOPs/sample for ``model`` on ``batch`` (excludes embedding lookups)."""
    batch_size = max(int(batch["label"].shape[0]), 1)

    if _HAS_FVCORE:
        try:
            model.eval()
            analysis = FlopCountAnalysis(model, (batch,))
            analysis.unsupported_ops_warnings(False)
            analysis.uncalled_modules_warnings(False)
            total = analysis.total()
            mflops = total / 1e6 / batch_size
            if mflops > 0 and mflops == mflops and mflops != float("inf"):
                return mflops
            # fvcore returned 0 / NaN / inf -> fall through to manual count.
        except Exception:
            # fvcore tracer failed on this model -> manual fallback.
            pass

    return _manual_linear_mflops(model, batch_size)
