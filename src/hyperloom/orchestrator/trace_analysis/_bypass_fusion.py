###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Kernel-fusion opportunity analysis for the bypass analysis backend."""

from __future__ import annotations

from typing import Any

# Point-wise / memory-bound categories that fuse well (chain -> one launch).
_FUSABLE_CATEGORIES = frozenset({"Elementwise", "Normalization", "Quantization", "KVCacheStore"})


def _fusable_cluster_durations(launches: list[dict[str, Any]], *, min_len: int = 2) -> list[float]:
    """Summed device time of each maximal run of consecutive fusable-category launches."""
    clusters: list[float] = []
    run: list[dict[str, Any]] = []
    for lc in [*launches, {}]:
        if str(lc.get("category") or "") in _FUSABLE_CATEGORIES:
            run.append(lc)
            continue
        if len(run) >= min_len:
            clusters.append(round(sum(float(r.get("dur") or 0.0) for r in run), 3))
        run = []
    return clusters


def analyze_fusion(launches: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the fusion analysis totals from a time-ordered launch sequence."""
    clusters = _fusable_cluster_durations(launches)
    return {
        "launch_count": len(launches),
        "fusable_cluster_count": len(clusters),
        "fusable_time_us": round(sum(clusters), 3),
    }


__all__ = ["analyze_fusion"]
