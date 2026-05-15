"""Runtime compatibility hooks for RoboTwin evaluation jobs."""

from __future__ import annotations

import types


def _patch_warp_torch_namespace() -> None:
    try:
        import warp as wp
    except Exception:
        return

    if hasattr(wp, "torch"):
        return

    torch_ns = types.SimpleNamespace()
    for name in (
        "device_from_torch",
        "device_to_torch",
        "dtype_from_torch",
        "dtype_to_torch",
        "from_torch",
        "to_torch",
        "stream_from_torch",
        "stream_to_torch",
    ):
        if hasattr(wp, name):
            setattr(torch_ns, name, getattr(wp, name))

    wp.torch = torch_ns


_patch_warp_torch_namespace()
