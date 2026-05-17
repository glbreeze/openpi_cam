"""Extract Pi3X's `ray_embed.proj` weights into a small .pt file.

The dumped file is consumed by `Pi0Config.ray_embed_pi3x_init_path` to warm-start
openpi's `ray_embed` (`Conv2d(2, 1152, k=14, s=14)`) from Pi3X's pretrained
`Conv2d(2, 1024, k=14, s=14)`. The dimension mismatch (1024 -> 1152) is handled
on the openpi side: first 1024 output channels are filled from Pi3X (scaled by
`ray_embed_pi3x_init_scale`); the remaining 128 stay at zero.

Run once per Pi3X checkpoint version.

Usage:
    uv run scripts/dump_pi3x_ray_embed.py \\
        --pi3x-repo ~/Research/Pi3X_Libero \\
        --output    ~/.cache/openpi/pi3x_init/ray_embed.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

logger = logging.getLogger("dump_pi3x_ray_embed")


def _build_pi3x(pi3x_repo: Path, ckpt: str | None, device: torch.device):
    if str(pi3x_repo) not in sys.path:
        sys.path.insert(0, str(pi3x_repo))
    from pi3.models.pi3x import Pi3X  # noqa: PLC0415

    if ckpt:
        model = Pi3X(use_multimodal=True).eval()
        if ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file

            state = load_file(ckpt)
        else:
            state = torch.load(ckpt, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning("Pi3X load_state_dict: missing=%d unexpected=%d", len(missing), len(unexpected))
    else:
        model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pi3x-repo", type=Path, default=Path("~/Research/Pi3X_Libero").expanduser())
    parser.add_argument("--ckpt", type=str, default=None, help="Optional local Pi3X ckpt; defaults to HF yyfz233/Pi3X.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("~/.cache/openpi/pi3x_init/ray_embed.pt").expanduser(),
        help="Where to write the {weight, bias} .pt file.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    device = torch.device(args.device)
    logger.info("Loading Pi3X (device=%s)", device)
    model = _build_pi3x(args.pi3x_repo, args.ckpt, device)

    if not hasattr(model, "ray_embed") or not hasattr(model.ray_embed, "proj"):
        raise SystemExit("Pi3X model has no `ray_embed.proj`; expected PatchEmbed-style layer.")

    state = {
        "weight": model.ray_embed.proj.weight.detach().cpu().clone(),
        "bias": model.ray_embed.proj.bias.detach().cpu().clone(),
    }
    expected_shape = (1024, 2, 14, 14)
    if tuple(state["weight"].shape) != expected_shape:
        logger.warning(
            "Pi3X ray_embed.weight shape %s differs from expected %s; downstream loader may need adjustment.",
            tuple(state["weight"].shape),
            expected_shape,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    logger.info(
        "Wrote %s  weight=%s  bias=%s",
        args.output,
        tuple(state["weight"].shape),
        tuple(state["bias"].shape),
    )


if __name__ == "__main__":
    main()
