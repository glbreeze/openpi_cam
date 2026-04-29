import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file
from safetensors.torch import save_file


DEFAULT_PATTERNS = (
    "paligemma_with_expert.ray_embed.",
    "paligemma_with_expert.cross_view_fusion.",
    "aux_point_head.",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", required=True)
    parser.add_argument("--dst-dir", required=True)
    parser.add_argument("--patterns", nargs="*", default=list(DEFAULT_PATTERNS))
    parser.add_argument("--copy-optimizer", action="store_true")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    src_model = src_dir / "model.safetensors"
    src_meta = src_dir / "metadata.pt"
    src_assets = src_dir / "assets"

    if not src_model.exists():
        raise FileNotFoundError(f"Missing source model: {src_model}")
    if not src_meta.exists():
        raise FileNotFoundError(f"Missing source metadata: {src_meta}")
    if not src_assets.exists():
        raise FileNotFoundError(f"Missing source assets: {src_assets}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_meta, dst_dir / "metadata.pt")

    if (dst_dir / "assets").exists():
        shutil.rmtree(dst_dir / "assets")
    shutil.copytree(src_assets, dst_dir / "assets", symlinks=True)

    if args.copy_optimizer and (src_dir / "optimizer.pt").exists():
        shutil.copy2(src_dir / "optimizer.pt", dst_dir / "optimizer.pt")

    state = load_file(str(src_model))
    zeroed = []
    total_numel = 0
    for key, tensor in state.items():
        if any(p in key for p in args.patterns):
            tensor.zero_()
            zeroed.append(key)
            total_numel += tensor.numel()

    dst_model = dst_dir / "model.safetensors"
    save_file(state, str(dst_model))

    summary = {
        "src_dir": str(src_dir),
        "dst_dir": str(dst_dir),
        "patterns": list(args.patterns),
        "zeroed_tensors": len(zeroed),
        "zeroed_numel": int(total_numel),
        "zeroed_keys": zeroed,
    }
    (dst_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote ablated checkpoint to: {dst_dir}")
    print(f"Zeroed tensors: {len(zeroed)}")
    print(f"Zeroed numel: {total_numel}")
    for key in zeroed[:20]:
        print(f"  - {key}")
    if len(zeroed) > 20:
        print(f"  ... and {len(zeroed) - 20} more")


if __name__ == "__main__":
    main()
