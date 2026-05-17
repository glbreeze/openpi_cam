import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file


TRAINABLE_MODULE_PATTERNS = ("ray_embed", "cross_view_fusion", "aux_point_head")


def _stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    d = a - b

    dsq = float(torch.dot(d, d))
    asq = float(torch.dot(a, a))
    bsq = float(torch.dot(b, b))
    dot = float(torch.dot(a, b))

    l2_diff = math.sqrt(dsq)
    l2_a = math.sqrt(asq)
    l2_b = math.sqrt(bsq)
    cosine = dot / (l2_a * l2_b + 1e-12)
    rel_l2_vs_a = l2_diff / (l2_a + 1e-12)

    return {
        "numel": int(d.numel()),
        "l2_diff": l2_diff,
        "l2_a": l2_a,
        "l2_b": l2_b,
        "rel_l2_vs_a": rel_l2_vs_a,
        "cosine": cosine,
        "max_abs_diff": float(d.abs().max()) if d.numel() else 0.0,
        "mean_abs_diff": float(d.abs().mean()) if d.numel() else 0.0,
    }


def _summarize(sd_a: dict[str, torch.Tensor], sd_b: dict[str, torch.Tensor], keys: list[str]) -> dict:
    total_dsq = 0.0
    total_asq = 0.0
    total_bsq = 0.0
    total_dot = 0.0
    total_numel = 0
    max_abs_diff = 0.0
    worst_param = None
    per_param = []

    for key in keys:
        a = sd_a[key].float().reshape(-1)
        b = sd_b[key].float().reshape(-1)
        d = a - b

        dsq = float(torch.dot(d, d))
        asq = float(torch.dot(a, a))
        bsq = float(torch.dot(b, b))
        dot = float(torch.dot(a, b))

        l2_diff = math.sqrt(dsq)
        rel_l2_vs_a = l2_diff / (math.sqrt(asq) + 1e-12)
        param_max_abs = float(d.abs().max()) if d.numel() else 0.0

        per_param.append(
            {
                "name": key,
                "numel": int(d.numel()),
                "l2_diff": l2_diff,
                "rel_l2_vs_a": rel_l2_vs_a,
                "max_abs_diff": param_max_abs,
            }
        )

        if param_max_abs > max_abs_diff:
            max_abs_diff = param_max_abs
            worst_param = key

        total_dsq += dsq
        total_asq += asq
        total_bsq += bsq
        total_dot += dot
        total_numel += int(d.numel())

    total_l2_diff = math.sqrt(total_dsq)
    total_l2_a = math.sqrt(total_asq)
    total_l2_b = math.sqrt(total_bsq)
    cosine = total_dot / (total_l2_a * total_l2_b + 1e-12)

    return {
        "num_tensors": len(keys),
        "numel": total_numel,
        "l2_diff": total_l2_diff,
        "l2_a": total_l2_a,
        "l2_b": total_l2_b,
        "rel_l2_vs_a": total_l2_diff / (total_l2_a + 1e-12),
        "cosine": cosine,
        "max_abs_diff": max_abs_diff,
        "worst_param": worst_param,
        "top_params_by_l2_diff": sorted(per_param, key=lambda x: x["l2_diff"], reverse=True)[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-a", required=True)
    parser.add_argument("--ckpt-b", required=True)
    parser.add_argument("--label-a", default="zero_init")
    parser.add_argument("--label-b", default="pi3x_init")
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    sd_a = load_file(args.ckpt_a)
    sd_b = load_file(args.ckpt_b)

    if sd_a.keys() != sd_b.keys():
        missing_a = sorted(set(sd_b) - set(sd_a))
        missing_b = sorted(set(sd_a) - set(sd_b))
        raise ValueError(f"State dict key mismatch. Missing in A: {missing_a[:10]}, missing in B: {missing_b[:10]}")

    all_keys = sorted(sd_a.keys())
    trainable_keys = [k for k in all_keys if any(f".{p}." in f".{k}." or k.startswith(f"{p}.") for p in TRAINABLE_MODULE_PATTERNS)]
    frozen_keys = [k for k in all_keys if k not in trainable_keys]

    results = {
        "labels": {"a": args.label_a, "b": args.label_b},
        "paths": {"a": args.ckpt_a, "b": args.ckpt_b},
        "all": _summarize(sd_a, sd_b, all_keys),
        "trainable_stage1": _summarize(sd_a, sd_b, trainable_keys),
        "frozen_rest": _summarize(sd_a, sd_b, frozen_keys),
    }

    for prefix in TRAINABLE_MODULE_PATTERNS:
        keys = [k for k in all_keys if f".{prefix}." in f".{k}." or k.startswith(f"{prefix}.")]
        results[prefix] = _summarize(sd_a, sd_b, keys)

    ray_w_key = "ray_embed.proj.weight"
    ray_b_key = "ray_embed.proj.bias"
    if ray_w_key in sd_a:
        results["ray_embed_slices"] = {
            "weight_first_1024": _stats(sd_a[ray_w_key][:1024], sd_b[ray_w_key][:1024]),
            "weight_last_128": _stats(sd_a[ray_w_key][1024:], sd_b[ray_w_key][1024:]),
            "bias_first_1024": _stats(sd_a[ray_b_key][:1024], sd_b[ray_b_key][:1024]),
            "bias_last_128": _stats(sd_a[ray_b_key][1024:], sd_b[ray_b_key][1024:]),
        }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")

    print("Wrote:", out_path)
    print()
    for name in ["all", "trainable_stage1", "ray_embed", "cross_view_fusion", "aux_point_head", "frozen_rest"]:
        block = results[name]
        print(
            f"[{name}] "
            f"l2_diff={block['l2_diff']:.6f} "
            f"rel_l2_vs_a={block['rel_l2_vs_a']:.6f} "
            f"cosine={block['cosine']:.9f} "
            f"max_abs_diff={block['max_abs_diff']:.6f}"
        )
        top = block["top_params_by_l2_diff"][:3]
        for item in top:
            print(
                f"  - {item['name']}: "
                f"l2_diff={item['l2_diff']:.6f}, "
                f"rel={item['rel_l2_vs_a']:.6f}, "
                f"max_abs={item['max_abs_diff']:.6f}"
            )
        print()

    if "ray_embed_slices" in results:
        print("[ray_embed_slices]")
        for name, block in results["ray_embed_slices"].items():
            print(
                f"  - {name}: "
                f"l2_diff={block['l2_diff']:.6f}, "
                f"rel_l2_vs_a={block['rel_l2_vs_a']:.6f}, "
                f"cosine={block['cosine']:.9f}, "
                f"max_abs_diff={block['max_abs_diff']:.6f}"
            )


if __name__ == "__main__":
    main()
