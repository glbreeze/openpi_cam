import dataclasses


@dataclasses.dataclass(frozen=True)
class CrossViewFusionConfig:
    type: str = "none"  # "none" | "simple" | "standard"
    aa_order: str = "fg"
    num_heads: int = 8
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    rope_freq: int = 100  # <= 0 disables RoPE.
    init_values: float = 0.01
