import dataclasses

@dataclasses.dataclass(frozen=True)
class CrossViewFusionConfig:
    enabled: bool = False
    aa_order: str = "fg"
    num_heads: int = 8
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    rope_freq: int = 100
    init_values: float = 0.01