import dataclasses


@dataclasses.dataclass(frozen=True)
class AuxPointHeadConfig:
    """Config for the auxiliary per-patch (xy, z) head used for pi3x distillation.

    The head taps post-cross-view-fusion, pre-projector vision tokens and predicts
    camera-frame ray direction (xy) and log-depth (z) per patch, mirroring pi3x's
    `point_decoder` + `point_head` at patch resolution (no ConvHead upsampling).
    """

    enabled: bool = False
    in_dim: int = 1152  # SigLIP hidden size (pre-projector tap).
    hidden_dim: int = 512
    depth: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    rope_freq: int = 100  # <= 0 disables 2D RoPE in the head.
    qk_norm: bool = True
    init_values: float = 0.01
    loss_weight: float = 1.0  # scalar weight on the point distillation loss.
    # Backward-compatible aliases used by older configs. "legacy_conf_mse"
    # corresponds to hard confidence masking with legacy_conf_threshold.
    loss_type: str | None = None
    legacy_conf_threshold: float | None = None
    # How Pi3X confidence controls per-patch supervision:
    # - "hard": keep only conf > threshold, all surviving patches weight equally.
    # - "soft": weight every patch by conf.
    # - "hybrid": keep only conf > threshold, then weight surviving patches by conf.
    conf_weight_mode: str = "soft"
    conf_threshold: float = 0.1
    # 16 -> patch-level prediction (matches the avg-pooled cache; cheap).
    # 224 -> Pi3X-matched full-resolution prediction with a ConvHead-style upsampler
    # (heavier; pair with `cache_pi3x_targets.py --output-resolution 224`).
    output_resolution: int = 16

    def __post_init__(self) -> None:
        if self.loss_type is None:
            return
        if self.loss_type != "legacy_conf_mse":
            raise ValueError(f"Unsupported AuxPointHeadConfig.loss_type={self.loss_type!r}")
        object.__setattr__(self, "conf_weight_mode", "hard")
        if self.legacy_conf_threshold is not None:
            object.__setattr__(self, "conf_threshold", self.legacy_conf_threshold)
