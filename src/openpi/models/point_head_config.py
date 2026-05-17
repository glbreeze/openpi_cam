import dataclasses
import typing


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
    loss_type: typing.Literal[
        "pi3x_local_pointmap",
        "legacy_conf_mse",
    ] = "pi3x_local_pointmap"
    # Original full-res best-recipe loss: hard confidence gate, then L_p on xy/logz.
    # `legacy_conf_loss_order` selects p: 2 = MSE (original), 1 = L1 (more robust to
    # noisy teacher predictions, especially at depth discontinuities).
    legacy_conf_threshold: float = 0.1
    legacy_conf_loss_order: int = 2
    # Pi3X-style scale-invariant local pointmap supervision: exp(log_z),
    # build local points (xy * z, z), align one scale per sample, then supervise rays
    # and aligned depth.
    ray_loss_weight: float = 1.0
    depth_loss_weight: float = 1.0
    depth_weight_min_frac: float = 0.1
    # Per-pixel weight applied to the depth loss (and scale-alignment).
    # "pi3x_inverse" -> 1 / clamp(target_depth, depth_weight_min_frac * mean_depth);
    #                   Pi3X-faithful, makes the loss roughly equivalent to relative
    #                   depth error. Tuned for natural scenes with wide depth range.
    # "uniform"      -> all-ones; appropriate for tabletop scenes where the depth
    #                   range is narrow (~5x near:far) so inverse-depth weighting is
    #                   nearly a no-op and only injects teacher-depth bias.
    depth_weighting: typing.Literal["pi3x_inverse", "uniform"] = "pi3x_inverse"
    # Pi3X aligns scale on a fixed-size sampled point set. <= 0 uses all points.
    scale_align_num_points: int = 4096
    # 16 -> patch-level prediction (matches the avg-pooled cache; cheap).
    # 224 -> Pi3X-matched full-resolution prediction with a ConvHead-style upsampler
    # (heavier; pair with `cache_pi3x_targets.py --output-resolution 224`).
    output_resolution: int = 16
