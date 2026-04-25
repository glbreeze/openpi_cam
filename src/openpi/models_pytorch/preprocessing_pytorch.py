from collections.abc import Sequence
import logging

import torch

from openpi.shared import image_tools

logger = logging.getLogger("openpi")

# Constants moved from model.py
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)

_IMAGE_TO_CAMERA_FIELDS = {
    "base_0_rgb": ("agent_intrinsic", "agent_extrinsic"),
    "left_wrist_0_rgb": ("wrist_intrinsic", "wrist_extrinsic"),
    "right_wrist_0_rgb": (None, None),
}


class SimpleProcessedObservation:
    """Lightweight observation container used by the PyTorch path."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _clone_optional_tensor(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.clone()
    return value


def _camera_field_names(image_key: str) -> tuple[str | None, str | None]:
    return _IMAGE_TO_CAMERA_FIELDS.get(image_key, (None, None))


def _intrinsic_dtype_device(intrinsic: torch.Tensor | None, image: torch.Tensor) -> tuple[torch.dtype, torch.device]:
    if intrinsic is not None:
        return intrinsic.dtype, intrinsic.device
    if image.dtype.is_floating_point:
        return image.dtype, image.device
    return torch.float32, image.device


def _scalar_tensor(value, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device).reshape(())


def _make_pixel_transform(
    row0: tuple[float | int | torch.Tensor, float | int | torch.Tensor, float | int | torch.Tensor],
    row1: tuple[float | int | torch.Tensor, float | int | torch.Tensor, float | int | torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.stack(
        [
            torch.stack([_scalar_tensor(v, dtype=dtype, device=device) for v in row0]),
            torch.stack([_scalar_tensor(v, dtype=dtype, device=device) for v in row1]),
            torch.stack(
                [
                    _scalar_tensor(0.0, dtype=dtype, device=device),
                    _scalar_tensor(0.0, dtype=dtype, device=device),
                    _scalar_tensor(1.0, dtype=dtype, device=device),
                ]
            ),
        ]
    )


def _apply_pixel_transform_to_intrinsic(
    intrinsic: torch.Tensor | None, pixel_transform: torch.Tensor
) -> torch.Tensor | None:
    if intrinsic is None:
        return None
    transform = pixel_transform.to(device=intrinsic.device, dtype=intrinsic.dtype)
    if intrinsic.ndim == 2:
        return transform @ intrinsic
    return transform.unsqueeze(0) @ intrinsic


def _resize_with_pad_pixel_transform(
    cur_height: int,
    cur_width: int,
    target_height: int,
    target_width: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    ratio = max(cur_width / target_width, cur_height / target_height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    pad_h0, _ = divmod(target_height - resized_height, 2)
    pad_w0, _ = divmod(target_width - resized_width, 2)
    scale_x = resized_width / cur_width
    scale_y = resized_height / cur_height
    return _make_pixel_transform(
        (scale_x, 0.0, pad_w0),
        (0.0, scale_y, pad_h0),
        dtype=dtype,
        device=device,
    )


def _crop_resize_pixel_transform(
    start_h: torch.Tensor,
    start_w: torch.Tensor,
    input_height: int,
    input_width: int,
    crop_height: int,
    crop_width: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    scale_x = input_width / crop_width
    scale_y = input_height / crop_height
    start_h = start_h.to(dtype=dtype, device=device).reshape(())
    start_w = start_w.to(dtype=dtype, device=device).reshape(())
    return _make_pixel_transform(
        (scale_x, 0.0, -scale_x * start_w),
        (0.0, scale_y, -scale_y * start_h),
        dtype=dtype,
        device=device,
    )


def _pixel_to_normalized_transform(
    height: int, width: int, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return _make_pixel_transform(
        (2.0 / width, 0.0, (1.0 / width) - 1.0),
        (0.0, 2.0 / height, (1.0 / height) - 1.0),
        dtype=dtype,
        device=device,
    )


def _normalized_to_pixel_transform(
    height: int, width: int, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return _make_pixel_transform(
        (width / 2.0, 0.0, (width / 2.0) - 0.5),
        (0.0, height / 2.0, (height / 2.0) - 0.5),
        dtype=dtype,
        device=device,
    )


def _rotation_pixel_transform(
    height: int,
    width: int,
    angle_rad: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    angle_rad = angle_rad.to(dtype=dtype, device=device).reshape(())
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    rotation_inv = _make_pixel_transform(
        (cos_a, sin_a, 0.0),
        (-sin_a, cos_a, 0.0),
        dtype=dtype,
        device=device,
    )
    return (
        _normalized_to_pixel_transform(height, width, dtype=dtype, device=device)
        @ rotation_inv
        @ _pixel_to_normalized_transform(height, width, dtype=dtype, device=device)
    )


def preprocess_observation_pytorch(
    observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
    disable_geometric_augs: bool = False,
):
    """Torch.compile-compatible observation preprocessing with camera-geometry updates."""
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    updated_intrinsics = {
        "agent_intrinsic": _clone_optional_tensor(getattr(observation, "agent_intrinsic", None)),
        "wrist_intrinsic": _clone_optional_tensor(getattr(observation, "wrist_intrinsic", None)),
    }
    updated_extrinsics = {
        "agent_extrinsic": _clone_optional_tensor(getattr(observation, "agent_extrinsic", None)),
        "wrist_extrinsic": _clone_optional_tensor(getattr(observation, "wrist_extrinsic", None)),
    }

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        intrinsic_field, _ = _camera_field_names(key)
        intrinsic = updated_intrinsics.get(intrinsic_field) if intrinsic_field is not None else None

        is_channels_first = image.shape[1] == 3
        if is_channels_first:
            image = image.permute(0, 2, 3, 1)

        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            matrix_dtype, matrix_device = _intrinsic_dtype_device(intrinsic, image)
            intrinsic = _apply_pixel_transform_to_intrinsic(
                intrinsic,
                _resize_with_pad_pixel_transform(
                    image.shape[1],
                    image.shape[2],
                    image_resolution[0],
                    image_resolution[1],
                    dtype=matrix_dtype,
                    device=matrix_device,
                ),
            )
            image = image_tools.resize_with_pad_torch(image, *image_resolution)

        if train:
            image = image / 2.0 + 0.5

            if "wrist" not in key and not disable_geometric_augs:
                height, width = image.shape[1:3]

                crop_height = int(height * 0.95)
                crop_width = int(width * 0.95)

                max_h = height - crop_height
                max_w = width - crop_width
                if max_h > 0 and max_w > 0:
                    start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
                    start_w = torch.randint(0, max_w + 1, (1,), device=image.device)

                    matrix_dtype, matrix_device = _intrinsic_dtype_device(intrinsic, image)
                    intrinsic = _apply_pixel_transform_to_intrinsic(
                        intrinsic,
                        _crop_resize_pixel_transform(
                            start_h,
                            start_w,
                            height,
                            width,
                            crop_height,
                            crop_width,
                            dtype=matrix_dtype,
                            device=matrix_device,
                        ),
                    )

                    image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]

                image = torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)

                angle = torch.rand(1, device=image.device) * 10 - 5
                if torch.abs(angle) > 0.1:
                    angle_rad = angle * torch.pi / 180.0
                    matrix_dtype, matrix_device = _intrinsic_dtype_device(intrinsic, image)
                    intrinsic = _apply_pixel_transform_to_intrinsic(
                        intrinsic,
                        _rotation_pixel_transform(
                            height,
                            width,
                            angle_rad,
                            dtype=matrix_dtype,
                            device=matrix_device,
                        ),
                    )

                    cos_a = torch.cos(angle_rad)
                    sin_a = torch.sin(angle_rad)

                    grid_x = torch.linspace(-1, 1, width, device=image.device)
                    grid_y = torch.linspace(-1, 1, height, device=image.device)
                    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
                    grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
                    grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)
                    grid_x_rot = grid_x * cos_a - grid_y * sin_a
                    grid_y_rot = grid_x * sin_a + grid_y * cos_a
                    grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

                    image = torch.nn.functional.grid_sample(
                        image.permute(0, 3, 1, 2),
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)

            brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6
            image = image * brightness_factor

            contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8
            mean = image.mean(dim=[1, 2, 3], keepdim=True)
            image = (image - mean) * contrast_factor + mean

            saturation_factor = 0.5 + torch.rand(1, device=image.device) * 1.0
            gray = image.mean(dim=-1, keepdim=True)
            image = gray + (image - gray) * saturation_factor

            image = torch.clamp(image, 0, 1)
            image = image * 2.0 - 1.0

        if is_channels_first:
            image = image.permute(0, 3, 1, 2)

        out_images[key] = image
        if intrinsic_field is not None:
            updated_intrinsics[intrinsic_field] = intrinsic

    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        else:
            out_masks[key] = observation.image_masks[key]

    return SimpleProcessedObservation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        agent_extrinsic=updated_extrinsics["agent_extrinsic"],
        wrist_extrinsic=updated_extrinsics["wrist_extrinsic"],
        agent_intrinsic=updated_intrinsics["agent_intrinsic"],
        wrist_intrinsic=updated_intrinsics["wrist_intrinsic"],
        pi3x_target_xy=_clone_optional_tensor(getattr(observation, "pi3x_target_xy", None)),
        pi3x_target_logz=_clone_optional_tensor(getattr(observation, "pi3x_target_logz", None)),
        pi3x_target_conf=_clone_optional_tensor(getattr(observation, "pi3x_target_conf", None)),
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )
