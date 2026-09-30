#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenPI-style image augmentation for LeRobot's dataset image-transform pipeline."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torchvision.transforms.v2 import Transform

from lerobot.utils.image_utils import resize_with_pad_torch


@dataclass
class OpenPIImageAugmentationConfig:
    crop_scale: float = 0.95
    rotation_degrees: float = 5.0
    color_probability: float = 0.5
    brightness: float = 0.3
    contrast: float = 0.4
    hue: float = 0.1


def _rgb_to_hsv(images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Convert float RGB images in NCHW layout to augmax-compatible HSV."""
    red, green, blue = images.unbind(dim=1)
    value = images.max(dim=1).values
    minimum = images.min(dim=1).values
    chroma = value - minimum
    safe_chroma = chroma.clamp_min(torch.finfo(images.dtype).eps)

    hue = torch.where(
        value == red,
        torch.remainder((green - blue) / safe_chroma, 6.0),
        torch.where(value == green, (blue - red) / safe_chroma + 2.0, (red - green) / safe_chroma + 4.0),
    )
    hue = torch.where(chroma > 0, hue / 6.0, torch.zeros_like(hue))
    saturation = torch.where(value > 0, chroma / value.clamp_min(torch.finfo(images.dtype).eps), 0.0)
    return hue, saturation, value


def _hsv_to_rgb(hue: Tensor, saturation: Tensor, value: Tensor) -> Tensor:
    """Convert HSV planes to RGB using augmax's piecewise conversion."""
    offsets = torch.tensor((5.0, 3.0, 1.0), device=hue.device, dtype=hue.dtype).view(1, 3, 1, 1)
    k = torch.remainder(offsets + hue.unsqueeze(1) * 6.0, 6.0)
    factor = torch.minimum(torch.minimum(k, 4.0 - k), torch.ones_like(k)).clamp_min(0.0)
    return value.unsqueeze(1) - value.unsqueeze(1) * saturation.unsqueeze(1) * factor


def _adjust_openpi_contrast(value: Tensor, amount: Tensor) -> Tensor:
    """Apply augmax's invertible contrast curve to an HSV value channel."""
    slant = torch.tan((amount + 1.0) * (torch.pi / 4.0))
    slant_squared = slant.square()
    p1 = (slant - slant_squared) / (2.0 * (1.0 - slant_squared))
    p2 = 1.0 - p1
    lower = value / slant
    middle = slant * (value - 0.5) + 0.5
    upper = value / slant + 1.0 - 1.0 / slant
    return torch.where(value < p1, lower, torch.where(value > p2, upper, middle))


def _augment_openpi_images(images: Tensor, camera_key: str, config: OpenPIImageAugmentationConfig) -> Tensor:
    """Augment a batch of HWC float RGB images in [0, 1] with OpenPI's policy.

    Each image gets independent crop, rotation, and color parameters. Augmax
    applies brightness, contrast, and hue with the configured probability.
    Its saturation branch is a no-op because it discards the returned value.
    """
    batch_size, height, width = images.shape[:3]
    images = images.permute(0, 3, 1, 2)

    if "wrist" not in camera_key and (config.crop_scale != 1.0 or config.rotation_degrees != 0.0):
        crop_height = max(1, int(height * config.crop_scale))
        crop_width = max(1, int(width * config.crop_scale))
        angle = (
            (torch.rand(batch_size, device=images.device) * 2.0 - 1.0)
            * config.rotation_degrees
            * (torch.pi / 180.0)
        )
        offset_y = (torch.rand(batch_size, device=images.device) - 0.5) * (height - crop_height)
        offset_x = (torch.rand(batch_size, device=images.device) - 0.5) * (width - crop_width)
        cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)
        theta = torch.zeros((batch_size, 2, 3), dtype=images.dtype, device=images.device)
        theta[:, 0, 0] = (crop_width / width) * cos_angle
        theta[:, 0, 1] = -(crop_height / height) * sin_angle
        theta[:, 0, 2] = 2.0 * offset_x / width
        theta[:, 1, 0] = (crop_width / width) * sin_angle
        theta[:, 1, 1] = (crop_height / height) * cos_angle
        theta[:, 1, 2] = 2.0 * offset_y / height
        grid = F.affine_grid(theta, images.shape, align_corners=False)
        images = F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    apply_color = torch.rand(batch_size, device=images.device) < config.color_probability
    hue, saturation, value = _rgb_to_hsv(images)
    brightness = ((torch.rand(batch_size, device=images.device) * 2.0 - 1.0) * config.brightness).view(
        -1, 1, 1
    )
    value = torch.where(
        brightness < 0,
        value * (1.0 + brightness),
        value * (1.0 - brightness) + brightness,
    )
    contrast = ((torch.rand(batch_size, device=images.device) * 2.0 - 1.0) * config.contrast).view(
        -1, 1, 1
    )
    value = _adjust_openpi_contrast(value, contrast)
    hue_shift = ((torch.rand(batch_size, device=images.device) * 2.0 - 1.0) * config.hue).view(
        -1, 1, 1
    )
    hue = torch.remainder(hue + hue_shift, 1.0)
    jittered = _hsv_to_rgb(hue, saturation, value)
    images = torch.where(apply_color.view(-1, 1, 1, 1), jittered, images)

    return images.clamp(0.0, 1.0).permute(0, 2, 3, 1)


class OpenPIImageAugmentation(Transform):
    """Apply the recipe to one camera's CHW image or TCHW image history."""

    def __init__(
        self,
        crop_scale: float = 0.95,
        rotation_degrees: float = 5.0,
        color_probability: float = 0.5,
        brightness: float = 0.3,
        contrast: float = 0.4,
        hue: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = OpenPIImageAugmentationConfig(
            crop_scale=crop_scale,
            rotation_degrees=rotation_degrees,
            color_probability=color_probability,
            brightness=brightness,
            contrast=contrast,
            hue=hue,
        )

    def forward(self, image: Tensor, camera_key: str) -> Tensor:
        single_frame = image.ndim == 3
        images = image.unsqueeze(0) if single_frame else image
        images = images.to(torch.float32) / 255.0 if images.dtype == torch.uint8 else images.to(torch.float32)
        if images.shape[-2:] != (224, 224):
            images = resize_with_pad_torch(images, 224, 224)
        images = _augment_openpi_images(images.permute(0, 2, 3, 1), camera_key, self.config)
        images = images.permute(0, 3, 1, 2)
        return images[0] if single_frame else images
