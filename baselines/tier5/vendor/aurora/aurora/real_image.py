"""Device bridge for Aurora's Hugging Face real-image processor."""

from __future__ import annotations

import torch


def process_real_images(processor, images: torch.Tensor) -> torch.Tensor:
    """Run the CPU image processor and return pixels to the caller's device."""
    device = images.device
    cpu_images = images.detach().cpu()
    pixels = processor(images=cpu_images, return_tensors="pt")["pixel_values"]
    return pixels.to(device)
