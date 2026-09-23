from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


# Dimensionality of the representation produced at each extraction point.
FEATURE_DIMS = {
    "stem": 64,
    "layer1": 256,
    "layer2": 512,
    "layer3": 1024,
    "layer4": 2048,
}


def get_default_device() -> torch.device:
    """Return the best available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class ResNet50Extractor(nn.Module):
    """
    Frozen ImageNet-pretrained ResNet-50 feature extractor.

    Available representations:

        stem   ->   64 dimensions
        layer1 ->  256 dimensions
        layer2 ->  512 dimensions
        layer3 -> 1024 dimensions
        layer4 -> 2048 dimensions

    Spatial feature maps are converted to vectors using
    global average pooling.
    """

    def __init__(
        self,
        layers: Iterable[str] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        # Extract every representation unless specific layers are requested.
        if layers is None:
            layers = FEATURE_DIMS.keys()

        self.layers = tuple(layers)

        invalid_layers = set(self.layers) - set(FEATURE_DIMS)

        if invalid_layers:
            raise ValueError(
                f"Unknown layers: {sorted(invalid_layers)}. "
                f"Valid layers are: {list(FEATURE_DIMS)}"
            )

        # Select CUDA, MPS, or CPU automatically.
        if device is None:
            device = get_default_device()

        self.device = torch.device(device)

        # Official torchvision ImageNet-pretrained weights.
        weights = ResNet50_Weights.DEFAULT

        # Save the exact preprocessing associated with those weights.
        self.preprocessing = weights.transforms()

        # Load pretrained ResNet-50.
        self.model = resnet50(weights=weights)

        # We are only extracting features.
        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.to(self.device)

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        """
        Convert:

            B x C x H x W

        into:

            B x C

        using global average pooling.
        """
        return F.adaptive_avg_pool2d(x, output_size=1).flatten(1)

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        x = x.to(self.device)

        features: dict[str, torch.Tensor] = {}

        # ---------------------------------------------------------
        # Stem
        # ---------------------------------------------------------

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        if "stem" in self.layers:
            features["stem"] = self._pool(x)

        # ---------------------------------------------------------
        # Residual stages
        # ---------------------------------------------------------

        x = self.model.layer1(x)

        if "layer1" in self.layers:
            features["layer1"] = self._pool(x)

        x = self.model.layer2(x)

        if "layer2" in self.layers:
            features["layer2"] = self._pool(x)

        x = self.model.layer3(x)

        if "layer3" in self.layers:
            features["layer3"] = self._pool(x)

        x = self.model.layer4(x)

        if "layer4" in self.layers:
            features["layer4"] = self._pool(x)

        return features