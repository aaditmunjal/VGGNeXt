# This is the base VGG model developed by Karen Simonyan & Andrew Zisserman with a few changes:
# 1) 7x7 adaptive average pooling after the convolution layers - makes the model input-size agnostic
# 2) Batch Normalization
# 3) Kaiming initialization

# This model will act as the baseline for experiments conducted in this project
# It is directly based on the code provided by PyTorch (TorchVision)

import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast


class LayerNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)   # to NHWC
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)   # back to NCHW
        return x

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, GELU: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                if GELU:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.GELU()]
                else:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                if GELU:
                    layers += [conv2d, LayerNorm(v), nn.GELU()]
                else:
                    layers += [conv2d, LayerNorm(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, "M"],
    "C": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, 256, 256, "M", 512, 512, 512, "M"],
}


def _vgg(cfg: str, batch_norm: bool, GELU: bool, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, GELU=GELU), **kwargs)
    return model

def vgg16_bn(**kwargs: Any) -> VGG:
    return _vgg("A", True, False, **kwargs)

def vgg16_bn_stage_ratio(**kwargs: Any) -> VGG:
    return _vgg("B", True, False, **kwargs)

def vgg16_bn_four_stage(**kwargs: Any) -> VGG:
    return _vgg("C", True, False, **kwargs)

def vgg16_bn_GELU(**kwargs: Any) -> VGG:
    return _vgg("C", True, True, **kwargs)

