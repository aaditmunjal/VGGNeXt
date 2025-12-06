import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast
from torchvision.ops import StochasticDepth

class ResidualBlockV2(nn.Module):
    def __init__(self, in_channels, drop_prob: float = 0.0):
        super().__init__()
        self.pre_norm = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        )
        self.stochastic_depth = StochasticDepth(drop_prob, "row")

    def forward(self, x):
        identity = x
        out = self.pre_norm(x)
        out = self.conv_block(out)
        out = self.stochastic_depth(out)
        out = identity + out 
        return out
    

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.GELU(),
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
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg: List[Union[str, int]]) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if type(v) == str and v.startswith("C"):
            v = int(v[1:])
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=2, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v)]
            in_channels = v
        elif v == "R2":
            layers.append(ResidualBlockV2(in_channels, drop_prob=0.1))
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": ["C64", "R2", "C128", "R2", "C320", "R2", "R2", "R2", "R2", "R2", "R2", "C512", "R2"]
}


def _vgg(cfg: str, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg]), **kwargs)
    return model

def VGGNeXt(**kwargs: Any) -> VGG:
    return _vgg("A", **kwargs)