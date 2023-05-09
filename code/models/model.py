from typing import Optional, Literal, Union

import torch.nn as nn
from torchvision.models import (
    ResNet50_Weights,
    resnet50,
    DenseNet121_Weights,
    densenet121,
)

NUM_CLASSES = 11


class Net(nn.Module):
    def __init__(
        self,
        network=Optional[Union[Literal["resnet"], Literal["densenet"]]],
        in_features: Optional[int] = None,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()

        if network == "resnet":
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif network == "densenet":
            self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        in_features = in_features or self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def freeze(self, freeze=True):
        if not freeze:
            return self
        for name, parameter in self.model.named_parameters():
            if "fc" not in name:
                parameter.requires_grad = False

        return self

    def forward(self, x):
        return self.model(x)
