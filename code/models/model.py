from typing import Optional

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

NUM_CLASSES = 11


class Net(nn.Module):
    def __init__(self, in_features: Optional[int] = None, num_classes=NUM_CLASSES):
        super().__init__()

        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
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
