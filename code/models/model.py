from typing import Optional, Literal, Union

import torch.nn as nn
from torchvision.models import (
    ResNet50_Weights,
    resnet50,
    DenseNet121_Weights,
    densenet121,
)

from models.IResnet.conv_iResNet import conv_iResNet as iResNet
from constants import IN_SHAPE


class Net(nn.Module):
    def __init__(
        self,
        network: Optional[Union[Literal["resnet"], Literal["densenet"]]],
        num_classes,
        in_features: Optional[int] = None,
    ):
        super().__init__()

        # in get_model(), CIFAR_main.py
        # default values from argument parser definition
        if network == "iresnet":
            self.model = iResNet(
                nBlocks=[4, 4, 4],
                nStrides=[1, 2, 2],
                nChannels=[16, 64, 256],
                nClasses=num_classes,
                init_ds=2,
                inj_pad=0,
                in_shape=IN_SHAPE,
                coeff=0.9,
                numTraceSamples=1,
                numSeriesTerms=1,
                n_power_iter=5,
                density_estimation=False,
                actnorm=False,
                learn_prior=False,
                nonlin="elu",
            )

        elif network == "resnet":
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
