from typing import Literal, Optional, Union

from torch import nn
from torchvision.models import DenseNet121_Weights, ResNet50_Weights, densenet121, resnet50

import src.constants

from .IResnet.conv_iResNet import conv_iResNet as iResNet


class Net(nn.Module):
    def __init__(
        self,
        network: Optional[Union[Literal["resnet"], Literal["densenet"]]],
        num_classes,
        in_features: Optional[int] = None,
    ):
        super().__init__()

        last_layer: nn.Module
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
                in_shape=src.constants.DEFAULT_IN_SHAPE,
                coeff=0.9,
                numTraceSamples=1,
                numSeriesTerms=1,
                n_power_iter=5,
                density_estimation=False,
                actnorm=False,
                learn_prior=False,
                nonlin="elu",
            )
            # TODO: Double-check if this is the final layer
            if not self.model.logits:
                raise TypeError(f"Type of final network layer is {type(None)}, expected {type(nn.Linear)}")

            last_layer = self.model.logits

        elif network == "resnet":
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            last_layer = self.model.fc

        elif network == "densenet":
            self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
            last_layer = self.model.classifier
        else:
            raise ValueError(
                f"Received unknown value {network} for argument 'network',"
                "expected one of: 'resnet', 'densenet', 'iresnet', 'idensenet"
            )

        if not in_features:
            in_features = last_layer.in_features

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
