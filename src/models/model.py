from typing import Literal, Optional, Union

import torch
import torch.utils.data
from torch import nn
from torchvision.models import DenseNet121_Weights, ResNet50_Weights, densenet121, resnet50

import src.constants

from .IResnet.conv_iResNet import conv_iResNet as iResNet


class Net(nn.Module):
    def __init__(
        self,
        network: Optional[Union[Literal["resnet"], Literal["densenet"], Literal["iresnet"], Literal["idensenet"]]],
        num_classes,
        in_shape: tuple[int, ...] = src.constants.DEFAULT_IN_SHAPE,
        in_features: Optional[int] = None,
    ):
        super().__init__()

        self.__last_layer_name: str
        self.__device = torch.device("cpu")
        # in get_model(), CIFAR_main.py
        # default values from argument parser definition
        if network == "iresnet":
            self.model = iResNet(
                nBlocks=[4, 4, 4],
                nStrides=[1, 2, 2],
                nChannels=[16, 64, 256],
                nClasses=num_classes,
                init_ds=1,
                inj_pad=0,
                in_shape=in_shape,
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

            self.__last_layer_name = "logits"

        elif network == "resnet":
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.__last_layer_name = "fc"

        elif network == "densenet":
            self.model = densenet121(weights=DenseNet121_Weights.DEFAULT, in_shape=in_shape)
            self.__last_layer_name = "classifier"
        else:
            raise ValueError(
                f"Received unknown value {network} for argument 'network',"
                "expected one of: 'resnet', 'densenet', 'iresnet', 'idensenet"
            )

        in_features = in_features or getattr(self.model, self.__last_layer_name).in_features

        setattr(self.model, self.__last_layer_name, nn.Linear(in_features, num_classes))

    def freeze(self, freeze=True):
        if not freeze:
            return self
        for name, parameter in self.model.named_parameters():
            if self.__last_layer_name not in name:
                parameter.requires_grad = False

        return self

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, tuple):
            return out[0]
        return out

    def init_actnorm(self, loader: torch.utils.data.DataLoader):
        if not isinstance(self.model, iResNet):
            raise Exception("Only initialize actnorm parameters for iResNet model.")

        batch = next(iter(loader))[0].to(self.__device)

        print("Initializing actnorm parameters...")
        with torch.no_grad():
            self.model(batch, ignore_logdet=True)

        print("Initialized.")

    def to(self, device: torch.device):
        if not isinstance(self.model, iResNet):
            self.model.to(device)
        self.model.cuda()

        self.__device = device

        return self
