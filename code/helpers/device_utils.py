from typing import Literal, Union

import torch_directml
from torch import cuda, device


class Device:
    def __init__(self, preferred: Union[Literal["cpu"], Literal["gpu"]] = "gpu"):
        if preferred == "cpu":
            self.__device = "cpu"
        elif torch_directml.is_available():
            self.__device = torch_directml.device()
        elif cuda.is_available():
            self.__device = cuda.get_device_name()
        else:
            self.__device = "cpu"

    def get_device(self):
        return device(self.__device)
