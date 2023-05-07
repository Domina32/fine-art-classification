from typing import Literal, Union
from torch import cuda, device

try:
    import torch_directml
    has_directml = True
except ImportError:
    has_directml = False
    print("Couldn't import directml... Ignoring...")

class Device:
    def __init__(self, preferred: Union[Literal["cpu"], Literal["gpu"]] = "gpu"):
        if preferred == "cpu":
            self.__device = "cpu"
        elif has_directml and torch_directml.is_available():
            self.__device = torch_directml.device()
        elif cuda.is_available():
            self.__device = "cuda"
        else:
            self.__device = "cpu"

    def get_device(self):
        return device(self.__device)
