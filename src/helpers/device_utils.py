from typing import Literal, Union

import torch
import torch.backends.mps

try:
    import torch_directml  # type: ignore

    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False
    print("Couldn't import directml... Ignoring...")


class Device:
    def __init__(self, preferred: Union[Literal["cpu"], Literal["gpu"]] = "gpu"):
        if preferred == "cpu":
            self.__device = "cpu"
        elif HAS_DIRECTML and torch_directml.is_available():
            self.__device = torch_directml.device()
        elif torch.cuda.is_available():
            self.__device = "torch.cuda"
        else:
            try:
                if torch.backends.mps.is_available():
                    self.__device = "mps"
                    return

                if not torch.backends.mps.is_built():
                    print("MPS not available because the current PyTorch install was not built with MPS enabled.")
                else:
                    print(
                        "MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine."
                    )
            except AttributeError:
                print("Apple metal isn't available... Ignoring...")

            self.__device = "cpu"

    def get_device(self):
        return torch.device(self.__device)
