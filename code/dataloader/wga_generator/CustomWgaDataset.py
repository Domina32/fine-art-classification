import os
from code.dataloader.dataloader import CustomDataset
from pathlib import Path, PurePath
from typing import Literal, Optional

import numpy as np
import torch
from torchvision.transforms import functional as fn

DATA_PATH = "wga"
current_dir = Path(__file__).parent.parent
local_DATA_PATH = os.path.join(current_dir.parent.parent, "data")


class CustomWgaDataset(CustomDataset):
    def __init__(
        self,
        chosen_label="genre",
        chunk_size=1,
    ):
        super().__init__(chosen_label=chosen_label, chunk_size=chunk_size)

        self.__data_path = PurePath(local_DATA_PATH, DATA_PATH)
        self.__data: dict[Literal["images", "labels"], np.ndarray] = {
            "images": np.load(
                os.path.join(self.__data_path, "wga_images.npy"), mmap_mode="r"
            ),
            "labels": np.load(
                os.path.join(self.__data_path, "wga_labels.npy"), mmap_mode="r"
            ),
        }

        labels_length = len(self.__data["labels"])
        images_length = len(self.__data["images"])

        if labels_length != images_length:
            raise ValueError(
                f"Length of labels ({labels_length}) and images ({images_length}) is mismatched"
            )

        self.__length = labels_length

    def __len__(self):
        return self.__length

    def __getitem__(self, key):
        return (fn.to_tensor(self.__data["images"][key]), self.__data["labels"][key])

    def __iter__(self):
        return self.generator()

    def generator(self):
        for slice_start in range(0, self.__length, self.chunk_size):
            slice_end = np.min((slice_start + self.chunk_size, self.__length))
            image_arrays = np.empty((self.chunk_size, *self.in_shape), dtype=np.uint8)

            for index, image_array in enumerate(
                self.__data["images"][slice_start:slice_end]
            ):
                tensor = fn.to_tensor(image_array)
                image_arrays[index] = tensor

            labels = self.__data["labels"][slice_start:slice_end]

            if self.chunk_size == 1:
                for row in range(self.chunk_size):
                    yield (torch.tensor(image_arrays[row]), labels[row])

            yield (image_arrays, np.asarray(labels))

    def slice(self, start: int = 0, stop: Optional[int] = None, step: int = 1):
        self.__data["images"] = self.__data["images"][start:stop:step]
        self.__data["labels"] = self.__data["labels"][start:stop:step]

        labels_length = len(self.__data["labels"])
        images_length = len(self.__data["images"])

        if labels_length != images_length:
            raise ValueError(
                f"New length of labels ({labels_length}) and images ({images_length}) is mismatched"
            )

        self.__length = labels_length

        return self
