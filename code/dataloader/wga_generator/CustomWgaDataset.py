import os
from code.dataloader.dataloader import CustomDataset
from pathlib import PurePath
from typing import Literal

import numpy as np
import torch


class CustomWgaDataset(CustomDataset):
    def __init__(
        self,
        WGA_DATA_PATH="wga",
        chosen_label="genre",
        chunk_size=1,
    ):
        super().__init__(chosen_label=chosen_label, chunk_size=chunk_size)

        self.data_path = PurePath(self.data_path, WGA_DATA_PATH)
        self.data: dict[Literal["images", "labels"], np.ndarray] = {
            "images": np.load(os.path.join(self.data_path, "wga_images.npy"), mmap_mode="r"),
            "labels": np.load(os.path.join(self.data_path, "wga_labels.npy")),
        }

        self.length = len(self.data["labels"])

    def __len__(self):
        return self.length

    def __getitem__(self, raw_row_id):
        return (torch.tensor(self.data["images"][raw_row_id]).permute(2, 0, 1), (self.data["labels"][raw_row_id]))

    def __iter__(self):
        return self.generator()

    def generator(self):
        for slice_start in range(0, self.length, self.chunk_size):
            slice_end = np.min((slice_start + self.chunk_size, self.length))
            image_arrays = np.empty((self.chunk_size, *self.in_shape), dtype=np.uint8)

            for index, image_array in enumerate(self.data["images"][slice_start:slice_end]):
                tensor = torch.tensor(image_array).permute(2, 0, 1)
                image_arrays[index] = tensor

            labels = self.data["labels"][slice_start:slice_end]

            if self.chunk_size == 1:
                for row in range(self.chunk_size):
                    yield (image_arrays[row], labels[row])

            yield (image_arrays, np.asarray(labels))
