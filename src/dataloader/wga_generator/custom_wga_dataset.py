import os
from pathlib import Path, PurePath
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from torchvision.transforms import functional as fn

from src.dataloader.dataloader import CustomDataset
from src.helpers.image_utils import url_to_numpy

DATA_PATH = "wga"
current_dir = Path(__file__).parent.parent
local_DATA_PATH = os.path.join(current_dir.parent.parent, "data")
wga_path = os.path.join(local_DATA_PATH, "wga")
data_chunks_path = os.path.join(wga_path, "data_chunks")


def make_wga_dataset_chunks():
    """
    Create wga data chunks and resize images.
    Resizing is done in resize_img() which accepts
    _extended_summary_
    """
    print("Starting...")

    NUMBER_OF_ARRAYS = 30
    temp = pd.read_csv(os.path.join(wga_path, "temp.csv"))
    # TODO: Fix types
    temps = np.array_split(temp, NUMBER_OF_ARRAYS)

    print("Starting data chunks")
    for i, t in enumerate(temps):
        if i <= -1:
            continue

        t["image_array"] = t["new_URL"].map(lambda url: url_to_numpy(url))  # type: ignore
        print(f"Url to npy done for chunk {i}/{NUMBER_OF_ARRAYS}")
        t.dropna(inplace=True, axis=0)  # type: ignore

        images_name = f"wga_imgs_{i}.npy"
        np.save(os.path.join(data_chunks_path, images_name), t["image_array"].values)  # type: ignore

        labels_name = f"wga_lbls_{i}.npy"
        np.save(os.path.join(data_chunks_path, labels_name), t["encoded_TYPE"].values)  # type: ignore

        print(f"Saved data chunk index: {i}/{NUMBER_OF_ARRAYS}")


def merge_wga_chunks():
    """
    Merge chunks of wga dataset into a full set of images and labels.
    """
    NUMBER_OF_ARRAYS = 30

    x = np.load(os.path.join(data_chunks_path, "wga_imgs_0.npy"), allow_pickle=True)
    print("Loaded first chunk")
    all_images = np.array([np.squeeze(xx) for xx in x])

    all_labels = np.load(os.path.join(data_chunks_path, "wga_lbls_0.npy"), allow_pickle=True)

    for i in range(1, NUMBER_OF_ARRAYS):
        x = np.load(os.path.join(data_chunks_path, f"wga_imgs_{i}.npy"), allow_pickle=True)
        print(f"Loaded chunk {i}")
        images_array = np.array([np.squeeze(xx) for xx in x])
        all_images = np.append(all_images, images_array, axis=0)

        labels_array = np.load(os.path.join(data_chunks_path, f"wga_lbls_{i}.npy"), allow_pickle=True)
        all_labels = np.append(all_labels, labels_array)
        print(f"len of all_labels = {len(all_labels)}")

    np.save(os.path.join(wga_path, "wga_images.npy"), all_images)
    np.save(os.path.join(wga_path, "wga_labels.npy"), all_labels)


class CustomWgaDataset(CustomDataset):
    def __init__(
        self,
        chosen_label="genre",
        chunk_size=1,
    ):
        super().__init__(chosen_label=chosen_label, chunk_size=chunk_size)

        self.__data_path = PurePath(local_DATA_PATH, DATA_PATH)
        self.__data: dict[Literal["images", "labels"], np.ndarray] = {
            "images": np.load(os.path.join(self.__data_path, "wga_images.npy"), mmap_mode="r"),
            "labels": np.load(os.path.join(self.__data_path, "wga_labels.npy"), mmap_mode="r"),
        }

        labels_length = len(self.__data["labels"])
        images_length = len(self.__data["images"])

        if labels_length != images_length:
            raise ValueError(f"Length of labels ({labels_length}) and images ({images_length}) is mismatched")

        self.__length = labels_length

    def __len__(self):
        return self.__length

    def __getitem__(self, key):
        # return (fn.to_tensor(self.__data["images"][key]), self.__data["labels"][key])
        return (
            fn.to_tensor(self.__data["images"][key]),
            torch.tensor(self.__data["labels"][key]),
        )
        # return torch.nested.nested_tensor([fn.to_tensor(self.__data["images"][key]), self.__data["labels"][key]])

    def __iter__(self):
        return self.generator()

    def generator(self):
        for slice_start in range(0, self.__length, self.chunk_size):
            slice_end = np.min((slice_start + self.chunk_size, self.__length))
            image_arrays = np.empty((self.chunk_size, *self.in_shape), dtype=np.uint8)

            for index, image_array in enumerate(self.__data["images"][slice_start:slice_end]):
                tensor = fn.to_tensor(image_array)
                image_arrays[index] = tensor

            labels = self.__data["labels"][slice_start:slice_end]

            if self.chunk_size == 1:
                for row in range(self.chunk_size):
                    # yield (torch.tensor(image_arrays[row]), labels[row])
                    yield (torch.tensor(image_arrays[row]), torch.tensor(labels[row]))
                    # yield torch.nested.nested_tensor([torch.tensor(image_arrays[row]), labels[row]])

            # yield (image_arrays, np.asarray(labels))
            yield (image_arrays, torch.tensor(labels))
            # yield torch.nested.nested_tensor([image_arrays, labels])

    def slice(self, start: int = 0, stop: Optional[int] = None, step: int = 1):
        self.__data["images"] = self.__data["images"][start:stop:step]
        self.__data["labels"] = self.__data["labels"][start:stop:step]

        labels_length = len(self.__data["labels"])
        images_length = len(self.__data["images"])

        if labels_length != images_length:
            raise ValueError(f"New length of labels ({labels_length}) and images ({images_length}) is mismatched")

        self.__length = labels_length

        return self
